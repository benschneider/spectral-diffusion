import time
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SpectralAdapter(nn.Module):
    """Applies frequency-domain weighting via FFT/iFFT with optional profiling."""

    def __init__(
        self,
        enabled: bool,
        weighting: str = "none",
        normalize: bool = True,
        bandpass_inner: float = 0.1,
        bandpass_outer: float = 0.6,
        learnable: bool = False,
        condition_dim: int = 0,
        mlp_hidden_dim: int = 64,
        learnable_mode: str = "bandpass",
        learnable_temperature: float = 50.0,
        learnable_gain_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.weighting = weighting
        self.normalize = normalize
        self.bandpass_inner = bandpass_inner
        self.bandpass_outer = bandpass_outer
        self.learnable = learnable
        self.learnable_mode = learnable_mode
        self.learnable_temperature = learnable_temperature
        self.condition_dim = condition_dim if learnable else 0

        self._total_calls: int = 0
        self._total_time: float = 0.0
        self._cpu_time: float = 0.0
        self._cuda_time: float = 0.0
        self._use_cuda_timer = torch.cuda.is_available()
        if self._use_cuda_timer:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
        else:
            self._start_evt = None
            self._end_evt = None

        if self.learnable:
            base_init = torch.tensor(
                [bandpass_inner, bandpass_outer, learnable_gain_init], dtype=torch.float32
            )
            self.base_params = nn.Parameter(base_init)
            if self.condition_dim > 0:
                self.mlp = nn.Sequential(
                    nn.Linear(self.condition_dim, mlp_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(mlp_hidden_dim, 3),
                )
            else:
                self.mlp = None
        else:
            self.register_parameter("base_params", None)
            self.mlp = None

    def _weight(self, h: int, w_fft: int, device: torch.device) -> torch.Tensor:
        key = (
            h,
            w_fft,
            self.weighting,
            self.bandpass_inner,
            self.bandpass_outer,
            device.type,
        )
        weight = self._weight_cache(key)
        if weight.device != device:
            weight = weight.to(device)
        return weight

    @lru_cache(maxsize=16)
    def _weight_cache(
        self, key: Tuple[int, int, str, float, float, str]
    ) -> torch.Tensor:
        h, w_fft, weighting, inner, outer, device_type = key
        device = torch.device(device_type)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w_fft, device=device),
            indexing="ij",
        )
        radius = torch.sqrt(xx**2 + yy**2).clamp_(0.0, 1.0)
        if weighting == "radial":
            weight = radius
        elif weighting == "bandpass":
            interior = torch.sigmoid(50.0 * (radius - inner))
            exterior = 1.0 - torch.sigmoid(50.0 * (radius - outer))
            weight = interior * exterior
        else:
            weight = torch.ones_like(radius)
        return weight.to(torch.float32)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return x
        cuda_timing = self._use_cuda_timer and x.device.type == "cuda"
        if cuda_timing:
            self._start_evt.record()
        else:
            start = time.perf_counter()

        norm = "ortho" if self.normalize else None
        x_fft = torch.fft.rfft2(x, norm=norm)
        if self.learnable:
            params = self._predict_learnable_params(condition, x_fft.shape[0], x.device)
            weight = self._dynamic_weight(
                x_fft.shape[-2],
                x_fft.shape[-1],
                x.device,
                params,
            )
        else:
            weight = self._weight(x_fft.shape[-2], x_fft.shape[-1], x.device)
        x_fft = x_fft * weight
        out = torch.fft.irfft2(x_fft, s=x.shape[-2:], norm=norm)

        if cuda_timing:
            self._end_evt.record()
            torch.cuda.synchronize()
            elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000.0
            self._cuda_time += elapsed
        else:
            elapsed = time.perf_counter() - start
            self._cpu_time += elapsed

        self._total_calls += 1
        self._total_time += elapsed
        return out

    def _predict_learnable_params(
        self, condition: Optional[torch.Tensor], batch: int, device: torch.device
    ) -> torch.Tensor:
        base = torch.tanh(self.base_params).to(device)
        base = base.unsqueeze(0).expand(batch, -1)
        if self.mlp is not None and condition is not None:
            cond = condition
            if cond.dim() > 2:
                cond = cond.view(cond.shape[0], -1)
            cond = cond.to(device)
            delta = self.mlp(cond)
            base = base + delta
        inner = torch.sigmoid(base[:, 0]).clamp(0.0, 0.95)
        outer = torch.sigmoid(base[:, 1]).clamp(0.0, 0.99)
        outer = torch.maximum(outer, inner + 0.02)
        gain = F.softplus(base[:, 2]) + 1e-6
        return torch.stack([inner, outer, gain], dim=1)

    @lru_cache(maxsize=32)
    def _radius_cache(self, key: Tuple[int, int, str]) -> torch.Tensor:
        h, w_fft, device_type = key
        device = torch.device(device_type)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w_fft, device=device),
            indexing="ij",
        )
        radius = torch.sqrt(xx**2 + yy**2).clamp_(0.0, 1.0)
        return radius

    def _dynamic_weight(
        self, h: int, w_fft: int, device: torch.device, params: torch.Tensor
    ) -> torch.Tensor:
        radius = self._radius_cache((h, w_fft, device.type)).to(device)
        temperature = self.learnable_temperature
        interior = torch.sigmoid(temperature * (radius[None, :, :] - params[:, 0].view(-1, 1, 1)))
        exterior = 1.0 - torch.sigmoid(
            temperature * (radius[None, :, :] - params[:, 1].view(-1, 1, 1))
        )
        band = interior * exterior
        gain = params[:, 2].view(-1, 1, 1)
        weight = gain * band
        weight = torch.clamp(weight, min=0.0)
        return weight.unsqueeze(1)

    def stats(self) -> dict:
        return {
            "spectral_calls": float(self._total_calls),
            "spectral_time_seconds": float(self._total_time),
            "spectral_cpu_time_seconds": float(self._cpu_time),
            "spectral_cuda_time_seconds": float(self._cuda_time),
        }

    def reset_stats(self) -> None:
        self._total_calls = 0
        self._total_time = 0.0
        self._cpu_time = 0.0
        self._cuda_time = 0.0
