import time
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import nn


class SpectralAdapter(nn.Module):
    """Applies frequency-domain weighting via FFT/iFFT with optional profiling."""

    def __init__(
        self,
        enabled: bool,
        weighting: str = "none",
        normalize: bool = True,
        bandpass_inner: float = 0.1,
        bandpass_outer: float = 0.6,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.weighting = weighting
        self.normalize = normalize
        self.bandpass_inner = bandpass_inner
        self.bandpass_outer = bandpass_outer

        self._total_calls: int = 0
        self._total_time: float = 0.0
        self._use_cuda_timer = torch.cuda.is_available()
        if self._use_cuda_timer:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
        else:
            self._start_evt = None
            self._end_evt = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        cuda_timing = self._use_cuda_timer and x.device.type == "cuda"
        if cuda_timing:
            self._start_evt.record()
        else:
            start = time.perf_counter()

        norm = "ortho" if self.normalize else None
        x_fft = torch.fft.rfft2(x, norm=norm)
        weight = self._weight(x_fft.shape[-2], x_fft.shape[-1], x.device)
        x_fft = x_fft * weight
        out = torch.fft.irfft2(x_fft, s=x.shape[-2:], norm=norm)

        if cuda_timing:
            self._end_evt.record()
            torch.cuda.synchronize()
            elapsed = self._start_evt.elapsed_time(self._end_evt) / 1000.0
        else:
            elapsed = time.perf_counter() - start

        self._total_calls += 1
        self._total_time += elapsed
        return out

    def stats(self) -> dict:
        return {
            "spectral_calls": float(self._total_calls),
            "spectral_time_seconds": float(self._total_time),
        }

    def reset_stats(self) -> None:
        self._total_calls = 0
        self._total_time = 0.0
