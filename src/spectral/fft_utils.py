from typing import Dict

import torch


def fft_transform(batch: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Apply a 2D FFT to each channel in an NCHW tensor.

    Returns a tensor with real/imag components stacked in the last dimension.
    Shape: [B, C, H, W, 2]
    """
    norm = "ortho" if normalize else None
    x_fft = torch.fft.fft2(batch, norm=norm)
    return torch.stack([x_fft.real, x_fft.imag], dim=-1)


def inverse_fft_transform(batch: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Invert the stacked real/imag representation produced by `fft_transform`.
    Expects shape [B, C, H, W, 2]; returns real part of the inverse FFT.
    """
    norm = "ortho" if normalize else None
    x_complex = torch.complex(batch[..., 0], batch[..., 1])
    x_rec = torch.fft.ifft2(x_complex, norm=norm)
    return x_rec.real


def configure_spectral_params(config: Dict) -> Dict:
    """Extract spectral configuration flags."""
    spectral_cfg = config.get("spectral", {}) if config else {}
    apply_to = spectral_cfg.get("apply_to", ["input", "output"])
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    return {
        "enabled": spectral_cfg.get("enabled", False),
        "normalize": spectral_cfg.get("normalize", True),
        "weighting": spectral_cfg.get("weighting", "none"),
        "apply_to": list(apply_to),
        "per_block": spectral_cfg.get("per_block", False),
        "bandpass_inner": spectral_cfg.get("bandpass_inner", 0.1),
        "bandpass_outer": spectral_cfg.get("bandpass_outer", 0.6),
    }


def apply_weight_map(batch: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    """
    Apply a broadcastable weight map to the FFT result.

    `batch` should be [B, C, H, W, 2]; `weight_map` should broadcast to [B, C, H, W].
    """
    if weight_map.ndim == 2:
        weight_map = weight_map[None, None, :, :]
    elif weight_map.ndim == 3:
        weight_map = weight_map[None, :, :, :]
    weight_map = weight_map.to(batch.device)
    return batch * weight_map[..., None]
