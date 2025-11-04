from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # pragma: no cover - Pillow is optional
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


@dataclass
class SyntheticSpectralConfig:
    size: int = 50_000
    image_size: int = 32
    channels: int = 3
    freq_mix: float = 0.5
    color_mix: float = 0.2
    use_text: bool = True
    include_gratings: bool = True
    include_shapes: bool = True
    log_fft_energy: bool = False
    seed: int = 0


class SyntheticSpectralDataset(Dataset):
    """Procedural dataset mixing textures, shapes and gratings."""

    def __init__(
        self,
        size: int = 50_000,
        image_size: int = 32,
        channels: int = 3,
        freq_mix: float = 0.5,
        color_mix: float = 0.2,
        use_text: bool = True,
        include_gratings: bool = True,
        include_shapes: bool = True,
        log_fft_energy: bool = False,
        seed: int = 0,
    ) -> None:
        if channels != 3:
            raise ValueError("SyntheticSpectralDataset currently supports only 3 channels (RGB)")
        self.size = int(size)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.freq_mix = float(freq_mix)
        self.color_mix = float(color_mix)
        self.use_text = bool(use_text) and Image is not None
        self.include_gratings = bool(include_gratings)
        self.include_shapes = bool(include_shapes)
        self.log_fft_energy = bool(log_fft_energy)
        self.seed = int(seed)

        self.device = torch.device("cpu")
        self.dtype = torch.float32

        coords = torch.linspace(-1.0, 1.0, self.image_size, dtype=self.dtype, device=self.device)
        self.yy, self.xx = torch.meshgrid(coords, coords, indexing="ij")
        freq = torch.fft.fftfreq(self.image_size, d=1.0, device=self.device)
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        self.radius = torch.sqrt(fx**2 + fy**2)
        self.radius[0, 0] = 1.0
        self.eps = 1e-5

        self._radial_bins = torch.clamp((self.radius * (self.image_size // 2)).round().to(torch.int64), min=0)
        self._num_radial_bins = int(self._radial_bins.max().item()) + 1
        flat_bins = self._radial_bins.view(-1)
        self._radial_counts = torch.zeros(self._num_radial_bins, dtype=self.dtype)
        ones = torch.ones_like(flat_bins, dtype=self.dtype)
        self._radial_counts.scatter_add_(0, flat_bins, ones)
        self._radial_counts = self._radial_counts.clamp_min(1.0)

        self._fft_energy_samples: List[Tensor] = []
        self._diag_ran = False

        if use_text and Image is None:
            warnings.warn("Pillow not available; synthetic text layer disabled")

        self._run_diagnostics()

    def __len__(self) -> int:  # pragma: no cover - simple
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image = self._generate(idx, log_energy=True)
        return image, image.clone()

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _make_generator(self, idx: int) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + int(idx))
        return gen

    def _rand(self, gen: torch.Generator, shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.rand(shape, generator=gen, device=self.device, dtype=self.dtype)

    def _randn(self, gen: torch.Generator, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype)

    def _randint(self, gen: torch.Generator, low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=gen, device=self.device).item())

    def _generate(self, idx: int, log_energy: bool) -> Tensor:
        gen = self._make_generator(idx)
        layers: List[Tensor] = []

        layers.append(self._spectral_noise(gen))
        layers.append(self._fractal_noise(gen))

        if self.include_shapes:
            layers.append(self._shapes_layer(gen))

        if self.include_gratings and self._rand(gen).item() > 0.25:
            layers.append(self._grating_layer(gen))

        if self.use_text and self._rand(gen).item() > 0.6:
            text_layer = self._text_layer(gen)
            if text_layer is not None:
                layers.append(text_layer)

        layers.append(self._blob_layer(gen))

        base = self._combine_layers(layers, gen)
        base = self._apply_color_mix(base, gen)
        base = base - base.mean(dim=(1, 2), keepdim=True)

        if log_energy and self.log_fft_energy:
            self._log_fft_energy(base)

        image = self._normalize_to_unit_interval(base)
        return image

    def _spectral_noise(self, gen: torch.Generator) -> Tensor:
        noise = self._randn(gen, (self.channels, self.image_size, self.image_size))
        noise = noise - noise.mean(dim=(1, 2), keepdim=True)
        fft = torch.fft.fftn(noise, dim=(-2, -1))
        fft[..., 0, 0] = 0.0
        low_weight = (self.radius + self.eps).pow(-0.75)
        high_weight = (self.radius + self.eps).pow(0.75)
        weight = (1.0 - self.freq_mix) * low_weight + self.freq_mix * high_weight
        fft = fft * weight
        filtered = torch.fft.ifftn(fft, dim=(-2, -1)).real
        filtered = filtered - filtered.mean(dim=(1, 2), keepdim=True)
        std = filtered.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        filtered = filtered / std
        return filtered

    def _fractal_noise(self, gen: torch.Generator) -> Tensor:
        beta = 0.5 + 1.5 * self._rand(gen).item()
        noise = self._randn(gen, (self.channels, self.image_size, self.image_size))
        noise = noise - noise.mean(dim=(1, 2), keepdim=True)
        fft = torch.fft.fftn(noise, dim=(-2, -1))
        fft[..., 0, 0] = 0.0
        weight = (self.radius + self.eps).pow(-beta)
        fft = fft * weight
        field = torch.fft.ifftn(fft, dim=(-2, -1)).real
        field = field - field.mean(dim=(1, 2), keepdim=True)
        std = field.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        field = field / std
        return field

    def _blob_layer(self, gen: torch.Generator) -> Tensor:
        num_blobs = self._randint(gen, 2, 6)
        layer = torch.zeros((self.channels, self.image_size, self.image_size), dtype=self.dtype)
        for _ in range(num_blobs):
            cx = self._rand(gen).item() * 2.0 - 1.0
            cy = self._rand(gen).item() * 2.0 - 1.0
            sx = 0.15 + 0.35 * self._rand(gen).item()
            sy = 0.15 + 0.35 * self._rand(gen).item()
            blob = torch.exp(-(((self.xx - cx) / sx) ** 2 + ((self.yy - cy) / sy) ** 2))
            blob = blob.unsqueeze(0).expand(self.channels, -1, -1)
            sign = 1.0 if self._rand(gen).item() > 0.5 else -1.0
            color = self._randn(gen, (self.channels, 1, 1)) * 0.5
            layer = layer + color * blob * sign
        layer = layer - layer.mean(dim=(1, 2), keepdim=True)
        std = layer.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        layer = layer / std
        return layer

    def _shapes_layer(self, gen: torch.Generator) -> Tensor:
        layer = torch.zeros((self.channels, self.image_size, self.image_size), dtype=self.dtype)
        num_shapes = self._randint(gen, 1, 6)
        for _ in range(num_shapes):
            mode = self._randint(gen, 0, 4)
            color = self._randn(gen, (self.channels, 1, 1))
            if mode == 0:
                layer += color * self._rectangle_mask(gen)
            elif mode == 1:
                layer += color * self._ellipse_mask(gen)
            elif mode == 2:
                layer += color * self._line_mask(gen)
            else:
                layer += color * self._ring_mask(gen)
        layer = layer - layer.mean(dim=(1, 2), keepdim=True)
        std = layer.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        layer = layer / std
        return layer

    def _rectangle_mask(self, gen: torch.Generator) -> Tensor:
        h = self.image_size
        w = self.image_size
        x0 = self._randint(gen, 0, w - 1)
        x1 = self._randint(gen, x0 + 1, w)
        y0 = self._randint(gen, 0, h - 1)
        y1 = self._randint(gen, y0 + 1, h)
        mask = torch.zeros((1, h, w), dtype=self.dtype)
        mask[:, y0:y1, x0:x1] = 1.0
        mask = mask - mask.mean(dim=(1, 2), keepdim=True)
        return mask

    def _ellipse_mask(self, gen: torch.Generator) -> Tensor:
        cx = self._rand(gen).item() * 0.6
        cy = self._rand(gen).item() * 0.6
        rx = 0.2 + 0.4 * self._rand(gen).item()
        ry = 0.2 + 0.4 * self._rand(gen).item()
        theta = 2 * math.pi * self._rand(gen).item()
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x = self.xx - (cx * 2 - 0.6)
        y = self.yy - (cy * 2 - 0.6)
        xr = cos_t * x - sin_t * y
        yr = sin_t * x + cos_t * y
        mask = ((xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0).to(self.dtype).unsqueeze(0)
        mask = mask - mask.mean(dim=(1, 2), keepdim=True)
        return mask

    def _line_mask(self, gen: torch.Generator) -> Tensor:
        theta = 2 * math.pi * self._rand(gen).item()
        width = 0.02 + 0.08 * self._rand(gen).item()
        offset = self._rand(gen).item() * 0.4 - 0.2
        projection = self.xx * math.cos(theta) + self.yy * math.sin(theta) - offset
        mask = (projection.abs() <= width).to(self.dtype).unsqueeze(0)
        mask = mask - mask.mean(dim=(1, 2), keepdim=True)
        return mask

    def _ring_mask(self, gen: torch.Generator) -> Tensor:
        radius = 0.2 + 0.5 * self._rand(gen).item()
        thickness = 0.03 + 0.12 * self._rand(gen).item()
        center_x = self._rand(gen).item() * 0.4 - 0.2
        center_y = self._rand(gen).item() * 0.4 - 0.2
        dist = torch.sqrt((self.xx - center_x) ** 2 + (self.yy - center_y) ** 2)
        mask = ((dist >= radius - thickness) & (dist <= radius + thickness)).to(self.dtype).unsqueeze(0)
        mask = mask - mask.mean(dim=(1, 2), keepdim=True)
        return mask

    def _grating_layer(self, gen: torch.Generator) -> Tensor:
        theta = 2 * math.pi * self._rand(gen).item()
        freq = 1.5 + (self.image_size / 4.0) * self._rand(gen).item()
        phase = 2 * math.pi * self._rand(gen).item()
        projection = self.xx * math.cos(theta) + self.yy * math.sin(theta)
        grating = torch.sin(freq * projection + phase)
        grating = grating.unsqueeze(0).expand(self.channels, -1, -1)
        grating = grating - grating.mean(dim=(1, 2), keepdim=True)
        std = grating.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        grating = grating / std
        return grating

    def _text_layer(self, gen: torch.Generator) -> Optional[Tensor]:
        if Image is None or ImageDraw is None:
            return None
        text_options = [
            "Lorem",
            "Ipsum",
            "Dolor",
            "Sit",
            "Amet",
            "Spectral",
            "Diffusion",
        ]
        text_idx = self._randint(gen, 0, len(text_options))
        text = text_options[text_idx]
        canvas_size = max(16, int(self.image_size * (0.6 + 0.8 * self._rand(gen).item())))
        canvas = Image.new("L", (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default() if ImageFont is not None else None
        x_pos = int(self._rand(gen).item() * canvas_size * 0.4)
        y_pos = int(self._rand(gen).item() * canvas_size * 0.6)
        draw.text((x_pos, y_pos), text, fill=255, font=font)
        resized = canvas.resize((self.image_size, self.image_size))
        tensor = torch.tensor(list(resized.getdata()), dtype=torch.uint8)
        tensor = tensor.to(self.dtype).view(self.image_size, self.image_size) / 255.0
        tensor = tensor.unsqueeze(0).expand(self.channels, -1, -1)
        tensor = tensor - tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        tensor = tensor / std
        return tensor

    def _combine_layers(self, layers: List[Tensor], gen: torch.Generator) -> Tensor:
        base = torch.zeros_like(layers[0])
        for layer in layers:
            mode = self._randint(gen, 0, 3)
            weight = 0.5 + 0.8 * self._rand(gen).item()
            if mode == 0:  # additive
                base = base + weight * layer
            elif mode == 1:  # multiplicative modulation
                base = base * (1.0 + 0.3 * weight * layer)
            else:  # overlay-like blend using tanh squashing
                base = torch.tanh(base) + weight * torch.tanh(layer)
            base = base - base.mean(dim=(1, 2), keepdim=True)
        std = base.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        base = base / std
        return base

    def _apply_color_mix(self, tensor: Tensor, gen: torch.Generator) -> Tensor:
        mix = float(torch.clamp(torch.tensor(self.color_mix, dtype=self.dtype), 0.0, 1.0).item())
        if mix <= 0.0:
            return tensor
        grey = tensor.mean(dim=0, keepdim=True)
        correlated = grey.repeat(self.channels, 1, 1)
        jitter = 0.1 * self._randn(gen, (self.channels, 1, 1))
        result = (1.0 - mix) * tensor + mix * correlated + jitter * mix
        result = result - result.mean(dim=(1, 2), keepdim=True)
        std = result.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        result = result / std
        return result

    def _normalize_to_unit_interval(self, tensor: Tensor) -> Tensor:
        min_val = tensor.amin(dim=(1, 2), keepdim=True)
        max_val = tensor.amax(dim=(1, 2), keepdim=True)
        scale = (max_val - min_val).clamp_min(1e-3)
        normalized = (tensor - min_val) / scale
        return normalized.clamp(0.0, 1.0)

    def _log_fft_energy(self, tensor: Tensor) -> None:
        fft = torch.fft.fftn(tensor, dim=(-2, -1))
        fft[..., 0, 0] = 0.0
        power = fft.abs().pow(2).mean(dim=0)
        radial = torch.zeros(self._num_radial_bins, dtype=self.dtype)
        radial.scatter_add_(0, self._radial_bins.view(-1), power.view(-1))
        radial = radial / self._radial_counts
        self._fft_energy_samples.append(radial)

    # ------------------------------------------------------------------
    # Diagnostics & utilities
    # ------------------------------------------------------------------
    def _run_diagnostics(self) -> None:
        if self._diag_ran:
            return
        if self.size == 0:
            return
        self._diag_ran = True
        sample_count = min(64, self.size)
        with torch.no_grad():
            batch = torch.stack([self._generate(i, log_energy=False) for i in range(sample_count)], dim=0)
        mean_val = batch.mean().item()
        std_val = batch.std().item()
        if not (0.45 <= mean_val <= 0.55):
            warnings.warn(
                f"Synthetic dataset mean {mean_val:.3f} outside expected range [0.45, 0.55]",
                RuntimeWarning,
            )
        if not (0.15 <= std_val <= 0.35):
            warnings.warn(
                f"Synthetic dataset std {std_val:.3f} outside expected range [0.15, 0.35]",
                RuntimeWarning,
            )

        sample = batch[0]
        spatial_energy = (sample**2).sum().item()
        freq = torch.fft.fftn(sample, dim=(-2, -1))
        freq_energy = freq.abs().pow(2).sum().item() / (self.image_size * self.image_size)
        if not math.isclose(spatial_energy, freq_energy, rel_tol=0.1, abs_tol=0.1):
            warnings.warn("Parseval energy check deviates beyond tolerance", RuntimeWarning)

        if self.channels > 1:
            channels_flat = batch.permute(1, 0, 2, 3).reshape(self.channels, -1)
            corr_matrix = torch.corrcoef(channels_flat)
            mask = ~torch.eye(self.channels, dtype=torch.bool, device=corr_matrix.device)
            off_diag = corr_matrix[mask]
            corr = off_diag.mean().item()
        else:
            corr = 1.0
        target = self.color_mix
        if abs(corr - target) > 0.3:
            warnings.warn(
                f"Channel correlation {corr:.2f} differs significantly from target {target:.2f}",
                RuntimeWarning,
            )

    def show_sample(self, idx: int = 0) -> None:  # pragma: no cover - visual helper
        import matplotlib.pyplot as plt

        img = self[idx][0].permute(1, 2, 0).clamp(0.0, 1.0)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


__all__ = ["SyntheticSpectralDataset", "SyntheticSpectralConfig"]
