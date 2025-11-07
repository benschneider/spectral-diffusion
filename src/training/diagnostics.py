from __future__ import annotations

import shutil
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn

from src.training.visualization import DiagnosticsPlotter
from src.utils.sanity_checks import check_fft_sanity


@dataclass
class TaguchiAggregator:
    """Resolve directories for Taguchi factor aggregation."""

    work_dir: Path
    factor_levels: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        self.factor_levels = dict(self.factor_levels or {})
        self.run_id = self.work_dir.name
        self.aggregate_base = self._resolve_base()
        self._factor_cache: Dict[str, Path] = {}

    def _resolve_base(self) -> Path:
        parent = self.work_dir.parent
        if parent.name == "runs":
            return parent.parent
        return self.work_dir

    @property
    def sanity_dir(self) -> Path:
        path = self.aggregate_base / "sanity"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def aggregate_dir(self) -> Path:
        path = self.aggregate_base / "diagnostics"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_factor_dir(self, factor: str) -> Optional[Path]:
        if factor not in self.factor_levels:
            return None
        if factor in self._factor_cache:
            return self._factor_cache[factor]
        meta = self.factor_levels[factor]
        level = str(meta.get("level_label", meta.get("level_index", "unknown")))
        path = self.aggregate_base / "factors" / factor / level
        path.mkdir(parents=True, exist_ok=True)
        self._factor_cache[factor] = path
        return path

    def iter_factor_dirs(self) -> Iterable[tuple[str, Path]]:
        for factor in self.factor_levels:
            factor_dir = self.get_factor_dir(factor)
            if factor_dir is not None:
                yield factor, factor_dir


class TrainingDiagnostics:
    """Capture intermediate artifacts and scalar histories during training."""

    def __init__(
        self,
        *,
        run_id: str,
        dataset_name: str,
        work_dir: Path,
        aggregator: TaguchiAggregator,
        plotter: Optional[DiagnosticsPlotter] = None,
    ) -> None:
        self.run_id = run_id
        self.dataset_name = dataset_name
        self.work_dir = work_dir
        self.aggregator = aggregator
        self.plotter = plotter or DiagnosticsPlotter()
        self.diagnostics_dir = work_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self.loss_history: list[float] = []
        self.loss_steps: list[int] = []
        self.mae_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.grad_norm_steps: list[int] = []
        self.noise_norm_history: list[float] = []
        self.noise_norm_steps: list[int] = []
        self.fft_feedback_steps: list[int] = []
        self.fft_history: Dict[str, list[float]] = defaultdict(list)
        self.coeff_steps: list[int] = []
        self.coeff_history: Dict[str, list[float]] = defaultdict(list)
        self.batch_steps: list[int] = []
        self.batch_history: Dict[str, list[float]] = defaultdict(list)
        self.weight_steps: list[int] = []
        self.weight_history: Dict[str, list[float]] = defaultdict(list)

        self._initial_batch_captured = False
        self._noisy_capture_done = False
        self._phase_demo_captured = False

    def capture_initial_batch(self, batch: torch.Tensor) -> None:
        if self._initial_batch_captured:
            return
        stats_path = check_fft_sanity(
            batch.detach().cpu(),
            self.dataset_name,
            self.aggregator.sanity_dir,
            prefix=f"{self.run_id}_",
        )
        base_name = stats_path.stem
        spatial_src = stats_path.with_name(f"{base_name}_spatial.png")
        fft_src = stats_path.with_name(f"{base_name}_fft_mag.png")

        for _, factor_dir in self.aggregator.iter_factor_dirs():
            if spatial_src.exists():
                shutil.copy(spatial_src, factor_dir / f"demo_spatial_{self.run_id}.png")
            if fft_src.exists():
                shutil.copy(fft_src, factor_dir / f"demo_fft_{self.run_id}.png")

        self._initial_batch_captured = True

    def capture_noisy_example(self, noisy_batch: torch.Tensor) -> None:
        if self._noisy_capture_done:
            return
        stats_path = check_fft_sanity(
            noisy_batch.detach().cpu(),
            f"{self.dataset_name}_noisy",
            self.aggregator.sanity_dir,
            prefix=f"{self.run_id}_noisy_",
        )
        base_name = stats_path.stem
        spatial_src = stats_path.with_name(f"{base_name}_spatial.png")
        fft_src = stats_path.with_name(f"{base_name}_fft_mag.png")

        for _, factor_dir in self.aggregator.iter_factor_dirs():
            if spatial_src.exists():
                shutil.copy(
                    spatial_src, factor_dir / f"demo_noisy_spatial_{self.run_id}.png"
                )
            if fft_src.exists():
                shutil.copy(fft_src, factor_dir / f"demo_noisy_fft_{self.run_id}.png")

        self._noisy_capture_done = True

    def capture_phase_demo(self, model: nn.Module) -> None:
        if self._phase_demo_captured:
            return
        factor_dir = self.aggregator.get_factor_dir("phase_attention_capacity")
        if factor_dir is None:
            self._phase_demo_captured = True
            return
        pcm = getattr(model, "pcm", None)
        if pcm is None:
            self._phase_demo_captured = True
            return
        weights = getattr(pcm, "last_attention_map", None)
        if weights is None:
            return
        attention = weights.detach().cpu().numpy()
        if attention.ndim == 3:
            attention = attention.mean(axis=0)
        if attention.ndim == 1:
            side = int(np.sqrt(attention.shape[0]))
            if side * side == attention.shape[0]:
                attention = attention.reshape(side, side)
        if attention.ndim == 2 and attention.shape[0] != attention.shape[1]:
            side = int(np.sqrt(attention.shape[-1]))
            if side * side == attention.shape[-1]:
                attention = attention.reshape(side, side)
        if attention.ndim != 2:
            self._phase_demo_captured = True
            return
        target_path = factor_dir / f"demo_phase_attention_{self.run_id}.png"
        self.plotter.phase_attention(attention, target_path)
        self._phase_demo_captured = True

    def record_loss(self, step: int, loss: float) -> None:
        self.loss_steps.append(step)
        self.loss_history.append(loss)

    def record_mae(self, step: int, mae: float) -> None:
        self.mae_history.append(mae)

    def record_noise_norm(self, step: int, norm: float) -> None:
        self.noise_norm_steps.append(step)
        self.noise_norm_history.append(norm)

    def record_fft_feedback(self, step: int, feedback: Mapping[str, float]) -> None:
        self.fft_feedback_steps.append(step)
        for key, value in feedback.items():
            self.fft_history[key].append(float(value))

    def record_coeff_stats(self, step: int, stats: Mapping[str, float]) -> None:
        self.coeff_steps.append(step)
        for key, value in stats.items():
            self.coeff_history[key].append(float(value))

    def record_batch_stats(self, step: int, stats: Mapping[str, float]) -> None:
        self.batch_steps.append(step)
        for key, value in stats.items():
            self.batch_history[key].append(float(value))

    def record_weight_stats(self, step: int, stats: Mapping[str, float]) -> None:
        self.weight_steps.append(step)
        for key, value in stats.items():
            self.weight_history[key].append(float(value))

    def record_gradients(self, model: nn.Module, step: int) -> Optional[float]:
        total_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_norm_sq += float(grad.pow(2).sum().cpu())
        if total_norm_sq == 0.0:
            return None
        total_norm = float(total_norm_sq ** 0.5)
        self.grad_norm_steps.append(step + 1)
        self.grad_norm_history.append(total_norm)
        return total_norm

    def finalise(self) -> None:
        if self.loss_history:
            loss_plot = self.plotter.loss_and_gradients(
                self.loss_steps,
                self.loss_history,
                self.grad_norm_steps,
                self.grad_norm_history,
                self.diagnostics_dir,
                self.run_id,
            )
            aggregate_target = self.aggregator.aggregate_dir / f"{self.run_id}_loss_gradients.png"
            shutil.copy(loss_plot, aggregate_target)
            for factor, factor_dir in self.aggregator.iter_factor_dirs():
                shutil.copy(loss_plot, factor_dir / f"demo_loss_grad_{self.run_id}.png")

            spectral_dir = self.aggregator.get_factor_dir("spectral_loss_weighting")
            self.plotter.recent_loss_tail(
                self.loss_steps,
                self.loss_history,
                spectral_dir,
                self.run_id,
            )

        if self.noise_norm_history:
            sampler_dir = self.aggregator.get_factor_dir("sampler_type")
            self.plotter.noise_norm(
                self.noise_norm_steps,
                self.noise_norm_history,
                sampler_dir,
                self.run_id,
            )

        if self.fft_feedback_steps:
            payload = {"steps": self.fft_feedback_steps}
            for key, values in self.fft_history.items():
                payload[key] = values
                payload[f"{key}_mean"] = mean(values) if values else None
            target = self.diagnostics_dir / "fft_feedback.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            for factor, factor_dir in self.aggregator.iter_factor_dirs():
                shutil.copy(target, factor_dir / f"fft_feedback_{self.run_id}.json")

        if self.coeff_history:
            payload = {"steps": self.coeff_steps}
            for key, values in self.coeff_history.items():
                payload[key] = values
                payload[f"{key}_mean"] = mean(values) if values else None
            target = self.diagnostics_dir / "diffusion_coefficients.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            for factor, factor_dir in self.aggregator.iter_factor_dirs():
                shutil.copy(
                    target,
                    factor_dir / f"diffusion_coefficients_{self.run_id}.json",
                )

        if self.batch_history:
            payload = {"steps": self.batch_steps}
            for key, values in self.batch_history.items():
                payload[key] = values
                payload[f"{key}_mean"] = mean(values) if values else None
            target = self.diagnostics_dir / "batch_signal_stats.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            for factor, factor_dir in self.aggregator.iter_factor_dirs():
                shutil.copy(
                    target,
                    factor_dir / f"batch_signal_stats_{self.run_id}.json",
                )

        if self.weight_history:
            payload = {"steps": self.weight_steps}
            for key, values in self.weight_history.items():
                payload[key] = values
                payload[f"{key}_mean"] = mean(values) if values else None
            target = self.diagnostics_dir / "snr_weights.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            for factor, factor_dir in self.aggregator.iter_factor_dirs():
                shutil.copy(target, factor_dir / f"snr_weights_{self.run_id}.json")
