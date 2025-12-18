from __future__ import annotations

import shutil
import json
import csv
import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Mapping, Optional

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
            grandparent = parent.parent
            if (grandparent / "pyproject.toml").exists() or (grandparent / ".git").exists():
                return parent
            return grandparent
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
        self.noise_stats_steps: list[int] = []
        self.noise_stats_history: Dict[str, list[float]] = defaultdict(list)
        self.fft_feedback_steps: list[int] = []
        self.fft_history: Dict[str, list[float]] = defaultdict(list)
        self.coeff_steps: list[int] = []
        self.coeff_history: Dict[str, list[float]] = defaultdict(list)
        self.batch_steps: list[int] = []
        self.batch_history: Dict[str, list[float]] = defaultdict(list)

        self._initial_batch_captured = False
        self._noisy_capture_done = False
        self.stability_csv_path = self.diagnostics_dir / "stability_metrics.csv"

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

    def capture_noisy_example(self, noisy_batch: torch.Tensor, eps: Optional[torch.Tensor] = None) -> None:
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

        if eps is not None:
            eps_path = check_fft_sanity(
                eps.detach().cpu(),
                f"{self.dataset_name}_eps",
                self.aggregator.sanity_dir,
                prefix=f"{self.run_id}_eps_",
            )
            base_name = eps_path.stem
            eps_spatial_src = eps_path.with_name(f"{base_name}_spatial.png")
            eps_fft_src = eps_path.with_name(f"{base_name}_fft_mag.png")
            for _, factor_dir in self.aggregator.iter_factor_dirs():
                if eps_spatial_src.exists():
                    shutil.copy(
                        eps_spatial_src,
                        factor_dir / f"demo_eps_spatial_{self.run_id}.png",
                    )
                if eps_fft_src.exists():
                    shutil.copy(
                        eps_fft_src,
                        factor_dir / f"demo_eps_fft_{self.run_id}.png",
                    )

        self._noisy_capture_done = True

    def record_loss(self, step: int, loss: float) -> None:
        self.loss_steps.append(step)
        self.loss_history.append(loss)

    def record_mae(self, step: int, mae: float) -> None:
        self.mae_history.append(mae)

    def record_noise_norm(self, step: int, norm: float) -> None:
        self.noise_norm_steps.append(step)
        self.noise_norm_history.append(norm)

    def record_noise_stats(self, step: int, stats: Mapping[str, float]) -> None:
        if not stats:
            return
        self.noise_stats_steps.append(step)
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.noise_stats_history[key].append(float(value))
        variance_sum = stats.get("variance_sum")
        if isinstance(variance_sum, (int, float)) and abs(float(variance_sum) - 1.0) > 1e-3:
            logging.warning(
                "[Diagnostics] variance_sum drift detected: step=%d variance_sum=%.6f",
                step,
                float(variance_sum),
            )
        snr_rel = stats.get("snr_rel")
        if isinstance(snr_rel, (int, float)) and not 0.3 <= float(snr_rel) <= 3.0:
            logging.warning(
                "[Diagnostics] snr_rel out of range: step=%d snr_rel=%.6f",
                step,
                float(snr_rel),
            )

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

            spectral_dir = self.aggregator.get_factor_dir("spectral_operator_mode")
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

        if self.noise_stats_history:
            payload = {"steps": self.noise_stats_steps}
            for key, values in self.noise_stats_history.items():
                payload[key] = values
                payload[f"{key}_mean"] = mean(values) if values else None
            target = self.diagnostics_dir / "noise_stats.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            for factor, factor_dir in self.aggregator.iter_factor_dirs():
                shutil.copy(target, factor_dir / f"noise_stats_{self.run_id}.json")

        stability_csv = self._export_stability_metrics()
        if stability_csv is not None:
            aggregate_target = (
                self.aggregator.aggregate_dir / f"{self.run_id}_stability_metrics.csv"
            )
            shutil.copy(stability_csv, aggregate_target)
            history_csv = self._export_training_history()
            if history_csv is not None:
                aggregate_history_target = (
                    self.aggregator.aggregate_dir / f"{self.run_id}_training_history.csv"
                )
                shutil.copy(history_csv, aggregate_history_target)

    def _export_stability_metrics(self) -> Optional[Path]:
        if not self.noise_stats_steps:
            return None

        header = [
            "step",
            "snr_theory",
            "snr_emp",
            "snr_rel",
            "variance_sum",
            "grad_norm",
            "noise_channel_std_min",
            "noise_channel_std_max",
        ]
        grad_lookup = {
            step: value for step, value in zip(self.grad_norm_steps, self.grad_norm_history)
        }
        keys = (
            "snr_theory",
            "snr_emp",
            "snr_rel",
            "variance_sum",
            "noise_channel_std_min",
            "noise_channel_std_max",
        )

        with self.stability_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for idx, step in enumerate(self.noise_stats_steps):
                row = [step]
                for key in keys:
                    values = self.noise_stats_history.get(key, [])
                    row.append(values[idx] if idx < len(values) else math.nan)
                grad_value = grad_lookup.get(step, math.nan)
                row.insert(5, grad_value)
                writer.writerow(row)

        return self.stability_csv_path

    def _export_training_history(self) -> Optional[Path]:
        if not self.noise_stats_steps:
            return None

        loss_lookup = {step: value for step, value in zip(self.loss_steps, self.loss_history)}
        mae_lookup = {step: value for step, value in zip(self.loss_steps, self.mae_history)}
        grad_lookup = {
            step: value for step, value in zip(self.grad_norm_steps, self.grad_norm_history)
        }

        header = [
            "step",
            "loss",
            "mae",
            "grad_norm",
            "snr_theory",
            "snr_emp",
            "snr_rel",
            "variance_sum",
            "noise_channel_std_min",
            "noise_channel_std_max",
        ]
        keys = (
            "snr_theory",
            "snr_emp",
            "snr_rel",
            "variance_sum",
            "noise_channel_std_min",
            "noise_channel_std_max",
        )
        history_path = self.diagnostics_dir / "training_history.csv"
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for idx, step in enumerate(self.noise_stats_steps):
                row = [
                    step,
                    loss_lookup.get(step, math.nan),
                    mae_lookup.get(step, math.nan),
                    grad_lookup.get(step, math.nan),
                ]
                for key in keys:
                    values = self.noise_stats_history.get(key, [])
                    row.append(values[idx] if idx < len(values) else math.nan)
                writer.writerow(row)
        return history_path
