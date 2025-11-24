import logging
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.core import build_model, get_loss_fn
from src.evaluation.metrics import compute_basic_metrics
from src.training.builders import build_dataloader, build_optimizer
from src.training.diagnostics import TaguchiAggregator, TrainingDiagnostics
from src.training.noise import NoisePreparer
from src.training.sampling import build_sampler
from src.training.scheduler import build_diffusion, loss_aware_timesteps, sample_timesteps
from src.training.steps import TrainingStepExecutor
from src.training.visualization import DiagnosticsPlotter


class TrainingPipeline:
    """Unified training pipeline for baseline and spectral diffusion models."""

    def __init__(
        self,
        config: Dict[str, Any],
        work_dir: Path,
        logger: Optional[logging.Logger] = None,
        *,
        noise_preparer: Optional[NoisePreparer] = None,
        step_executor: Optional[TrainingStepExecutor] = None,
        diagnostics: Optional[TrainingDiagnostics] = None,
        plotter: Optional[DiagnosticsPlotter] = None,
    ) -> None:
        self.config = config
        self.work_dir = work_dir
        self.logger = logger or logging.getLogger(__name__)
        self.model = build_model(config.get("model", {}))
        self.loss_fn = get_loss_fn(config.get("loss", {}))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._noise_preparer = noise_preparer
        self._step_executor = step_executor
        self._diagnostics = diagnostics
        self._plotter = plotter

    def setup(self) -> None:
        """Prepare datasets, optimizers, and other resources."""
        self.logger.debug("Setting up training pipeline with config: %s", self.config)
        self.loader = build_dataloader(self.config)
        self.optimizer = build_optimizer(self.model, self.config)

    def run(self) -> Dict[str, Any]:
        """Execute the (placeholder) training loop and return metrics."""
        self.setup()
        self.model.train()
        epochs = int(self.config.get("training", {}).get("epochs", 1))
        log_every = int(self.config.get("training", {}).get("log_every", 10))
        max_batches = self.config.get("training", {}).get("num_batches")
        batch_limit = int(max_batches) if str(max_batches).isdigit() else None

        metrics_cfg = self.config.get("metrics", {})
        loss_threshold = metrics_cfg.get("loss_threshold")

        self._prepare_instrumentation()

        diagnostics = self._diagnostics
        noise_preparer = self._noise_preparer
        step_executor = self._step_executor
        if diagnostics is None or noise_preparer is None or step_executor is None:
            raise RuntimeError("Training pipeline instrumentation was not initialised")

        step = 0
        threshold_steps: Optional[int] = None
        threshold_time: Optional[float] = None
        wall_start = perf_counter()
        T, schedule, schedule_kwargs = self._diffusion_params()
        coeffs = build_diffusion(T, schedule, schedule_kwargs)
        schedule_key = schedule.replace("-", "_").lower()
        if schedule_key == "logsnr_cosine":
            lam_min = float(schedule_kwargs.get("lambda_min", -13.0))
            lam_max = float(schedule_kwargs.get("lambda_max", 13.0))
            delta = float(schedule_kwargs.get("delta", 0.008))
            self.logger.info(
                "[Schedule] mode=logsnr_cosine λ∈[%.3f, %.3f] δ=%.3f",
                lam_min,
                lam_max,
                delta,
            )
        total_timesteps = coeffs.num_timesteps
        snr_ratio_value = noise_preparer.snr_ratio

        fft_feedback_history: Dict[str, list[float]] = defaultdict(list)
        coeff_history: Dict[str, list[float]] = {}
        batch_history: Dict[str, list[float]] = {}
        diffusion_cfg = self.config.get("diffusion", {})
        loss_aware_enabled = bool(diffusion_cfg.get("loss_aware_sampling", False))
        lais_temperature = float(diffusion_cfg.get("lais_temperature", 1.2))
        lais_decay = float(diffusion_cfg.get("lais_decay", 0.95))
        loss_landscape = (
            torch.ones(coeffs.num_timesteps, device=self.device)
            if loss_aware_enabled
            else None
        )
        warmup_repeats = int(self.config.get("training", {}).get("warmup_repeats", 1))
        warmup_repeats = max(1, warmup_repeats)

        for epoch in range(epochs):
            for batch_idx, (xb, _) in enumerate(self.loader):
                xb = xb.to(self.device)

                diagnostics.capture_initial_batch(xb)

                if step == 0:
                    schedule_snr = (
                        coeffs.sqrt_alphas_cumprod.pow(2) / coeffs.sqrt_one_minus_alphas_cumprod.pow(2).clamp_min(1e-12)
                    )
                    snr_min = float(schedule_snr.min().item())
                    snr_max = float(schedule_snr.max().item())
                sample_noise = noise_preparer.prepare(
                    xb,
                    coeffs,
                    torch.zeros(xb.shape[0], device=xb.device, dtype=torch.long),
                    base_noise=torch.randn_like(xb),
                )
                effective = sample_noise.stats.get("snr_emp") if sample_noise.stats else None
                self.logger.info(
                    "[SNR-DIAG] schedule_snr_range=(%.3f, %.3f) snr_emp_sample=%.3f snr_ratio=%s",
                    snr_min,
                    snr_max,
                    float(effective) if effective is not None else float("nan"),
                    noise_preparer.snr_ratio,
                )

                B = xb.shape[0]
                if loss_aware_enabled and warmup_repeats > 1:
                    for _ in range(warmup_repeats - 1):
                        warm_timesteps = loss_aware_timesteps(
                            B,
                            loss_landscape,
                            device=xb.device,
                            temperature=lais_temperature,
                        )
                        warm_noise = noise_preparer.prepare(
                            xb,
                            coeffs,
                            warm_timesteps,
                            base_noise=torch.randn_like(xb),
                        )
                        step_executor.run_step(
                            xb,
                            warm_noise,
                            warm_timesteps,
                        )

                if loss_aware_enabled and loss_landscape is not None:
                    timesteps = loss_aware_timesteps(
                        B,
                        loss_landscape,
                        device=xb.device,
                        temperature=lais_temperature,
                    )
                else:
                    timesteps = sample_timesteps(
                        B,
                        total_timesteps,
                        xb.device,
                    )
                noise_batch = noise_preparer.prepare(
                    xb,
                    coeffs,
                    timesteps,
                    base_noise=torch.randn_like(xb),
                )

                diagnostics.capture_noisy_example(noise_batch.noisy)
                diagnostics.record_noise_stats(step + 1, noise_batch.stats or {})

                outcome = step_executor.run_step(
                    xb,
                    noise_batch,
                    timesteps,
                    grad_callback=lambda: diagnostics.record_gradients(self.model, step),
                )

                step += 1
                diagnostics.record_loss(step, outcome.loss)
                diagnostics.record_mae(step, outcome.mae)
                diagnostics.record_noise_norm(step, noise_batch.eps_norm)
                diagnostics.record_fft_feedback(step, outcome.fft_feedback)
                diagnostics.record_coeff_stats(step, outcome.coeff_stats)
                diagnostics.record_batch_stats(step, outcome.batch_stats)

                for key, metric_val in outcome.fft_feedback.items():
                    fft_feedback_history[key].append(float(metric_val))
                for key, value in outcome.coeff_stats.items():
                    coeff_history.setdefault(key, []).append(float(value))
                for key, value in outcome.batch_stats.items():
                    batch_history.setdefault(key, []).append(float(value))

                if (
                    loss_aware_enabled
                    and loss_landscape is not None
                    and outcome.per_example_mse is not None
                ):
                    per_example = outcome.per_example_mse.to(device=loss_landscape.device, dtype=loss_landscape.dtype)
                    buckets = torch.zeros_like(loss_landscape)
                    buckets.scatter_add_(0, timesteps.to(loss_landscape.device), per_example)
                    counts = torch.bincount(
                        timesteps.to(loss_landscape.device), minlength=loss_landscape.numel()
                    ).to(loss_landscape.dtype)
                    mask = counts > 0
                    averages = torch.zeros_like(loss_landscape)
                    averages[mask] = buckets[mask] / counts[mask]
                    loss_landscape = torch.where(
                        mask,
                        loss_landscape * lais_decay + averages * (1.0 - lais_decay),
                        loss_landscape,
                    )

                if step % log_every == 0:
                    snr_theory = outcome.coeff_stats.get("snr_theory", float("nan"))
                    snr_emp = outcome.coeff_stats.get("snr_emp", float("nan"))
                    snr_rel = outcome.coeff_stats.get("snr_rel", float("nan"))
                    variance_sum = outcome.coeff_stats.get("variance_sum", float("nan"))
                    self.logger.debug(
                        "epoch %d step %d loss %.5f snr_theory %.3f snr_emp %.3f snr_rel %.3f var_sum %.4f",
                        epoch,
                        step,
                        outcome.loss,
                        snr_theory,
                        snr_emp,
                        snr_rel,
                        variance_sum,
                    )

                if (
                    loss_threshold is not None
                    and threshold_steps is None
                    and outcome.loss <= loss_threshold
                ):
                    threshold_steps = step
                    threshold_time = perf_counter() - wall_start

                if batch_limit is not None and (batch_idx + 1) >= batch_limit:
                    break

        runtime_seconds = perf_counter() - wall_start

        steps_per_second = step / runtime_seconds if runtime_seconds > 0 else 0.0
        images_per_second = (
            (step * self.loader.batch_size) / runtime_seconds if runtime_seconds > 0 else 0.0
        )
        runtime_per_epoch = runtime_seconds / epochs if epochs > 0 else None
        loss_history = diagnostics.loss_history
        mae_history = diagnostics.mae_history
        initial_loss = loss_history[0] if loss_history else None
        final_loss = loss_history[-1] if loss_history else None
        loss_drop = (
            (initial_loss - final_loss)
            if initial_loss is not None and final_loss is not None
            else None
        )
        loss_drop_per_second = (
            (loss_drop / runtime_seconds)
            if loss_drop is not None and runtime_seconds > 0
            else None
        )
        fft_means = {
            f"fft_{key}_mean": (mean(vals) if vals else None)
            for key, vals in fft_feedback_history.items()
        }
        coeff_means = {
            f"diffusion_{key}_mean": (mean(vals) if vals else None)
            for key, vals in coeff_history.items()
        }
        batch_means = {
            f"batch_{key}_mean": (mean(vals) if vals else None)
            for key, vals in batch_history.items()
        }
        metrics = compute_basic_metrics(
            loss_history=loss_history,
            mae_history=mae_history,
            runtime_seconds=runtime_seconds,
            extra={
                "status": "ok",
                "num_steps": step,
                "epochs": epochs,
                "steps_per_second": steps_per_second,
                "images_per_second": images_per_second,
                "runtime_per_epoch": runtime_per_epoch,
                "loss_initial": initial_loss,
                "loss_final": final_loss,
                "loss_drop": loss_drop,
                "loss_drop_per_second": loss_drop_per_second,
                "loss_threshold": loss_threshold,
                "loss_threshold_steps": threshold_steps,
                "loss_threshold_time": threshold_time,
                "loss_history": [float(v) for v in loss_history],
                "mae_history": [float(v) for v in mae_history],
                "snr_ratio": snr_ratio_value,
                **{
                    f"fft_{key}_history": [float(v) for v in vals]
                    for key, vals in fft_feedback_history.items()
                    if vals
                },
                **fft_means,
                **{
                    f"diffusion_{key}_history": [float(v) for v in vals]
                    for key, vals in coeff_history.items()
                    if vals
                },
                **coeff_means,
                **{
                    f"batch_{key}_history": [float(v) for v in vals]
                    for key, vals in batch_history.items()
                    if vals
                },
                **batch_means,
            },
        )
        diagnostics.finalise()
        training_stats = {}
        if hasattr(self.model, "spectral_stats"):
            training_stats = dict(self.model.spectral_stats())
            metrics.update(training_stats)
            if hasattr(self.model, "reset_spectral_stats"):
                self.model.reset_spectral_stats()

        log_summary_keys = [
            "status",
            "loss_initial",
            "loss_final",
            "loss_drop",
            "loss_drop_per_second",
            "runtime_seconds",
            "steps_per_second",
            "images_per_second",
            "snr_ratio",
        ]
        log_summary = {key: metrics.get(key) for key in log_summary_keys if key in metrics}
        self.logger.info("Training metrics: %s", log_summary)
        return metrics

    def _diffusion_params(self) -> Tuple[int, str, Dict[str, float]]:
        diffusion_cfg = self.config.get("diffusion", {})
        T = int(diffusion_cfg.get("num_timesteps", 1000))
        schedule = diffusion_cfg.get("beta_schedule", "cosine")
        schedule_kwargs: Dict[str, float] = dict(
            diffusion_cfg.get("schedule_kwargs", {}) or {}
        )
        schedule_key = schedule.replace("-", "_").lower()
        if schedule_key == "logsnr_cosine":
            logsnr_cfg = diffusion_cfg.get("logsnr", {}) or {}
            for key in ("lambda_min", "lambda_max", "delta"):
                if key in logsnr_cfg and key not in schedule_kwargs:
                    schedule_kwargs[key] = float(logsnr_cfg[key])
        return T, schedule, schedule_kwargs

    def _prepare_instrumentation(self) -> None:
        factor_levels = self.config.get("taguchi", {}).get("factor_levels", {}) or {}
        aggregator = TaguchiAggregator(self.work_dir, factor_levels)
        dataset_name = str(self.config.get("data", {}).get("source", "unknown"))
        plotter = self._plotter or DiagnosticsPlotter()

        if self._diagnostics is None:
            self._diagnostics = TrainingDiagnostics(
                run_id=self.work_dir.name,
                dataset_name=dataset_name,
                work_dir=self.work_dir,
                aggregator=aggregator,
                plotter=plotter,
            )
        else:
            self._diagnostics.aggregator = aggregator
            self._diagnostics.dataset_name = dataset_name
            self._diagnostics.work_dir = self.work_dir
            self._diagnostics.plotter = plotter
            self._diagnostics.diagnostics_dir = self.work_dir / "diagnostics"
            self._diagnostics.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        if self._noise_preparer is None:
            self._noise_preparer = NoisePreparer.from_config(self.config)

        if self._step_executor is None:
            diffusion_cfg = self.config.get("diffusion", {}) or {}
            prediction_type = diffusion_cfg.get("prediction_type", "eps")
            fft_norm = (
                getattr(self._noise_preparer, "fft_norm", None)
                or str(diffusion_cfg.get("fft_norm", "ortho"))
            )
            self._step_executor = TrainingStepExecutor(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                prediction_type=str(prediction_type),
                fft_norm=str(fft_norm),
            )


    def generate_samples(
        self,
        num_samples: Optional[int] = None,
        num_steps: Optional[int] = None,
        sampler_type: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        sampling_cfg = dict(self.config.get("sampling", {}) or {})
        sampling_cfg["enabled"] = True
        if num_samples is not None:
            sampling_cfg["num_samples"] = int(num_samples)
        if num_steps is not None:
            sampling_cfg["num_steps"] = int(num_steps)
        if sampler_type is not None:
            sampling_cfg["sampler_type"] = str(sampler_type)

        T, schedule, schedule_kwargs = self._diffusion_params()
        coeffs = build_diffusion(T, schedule, schedule_kwargs)
        if schedule.replace("-", "_").lower() == "logsnr_cosine":
            lam_min = float(schedule_kwargs.get("lambda_min", -13.0))
            lam_max = float(schedule_kwargs.get("lambda_max", 13.0))
            delta = float(schedule_kwargs.get("delta", 0.008))
            self.logger.info(
                "[Schedule] mode=logsnr_cosine λ∈[%.3f, %.3f] δ=%.3f",
                lam_min,
                lam_max,
                delta,
            )

        sampler = sampling_cfg.get("sampler_type", "ddpm").lower()
        try:
            sampler_impl = build_sampler(sampler, model=self.model, coeffs=coeffs)
        except ValueError:
            self.logger.warning("Sampler '%s' not supported; falling back to ddpm", sampler)
            sampler_impl = build_sampler("ddpm", model=self.model, coeffs=coeffs)
            sampler = "ddpm"

        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})
        channels = int(model_cfg.get("channels") or data_cfg.get("channels", 3))
        height = int(data_cfg.get("height", 32))
        width = int(data_cfg.get("width", 32))
        shape = (channels, height, width)

        requested_samples = int(sampling_cfg.get("num_samples", 16))
        requested_steps = int(sampling_cfg.get("num_steps", coeffs.betas.shape[0]))

        images_dir = output_dir or (self.work_dir / "images")
        images_dir.mkdir(parents=True, exist_ok=True)

        samples = sampler_impl.sample(
            num_samples=requested_samples,
            shape=shape,
            num_steps=requested_steps,
            device=self.device,
        )

        grid_path = images_dir / "grid.png"
        save_image((samples + 1) / 2.0, grid_path, nrow=max(1, int(requested_samples**0.5)))

        for idx, img in enumerate(samples):
            save_image((img + 1) / 2.0, images_dir / f"sample_{idx:03d}.png")

        return {
            "images_dir": images_dir,
            "num_samples": requested_samples,
            "num_steps": requested_steps,
            "sampler_type": sampler,
        }

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_dir = self.work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save({"model": self.model.state_dict()}, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
