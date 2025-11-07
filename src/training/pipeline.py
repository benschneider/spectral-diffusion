import logging
import os
from pathlib import Path
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
from src.training.scheduler import build_diffusion, sample_timesteps
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
        T, schedule = self._diffusion_params()
        coeffs = build_diffusion(T, schedule)
        snr_ratio_value = noise_preparer.snr_ratio
        dc_scale_factor = noise_preparer.dc_scale_factor

        for epoch in range(epochs):
            for batch_idx, (xb, _) in enumerate(self.loader):
                xb = xb.to(self.device)

                diagnostics.capture_initial_batch(xb)

                B = xb.shape[0]
                timesteps = sample_timesteps(B, T, xb.device)
                noise_batch = noise_preparer.prepare(
                    xb,
                    coeffs,
                    timesteps,
                    base_noise=torch.randn_like(xb),
                )

                diagnostics.capture_noisy_example(noise_batch.noisy)

                outcome = step_executor.run_step(
                    xb,
                    noise_batch,
                    timesteps,
                    grad_callback=lambda: diagnostics.record_gradients(self.model, step),
                )

                diagnostics.capture_phase_demo(self.model)

                step += 1
                diagnostics.record_loss(step, outcome.loss)
                diagnostics.record_mae(step, outcome.mae)
                diagnostics.record_noise_norm(step, noise_batch.eps_norm)

                if step % log_every == 0:
                    mean_val = noise_batch.stats.get("noisy_mean") if noise_batch.stats else None
                    std_val = noise_batch.stats.get("noisy_std") if noise_batch.stats else None
                    self.logger.info("epoch %d step %d loss %.5f", epoch, step, outcome.loss)
                    self.logger.debug(
                        "spectral noise stats: snr_ratio=%.3f mean=%.3f std=%.3f",
                        snr_ratio_value if snr_ratio_value is not None else float("nan"),
                        mean_val if mean_val is not None else float("nan"),
                        std_val if std_val is not None else float("nan"),
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
            (final_loss - initial_loss)
            if initial_loss is not None and final_loss is not None
            else None
        )
        loss_drop_per_second = (
            (loss_drop / runtime_seconds)
            if loss_drop is not None and runtime_seconds > 0
            else None
        )
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
                "dc_scale_factor": dc_scale_factor,
            },
        )
        diagnostics.finalise()
        training_stats = {}
        if hasattr(self.model, "spectral_stats"):
            training_stats = dict(self.model.spectral_stats())
            metrics.update(training_stats)
            if hasattr(self.model, "reset_spectral_stats"):
                self.model.reset_spectral_stats()

        self.logger.info("Training metrics: %s", metrics)
        return metrics

    def _diffusion_params(self) -> Tuple[int, str]:
        diffusion_cfg = self.config.get("diffusion", {})
        T = int(diffusion_cfg.get("num_timesteps", 1000))
        schedule = diffusion_cfg.get("beta_schedule", "cosine")
        return T, schedule

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
            snr_weighting = bool(diffusion_cfg.get("snr_weighting", False))
            snr_transform = str(diffusion_cfg.get("snr_transform", "snr"))
            self._step_executor = TrainingStepExecutor(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                prediction_type=str(prediction_type),
                snr_weighting=snr_weighting,
                snr_transform=snr_transform,
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

        T, schedule = self._diffusion_params()
        coeffs = build_diffusion(T, schedule)

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
