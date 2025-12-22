import json
import logging
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
from src.training.scheduler import build_diffusion, sample_timesteps
from src.training.steps import TrainingStepExecutor
from src.training.visualization import DiagnosticsPlotter
from src.utils.sanity_checks import check_fft_sanity


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
        self.loss_fn = get_loss_fn()
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
        training_cfg = self.config.get("training", {}) or {}
        epochs_cfg = int(training_cfg.get("epochs", 1))
        log_every = max(1, int(training_cfg.get("log_every", 10)))
        train_steps_raw = training_cfg.get("train_steps")
        train_steps: Optional[int] = None
        if train_steps_raw is not None:
            try:
                train_steps = int(train_steps_raw)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"training.train_steps must be an integer or null, got {train_steps_raw!r}") from exc

        self._prepare_instrumentation()

        diagnostics = self._diagnostics
        noise_preparer = self._noise_preparer
        step_executor = self._step_executor
        if diagnostics is None or noise_preparer is None or step_executor is None:
            raise RuntimeError("Training pipeline instrumentation was not initialised")

        batches_per_epoch = len(self.loader)
        if batches_per_epoch <= 0:
            raise ValueError("Dataloader produced zero batches; check data configuration.")
        total_steps = (
            train_steps
            if train_steps is not None
            else int(epochs_cfg) * batches_per_epoch
        )
        if total_steps <= 0:
            raise ValueError("Total training steps must be positive.")
        effective_epochs = max(1, (total_steps + batches_per_epoch - 1) // batches_per_epoch)

        step = 0
        epochs_completed = 0
        wall_start = perf_counter()
        T, schedule = self._diffusion_params()
        coeffs = build_diffusion(T, schedule)
        total_timesteps = coeffs.num_timesteps
        snr_ratio_value = noise_preparer.snr_ratio

        checkpoint_every_raw = training_cfg.get("checkpoint_every")
        checkpoint_every = int(checkpoint_every_raw) if checkpoint_every_raw is not None else None
        if checkpoint_every is not None and checkpoint_every <= 0:
            raise ValueError("training.checkpoint_every must be positive when set.")

        eval_every_raw = training_cfg.get("eval_every")
        eval_every = int(eval_every_raw) if eval_every_raw is not None else None
        if eval_every is not None and eval_every <= 0:
            raise ValueError("training.eval_every must be positive when set.")

        eval_num_samples_raw = training_cfg.get("eval_num_samples")
        eval_num_samples = (
            int(eval_num_samples_raw)
            if eval_num_samples_raw is not None
            else None
        )
        if eval_num_samples is not None and eval_num_samples <= 0:
            raise ValueError("training.eval_num_samples must be positive when set.")

        eval_sampling_steps_raw = training_cfg.get("eval_sampling_steps")
        eval_sampling_steps = (
            int(eval_sampling_steps_raw)
            if eval_sampling_steps_raw is not None
            else None
        )
        if eval_sampling_steps is not None and eval_sampling_steps <= 0:
            raise ValueError("training.eval_sampling_steps must be positive when set.")

        eval_seed_raw = training_cfg.get("eval_seed")
        eval_seed = int(eval_seed_raw) if eval_seed_raw is not None else int(self.config.get("seed", 1337))
        eval_artifacts: list[dict[str, Any]] = []

        fft_feedback_history: Dict[str, list[float]] = defaultdict(list)
        coeff_history: Dict[str, list[float]] = {}
        batch_history: Dict[str, list[float]] = {}
        schedule_snr: Optional[torch.Tensor] = None

        for epoch in range(effective_epochs):
            for batch_idx, (xb, _) in enumerate(self.loader):
                if step >= total_steps:
                    break
                xb = xb.to(self.device)

                diagnostics.capture_initial_batch(xb)

                if step == 0:
                    schedule_snr = (
                        coeffs.sqrt_alphas_cumprod.pow(2)
                        / coeffs.sqrt_one_minus_alphas_cumprod.pow(2).clamp_min(1e-12)
                    )
                    snr_min = float(schedule_snr.min().item())
                    snr_max = float(schedule_snr.max().item())
                    self.logger.info(
                        "[SNR-DIAG] schedule_snr_range=(%.3f, %.3f) snr_ratio=%s",
                        snr_min,
                        snr_max,
                        noise_preparer.snr_ratio,
                    )

                B = xb.shape[0]
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

                diagnostics.capture_noisy_example(noise_batch.noisy, eps=noise_batch.eps)
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

                if checkpoint_every is not None and step % checkpoint_every == 0:
                    self.save_checkpoint(step)

                if eval_every is not None and step % eval_every == 0:
                    eval_dir = self.work_dir / "eval" / f"step_{step:06d}"
                    samples_dir = eval_dir / "samples"
                    diag_dir = eval_dir / "diagnostics"
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    # Preserve training RNG streams while using deterministic sampling seeds.
                    cpu_state = torch.random.get_rng_state()
                    cuda_state = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    )
                    try:
                        torch.manual_seed(eval_seed + step)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(eval_seed + step)
                        requested_samples, requested_steps, sampler = self._resolve_sampling_request(
                            num_samples=eval_num_samples,
                            sampling_steps=eval_sampling_steps,
                            sampler_type=None,
                        )
                        samples = self._sample_tensor(
                            num_samples=requested_samples,
                            sampling_steps=requested_steps,
                            sampler_type=sampler,
                            coeffs=coeffs,
                        )
                        self._write_samples(samples_dir, samples)
                        sanity_json = check_fft_sanity(
                            inputs=samples.detach().cpu(),
                            dataset_name="generated_samples",
                            out_dir=diag_dir,
                            prefix="eval_",
                        )
                        checkpoint_path = None
                        if checkpoint_every is not None:
                            checkpoint_path = self.work_dir / "checkpoints" / f"checkpoint_step_{step}.pt"
                            if not checkpoint_path.exists():
                                checkpoint_path = self.save_checkpoint(step)
                        eval_payload = {
                            "step": step,
                            "seed": eval_seed + step,
                            "sampler_type": sampler,
                            "sampling_steps": requested_steps,
                            "num_samples": requested_samples,
                            "samples_dir": str(samples_dir),
                            "diagnostics_dir": str(diag_dir),
                            "sanity_json": str(sanity_json),
                            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                        }
                        eval_path = eval_dir / "eval.json"
                        eval_dir.mkdir(parents=True, exist_ok=True)
                        eval_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
                        eval_artifacts.append(eval_payload)
                    finally:
                        torch.random.set_rng_state(cpu_state)
                        if cuda_state is not None:
                            torch.cuda.set_rng_state_all(cuda_state)

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

            epochs_completed += 1
            if step >= total_steps:
                break

        runtime_seconds = perf_counter() - wall_start

        steps_per_second = step / runtime_seconds if runtime_seconds > 0 else 0.0
        images_per_second = (
            (step * self.loader.batch_size) / runtime_seconds if runtime_seconds > 0 else 0.0
        )
        runtime_per_epoch = runtime_seconds / epochs_completed if epochs_completed > 0 else None
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
                "epochs": epochs_completed,
                "steps_per_second": steps_per_second,
                "images_per_second": images_per_second,
                "runtime_per_epoch": runtime_per_epoch,
                "loss_initial": initial_loss,
                "loss_final": final_loss,
                "loss_drop": loss_drop,
                "loss_drop_per_second": loss_drop_per_second,
                "loss_history": [float(v) for v in loss_history],
                "mae_history": [float(v) for v in mae_history],
                "snr_ratio": snr_ratio_value,
                "eval_artifacts": eval_artifacts,
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

    def _diffusion_params(self) -> Tuple[int, str]:
        diffusion_cfg = self.config.get("diffusion", {}) or {}
        T = int(diffusion_cfg.get("num_timesteps", 1000))
        if T <= 0:
            raise ValueError("diffusion.num_timesteps must be positive.")
        schedule = str(diffusion_cfg.get("beta_schedule", "cosine")).lower()
        if schedule not in {"cosine", "linear"}:
            raise ValueError(
                f"Unsupported beta_schedule '{schedule}'. Expected one of ['cosine', 'linear']."
            )
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

    def _resolve_sampling_request(
        self,
        *,
        num_samples: Optional[int],
        sampling_steps: Optional[int],
        sampler_type: Optional[str],
    ) -> tuple[int, int, str]:
        sampling_cfg = dict(self.config.get("sampling", {}) or {})
        enabled = bool(sampling_cfg.get("enabled", False))
        if not enabled:
            raise ValueError("sampling.enabled must be true for sampling/evaluation.")

        if sampler_type is None:
            sampler_raw = sampling_cfg.get("sampler_type")
            if sampler_raw is None:
                raise ValueError("sampling.sampler_type is required for sampling.")
            sampler_type = str(sampler_raw)

        if sampling_steps is None:
            if "sampling_steps" in sampling_cfg:
                sampling_steps = int(sampling_cfg["sampling_steps"])
            elif "num_steps" in sampling_cfg:
                raise ValueError(
                    "sampling.num_steps is deprecated; use sampling.sampling_steps instead."
                )
            else:
                raise ValueError("sampling.sampling_steps is required for sampling.")

        if num_samples is None:
            num_samples = int(sampling_cfg.get("num_samples", 16))

        if int(sampling_steps) <= 0:
            raise ValueError("sampling.sampling_steps must be positive.")
        if int(num_samples) <= 0:
            raise ValueError("sampling.num_samples must be positive.")

        return int(num_samples), int(sampling_steps), str(sampler_type).strip().lower()

    def _sample_tensor(
        self,
        *,
        num_samples: int,
        sampling_steps: int,
        sampler_type: str,
        coeffs: Any,
    ) -> torch.Tensor:
        sampler_impl = build_sampler(sampler_type, model=self.model, coeffs=coeffs)

        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})
        channels = int(model_cfg.get("channels") or data_cfg.get("channels", 3))
        height = int(data_cfg.get("height", 32))
        width = int(data_cfg.get("width", 32))
        shape = (channels, height, width)

        return sampler_impl.sample(
            num_samples=int(num_samples),
            shape=shape,
            num_steps=int(sampling_steps),
            device=self.device,
        )

    @staticmethod
    def _write_samples(images_dir: Path, samples: torch.Tensor) -> None:
        images_dir.mkdir(parents=True, exist_ok=True)
        nrow = max(1, int(samples.shape[0] ** 0.5))
        grid_path = images_dir / "grid.png"
        save_image((samples + 1) / 2.0, grid_path, nrow=nrow)
        for idx, img in enumerate(samples):
            save_image((img + 1) / 2.0, images_dir / f"sample_{idx:03d}.png")

    def generate_samples(
        self,
        num_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        sampler_type: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        sampling_cfg = dict(self.config.get("sampling", {}) or {})
        sampling_cfg["enabled"] = True
        self.config.setdefault("sampling", {})
        self.config["sampling"]["enabled"] = True
        if num_samples is not None:
            self.config["sampling"]["num_samples"] = int(num_samples)
        if sampling_steps is not None:
            self.config["sampling"]["sampling_steps"] = int(sampling_steps)
        if sampler_type is not None:
            self.config["sampling"]["sampler_type"] = str(sampler_type)

        T, schedule = self._diffusion_params()
        coeffs = build_diffusion(T, schedule)

        requested_samples, requested_steps, sampler = self._resolve_sampling_request(
            num_samples=num_samples,
            sampling_steps=sampling_steps,
            sampler_type=sampler_type,
        )
        samples = self._sample_tensor(
            num_samples=requested_samples,
            sampling_steps=requested_steps,
            sampler_type=sampler,
            coeffs=coeffs,
        )
        images_dir = output_dir or (self.work_dir / "images")
        self._write_samples(images_dir, samples)

        return {
            "images_dir": images_dir,
            "num_samples": requested_samples,
            "sampling_steps": requested_steps,
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
