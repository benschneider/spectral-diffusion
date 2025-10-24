import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.core import build_model, get_loss_fn
from src.core.functional import compute_snr_weight, compute_target
from src.evaluation.metrics import compute_basic_metrics
from src.training.builders import build_dataloader, build_optimizer
from src.training.sampling import build_sampler
from src.training.scheduler import build_diffusion, sample_timesteps


class TrainingPipeline:
    """Unified training pipeline for baseline and spectral diffusion models."""

    def __init__(
        self,
        config: Dict[str, Any],
        work_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.work_dir = work_dir
        self.logger = logger or logging.getLogger(__name__)
        self.model = build_model(config.get("model", {}))
        self.loss_fn = get_loss_fn(config.get("loss", {}))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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

        step = 0
        loss_history, mae_history = [], []
        threshold_steps: Optional[int] = None
        threshold_time: Optional[float] = None
        wall_start = perf_counter()
        diffusion_cfg = self.config.get("diffusion", {})
        T = int(diffusion_cfg.get("num_timesteps", 1000))
        schedule = diffusion_cfg.get("beta_schedule", "cosine")
        prediction_type = diffusion_cfg.get("prediction_type", "eps")
        snr_weighting = diffusion_cfg.get("snr_weighting", False)
        snr_transform = diffusion_cfg.get("snr_transform", "snr")

        coeffs = build_diffusion(T, schedule)

        for epoch in range(epochs):
            for batch_idx, (xb, _) in enumerate(self.loader):
                xb = xb.to(self.device)
                B = xb.shape[0]
                t = sample_timesteps(B, T, xb.device)

                alpha_t = coeffs.sqrt_alphas_cumprod[t].view(B, 1, 1, 1).to(self.device)
                sigma_t = coeffs.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1).to(self.device)

                eps = torch.randn_like(xb)
                x_t = alpha_t * xb + sigma_t * eps

                pred = self.model(x_t, t)
                target = compute_target(prediction_type, xb, x_t, eps, alpha_t, sigma_t)

                residual = pred - target
                weight = None
                if snr_weighting:
                    weight = compute_snr_weight(alpha_t, sigma_t, transform=snr_transform)

                loss = self.loss_fn(residual, weight)
                mae = F.l1_loss(pred, target)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                step += 1
                loss_val = float(loss.detach().cpu())
                loss_history.append(loss_val)
                mae_history.append(float(mae.detach().cpu()))
                if step % log_every == 0:
                    self.logger.info("epoch %d step %d loss %.5f", epoch, step, loss_val)
                if (
                    loss_threshold is not None
                    and threshold_steps is None
                    and loss_val <= loss_threshold
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
            },
        )
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
