import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import shutil

from src.core import build_model, get_loss_fn
from src.core.functional import compute_snr_weight, compute_target
from src.evaluation.metrics import compute_basic_metrics
from src.spectral.fft_adapter import add_uniform_frequency_noise
from src.training.builders import build_dataloader, build_optimizer
from src.training.sampling import build_sampler
from src.training.scheduler import build_diffusion, sample_timesteps
from src.utils.sanity_checks import check_fft_sanity


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

        self._prepare_instrumentation()

        step = 0
        threshold_steps: Optional[int] = None
        threshold_time: Optional[float] = None
        wall_start = perf_counter()
        diffusion_cfg = self.config.get("diffusion", {})
        T = int(diffusion_cfg.get("num_timesteps", 1000))
        schedule = diffusion_cfg.get("beta_schedule", "cosine")
        prediction_type = diffusion_cfg.get("prediction_type", "eps")
        snr_weighting = diffusion_cfg.get("snr_weighting", False)
        snr_transform = diffusion_cfg.get("snr_transform", "snr")
        uniform_corruption = bool(diffusion_cfg.get("uniform_corruption", False))
        corruption_scale = float(
            diffusion_cfg.get(
                "uniform_corruption_scale",
                self.config.get("spectral", {}).get("uniform_corruption_scale", 1.0),
            )
        )
        target_corr = diffusion_cfg.get(
            "similarity_target", self.config.get("spectral", {}).get("similarity_target")
        )
        target_corr = float(target_corr) if target_corr is not None else None
        adaptive_rescale = bool(
            diffusion_cfg.get(
                "adaptive_rescale",
                self.config.get("spectral", {}).get("adaptive_rescale", False),
            )
        )
        fft_norm = diffusion_cfg.get(
            "fft_norm", self.config.get("spectral", {}).get("fft_norm", "ortho")
        )
        corruption_mode = diffusion_cfg.get(
            "corruption_mode",
            self.config.get("spectral", {}).get("corruption_mode", "magnitude"),
        )
        phase_std = float(
            diffusion_cfg.get(
                "phase_std", self.config.get("spectral", {}).get("phase_std", 0.0)
            )
        )
        snr_ratio_cfg = diffusion_cfg.get(
            "snr_ratio",
            self.config.get("spectral", {}).get("snr_ratio"),
        )
        snr_ratio = float(snr_ratio_cfg) if snr_ratio_cfg is not None else None
        dc_scale_cfg = diffusion_cfg.get(
            "dc_scale_factor",
            self.config.get("spectral", {}).get("dc_scale_factor", 0.1),
        )
        dc_scale_factor = float(dc_scale_cfg)

        coeffs = build_diffusion(T, schedule)

        for epoch in range(epochs):
            for batch_idx, (xb, _) in enumerate(self.loader):
                xb = xb.to(self.device)

                if not self._initial_batch_captured:
                    self._capture_initial_batch(xb)

                B = xb.shape[0]
                t = sample_timesteps(B, T, xb.device)

                sqrt_alpha_t = coeffs.sqrt_alphas_cumprod[t].view(B, 1, 1, 1).to(self.device)
                sqrt_one_minus_t = (
                    coeffs.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1).to(self.device)
                )

                base_noise = torch.randn_like(xb)
                noise_stats: Dict[str, float] = {}
                x_t, eps = add_uniform_frequency_noise(
                    xb,
                    base_noise,
                    sqrt_alpha_t=sqrt_alpha_t,
                    sqrt_one_minus_alpha_t=sqrt_one_minus_t,
                    uniform_corruption=uniform_corruption,
                    strength=corruption_scale,
                    mode=corruption_mode,
                    phase_std=phase_std,
                    target_corr=target_corr,
                    adaptive_rescale=adaptive_rescale,
                    stats=noise_stats,
                    fft_norm=fft_norm,
                    snr_ratio=snr_ratio,
                    dc_scale_factor=dc_scale_factor,
                    return_noise=True,
                )

                if not self._noisy_capture_done:
                    self._capture_noisy_example(x_t)

                pred = self.model(x_t, t)
                if not self._phase_demo_captured:
                    self._capture_phase_demo()

                eps_norm = float(eps.view(B, -1).norm(dim=1).mean().detach().cpu())
                self.noise_norm_history.append(eps_norm)
                self.noise_norm_steps.append(step + 1)

                target = compute_target(
                    prediction_type, xb, x_t, eps, sqrt_alpha_t, sqrt_one_minus_t
                )

                residual = pred - target
                weight = None
                if snr_weighting:
                    weight = compute_snr_weight(
                        sqrt_alpha_t, sqrt_one_minus_t, transform=snr_transform
                    )

                loss = self.loss_fn(residual, weight)
                mae = F.l1_loss(pred, target)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = self._record_gradients(step)
                self.optimizer.step()

                step += 1
                loss_val = float(loss.detach().cpu())
                self.loss_history.append(loss_val)
                self.mae_history.append(float(mae.detach().cpu()))
                self.loss_steps.append(step)
                if step % log_every == 0:
                    self.logger.info("epoch %d step %d loss %.5f", epoch, step, loss_val)
                    mean_val = noise_stats.get("noisy_mean") if noise_stats else None
                    std_val = noise_stats.get("noisy_std") if noise_stats else None
                    self.logger.debug(
                        "spectral noise stats: snr_ratio=%.3f mean=%.3f std=%.3f",
                        snr_ratio,
                        mean_val if mean_val is not None else float("nan"),
                        std_val if std_val is not None else float("nan"),
                    )
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
        initial_loss = self.loss_history[0] if self.loss_history else None
        final_loss = self.loss_history[-1] if self.loss_history else None
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
            loss_history=self.loss_history,
            mae_history=self.mae_history,
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
                "loss_history": [float(v) for v in self.loss_history],
                "mae_history": [float(v) for v in self.mae_history],
                "snr_ratio": snr_ratio,
                "dc_scale_factor": dc_scale_factor,
            },
        )
        self._finalise_diagnostics()
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

    def _resolve_taguchi_root(self) -> Optional[Path]:
        parent = self.work_dir.parent
        if parent.name == "runs":
            return parent.parent
        return None

    def _prepare_instrumentation(self) -> None:
        self.loss_history: List[float] = []
        self.mae_history: List[float] = []
        self.loss_steps: List[int] = []
        self.grad_norm_history: List[float] = []
        self.grad_norm_steps: List[int] = []
        self.noise_norm_history: List[float] = []
        self.noise_norm_steps: List[int] = []
        self._initial_batch_captured = False
        self._phase_demo_captured = False
        self._noisy_capture_done = False
        self._factor_levels: Dict[str, Dict[str, Any]] = (
            self.config.get("taguchi", {}).get("factor_levels", {}) or {}
        )
        self._taguchi_root = self._resolve_taguchi_root()
        self.dataset_name = str(self.config.get("data", {}).get("source", "unknown"))
        self.run_id = self.work_dir.name
        self._aggregate_base = self._taguchi_root or self.work_dir
        self._sanity_dir = self._aggregate_base / "sanity"
        self._sanity_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostics_dir = self.work_dir / "diagnostics"
        self._diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def _get_factor_dir(self, factor: str) -> Optional[Path]:
        if factor not in self._factor_levels:
            return None
        meta = self._factor_levels[factor]
        level = str(meta.get("level_label", meta.get("level_index", "unknown")))
        path = self._aggregate_base / "factors" / factor / level
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _capture_initial_batch(self, xb: torch.Tensor) -> None:
        batch_cpu = xb.detach().cpu()
        prefix = f"{self.run_id}_"
        stats_path = check_fft_sanity(
            batch_cpu,
            self.dataset_name,
            self._sanity_dir,
            prefix=prefix,
        )

        base_name = stats_path.stem
        spatial_src = stats_path.with_name(f"{base_name}_spatial.png")
        fft_src = stats_path.with_name(f"{base_name}_fft_mag.png")

        if not self._factor_levels:
            self._initial_batch_captured = True
            return

        for factor in self._factor_levels.keys():
            factor_dir = self._get_factor_dir(factor)
            if factor_dir is None:
                continue
            if spatial_src.exists():
                dest = factor_dir / f"demo_spatial_{self.run_id}.png"
                shutil.copy(spatial_src, dest)
            if fft_src.exists():
                dest = factor_dir / f"demo_fft_{self.run_id}.png"
                shutil.copy(fft_src, dest)

        self._initial_batch_captured = True

    def _capture_phase_demo(self) -> None:
        if self._phase_demo_captured:
            return
        factor_dir = self._get_factor_dir("phase_attention_capacity")
        if factor_dir is None:
            self._phase_demo_captured = True
            return
        pcm = getattr(self.model, "pcm", None)
        if pcm is None:
            self._phase_demo_captured = True
            return
        weights = getattr(pcm, "last_attention_map", None)
        if weights is None:
            return
        attn = weights.detach().cpu()
        attn = attn.mean(dim=0)
        if attn.ndim == 2:
            seq_len = attn.shape[-1]
            side = int(seq_len ** 0.5)
            if side * side == seq_len:
                attn = attn.reshape(side, side)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(attn.numpy(), cmap="magma")
        ax.set_title("Phase Attention")
        ax.axis("off")
        fig.tight_layout()
        out_path = factor_dir / f"demo_phase_attention_{self.run_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self._phase_demo_captured = True

    def _capture_noisy_example(self, x_t: torch.Tensor) -> None:
        if self._noisy_capture_done:
            return
        prefix = f"{self.run_id}_noisy_"
        stats_path = check_fft_sanity(
            x_t.detach().cpu(),
            f"{self.dataset_name}_noisy",
            self._sanity_dir,
            prefix=prefix,
        )
        base_name = stats_path.stem
        spatial_src = stats_path.with_name(f"{base_name}_spatial.png")
        fft_src = stats_path.with_name(f"{base_name}_fft_mag.png")

        for factor in self._factor_levels.keys():
            factor_dir = self._get_factor_dir(factor)
            if factor_dir is None:
                continue
            if spatial_src.exists():
                shutil.copy(spatial_src, factor_dir / f"demo_noisy_spatial_{self.run_id}.png")
            if fft_src.exists():
                shutil.copy(fft_src, factor_dir / f"demo_noisy_fft_{self.run_id}.png")

        self._noisy_capture_done = True

    def _record_gradients(self, step: int) -> Optional[float]:
        total_norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total_norm_sq += float(grad.pow(2).sum().cpu())
        if total_norm_sq == 0.0:
            return None
        total_norm = float(total_norm_sq ** 0.5)
        self.grad_norm_history.append(total_norm)
        self.grad_norm_steps.append(step + 1)
        return total_norm

    def _finalise_diagnostics(self) -> None:
        if self.loss_history:
            self._plot_loss_and_gradients()
        self._write_factor_loss_snapshot()
        self._write_noise_norm_plot()

    def _plot_loss_and_gradients(self) -> None:
        steps = np.array(self.loss_steps, dtype=float)
        losses = np.array(self.loss_history, dtype=float)
        if steps.size == 0:
            return
        loss_grad = np.gradient(losses, steps) if steps.size > 1 else np.zeros_like(losses)

        fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
        axes[0].plot(steps, losses, label="Loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss over Steps")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, loss_grad, label="d(Loss)/d(step)", color="orange")
        axes[1].set_ylabel("Loss Slope")
        axes[1].set_title("Loss Slope per Step")
        axes[1].grid(True, alpha=0.3)

        if self.grad_norm_history:
            g_steps = np.array(self.grad_norm_steps, dtype=float)
            g_norms = np.array(self.grad_norm_history, dtype=float)
            axes[2].plot(g_steps, g_norms, label="Grad Norm", color="green")
        axes[2].set_ylabel("Grad Norm")
        axes[2].set_xlabel("Step")
        axes[2].set_title("Gradient Norm Evolution")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        diag_path = self._diagnostics_dir / "loss_gradients.png"
        fig.savefig(diag_path, dpi=150)
        plt.close(fig)

        aggregate_dir = self._aggregate_base / "diagnostics"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(diag_path, aggregate_dir / f"{self.run_id}_loss_gradients.png")

        for factor in self._factor_levels:
            factor_dir = self._get_factor_dir(factor)
            if factor_dir is None:
                continue
            shutil.copy(diag_path, factor_dir / f"demo_loss_grad_{self.run_id}.png")

    def _write_factor_loss_snapshot(self) -> None:
        factor_dir = self._get_factor_dir("spectral_loss_weighting")
        if factor_dir is None or not self.loss_history:
            return
        last_steps = np.array(self.loss_steps[-50:], dtype=float)
        last_losses = np.array(self.loss_history[-50:], dtype=float)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(last_steps, last_losses, marker="o")
        ax.set_title("Recent Loss (50 steps)")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = factor_dir / f"demo_loss_tail_{self.run_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _write_noise_norm_plot(self) -> None:
        factor_dir = self._get_factor_dir("sampler_type")
        if factor_dir is None or not self.noise_norm_history:
            return
        steps = np.array(self.noise_norm_steps, dtype=float)
        norms = np.array(self.noise_norm_history, dtype=float)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(steps, norms, color="purple")
        ax.set_title("Noise Norm vs Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("‖ε‖")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = factor_dir / f"demo_noise_norm_{self.run_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

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
