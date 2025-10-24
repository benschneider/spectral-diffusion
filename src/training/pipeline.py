import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from src.core import build_model, get_loss_fn
from src.core.functional import compute_snr_weight, compute_target
from src.evaluation.metrics import compute_basic_metrics, compute_dataset_metrics
from src.training.sampling import sample_ddpm
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

    def _make_dataloader(self) -> DataLoader:
        """Build a dataloader based on configuration."""
        data_cfg = self.config.get("data", {})
        source = str(data_cfg.get("source", "synthetic")).lower()
        if source == "synthetic":
            return self._make_synthetic_dataloader(data_cfg=data_cfg)
        if source == "cifar10":
            return self._make_cifar10_dataloader(data_cfg=data_cfg)
        raise ValueError(f"Unsupported data source: {source}")

    def _make_synthetic_dataloader(self, data_cfg: Dict[str, Any]) -> DataLoader:
        bs = int(self.config.get("training", {}).get("batch_size", 32))
        n_setting = self.config.get("training", {}).get("num_batches", 50)
        n = int(n_setting) if str(n_setting).isdigit() else 50
        c = int(data_cfg.get("channels", 3))
        h = int(data_cfg.get("height", 32))
        w = int(data_cfg.get("width", 32))
        x = torch.randn(n * bs, c, h, w)
        y = torch.randn_like(x)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    def _make_cifar10_dataloader(self, data_cfg: Dict[str, Any]) -> DataLoader:
        bs = int(self.config.get("training", {}).get("batch_size", 32))
        target_h = int(data_cfg.get("height", 32))
        target_w = int(data_cfg.get("width", 32))
        num_workers = int(data_cfg.get("num_workers", 0))
        download = bool(data_cfg.get("download", False))

        transform = transforms.Compose(
            [
                transforms.Resize((target_h, target_w)),
                transforms.ToTensor(),
            ]
        )
        try:
            base_dataset = datasets.CIFAR10(
                root=data_cfg.get("root", "data"),
                train=True,
                download=download,
                transform=transform,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "CIFAR-10 dataset not found. Download it manually with:\n"
                "  mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
                "-o data/cifar-10-python.tar.gz && tar -xzf data/cifar-10-python.tar.gz -C data\n"
                "Then rerun training or set data.source to 'synthetic'."
            ) from exc

        class ReconstructionWrapper(Dataset):
            def __init__(self, dataset) -> None:
                self.dataset = dataset

            def __len__(self) -> int:
                return len(self.dataset)

            def __getitem__(self, idx: int):
                img, _ = self.dataset[idx]
                return img, img

        return DataLoader(
            ReconstructionWrapper(base_dataset),
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.config.get("optim", {}).get("lr", 1e-4))
        wd = float(self.config.get("optim", {}).get("weight_decay", 0.0))
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def setup(self) -> None:
        """Prepare datasets, optimizers, and other resources."""
        self.logger.debug("Setting up training pipeline with config: %s", self.config)
        self.loader = self._make_dataloader()
        self.optimizer = self._make_optimizer()

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
        coeffs = coeffs

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

        # Sampling (optional)
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        channels = int(model_cfg.get("channels") or data_cfg.get("channels", 3))
        height = int(data_cfg.get("height", 32))
        width = int(data_cfg.get("width", 32))
        images_dir = self.run_sampling(coeffs, (channels, height, width))

        if images_dir:
            metrics["sampling_images_dir"] = str(images_dir)

        evaluation_cfg = self.config.get("evaluation", {})
        if images_dir and evaluation_cfg.get("reference_dir"):
            eval_metrics = compute_dataset_metrics(
                generated_dir=images_dir,
                reference_dir=evaluation_cfg["reference_dir"],
                image_size=evaluation_cfg.get("image_size"),
                use_fid=bool(evaluation_cfg.get("use_fid", False)),
                strict_filenames=False,
            )
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        if training_stats and hasattr(self.model, "spectral_stats"):
            sampling_stats = self.model.spectral_stats()
            metrics["sampling_spectral_calls"] = sampling_stats.get("spectral_calls", 0.0)
            metrics["sampling_spectral_time_seconds"] = sampling_stats.get("spectral_time_seconds", 0.0)
            metrics["sampling_spectral_cpu_time_seconds"] = sampling_stats.get("spectral_cpu_time_seconds", 0.0)
            metrics["sampling_spectral_cuda_time_seconds"] = sampling_stats.get("spectral_cuda_time_seconds", 0.0)
            metrics["spectral_calls"] = training_stats.get("spectral_calls", 0.0)
            metrics["spectral_time_seconds"] = training_stats.get("spectral_time_seconds", 0.0)
            metrics["spectral_cpu_time_seconds"] = training_stats.get("spectral_cpu_time_seconds", 0.0)
            metrics["spectral_cuda_time_seconds"] = training_stats.get("spectral_cuda_time_seconds", 0.0)

        self.logger.info("Training metrics: %s", metrics)
        return metrics

    def run_sampling(self, coeffs, shape) -> Optional[Path]:
        sampling_cfg = self.config.get("sampling", {})
        if not sampling_cfg.get("enabled", False):
            return None

        sampler_type = sampling_cfg.get("sampler_type", "ddpm").lower()
        num_samples = int(sampling_cfg.get("num_samples", 16))
        num_steps = int(sampling_cfg.get("num_steps", coeffs.betas.shape[0]))

        images_dir = self.work_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        samples = sample_ddpm(
            model=self.model,
            coeffs=coeffs,
            num_samples=num_samples,
            shape=shape,
            num_steps=num_steps,
            device=self.device,
        )

        from torchvision.utils import save_image

        grid_path = images_dir / "grid.png"
        save_image((samples + 1) / 2.0, grid_path, nrow=max(1, int(num_samples**0.5)))

        for idx, img in enumerate(samples):
            save_image((img + 1) / 2.0, images_dir / f"sample_{idx:03d}.png")

        return images_dir

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_dir = self.work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save({"model": self.model.state_dict()}, checkpoint_path)
        return checkpoint_path
