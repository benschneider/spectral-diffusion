from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class DiagnosticsPlotter:
    """Render diagnostic plots for training runs."""

    def loss_and_gradients(
        self,
        loss_steps: Sequence[int],
        loss_history: Sequence[float],
        grad_steps: Sequence[int],
        grad_history: Sequence[float],
        output_dir: Path,
        run_id: str,
        filename: str = "loss_gradients.png",
    ) -> Path:
        if not loss_steps or not loss_history:
            raise ValueError("Loss history is required to plot diagnostics")

        steps = np.asarray(loss_steps, dtype=float)
        losses = np.asarray(loss_history, dtype=float)
        gradients = (
            np.gradient(losses, steps)
            if steps.size > 1
            else np.zeros_like(losses, dtype=float)
        )

        fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

        axes[0].plot(steps, losses, label="Loss")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss over Steps")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, gradients, label="d(Loss)/d(step)", color="orange")
        axes[1].set_ylabel("Loss Slope")
        axes[1].set_title("Loss Slope per Step")
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel("Grad Norm")
        axes[2].set_xlabel("Step")
        axes[2].set_title("Gradient Norm Evolution")
        axes[2].grid(True, alpha=0.3)
        if grad_steps and grad_history:
            g_steps = np.asarray(grad_steps, dtype=float)
            g_norms = np.asarray(grad_history, dtype=float)
            axes[2].plot(g_steps, g_norms, label="Grad Norm", color="green")

        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def recent_loss_tail(
        self,
        loss_steps: Sequence[int],
        loss_history: Sequence[float],
        target_dir: Optional[Path],
        run_id: str,
        window: int = 50,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        if target_dir is None or not loss_history:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = filename or f"demo_loss_tail_{run_id}.png"
        steps = np.asarray(loss_steps[-window:], dtype=float)
        losses = np.asarray(loss_history[-window:], dtype=float)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(steps, losses, marker="o")
        ax.set_title("Recent Loss (50 steps)")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = target_dir / file_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def noise_norm(
        self,
        noise_steps: Sequence[int],
        noise_history: Sequence[float],
        target_dir: Optional[Path],
        run_id: str,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        if target_dir is None or not noise_history:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = filename or f"demo_noise_norm_{run_id}.png"
        steps = np.asarray(noise_steps, dtype=float)
        norms = np.asarray(noise_history, dtype=float)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(steps, norms, color="purple")
        ax.set_title("Noise Norm vs Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("‖ε‖")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = target_dir / file_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def phase_attention(self, attention: np.ndarray, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(attention, cmap="magma")
        ax.set_title("Phase Attention")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(target_path, dpi=150)
        plt.close(fig)
        return target_path
