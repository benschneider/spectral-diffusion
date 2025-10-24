"""Command-line entry points for Spectral Diffusion."""

from .evaluate import evaluate_directory, evaluate_images, main as evaluate_main
from .sample import main as sample_main, sample_from_run
from .train import main as train_main, train_from_config

__all__ = [
    "train_main",
    "train_from_config",
    "sample_main",
    "sample_from_run",
    "evaluate_main",
    "evaluate_directory",
    "evaluate_images",
]
