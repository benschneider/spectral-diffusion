"""Compatibility wrapper for the legacy training CLI."""

from src.cli.common import cleanup_run_artifacts
from src.cli.train import main, train_from_config

__all__ = ["main", "train_from_config", "cleanup_run_artifacts"]

if __name__ == "__main__":
    main()
