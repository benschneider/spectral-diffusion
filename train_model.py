"""Compatibility wrapper for the legacy training CLI."""

from src.cli.train import main, train_from_config

__all__ = ["main", "train_from_config"]

if __name__ == "__main__":
    main()
