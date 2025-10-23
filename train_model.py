"""Command-line script for training Spectral Diffusion models."""

import argparse
import csv
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from src.training import TrainingPipeline

DEFAULT_OUTPUT_DIR = Path("results")
SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "summary.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(description="Train Spectral Diffusion models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["baseline", "spectral"],
        default=None,
        help="Optional override for the model variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for logs, checkpoints, and metrics.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip intensive work; useful for testing the pipeline setup.",
    )
    return parser


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_variant_override(config: Dict[str, Any], variant: Optional[str]) -> None:
    """Mutate the configuration to select a specific model variant."""
    if variant:
        config.setdefault("model", {})
        config["model"]["type"] = variant


def ensure_directories(output_dir: Path, run_id: str) -> Dict[str, Path]:
    """Create directories for logs, metrics, and checkpoints."""
    run_log_dir = output_dir / "logs" / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_log_dir,
        "metrics_dir": metrics_dir,
        "images_dir": images_dir,
    }


def save_config_snapshot(config: Dict[str, Any], destination: Path) -> Path:
    """Persist a copy of the run configuration for reproducibility."""
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return destination


def save_metrics(metrics: Dict[str, Any], destination: Path) -> Path:
    """Store metrics produced by the training pipeline."""
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return destination


def append_run_summary(
    run_id: str,
    config_path: Path,
    metrics_path: Path,
    summary_path: Path = SUMMARY_PATH,
) -> None:
    """Append a new row to the experiment summary log."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([run_id, str(config_path), str(metrics_path), datetime.now(timezone.utc).isoformat()])


def configure_run_logger(logger: logging.Logger, log_file: Path) -> None:
    """Attach a file handler so each run captures logs in its own directory."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logger.level or logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def train_from_config(
    config_path: Path,
    variant: Optional[str] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Load configuration, execute training, and log artifacts."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("spectral_diffusion.train")

    config = load_config(config_path=config_path)
    apply_variant_override(config=config, variant=variant)

    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dirs = ensure_directories(output_dir=output_dir, run_id=run_identifier)

    config_copy_path = dirs["run_dir"] / "config.yaml"
    save_config_snapshot(config=config, destination=config_copy_path)

    configure_run_logger(logger, dirs["run_dir"] / "run.log")

    pipeline = TrainingPipeline(config=config, work_dir=dirs["run_dir"], logger=logger)
    if dry_run:
        logger.info("Dry run requested; skipping training loop execution.")
        metrics: Dict[str, Any] = {"status": "dry_run"}
    else:
        metrics = pipeline.run()

    metrics_path = dirs["metrics_dir"] / f"{run_identifier}.json"
    save_metrics(metrics=metrics, destination=metrics_path)
    append_run_summary(run_id=run_identifier, config_path=config_copy_path, metrics_path=metrics_path)

    return {
        "run_id": run_identifier,
        "config_path": config_copy_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def main(argv: Optional[Any] = None) -> None:
    """Parse CLI arguments and launch a training run."""
    parser = build_arg_parser()
    args = parser.parse_args(args=argv)
    result = train_from_config(
        config_path=args.config,
        variant=args.variant,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    logging.getLogger("spectral_diffusion.train").info(
        "Completed run %s with metrics at %s", result["run_id"], result["metrics_path"]
    )


if __name__ == "__main__":
    main()
