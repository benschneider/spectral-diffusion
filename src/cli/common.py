from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

DEFAULT_OUTPUT_DIR = Path("results")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_directories(output_dir: Path, run_id: str) -> Dict[str, Path]:
    """Create directories for a new training run."""
    run_root = output_dir / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_root,
        "logs_dir": logs_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "images_dir": images_dir,
    }


def save_config_snapshot(config: Dict[str, Any], destination: Path) -> Path:
    """Persist a copy of the run configuration for reproducibility."""
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return destination


def seed_everything(config: Dict[str, Any]) -> None:
    """Seed python, numpy, and torch RNGs for reproducibility."""
    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics: Dict[str, Any], destination: Path) -> Path:
    """Store metrics produced by the training pipeline."""
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return destination


def write_system_info(run_dir: Path, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Persist basic system metadata for the run."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }
    if extra:
        info.update(extra)

    path = run_dir / "system.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    return path


SUMMARY_HEADER = [
    "run_id",
    "config_path",
    "metrics_path",
    "timestamp",
    "loss_mean",
    "loss_initial",
    "loss_final",
    "loss_drop",
    "loss_drop_per_second",
    "runtime_seconds",
    "steps_per_second",
    "images_per_second",
    "runtime_per_epoch",
    "loss_threshold",
    "loss_threshold_steps",
    "loss_threshold_time",
    "spectral_calls",
    "spectral_time_seconds",
    "spectral_cpu_time_seconds",
    "spectral_cuda_time_seconds",
    "sampling_images_dir",
    "eval_mse",
    "eval_mae",
    "eval_psnr",
    "eval_fid",
]


def append_run_summary(
    run_id: str,
    config_path: Path,
    metrics_path: Path,
    summary_path: Path,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a new row to the experiment summary log."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not summary_path.exists() or summary_path.stat().st_size == 0

    row = [
        run_id,
        str(config_path),
        str(metrics_path),
        datetime.now(timezone.utc).isoformat(),
    ]

    def _metric_value(key: str) -> Any:
        return metrics.get(key) if metrics else None

    row.extend(
        [
            _metric_value("loss_mean"),
            _metric_value("loss_initial"),
            _metric_value("loss_final"),
            _metric_value("loss_drop"),
            _metric_value("loss_drop_per_second"),
            _metric_value("runtime_seconds"),
            _metric_value("steps_per_second"),
            _metric_value("images_per_second"),
            _metric_value("runtime_per_epoch"),
            _metric_value("loss_threshold"),
            _metric_value("loss_threshold_steps"),
            _metric_value("loss_threshold_time"),
            _metric_value("spectral_calls"),
            _metric_value("spectral_time_seconds"),
            _metric_value("spectral_cpu_time_seconds"),
            _metric_value("spectral_cuda_time_seconds"),
            _metric_value("sampling_images_dir"),
            _metric_value("eval_mse"),
            _metric_value("eval_mae"),
            _metric_value("eval_psnr"),
            _metric_value("eval_fid"),
        ]
    )

    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(SUMMARY_HEADER)
        writer.writerow(row)


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


def cleanup_run_artifacts(
    run_id: str,
    run_dir: Path,
    metrics_path: Path,
    summary_path: Path,
) -> None:
    """Remove artifacts associated with a specific run (useful for dry runs)."""
    logger = logging.getLogger("spectral_diffusion.cli")
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        logger.debug("Removed run directory %s", run_dir)
    if metrics_path.exists():
        try:
            metrics_path.unlink()
            logger.debug("Removed metrics file %s", metrics_path)
        except OSError as exc:
            logger.warning("Failed to remove metrics file %s: %s", metrics_path, exc)
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        except OSError as exc:
            logger.warning("Unable to read summary file %s: %s", summary_path, exc)
            rows = []
        if rows:
            header, *data = rows
            updated = [row for row in data if not row or row[0] != run_id]
            if len(updated) != len(data):
                try:
                    with summary_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(header)
                        writer.writerows(updated)
                    logger.debug("Removed run %s from summary file", run_id)
                except OSError as exc:
                    logger.warning("Unable to update summary file %s: %s", summary_path, exc)
