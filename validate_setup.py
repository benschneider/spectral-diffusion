"""Sanity-check script to ensure Spectral Diffusion training pipeline runs end-to-end."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from train_model import cleanup_run_artifacts, train_from_config

FAST_TRAINING_OVERRIDES: Dict[str, Any] = {
    "training": {
        "epochs": 1,
        "num_batches": 2,
        "log_every": 1,
    }
}


def _assert_exists(path: Path, run_id: Optional[str] = None) -> None:
    if not path.exists():
        suffix = f" for run {run_id}" if run_id else ""
        raise AssertionError(f"Expected artifact missing: {path}{suffix}")


def _merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _prepare_temp_config(base_path: Path, overrides: Dict[str, Any]) -> Path:
    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    merged = _merge_dict(config, overrides)
    temp_path = base_path.parent / f"{base_path.stem}_validate_{uuid4().hex}.yaml"
    with temp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle)
    return temp_path


def _run_training(
    config_path: Path,
    dry_run: bool,
    overrides: Optional[Dict[str, Any]],
    temp_configs: List[Path],
) -> Dict[str, Any]:
    effective_config = config_path
    if overrides:
        effective_config = _prepare_temp_config(config_path, overrides)
        temp_configs.append(effective_config)

    result = train_from_config(config_path=effective_config, dry_run=dry_run)
    run_id = result["run_id"]
    metrics_path = Path(result["metrics_path"])
    log_dir = Path(result["config_path"]).parent
    run_log = log_dir / "run.log"

    _assert_exists(metrics_path, run_id)
    _assert_exists(run_log, run_id)

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    if dry_run:
        if metrics.get("status") != "dry_run":
            raise AssertionError(f"Dry run should report status 'dry_run' (run {run_id}).")
    else:
        if "loss_mean" not in metrics:
            raise AssertionError(f"Expected 'loss_mean' in metrics for run {run_id}.")
        if metrics.get("status") != "ok":
            raise AssertionError(f"Unexpected run status '{metrics.get('status')}' for run {run_id}.")

    return result


def _cifar10_available(root: Path) -> bool:
    return (root / "cifar-10-batches-py").exists()


def validate(cleanup_temp: bool = True, cleanup_runs: bool = True) -> Dict[str, Any]:
    config_path = Path("configs/baseline.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Baseline config not found: configs/baseline.yaml")

    with config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    data_cfg = base_config.get("data", {})
    data_source = data_cfg.get("source", "synthetic").lower()
    data_root = Path(data_cfg.get("root", "data"))
    cifar_ready = data_source != "cifar10" or _cifar10_available(data_root)

    temp_configs: List[Path] = []
    try:
        print("Running dry-run validation...")
        dry_result = _run_training(config_path=config_path, dry_run=True, overrides=None, temp_configs=temp_configs)

        print("Running short training validation...")
        overrides = FAST_TRAINING_OVERRIDES
        if not cifar_ready:
            print(
                "⚠️ CIFAR-10 dataset not detected; falling back to synthetic data for validation.\n"
                "   Download dataset manually with:\n"
                "     mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
                "-o data/cifar-10-python.tar.gz && tar -xzf data/cifar-10-python.tar.gz -C data"
            )
            overrides = _merge_dict(
                overrides.copy(),
                {"data": {"source": "synthetic"}},
            )

        result = _run_training(
            config_path=config_path,
            dry_run=False,
            overrides=overrides,
            temp_configs=temp_configs,
        )

        summary_path = Path("results/summary.csv")
        _assert_exists(summary_path)

        print("Validation metrics:", result["metrics"])

        if cleanup_runs and dry_result["metrics"].get("status") == "dry_run":
            run_dir = Path(dry_result["config_path"]).parent
            cleanup_run_artifacts(
                run_id=dry_result["run_id"],
                run_dir=run_dir,
                metrics_path=Path(dry_result["metrics_path"]),
            )
            print(f"Removed dry-run artifacts for run: {dry_result['run_id']}")

        return result
    finally:
        if cleanup_temp:
            for temp_path in temp_configs:
                temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Spectral Diffusion training setup.")
    parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Preserve temporary config files generated during validation.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve dry-run and fast-run artifacts generated during validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        validate(cleanup_temp=not args.keep_temp_configs, cleanup_runs=not args.keep_artifacts)
    except Exception as exc:
        print(f"❌ Validation failed: {exc}")
        sys.exit(1)
    else:
        print("✅ Spectral Diffusion setup validation passed.")
