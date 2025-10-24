from __future__ import annotations

import argparse
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.cli.common import (
    DEFAULT_OUTPUT_DIR,
    append_run_summary,
    cleanup_run_artifacts,
    configure_run_logger,
    ensure_directories,
    load_config,
    save_config_snapshot,
    save_metrics,
    seed_everything,
    write_system_info,
)
from src.training import TrainingPipeline


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
        choices=["baseline", "spectral", "spectral_deep"],
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
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="Emit structured JSONL logs alongside the plain text train.log.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove generated artifacts after completion (useful for dry runs).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def apply_variant_override(config: Dict[str, Any], variant: Optional[str]) -> None:
    """Mutate the configuration to select a specific model variant."""
    if variant:
        alias = {
            "baseline": "baseline",
            "spectral": "unet_spectral",
            "spectral_deep": "unet_spectral_deep",
        }
        config.setdefault("model", {})
        config["model"]["type"] = alias.get(variant, variant)


def train_from_config(
    config_path: Path,
    variant: Optional[str] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_id: Optional[str] = None,
    dry_run: bool = False,
    cleanup: bool = False,
    log_level: str = "INFO",
    json_log: bool = False,
) -> Dict[str, Any]:
    """Load configuration, execute training, and log artifacts."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger("train")

    config = load_config(config_path=config_path)
    apply_variant_override(config=config, variant=variant)
    seed_everything(config)

    run_identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dirs = ensure_directories(output_dir=output_dir, run_id=run_identifier)
    summary_path = output_dir / "summary.csv"

    config_copy_path = dirs["run_dir"] / "config.yaml"
    save_config_snapshot(config=config, destination=config_copy_path)

    write_system_info(
        dirs["run_dir"],
        extra={
            "config_path": str(config_path),
            "run_id": run_identifier,
        },
    )

    train_log_path = dirs["logs_dir"] / "train.log"
    json_log_path = dirs["logs_dir"] / "train.jsonl" if json_log else None
    configure_run_logger(logger, train_log_path, json_log_file=json_log_path)
    if json_log_path is not None:
        json_log_path.parent.mkdir(parents=True, exist_ok=True)
        with json_log_path.open("a", encoding="utf-8") as handle:
            json.dump(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "name": "train",
                    "event": "run_start",
                    "run_id": run_identifier,
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
    logger.info("Starting training run %s", run_identifier)

    pipeline = TrainingPipeline(config=config, work_dir=dirs["run_dir"], logger=logger)
    if dry_run:
        logger.info("Dry run requested; skipping training loop execution.")
        metrics: Dict[str, Any] = {"status": "dry_run"}
        detailed_metrics = None
        checkpoint_path: Optional[Path] = None
    else:
        metrics = pipeline.run()
        detailed_metrics = metrics
        checkpoint_path = pipeline.save_checkpoint(step=int(metrics.get("num_steps", 0)))

    metrics_path = dirs["metrics_dir"] / f"{run_identifier}.json"
    save_metrics(metrics=metrics, destination=metrics_path)
    append_run_summary(
        run_id=run_identifier,
        config_path=config_copy_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        metrics=detailed_metrics,
    )

    if cleanup or dry_run:
        cleanup_run_artifacts(
            run_id=run_identifier,
            run_dir=dirs["run_dir"],
            metrics_path=metrics_path,
            summary_path=summary_path,
        )
        checkpoint_path = None

    for handler in logger.handlers:  # pragma: no cover - harmless flush
        if hasattr(handler, "flush"):
            handler.flush()

    return {
        "run_id": run_identifier,
        "config_path": config_copy_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
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
        cleanup=args.cleanup,
        log_level=args.log_level,
        json_log=args.json_log,
    )
    logging.getLogger("spectral_diffusion.train").info(
        "Completed run %s with metrics at %s", result["run_id"], result["metrics_path"]
    )


if __name__ == "__main__":
    main()
