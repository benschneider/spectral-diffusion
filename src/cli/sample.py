from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import json

from src.cli.common import load_config, seed_everything
from src.training import TrainingPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate samples from a trained diffusion model.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing checkpoints and config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to latest in <run-dir>/checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override. Defaults to <run-dir>/config.yaml.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Name for the sampling output folder (defaults to timestamp).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override sample count.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override number of diffusion steps.",
    )
    parser.add_argument(
        "--sampler-type",
        type=str,
        default=None,
        help="Override sampler type (defaults to ddpm).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def _find_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def sample_from_run(
    run_dir: Path,
    checkpoint: Optional[Path] = None,
    config_path: Optional[Path] = None,
    tag: Optional[str] = None,
    num_samples: Optional[int] = None,
    num_steps: Optional[int] = None,
    sampler_type: Optional[str] = None,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint_path = checkpoint or _find_latest_checkpoint(run_dir)
    config_file = config_path or (run_dir / "config.yaml")
    if not config_file.exists():
        raise FileNotFoundError(f"Config snapshot not found at {config_file}")

    config = load_config(config_file)
    seed_everything(config)

    sample_tag = tag or datetime.now(timezone.utc).strftime("samples_%Y%m%d_%H%M%S")
    sample_dir = run_dir / "samples" / sample_tag
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("spectral_diffusion.sample")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    pipeline = TrainingPipeline(config=config, work_dir=sample_dir, logger=logger)
    pipeline.load_checkpoint(checkpoint_path)

    sample_info = pipeline.generate_samples(
        num_samples=num_samples,
        num_steps=num_steps,
        sampler_type=sampler_type,
        output_dir=sample_dir,
    )

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_file),
        "num_samples": sample_info["num_samples"],
        "num_steps": sample_info["num_steps"],
        "sampler_type": sample_info["sampler_type"],
        "images_dir": str(sample_info["images_dir"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = sample_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "sample_dir": sample_dir,
        "checkpoint_path": checkpoint_path,
        "images_dir": sample_info["images_dir"],
        "num_samples": sample_info["num_samples"],
        "num_steps": sample_info["num_steps"],
        "metadata_path": metadata_path,
    }


def main(argv: Optional[Any] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    result = sample_from_run(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        config_path=args.config,
        tag=args.tag,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        sampler_type=args.sampler_type,
        log_level=args.log_level,
    )
    logging.getLogger("spectral_diffusion.sample").info(
        "Samples written to %s using checkpoint %s",
        result["images_dir"],
        result["checkpoint_path"],
    )


if __name__ == "__main__":
    main()
