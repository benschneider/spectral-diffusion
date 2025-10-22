"""Entry point for running Spectral Diffusion experiments."""

import logging
from typing import Any, Optional

import train_model


def main(argv: Optional[Any] = None) -> None:
    """Load configuration, launch training, and report metrics."""
    parser = train_model.build_arg_parser()
    parser.description = "Spectral Diffusion unified training entry point."
    args = parser.parse_args(args=argv)
    result = train_model.train_from_config(
        config_path=args.config,
        variant=args.variant,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    logging.getLogger("spectral_diffusion.main").info(
        "Run %s completed. Metrics stored at %s", result["run_id"], result["metrics_path"]
    )


if __name__ == "__main__":
    main()
