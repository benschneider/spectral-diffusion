from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.evaluation.metrics import compute_dataset_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated images against a reference set.")
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Directory containing generated images.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory containing reference images.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize images to this resolution before evaluation.",
    )
    parser.add_argument(
        "--use-fid",
        action="store_true",
        help="Compute FID using torchmetrics if available.",
    )
    parser.add_argument(
        "--strict-filenames",
        action="store_true",
        help="Require generated/ref directories to share identical filenames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the metrics JSON.",
    )
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Update generated_dir/metadata.json with evaluation results if present.",
    )
    return parser


def evaluate_images(
    generated_dir: Path,
    reference_dir: Path,
    image_size: Optional[Sequence[int]] = None,
    use_fid: bool = False,
    strict_filenames: bool = False,
) -> Dict[str, Any]:
    return compute_dataset_metrics(
        generated_dir=generated_dir,
        reference_dir=reference_dir,
        image_size=image_size,
        use_fid=use_fid,
        strict_filenames=strict_filenames,
    )


def evaluate_directory(
    generated_dir: Path,
    reference_dir: Path,
    image_size: Optional[Sequence[int]] = None,
    use_fid: bool = False,
    strict_filenames: bool = False,
    output_path: Optional[Path] = None,
    update_metadata: bool = False,
) -> Dict[str, Any]:
    metrics = evaluate_images(
        generated_dir=generated_dir,
        reference_dir=reference_dir,
        image_size=image_size,
        use_fid=use_fid,
        strict_filenames=strict_filenames,
    )
    metrics_path = output_path or generated_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "image_size": list(image_size) if image_size is not None else None,
        "use_fid": use_fid,
        "strict_filenames": strict_filenames,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if update_metadata:
        metadata_path = generated_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except json.JSONDecodeError:
                metadata = {}
            metadata.setdefault("evaluation", {})
            metadata["evaluation"].update(
                {
                    "metrics_path": str(metrics_path),
                    "metrics": metrics,
                }
            )
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

    return {"metrics": metrics, "metrics_path": metrics_path}


def main(argv: Optional[Any] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=argv)
    result = evaluate_directory(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        image_size=args.image_size,
        use_fid=args.use_fid,
        strict_filenames=args.strict_filenames,
        output_path=args.output,
        update_metadata=args.update_metadata,
    )

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("spectral_diffusion.evaluate").info(
        "Evaluation metrics stored at %s", result["metrics_path"]
    )


if __name__ == "__main__":
    main()
