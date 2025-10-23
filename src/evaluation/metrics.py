from statistics import mean
from typing import Any, Dict, Iterable, Optional


def compute_basic_metrics(
    loss_history: Iterable[float],
    mae_history: Iterable[float],
    runtime_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate basic scalar metrics collected during training."""
    losses = list(loss_history)
    maes = list(mae_history)
    metrics: Dict[str, Any] = {
        "loss_mean": mean(losses) if losses else None,
        "loss_last": losses[-1] if losses else None,
        "mae_mean": mean(maes) if maes else None,
        "runtime_seconds": runtime_seconds,
    }
    if extra:
        metrics.update(extra)
    return metrics


def compute_fid(generated_path: str, reference_path: str) -> Optional[float]:
    """Placeholder for Frechet Inception Distance calculation."""
    _ = (generated_path, reference_path)
    return None


def compute_lpips(generated_path: str, reference_path: str) -> Optional[float]:
    """Placeholder for LPIPS metric calculation."""
    _ = (generated_path, reference_path)
    return None


def compute_runtime_stats(runtime_seconds: float, num_steps: int) -> Dict[str, Any]:
    """Compute throughput-style runtime statistics."""
    if runtime_seconds <= 0:
        return {"runtime_seconds": 0.0, "steps_per_second": None}
    steps_per_second = num_steps / runtime_seconds if num_steps > 0 else None
    return {
        "runtime_seconds": runtime_seconds,
        "steps_per_second": steps_per_second,
    }


def aggregate_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate raw metric outputs into a standardized dictionary."""
    return dict(raw_metrics)
