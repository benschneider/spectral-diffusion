from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

logger = logging.getLogger(__name__)

_CROSS_DOMAIN_CACHE: Dict[str, torch.Tensor] = {}


def _flatten_parameters(params: Iterable[torch.Tensor]) -> torch.Tensor:
    flat_tensors = [p.detach().to(dtype=torch.float32, device="cpu").reshape(-1) for p in params]
    if not flat_tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_tensors)


def _load_tensor_from_file(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data.reshape(-1).to(dtype=torch.float32)
    if isinstance(data, dict):
        tensors = []
        for value in data.values():
            if torch.is_tensor(value):
                tensors.append(value.reshape(-1).to(dtype=torch.float32))
        if tensors:
            return torch.cat(tensors)
    raise ValueError(f"Unsupported tensor format at {path}")


def _load_cross_domain_vector(cfg: Dict[str, str]) -> torch.Tensor:
    cache_key = repr(sorted(cfg.items()))
    if cache_key in _CROSS_DOMAIN_CACHE:
        return _CROSS_DOMAIN_CACHE[cache_key]

    source_type = cfg.get("type", "gpt2")
    if source_type == "file":
        path = Path(cfg["path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Cross-domain weight file not found: {path}")
        vector = _load_tensor_from_file(path)
    elif source_type == "constant":
        values = cfg.get("values")
        length = int(cfg.get("length", 0))
        if values is None:
            value = float(cfg.get("value", 0.0))
            if length <= 0:
                length = 1
            vector = torch.full((length,), value, dtype=torch.float32)
        else:
            base = torch.tensor(values, dtype=torch.float32)
            if length and length > base.numel():
                repeats = (length // base.numel()) + 1
                base = base.repeat(repeats)
            if length > 0:
                base = base[:length]
            vector = base
        if vector.numel() == 0:
            raise ValueError("Constant source produced empty vector")
    elif source_type == "random_normal":
        length = int(cfg.get("length", 0))
        if length <= 0:
            raise ValueError("random_normal source requires positive 'length'")
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 1.0))
        generator = torch.Generator(device="cpu")
        seed = cfg.get("seed")
        if seed is not None:
            generator.manual_seed(int(seed))
        vector = torch.normal(mean, std, size=(length,), generator=generator)
    elif source_type == "gpt2":
        model_name = cfg.get("model_name", "gpt2")
        try:
            from transformers import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers not available; install it to use GPT-based initialization"
            ) from exc
        model = AutoModel.from_pretrained(model_name)  # type: ignore
        vector = _flatten_parameters(model.parameters())
    else:
        raise ValueError(f"Unknown cross-domain weight source '{source_type}'")

    _CROSS_DOMAIN_CACHE[cache_key] = vector
    return vector


def apply_initialization(model: torch.nn.Module, cfg: Optional[Dict[str, any]]) -> None:
    if not cfg:
        return

    strategy = cfg.get("strategy", "default")
    if strategy == "default":
        return

    if strategy == "zeros":
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        return

    if strategy == "cross_domain_flat":
        source_cfg = cfg.get("source") or {}
        try:
            vector = _load_cross_domain_vector(source_cfg)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Cross-domain initialization failed (%s); falling back to default init", exc
            )
            return

        if vector.numel() == 0:
            logger.warning("Cross-domain vector is empty; skipping initialization")
            return

        scale = float(cfg.get("scale", 1.0))
        recycle = bool(cfg.get("recycle", True))

        with torch.no_grad():
            cursor = 0
            total = vector.numel()
            for param in model.parameters():
                num = param.numel()
                if cursor >= total:
                    if recycle:
                        cursor = cursor % total
                    else:
                        break
                end = cursor + num
                if end <= total:
                    slice_ = vector[cursor:end]
                    cursor = end
                else:
                    if not recycle:
                        slice_ = torch.zeros(num, dtype=vector.dtype)
                    else:
                        needed = num
                        expanded = torch.tile(vector, (needed // total + 2,))
                        slice_ = expanded[cursor : cursor + needed]
                        cursor = (cursor + needed) % total
                param.copy_(slice_.view_as(param) * scale)
        logger.info(
            "Applied cross-domain initialization (source=%s, scale=%.4f, recycle=%s)",
            source_cfg.get("type", "unknown"),
            scale,
            recycle,
        )
        return

    raise ValueError(f"Unknown initialization strategy '{strategy}'")
