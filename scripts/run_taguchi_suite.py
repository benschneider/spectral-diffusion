#!/usr/bin/env python
"""Build and execute configurable Taguchi suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import yaml

DEFAULT_GENERATED_DIR = Path("configs/taguchi/generated")
OA_DESIGN_MAP = {
    "L27": Path("configs/taguchi/L27_extended.csv"),
    "L23": Path("configs/taguchi/L23_synthetic.csv"),
}


class SuitePlan(TypedDict):
    suite: str
    description: str
    include: List[str]
    fixed: Dict[str, Any]
    oa: str
    oa_path: Path
    randomize: bool
    seeds: List[int]
    base_config: Optional[Path]


class ArtifactPaths(TypedDict):
    registry: Path
    design: Path
    manifest: Path
    base_dir: Path


class RuntimeEstimate(TypedDict):
    per_run_seconds: float
    runs: int
    total_seconds: float


def _ordered_active_factors(plan: SuitePlan) -> List[str]:
    seen = set()
    ordered = []
    for factor in plan["include"] + list(plan["fixed"].keys()):
        if factor not in seen:
            ordered.append(factor)
            seen.add(factor)
    return ordered


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        __import__("yaml").safe_dump(data, handle, sort_keys=False)
    return path


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def cast_value(value: str, kind: str) -> Any:
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return value


def build_plan(
    suite_name: str,
    suites: Dict[str, Any],
    catalog: Dict[str, Any],
    overrides: Iterable[str],
    base_config_override: Optional[Path] = None,
    oa_override: Optional[str] = None,
    oa_path_override: Optional[Path] = None,
    randomize_override: Optional[bool] = None,
) -> SuitePlan:
    suite_data = suites["suites"].get(suite_name)
    if suite_data is None:
        raise ValueError(f"Suite '{suite_name}' not found")
    include = list(suite_data.get("include", []))
    fixed: Dict[str, Any] = dict(suite_data.get("fixed", {}))
    base_config = Path(suite_data["base_config"]) if suite_data.get("base_config") else None
    if base_config_override is not None:
        base_config = base_config_override
    oa_name = oa_override or suite_data.get("oa")
    if not oa_name:
        raise ValueError("No OA name defined for suite and no --oa override provided")
    oa_path = oa_path_override or Path(suite_data.get("oa_path") or OA_DESIGN_MAP.get(oa_name))
    if oa_path is None or not oa_path.exists():
        raise FileNotFoundError(f"OA design for '{oa_name}' not found at {oa_path}")
    randomize = suite_data.get("randomize", True)
    if randomize_override is not None:
        randomize = randomize_override
    seeds = list(suite_data.get("seeds", []))

    catalog_factors = catalog.get("factors", {})
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be in key=value form")
        key, value = override.split("=", 1)
        if key not in catalog_factors:
            raise ValueError(f"Unknown factor '{key}' (not in catalog)")
        info = catalog_factors[key]
        kind = info.get("kind", "enum")
        fixed[key] = cast_value(value, kind)
        if key not in include:
            include.append(key)

    active_factors = []
    seen = set()
    for entry in include + list(fixed.keys()):
        if entry in catalog_factors and entry not in seen:
            active_factors.append(entry)
            seen.add(entry)
    plan: SuitePlan = {
        "suite": suite_name,
        "description": suite_data.get("description", ""),
        "include": include,
        "fixed": fixed,
        "oa": oa_name,
        "oa_path": oa_path,
        "randomize": bool(randomize),
        "seeds": seeds,
        "base_config": base_config,
    }
    return plan


def _workspace_for(plan: SuitePlan, generated_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return generated_dir / f"{plan['suite']}_{timestamp}"


def generate_factor_registry(plan: SuitePlan, catalog: Dict[str, Any], out_path: Path) -> Path:
    catalog_factors = catalog.get("factors", {})
    data: Dict[str, Any] = {"factors": {}}
    for factor in _ordered_active_factors(plan):
        info = catalog_factors[factor]
        levels = info.get("levels", [])
        if factor in plan["fixed"]:
            levels = [plan["fixed"][factor]]
        data["factors"][factor] = {
            "description": info.get("doc", ""),
            "levels": levels,
        }
    return save_yaml(data, out_path)


def generate_design_csv(plan: SuitePlan, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(plan["oa_path"], out_path)
    return out_path


def estimate_runtime(plan: SuitePlan, catalog: Dict[str, Any], design_path: Path) -> RuntimeEstimate:
    estimation = catalog.get("estimation", {})
    base = float(estimation.get("base_seconds_per_run", 120.0))
    multipliers = estimation.get("multipliers", {})
    catalog_factors = catalog.get("factors", {})
    total_multiplier = 1.0
    level_map: Dict[str, Any] = {}
    for factor in plan["include"] + list(plan["fixed"].keys()):
        info = catalog_factors[factor]
        if factor in plan["fixed"]:
            level = plan["fixed"][factor]
        else:
            level = info.get("fixed_default")
            if level is None:
                level = info.get("levels", [None])[0]
        level_map[factor] = level
        entries = multipliers.get(factor, {})
        mult = entries.get(str(level)) if level is not None else None
        if mult is None and isinstance(level, (float, int)):
            mult = entries.get(str(int(level)))
        total_multiplier *= mult or 1.0
    per_run = base * total_multiplier
    with open(design_path, "r", encoding="utf-8") as handle:
        runs = sum(1 for _ in handle) - 1
    return {
        "per_run_seconds": per_run,
        "runs": max(1, runs),
        "total_seconds": per_run * max(1, runs),
    }


def build_manifest(
    plan: SuitePlan,
    catalog_path: Path,
    registry_path: Path,
    design_path: Path,
    estimate: RuntimeEstimate,
    workspace: Path,
) -> Dict[str, Any]:
    return {
        "suite": plan["suite"],
        "description": plan["description"],
        "oa": plan["oa"],
        "oa_path": str(plan["oa_path"]),
        "base_config": str(plan["base_config"]) if plan["base_config"] else None,
        "registry": str(registry_path),
        "design": str(design_path),
        "randomize": plan["randomize"],
        "seeds": plan["seeds"],
        "estimate": estimate,
        "catalog_hash": hash_file(catalog_path),
        "workspace": str(workspace),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def dump_manifest(manifest: Dict[str, Any], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def generate_artifacts(
    plan: SuitePlan,
    catalog_path: Path,
    catalog: Dict[str, Any],
    base_dir: Path,
) -> Tuple[ArtifactPaths, RuntimeEstimate]:
    workspace = _workspace_for(plan, base_dir)
    registry_path = workspace / f"factor_registry_{plan['suite']}.yaml"
    design_path = workspace / f"design_{plan['oa']}_{plan['suite']}.csv"
    manifest_path = workspace / f"run_manifest_{plan['suite']}.json"
    registry = generate_factor_registry(plan, catalog, registry_path)
    design = generate_design_csv(plan, design_path)
    estimate = estimate_runtime(plan, catalog, design)
    manifest = build_manifest(plan, catalog_path, registry, design, estimate, workspace)
    dump_manifest(manifest, manifest_path)
    return (
        {
            "registry": registry,
            "design": design,
            "manifest": manifest_path,
            "base_dir": workspace / "results",
        },
        estimate,
    )


def run_suite(
    plan: SuitePlan,
    artifacts: ArtifactPaths,
    runner: Path,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print("Dry run requested; skipping execution step.")
        return
    env = os.environ.copy()
    env["TAGUCHI_FACTOR_REGISTRY"] = str(artifacts["registry"])
    env["TAGUCHI_ARRAY_PATH"] = str(artifacts["design"])
    if plan["base_config"]:
        env["TAGUCHI_BASE_CONFIG"] = str(plan["base_config"])
    env["TAGUCHI_RANDOMIZE"] = "1" if plan["randomize"] else "0"
    if plan["seeds"]:
        env["TAGUCHI_MAPPING_SEED"] = str(plan["seeds"][0])
    else:
        env.pop("TAGUCHI_MAPPING_SEED", None)
    artifacts["base_dir"].mkdir(parents=True, exist_ok=True)
    cmd = [str(runner), str(artifacts["base_dir"])]
    print("Executing suite with command:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Taguchi experiment suite")
    parser.add_argument("--suite", required=True, help="Suite name from configs/taguchi/suites.yaml")
    parser.add_argument(
        "--catalog", type=Path, default=Path("configs/taguchi/factor_catalog.yaml"), help="Factor catalog path"
    )
    parser.add_argument(
        "--suites", type=Path, default=Path("configs/taguchi/suites.yaml"), help="Suite definitions"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a factor level (format key=value)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate artifacts without executing the suite")
    parser.add_argument("--estimate-only", action="store_true", help="Only print runtime estimates")
    parser.add_argument("--runner", type=Path, default=Path("scripts/run_full_report_32x32.sh"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--base-config", type=Path, help="Override base config path for the suite")
    parser.add_argument("--oa", type=str, help="Override the OA design name used by the suite")
    parser.add_argument("--oa-path", type=Path, help="Direct path to the Taguchi array CSV")
    rand_group = parser.add_mutually_exclusive_group()
    rand_group.add_argument("--randomize", dest="randomize", action="store_true")
    rand_group.add_argument("--no-randomize", dest="randomize", action="store_false")
    parser.set_defaults(randomize=None)
    return parser.parse_args()


def print_summary(plan: SuitePlan, artifacts: ArtifactPaths, estimate: RuntimeEstimate) -> None:
    print("Suite:", plan["suite"])
    print("Description:", plan["description"])
    print("OA:", plan["oa"], "->", plan["oa_path"])
    print("Registry:", artifacts["registry"])
    print("Design:", artifacts["design"])
    print("Manifest:", artifacts["manifest"])
    print("Randomize mapping:", plan["randomize"])
    print("Estimated runs:", estimate["runs"])
    print("Per-run seconds:", f"{estimate['per_run_seconds']:.2f}")
    print("Total seconds:", f"{estimate['total_seconds']:.2f}")


def main() -> None:
    args = parse_args()
    catalog = load_yaml(args.catalog)
    suites = load_yaml(args.suites)
    plan = build_plan(
        suite_name=args.suite,
        suites=suites,
        catalog=catalog,
        overrides=args.override,
        base_config_override=args.base_config,
        oa_override=args.oa,
        oa_path_override=args.oa_path,
        randomize_override=args.randomize,
    )
    workspace = args.output_dir
    artifacts, estimate = generate_artifacts(plan, args.catalog, catalog, workspace)
    print_summary(plan, artifacts, estimate)
    if args.estimate_only:
        return
    run_suite(plan, artifacts, runner=args.runner, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
