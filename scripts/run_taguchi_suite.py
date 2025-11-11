#!/usr/bin/env python
"""Build and execute configurable Taguchi suites."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, TypedDict

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

from taguchi_suite import oa

DEFAULT_GENERATED_DIR = Path("configs/taguchi/generated")
OA_DESIGN_MAP = {
    "L27": Path("configs/taguchi/L27_extended.csv"),
    "L23": Path("configs/taguchi/L23_synthetic.csv"),
    "L18": Path("configs/taguchi/L18_mixed.csv"),
}


class SuitePlan(TypedDict):
    suite: str
    description: str
    include: List[str]
    fixed: Dict[str, Any]
    oa: str
    oa_path: Optional[Path]
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


class ConstraintStatus(TypedDict):
    expr: str
    status: Literal["OK", "WARNING", "BLOCKED"]
    severity: str
    reason: str
    triggered: bool


class ConstraintEngine:
    ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Subscript,
        ast.Index,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.Gt,
        ast.LtE,
        ast.GtE,
        ast.In,
        ast.NotIn,
    )

    def __init__(self, catalog: Dict[str, Any]) -> None:
        self.catalog = catalog
        self.constraints = catalog.get("constraints", [])

    def _context(self, plan: SuitePlan) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        catalog_factors = self.catalog.get("factors", {})
        for factor, info in catalog_factors.items():
            if factor in plan["fixed"]:
                context[factor] = plan["fixed"][factor]
                continue
            levels = info.get("levels")
            context[factor] = levels[0] if isinstance(levels, list) and levels else None
        return context

    def _ensure_safe(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if not isinstance(child, self.ALLOWED_NODES):
                raise ValueError(f"Unsafe expression element: {type(child).__name__}")

    def _eval_expr(self, expr: str, context: Dict[str, Any]) -> bool:
        tree = ast.parse(expr, mode="eval")
        self._ensure_safe(tree)
        code = compile(tree, "<constraint>", "eval")
        return bool(eval(code, {"__builtins__": {}}, context))

    def _dependency_statuses(
        self, plan: SuitePlan, context: Dict[str, Any]
    ) -> List[ConstraintStatus]:
        statuses: List[ConstraintStatus] = []
        catalog_factors = self.catalog.get("factors", {})
        active_factors = set(plan["include"]) | set(plan["fixed"].keys())
        for factor in active_factors:
            info = catalog_factors.get(factor)
            if not info:
                continue
            if factor not in plan["fixed"]:
                continue
            depends = info.get("depends_on", {})
            for dep, expected in (depends or {}).items():
                if dep not in context:
                    continue
                expected_values = expected if isinstance(expected, list) else [expected]
                actual = context.get(dep)
                triggered = actual not in expected_values
                if not triggered:
                    continue
                statuses.append(
                    {
                        "expr": f"{factor} depends_on {dep}",
                        "status": "BLOCKED",
                        "severity": "dependency",
                        "reason": (
                            f"{factor} requires {dep} ∈ {expected_values}, got {actual}"
                        ),
                        "triggered": True,
                    }
                )
        return statuses

    def evaluate(self, plan: SuitePlan) -> List[ConstraintStatus]:
        context = self._context(plan)
        statuses: List[ConstraintStatus] = []
        for rule in self.constraints:
            expr = rule.get("expr")
            if not expr:
                continue
            severity = rule.get("severity", "block")
            reason = rule.get("reason", "")
            try:
                triggered = self._eval_expr(expr, context)
            except Exception as exc:  # pragma: no cover - defensive
                statuses.append(
                    {
                        "expr": expr,
                        "status": "WARNING",
                        "severity": "error",
                        "reason": f"{reason} (eval error: {exc})",
                        "triggered": False,
                    }
                )
                continue
            status = "OK"
            if triggered:
                status = "WARNING" if severity == "warn" else "BLOCKED"
            statuses.append(
                {
                    "expr": expr,
                    "status": status,
                    "severity": severity,
                    "reason": reason,
                    "triggered": triggered,
                }
            )
        statuses.extend(self._dependency_statuses(plan, context))
        return statuses


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


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


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
    oa_path = oa_path_override or suite_data.get("oa_path")
    if oa_path:
        oa_path = Path(oa_path)
        if not oa_path.exists():
            raise FileNotFoundError(f"OA design for '{oa_name}' not found at {oa_path}")
    elif oa_name in OA_DESIGN_MAP:
        oa_path = OA_DESIGN_MAP[oa_name]
    else:
        oa_path = None
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
    if plan["oa_path"] and plan["oa_path"].exists():
        shutil.copy(plan["oa_path"], out_path)
    else:
        design = oa.select_oa(num_factors=len(plan["include"]), levels=3)
        design.to_csv(out_path, index=False)
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
    suites_path: Path,
    registry_path: Path,
    design_path: Path,
    estimate: RuntimeEstimate,
    workspace: Path,
    constraint_statuses: List[ConstraintStatus],
) -> Dict[str, Any]:
    return {
        "suite": plan["suite"],
        "description": plan["description"],
        "oa": plan["oa"],
        "oa_path": str(plan["oa_path"]) if plan["oa_path"] else None,
        "base_config": str(plan["base_config"]) if plan["base_config"] else None,
        "registry": str(registry_path),
        "design": str(design_path),
        "randomize": plan["randomize"],
        "seeds": plan["seeds"],
        "estimate": estimate,
        "catalog_sha": hash_file(catalog_path),
        "suite_sha": hash_file(suites_path),
        "runtime_estimate_sec": estimate["total_seconds"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "constraints": constraint_statuses,
        "workspace": str(workspace),
    }


def dump_manifest(manifest: Dict[str, Any], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def generate_artifacts(
    plan: SuitePlan,
    catalog_path: Path,
    suites_path: Path,
    catalog: Dict[str, Any],
    base_dir: Path,
    constraint_statuses: List[ConstraintStatus],
) -> Tuple[ArtifactPaths, RuntimeEstimate]:
    workspace = _workspace_for(plan, base_dir)
    registry_path = workspace / f"factor_registry_{plan['suite']}.yaml"
    design_path = workspace / f"design_{plan['oa']}_{plan['suite']}.csv"
    manifest_path = workspace / f"run_manifest_{plan['suite']}.json"
    registry = generate_factor_registry(plan, catalog, registry_path)
    design = generate_design_csv(plan, design_path)
    estimate = estimate_runtime(plan, catalog, design)
    manifest = build_manifest(
        plan,
        catalog_path,
        suites_path,
        registry,
        design,
        estimate,
        workspace,
        constraint_statuses,
    )
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
    parser.add_argument("--suite", help="Suite name from configs/taguchi/suites.yaml")
    parser.add_argument("--list-suites", action="store_true", help="List available suites and their default runtimes")
    parser.add_argument("--list-factors", action="store_true", help="List all catalog factors with cost weights")
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
    parser.add_argument("--validate-only", action="store_true", help="Validate constraints without emitting files")
    parser.add_argument("--randomize-seed", type=int, help="Override the deterministic seed when randomize is false")
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


def list_factors(catalog: Dict[str, Any]) -> None:
    print("Available factors:")
    for name, info in catalog.get("factors", {}).items():
        print(f"  - {name} (kind={info.get('kind')}, cost={info.get('cost_weight')})")
        doc = info.get("doc")
        if doc:
            print(f"      {doc}")


def _resolve_design_path(plan: SuitePlan) -> Tuple[Path, Optional[Path]]:
    if plan["oa_path"] and plan["oa_path"].exists():
        return plan["oa_path"], None
    df = oa.select_oa(num_factors=len(plan["include"]), levels=3)
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    file_path = Path(temp_path)
    df.to_csv(file_path, index=False)
    return file_path, file_path


def list_suites(catalog: Dict[str, Any], suites: Dict[str, Any]) -> None:
    for name, data in suites.get("suites", {}).items():
        plan = build_plan(
            suite_name=name,
            suites=suites,
            catalog=catalog,
            overrides=[],
            base_config_override=None,
        )
        design_path, temp = _resolve_design_path(plan)
        try:
            estimate = estimate_runtime(plan, catalog, design_path)
        finally:
            if temp:
                temp.unlink()
        total_hours = estimate["total_seconds"] / 3600.0
        print(
            f"- {name}: {data.get('description','')} -> runs={estimate['runs']} total≈{total_hours:.2f}h"
        )


def print_constraints(statuses: List[ConstraintStatus]) -> None:
    if not statuses:
        return
    print("Constraint summary:")
    for status in statuses:
        if status["status"] == "OK":
            continue
        prefix = {
            "WARNING": "⚠️",
            "BLOCKED": "⛔",
        }.get(status["status"], status["status"])
        print(
            f"  {prefix} {status['expr']} -> {status['status']} ({status['severity']}): {status['reason']}"
        )


def print_summary(
    plan: SuitePlan,
    artifacts: ArtifactPaths,
    estimate: RuntimeEstimate,
    statuses: List[ConstraintStatus],
) -> None:
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
    if args.list_factors:
        list_factors(catalog)
        return
    if args.list_suites:
        list_suites(catalog, suites)
        return
    if not args.suite:
        raise ValueError("--suite is required unless listing factors or suites")
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
    if args.randomize_seed is not None:
        plan["seeds"] = [args.randomize_seed]
    engine = ConstraintEngine(catalog)
    statuses = engine.evaluate(plan)
    print_constraints(statuses)
    blocked = any(status["status"] == "BLOCKED" for status in statuses)
    if blocked:
        print("One or more constraints are blocked; aborting.")
        sys.exit(1)
    if args.validate_only:
        return
    workspace = args.output_dir
    artifacts, estimate = generate_artifacts(
        plan, args.catalog, args.suites, catalog, workspace, statuses
    )
    print_summary(plan, artifacts, estimate, statuses)
    if args.estimate_only:
        return
    run_suite(plan, artifacts, runner=args.runner, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
