#!/usr/bin/env python
"""Build and execute configurable Taguchi suites."""

from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(ROOT))

# Optional: pre-baked OA CSVs. If not found, we will synthesize a simple L27-like CSV.
OA_DESIGN_MAP: Dict[str, Path] = {
    "L27": Path("configs/taguchi/L27_extended.csv"),
}

DEFAULT_GENERATED_DIR = Path("configs/taguchi/generated")

class ConstraintStatus(TypedDict):
    expr: str
    status: Literal["OK", "WARNING", "BLOCKED"]
    severity: str
    reason: str
    triggered: bool

@dataclass
class SuitePlan:
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
    registry: str
    design: str
    manifest: str
    base_dir: str

class RuntimeEstimate(TypedDict):
    runs: int
    seconds_per_run: float
    total_seconds: float

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

    def _context(self, plan: "SuitePlan") -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        catalog_factors = self.catalog.get("factors", {})
        for factor, info in catalog_factors.items():
            if factor in plan.fixed:
                context[factor] = plan.fixed[factor]
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

    def _dependency_statuses(self, plan: "SuitePlan", context: Dict[str, Any]) -> List[ConstraintStatus]:
        statuses: List[ConstraintStatus] = []
        catalog_factors = self.catalog.get("factors", {})
        active_factors = set(plan.include) | set(plan.fixed.keys())
        for factor in active_factors:
            info = catalog_factors.get(factor)
            if not info:
                continue
            if factor not in plan.fixed:
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

    def evaluate(self, plan: "SuitePlan") -> List[ConstraintStatus]:
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


def _ordered_active_factors(plan: "SuitePlan") -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for f in plan.include + list(plan.fixed.keys()):
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}

def save_yaml(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return path

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

def _synthesize_l27_csv(num_cols: int, out_path: Path) -> Path:
    """
    Fallback: create a simple 27-row, num_cols-column CSV cycling 0/1/2.
    Not a perfect OA, but adequate placeholder when a real OA file isn't present.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"C{i+1}" for i in range(num_cols)])
        for r in range(27):
            row = [(r // (3**c)) % 3 for c in range(num_cols)]
            writer.writerow(row)
    return out_path

def generate_design_csv(plan: "SuitePlan", out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.oa_path and Path(plan.oa_path).exists():
        shutil.copy(Path(plan.oa_path), out_path)
        return out_path
    # Fallback to a synthetic L27 if no OA CSV on disk.
    return _synthesize_l27_csv(num_cols=len(plan.include), out_path=out_path)

def build_manifest(
    plan: "SuitePlan",
    catalog_path: Path,
    suites_path: Path,
    registry_path: Path,
    design_path: Path,
    estimate: RuntimeEstimate,
    workspace: Path,
    constraint_statuses: List[ConstraintStatus],
) -> Dict[str, Any]:
    return {
        "suite": plan.suite,
        "description": plan.description,
        "oa": plan.oa,
        "oa_path": str(plan.oa_path) if plan.oa_path else None,
        "base_config": str(plan.base_config) if plan.base_config else None,
        "registry": str(registry_path),
        "design": str(design_path),
        "randomize": plan.randomize,
        "seeds": plan.seeds,
        "estimate": estimate,
        "catalog_sha": hash_file(catalog_path),
        "suite_sha": hash_file(suites_path),
        "runtime_estimate_sec": estimate["total_seconds"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "constraints": constraint_statuses,
        "workspace": str(workspace),
    }

def _workspace_for(plan: SuitePlan, base_dir: Path) -> Path:
    ws = base_dir / plan.suite / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ws / "results").mkdir(parents=True, exist_ok=True)
    return ws

def generate_factor_registry(plan: SuitePlan, catalog: Dict[str, Any], out_path: Path) -> Path:
    data: Dict[str, Any] = {"factors": {}, "fixed_values": {}}
    catalog_factors = catalog.get("factors", {})
    for name in plan.include:
        info = catalog_factors[name]
        data["factors"][name] = {
            k: v for k, v in info.items() if k in ("kind", "levels", "doc")
        }
    for name, val in plan.fixed.items():
        data["fixed_values"][name] = val
    return save_yaml(data, out_path)

def estimate_runtime(plan: SuitePlan, catalog: Dict[str, Any], design_csv: Path) -> RuntimeEstimate:
    est = catalog.get("estimation", {}) or {}
    base = float(est.get("base_seconds_per_run", 120.0))
    mults = est.get("multipliers", {}) or {}
    # count rows in design
    with design_csv.open("r", encoding="utf-8") as fh:
        runs = sum(1 for _ in fh) - 1  # minus header
    # compute per-run multiplier from fixed values when known
    per_run = base
    for k, v in plan.fixed.items():
        m = mults.get(k)
        if isinstance(m, dict):
            key = str(v)
            if key in m:
                per_run *= float(m[key])
    total = per_run * runs * max(1, len(plan.seeds))
    return {"runs": runs, "seconds_per_run": per_run, "total_seconds": total}

def dump_manifest(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path

def generate_artifacts(
    plan: "SuitePlan",
    catalog_path: Path,
    suites_path: Path,
    catalog: Dict[str, Any],
    base_dir: Path,
    constraint_statuses: List[ConstraintStatus],
) -> Tuple[ArtifactPaths, RuntimeEstimate]:
    workspace = _workspace_for(plan, base_dir)
    registry_path = workspace / f"factor_registry_{plan.suite}.yaml"
    design_path = workspace / f"design_{plan.oa}_{plan.suite}.csv"
    manifest_path = workspace / f"manifest_{plan.suite}.json"
    registry = generate_factor_registry(plan, catalog, registry_path)
    design = generate_design_csv(plan, design_path)
    estimate = estimate_runtime(plan, catalog, design)
    manifest = build_manifest(plan, catalog_path, suites_path, registry, design, estimate, workspace, constraint_statuses)
    dump_manifest(manifest, manifest_path)
    return (
        {
            "registry": str(registry),
            "design": str(design),
            "manifest": str(manifest_path),
            "base_dir": str(workspace / "results"),
        },
        estimate,
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Taguchi experiment suite")
    p.add_argument("--suite", help="Suite name from configs/taguchi/suites.yaml")
    p.add_argument("--list-suites", action="store_true")
    p.add_argument("--list-factors", action="store_true")
    p.add_argument("--catalog", type=Path, default=Path("configs/taguchi/factor_catalog.yaml"))
    p.add_argument("--suites", type=Path, default=Path("configs/taguchi/suites.yaml"))
    p.add_argument("--oa", help="Override OA name (e.g., L27)")
    p.add_argument("--oa-path", type=Path, help="Override OA CSV path")
    p.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--randomize-seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--runner", type=Path, default=Path("scripts/run_full_report_32x32.sh"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    p.add_argument("--base-config", type=Path)
    return p.parse_args()

def list_factors(catalog: Dict[str, Any]) -> None:
    print("Available factors:")
    for name, info in catalog.get("factors", {}).items():
        print(f"  - {name} (kind={info.get('kind')}, cost={info.get('cost_weight')})")
        doc = info.get("doc")
        if doc:
            print(f"      {doc}")

def list_suites(catalog: Dict[str, Any], suites: Dict[str, Any]) -> None:
    for name, data in suites.get("suites", {}).items():
        plan = build_plan(name, suites, catalog, oa_override=data.get("oa"), oa_path_override=data.get("oa_path"))
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        try:
            design = generate_design_csv(plan, tmp)
            est = estimate_runtime(plan, catalog, design)
        finally:
            tmp.unlink(missing_ok=True)
        hours = est["total_seconds"] / 3600.0
        print(f"- {name}: {data.get('description','')} -> runs={est['runs']} total≈{hours:.2f}h")

def print_summary(plan: SuitePlan, artifacts: ArtifactPaths, estimate: RuntimeEstimate, statuses: List[ConstraintStatus]) -> None:
    print("Suite:", plan.suite)
    print("Description:", plan.description)
    print("OA:", plan.oa, "->", plan.oa_path)
    print_constraints(statuses)
    hrs = estimate["total_seconds"] / 3600.0
    print(f"Runs: {estimate['runs']}  per-run≈{estimate['seconds_per_run']:.1f}s  total≈{hrs:.2f}h")
    print("Artifacts:")
    print("  factor_registry:", artifacts["registry"])
    print("  design_csv     :", artifacts["design"])
    print("  manifest       :", artifacts["manifest"])

def print_constraints(statuses: List[ConstraintStatus]) -> None:
    if not statuses:
        return
    print("Constraint summary:")
    for s in statuses:
        if s["status"] == "OK":
            continue
        icon = "⚠️" if s["status"] == "WARNING" else "⛔"
        print(f"  {icon} {s['expr']} -> {s['status']} ({s['severity']}): {s['reason']}")

def build_plan(
    suite_name: str,
    suites: Dict[str, Any],
    catalog: Dict[str, Any],
    oa_override: Optional[str] = None,
    oa_path_override: Optional[str | Path] = None,
    randomize_override: Optional[bool] = None,
    base_config_override: Optional[Path] = None,
) -> SuitePlan:
    sdata = suites["suites"][suite_name]
    include = list(sdata.get("include", []))
    fixed = dict(sdata.get("fixed", {}))
    oa_name = oa_override or sdata.get("oa") or "L27"
    oa_path = Path(oa_path_override) if oa_path_override else OA_DESIGN_MAP.get(oa_name)
    randomize = sdata.get("randomize", False) if randomize_override is None else bool(randomize_override)
    seeds = list(sdata.get("seeds", [42]))
    base_cfg = base_config_override or (Path(sdata["base_config"]) if sdata.get("base_config") else None)
    return SuitePlan(
        suite=suite_name,
        description=sdata.get("description", ""),
        include=include,
        fixed=fixed,
        oa=oa_name,
        oa_path=oa_path,
        randomize=randomize,
        seeds=seeds,
        base_config=base_cfg,
    )

def run_suite(plan: SuitePlan, artifacts: ArtifactPaths, runner: Path, dry_run: bool) -> None:
    if dry_run:
        print("Dry-run: not executing runner.")
        return
    env = os.environ.copy()
    env["TAGUCHI_FACTOR_REGISTRY"] = artifacts["registry"]
    env["TAGUCHI_ARRAY_PATH"] = artifacts["design"]
    env["TAGUCHI_RANDOMIZE"] = "1" if plan.randomize else "0"
    if plan.base_config:
        env["TAGUCHI_BASE_CONFIG"] = str(plan.base_config)
    subprocess.check_call([str(runner)], env=env)

def main() -> None:
    args = parse_args()
    catalog = load_yaml(args.catalog)
    suites = load_yaml(args.suites)
    if args.list_factors:
        list_factors(catalog); return
    if args.list_suites:
        list_suites(catalog, suites); return
    if not args.suite:
        raise SystemExit("--suite is required (or use --list-suites / --list-factors)")
    plan = build_plan(
        suite_name=args.suite,
        suites=suites,
        catalog=catalog,
        oa_override=args.oa,
        oa_path_override=args.oa_path,
        randomize_override=args.randomize,
        base_config_override=args.base_config,
    )
    if args.randomize_seed is not None:
        plan.seeds = [int(args.randomize_seed)]
    engine = ConstraintEngine(catalog)
    statuses = engine.evaluate(plan)
    if any(s["status"] == "BLOCKED" for s in statuses):
        print_constraints(statuses)
        raise SystemExit("Constraint violation: BLOCKED")
    if args.validate_only:
        print_constraints(statuses); return
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
