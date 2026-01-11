#!/usr/bin/env python3
"""
Lightweight training monitor UI (stdlib only).

Usage:
  python scripts/debug/train_monitor_web.py
  open http://127.0.0.1:8790
"""

from __future__ import annotations

import csv
import json
import mimetypes
import sys
from collections import deque
from datetime import datetime
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config  # noqa: E402


RUN_ROOTS = [ROOT / "runs", ROOT / "results"]


def _discover_runs() -> List[Path]:
    runs: List[Path] = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for config_path in root.rglob("config.yaml"):
            runs.append(config_path.parent)
        for log_path in root.rglob("logs/train.log"):
            runs.append(log_path.parent.parent)
        for history_path in root.rglob("diagnostics/training_history.csv"):
            runs.append(history_path.parent.parent)
        for ckpt_path in root.rglob("checkpoints/checkpoint_step_*.pt"):
            runs.append(ckpt_path.parent.parent)
    return sorted({p for p in runs if p.exists()}, key=lambda p: str(p).lower())


def _relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _safe_path(path_str: str) -> Path:
    candidate = (ROOT / path_str).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError("Path escapes repo root.")
    return candidate


def _read_csv_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: Deque[Dict[str, Any]] = deque(maxlen=limit)
        for row in reader:
            rows.append(row)
    return list(rows)


def _coerce_row(row: Dict[str, Any]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            coerced[key] = None
            continue
        try:
            if "." in value or "e" in value.lower():
                coerced[key] = float(value)
            else:
                coerced[key] = int(value)
        except (ValueError, AttributeError):
            coerced[key] = value
    return coerced


def _run_info(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {}
    diffusion_cfg = config.get("diffusion", {}) or {}
    training_cfg = config.get("training", {}) or {}
    model_cfg = config.get("model", {}) or {}
    data_cfg = config.get("data", {}) or {}
    train_history = run_dir / "diagnostics" / "training_history.csv"
    rows = _read_csv_tail(train_history, limit=1)
    last_row = _coerce_row(rows[-1]) if rows else {}
    return {
        "run_dir": _relpath(run_dir),
        "model": model_cfg.get("name", model_cfg.get("type", "unknown")),
        "dataset": data_cfg.get("name", data_cfg.get("dataset", "unknown")),
        "num_timesteps": int(diffusion_cfg.get("num_timesteps", 1000)),
        "snr_ratio": float(diffusion_cfg.get("snr_ratio", 1.0)),
        "spectral_operator_mode": str(diffusion_cfg.get("spectral_operator_mode", "none")),
        "train_steps": int(training_cfg.get("train_steps", 0)),
        "eval_every": int(training_cfg.get("eval_every", 0)),
        "checkpoint_every": int(training_cfg.get("checkpoint_every", 0)),
        "last_step": last_row.get("step"),
        "last_loss": last_row.get("loss"),
        "last_grad_norm": last_row.get("grad_norm"),
    }


def _discover_configs() -> List[Path]:
    configs_dir = ROOT / "configs"
    if not configs_dir.exists():
        return []
    return sorted(configs_dir.glob("*.yaml"), key=lambda p: str(p).lower())


def _config_summary(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    model_cfg = config.get("model", {}) or {}
    data_cfg = config.get("data", {}) or {}
    diffusion_cfg = config.get("diffusion", {}) or {}
    training_cfg = config.get("training", {}) or {}
    sampling_cfg = config.get("sampling", {}) or {}
    return {
        "config": _relpath(config_path),
        "model": model_cfg.get("name", model_cfg.get("type", "unknown")),
        "dataset": data_cfg.get("name", data_cfg.get("dataset", "unknown")),
        "num_timesteps": int(diffusion_cfg.get("num_timesteps", 1000)),
        "seed": config.get("seed"),
        "snr_ratio": float(diffusion_cfg.get("snr_ratio", 1.0)),
        "spectral_operator_mode": str(diffusion_cfg.get("spectral_operator_mode", "none")),
        "train_steps": int(training_cfg.get("train_steps", 0)),
        "eval_every": int(training_cfg.get("eval_every", 0)),
        "eval_num_samples": training_cfg.get("eval_num_samples"),
        "eval_sampling_steps": training_cfg.get("eval_sampling_steps"),
        "eval_seed": training_cfg.get("eval_seed"),
        "checkpoint_every": int(training_cfg.get("checkpoint_every", 0)),
        "sampler_type": str(sampling_cfg.get("sampler_type", "unknown")),
        "sampling_steps": int(sampling_cfg.get("sampling_steps", 0)),
    }


def _eval_steps(run_dir: Path) -> List[int]:
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return []
    steps = []
    for step_dir in eval_dir.glob("step_*"):
        try:
            steps.append(int(step_dir.name.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(steps))


def _latest_eval(run_dir: Path) -> Dict[str, Any]:
    steps = _eval_steps(run_dir)
    if not steps:
        return {"steps": []}
    latest = steps[-1]
    eval_dir = run_dir / "eval" / f"step_{latest:06d}"
    eval_json = eval_dir / "eval.json"
    payload: Dict[str, Any] = {"steps": steps, "latest_step": latest}
    if eval_json.exists():
        payload["eval_json"] = json.loads(eval_json.read_text(encoding="utf-8"))
    artifacts = []
    candidates = [
        ("samples grid", eval_dir / "samples" / "grid.png"),
        ("sanity spatial", eval_dir / "diagnostics" / "eval_sanity_generated_samples_spatial.png"),
        ("sanity fft", eval_dir / "diagnostics" / "eval_sanity_generated_samples_fft_mag.png"),
        ("denoise x0", eval_dir / "diagnostics" / "eval_denoise_x0.png"),
        ("denoise xt", eval_dir / "diagnostics" / "eval_denoise_xt.png"),
        ("denoise pred x0", eval_dir / "diagnostics" / "eval_denoise_pred_x0.png"),
    ]
    for label, path in candidates:
        if path.exists():
            artifacts.append({"label": label, "path": _relpath(path)})
    payload["artifacts"] = artifacts
    return payload


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Training Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f2f4f6;
      --panel: #ffffff;
      --ink: #1d1d1d;
      --muted: #5f6b75;
      --accent: #1f4e79;
      --line: #d7dde3;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(160deg, #f2f4f6 0%, #f8fafc 100%);
      color: var(--ink);
    }
    header {
      padding: 22px 26px 10px 26px;
    }
    h1 {
      margin: 0 0 6px 0;
      font-size: 22px;
      font-weight: 600;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    .stage-bar {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .stage {
      padding: 8px 10px;
      border-radius: 10px;
      background: #e8edf2;
      color: #3b4b5a;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      text-align: center;
      border: 1px solid var(--line);
    }
    .stage.active {
      background: #1f4e79;
      color: #fff;
      border-color: #1f4e79;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      padding: 14px 24px 32px 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.05);
    }
    label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      display: block;
      margin-top: 14px;
    }
    select, input[type="number"] {
      width: 100%;
      padding: 8px 10px;
      margin-top: 6px;
      border-radius: 8px;
      border: 1px solid var(--line);
      font-size: 14px;
    }
    button {
      margin-top: 16px;
      width: 100%;
      padding: 10px 12px;
      border: none;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: #6c7a89;
    }
    .status {
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .toolbar a {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 8px;
      background: #eef2f6;
      border: 1px solid var(--line);
      color: var(--ink);
      text-decoration: none;
      font-size: 12px;
    }
    .status-pill {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #eef2f6;
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
    }
    .error-banner {
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #fff1f0;
      border: 1px solid #f5c2c7;
      color: #9a2c2c;
      font-size: 12px;
      display: none;
      white-space: pre-wrap;
    }
    .launch-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .field {
      background: #f7f9fb;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
    }
    .field p {
      font-size: 12px;
      color: var(--muted);
      margin: 6px 0 0 0;
    }
    .metric {
      padding: 12px;
      border-radius: 10px;
      background: #f7f9fb;
      border: 1px solid var(--line);
    }
    canvas {
      width: 100%;
      height: 120px;
      border-radius: 8px;
      background: #fff;
      border: 1px solid var(--line);
    }
    pre {
      background: #111;
      color: #d7d7d7;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
      font-size: 12px;
    }
    figure {
      margin: 0;
      background: #fafbfc;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
    }
    figcaption {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    img {
      width: 100%;
      display: block;
      border-radius: 6px;
    }
    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Training Monitor</h1>
    <p>Launch, track, and inspect runs with a single lightweight UI.</p>
    <div class="stage-bar">
      <div class="stage" id="stageLaunch">1. Launch</div>
      <div class="stage" id="stageMonitor">2. Monitor</div>
      <div class="stage" id="stageInspect">3. Inspect</div>
    </div>
    <div class="error-banner" id="errorBanner"></div>
  </header>
  <div class="layout">
    <div class="panel">
      <label for="runSelect">Run directory</label>
      <select id="runSelect"></select>

      <label for="limitInput">Metrics window (rows)</label>
      <input id="limitInput" type="number" min="10" max="5000" value="500" />

      <button id="refreshBtn">Refresh</button>
      <button id="autoBtn" class="secondary">Auto refresh: off</button>
      <div class="status" id="status">Idle.</div>

      <div class="metric" style="margin-top: 16px;">
        <strong>Run summary</strong>
        <div id="runSummary"></div>
        <div class="toolbar">
          <span class="status-pill" id="progressPill">No run selected</span>
        </div>
      </div>

      <div class="metric" style="margin-top: 16px;">
        <strong>Launch training</strong>
        <div class="status" id="launchStatus">Idle.</div>
        <label for="configSelect">Config</label>
        <select id="configSelect"></select>
        <div class="launch-grid" style="margin-top: 10px;">
          <div class="field">
            <label for="runIdInput">run_id</label>
            <input id="runIdInput" type="text" placeholder="e.g. fig2_long_run" />
            <p>Optional run identifier. Defaults to a timestamp.</p>
          </div>
          <div class="field">
            <label for="seedInput">seed</label>
            <input id="seedInput" type="number" min="0" />
            <p>Override config.seed for deterministic repeats.</p>
          </div>
          <div class="field">
            <label for="trainStepsInput">train_steps</label>
            <input id="trainStepsInput" type="number" min="1" />
            <p>Override training.train_steps (total gradient steps).</p>
          </div>
          <div class="field">
            <label for="ckptEveryInput">checkpoint_every</label>
            <input id="ckptEveryInput" type="number" min="1" />
            <p>Override training.checkpoint_every (steps).</p>
          </div>
          <div class="field">
            <label for="evalEveryInput">eval_every</label>
            <input id="evalEveryInput" type="number" min="1" />
            <p>Override training.eval_every (steps).</p>
          </div>
          <div class="field">
            <label for="evalNumInput">eval_num_samples</label>
            <input id="evalNumInput" type="number" min="1" />
            <p>Override training.eval_num_samples (per eval).</p>
          </div>
          <div class="field">
            <label for="evalStepsInput">eval_sampling_steps</label>
            <input id="evalStepsInput" type="number" min="1" />
            <p>Override training.eval_sampling_steps (sampling steps per eval).</p>
          </div>
          <div class="field">
            <label for="evalSeedInput">eval_seed</label>
            <input id="evalSeedInput" type="number" min="0" />
            <p>Override training.eval_seed (base seed for eval sampling).</p>
          </div>
          <div class="field">
            <label for="snrInput">snr_ratio</label>
            <input id="snrInput" type="number" min="0.0" step="0.01" />
            <p>Override diffusion.snr_ratio (signal-to-noise ratio for spectral noise).</p>
          </div>
          <div class="field">
            <label for="spectralSelect">spectral_operator_mode</label>
            <select id="spectralSelect">
              <option value="">(config default)</option>
              <option value="none">none</option>
              <option value="radial">radial</option>
              <option value="radial_squared">radial_squared</option>
            </select>
            <p>Override diffusion.spectral_operator_mode.</p>
          </div>
          <div class="field">
            <label for="logLevelSelect">log_level</label>
            <select id="logLevelSelect">
              <option value="INFO">INFO</option>
              <option value="WARNING" selected>WARNING</option>
              <option value="DEBUG">DEBUG</option>
              <option value="ERROR">ERROR</option>
            </select>
            <p>Logging level (DEBUG, INFO, WARNING, ERROR).</p>
          </div>
        </div>
        <button id="launchBtn">Launch training</button>
        <pre id="launchCommand">Command preview...</pre>
      </div>

      <div class="metric" style="margin-top: 16px;">
        <strong>Run artifacts</strong>
        <div id="artifactsBox">
          <div>Run dir: <span id="artifactRunDir">n/a</span></div>
          <div>Log: <span id="artifactLog">n/a</span></div>
          <div>Inspect CLI: <span id="artifactInspectCli">n/a</span></div>
          <div>Inspect UI: <span id="artifactInspectUi">http://127.0.0.1:8787</span></div>
          <div>Latest eval grid: <span id="artifactEvalGrid">n/a</span></div>
          <div class="toolbar">
            <a href="http://127.0.0.1:8787" target="_blank" rel="noreferrer">Open inspect UI</a>
            <a href="#" id="openEvalGrid">Open eval grid</a>
          </div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="grid">
        <div class="metric">
          <strong>Loss</strong>
          <canvas id="lossChart"></canvas>
        </div>
        <div class="metric">
          <strong>MAE</strong>
          <canvas id="maeChart"></canvas>
        </div>
        <div class="metric">
          <strong>Grad norm</strong>
          <canvas id="gradChart"></canvas>
        </div>
        <div class="metric">
          <strong>SNR rel</strong>
          <canvas id="snrChart"></canvas>
        </div>
      </div>

      <h3>Latest eval</h3>
      <div class="grid" id="evalGrid"></div>

      <h3>Log tail</h3>
      <pre id="logBox">No log loaded.</pre>
    </div>
  </div>
<script>
  const runSelect = document.getElementById("runSelect");
  const limitInput = document.getElementById("limitInput");
  const refreshBtn = document.getElementById("refreshBtn");
  const autoBtn = document.getElementById("autoBtn");
  const statusEl = document.getElementById("status");
  const runSummary = document.getElementById("runSummary");
  const evalGrid = document.getElementById("evalGrid");
  const logBox = document.getElementById("logBox");
  const configSelect = document.getElementById("configSelect");
  const launchBtn = document.getElementById("launchBtn");
  const launchStatus = document.getElementById("launchStatus");
  const launchCommand = document.getElementById("launchCommand");
  const artifactRunDir = document.getElementById("artifactRunDir");
  const artifactLog = document.getElementById("artifactLog");
  const artifactInspectCli = document.getElementById("artifactInspectCli");
  const artifactInspectUi = document.getElementById("artifactInspectUi");
  const artifactEvalGrid = document.getElementById("artifactEvalGrid");
  const openEvalGrid = document.getElementById("openEvalGrid");
  const progressPill = document.getElementById("progressPill");
  const stageLaunch = document.getElementById("stageLaunch");
  const stageMonitor = document.getElementById("stageMonitor");
  const stageInspect = document.getElementById("stageInspect");
  const errorBanner = document.getElementById("errorBanner");
  const runIdInput = document.getElementById("runIdInput");
  const seedInput = document.getElementById("seedInput");
  const trainStepsInput = document.getElementById("trainStepsInput");
  const ckptEveryInput = document.getElementById("ckptEveryInput");
  const evalEveryInput = document.getElementById("evalEveryInput");
  const evalNumInput = document.getElementById("evalNumInput");
  const evalStepsInput = document.getElementById("evalStepsInput");
  const evalSeedInput = document.getElementById("evalSeedInput");
  const snrInput = document.getElementById("snrInput");
  const spectralSelect = document.getElementById("spectralSelect");
  const logLevelSelect = document.getElementById("logLevelSelect");

  let currentConfigDefaults = {};

  const charts = {
    loss: document.getElementById("lossChart"),
    mae: document.getElementById("maeChart"),
    grad_norm: document.getElementById("gradChart"),
    snr_rel: document.getElementById("snrChart"),
  };

  let autoTimer = null;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function showError(msg) {
    if (!errorBanner) return;
    errorBanner.textContent = msg;
    errorBanner.style.display = "block";
  }

  window.addEventListener("error", (event) => {
    showError(`UI error: ${event.message}`);
  });
  window.addEventListener("unhandledrejection", (event) => {
    showError(`UI rejection: ${event.reason}`);
  });

  function setStage(activeLaunch, activeMonitor, activeInspect) {
    stageLaunch.classList.toggle("active", activeLaunch);
    stageMonitor.classList.toggle("active", activeMonitor);
    stageInspect.classList.toggle("active", activeInspect);
  }

  function drawLine(canvas, data, color) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width = canvas.clientWidth * 2;
    const height = canvas.height = canvas.clientHeight * 2;
    ctx.clearRect(0, 0, width, height);
    if (!data.length) {
      ctx.fillStyle = "#999";
      ctx.fillText("no data", 10, 20);
      return;
    }
    const min = Math.min(...data);
    const max = Math.max(...data);
    const span = max - min || 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((val, idx) => {
      const x = (idx / (data.length - 1)) * (width - 20) + 10;
      const y = height - 10 - ((val - min) / span) * (height - 20);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  async function loadRuns() {
    const resp = await fetch("/api/runs");
    const data = await resp.json();
    runSelect.innerHTML = "";
    data.runs.forEach((run) => {
      const opt = document.createElement("option");
      opt.value = run;
      opt.textContent = run;
      runSelect.appendChild(opt);
    });
    if (data.runs.length) {
      setStage(true, true, false);
      await refreshAll();
    } else {
      setStage(true, false, false);
      progressPill.textContent = "No runs found";
      runSummary.innerHTML = "<div>No runs yet. Launch one to start monitoring.</div>";
      evalGrid.innerHTML = "";
      logBox.textContent = "No log loaded.";
    }
  }

  async function loadConfigs() {
    const resp = await fetch("/api/configs");
    const data = await resp.json();
    configSelect.innerHTML = "";
    data.configs.forEach((config) => {
      const opt = document.createElement("option");
      opt.value = config;
      opt.textContent = config;
      configSelect.appendChild(opt);
    });
    await loadConfigDefaults();
  }

  async function loadConfigDefaults() {
    const cfg = configSelect.value;
    if (!cfg) return;
    const resp = await fetch(`/api/config_summary?config=${encodeURIComponent(cfg)}`);
    const data = await resp.json();
    if (data.error) {
      launchStatus.textContent = `Error: ${data.error}`;
      return;
    }
    currentConfigDefaults = data;
    seedInput.value = data.seed ?? "";
    trainStepsInput.value = data.train_steps ?? "";
    ckptEveryInput.value = data.checkpoint_every ?? "";
    evalEveryInput.value = data.eval_every ?? "";
    evalNumInput.value = data.eval_num_samples ?? "";
    evalStepsInput.value = data.eval_sampling_steps ?? "";
    evalSeedInput.value = data.eval_seed ?? "";
    snrInput.value = data.snr_ratio ?? "";
    spectralSelect.value = data.spectral_operator_mode || "";
    updateCommandPreview();
  }

  function updateCommandPreview() {
    const cfg = configSelect.value;
    const args = ["python", "train.py", "--config", cfg, "--output-dir", "./runs"];
    if (runIdInput.value) args.push("--run-id", runIdInput.value);
    if (seedInput.value && String(seedInput.value) !== String(currentConfigDefaults.seed ?? "")) {
      args.push("--seed", seedInput.value);
    }
    if (trainStepsInput.value && String(trainStepsInput.value) !== String(currentConfigDefaults.train_steps ?? "")) {
      args.push("--train-steps", trainStepsInput.value);
    }
    if (ckptEveryInput.value && String(ckptEveryInput.value) !== String(currentConfigDefaults.checkpoint_every ?? "")) {
      args.push("--checkpoint-every", ckptEveryInput.value);
    }
    if (evalEveryInput.value && String(evalEveryInput.value) !== String(currentConfigDefaults.eval_every ?? "")) {
      args.push("--eval-every", evalEveryInput.value);
    }
    if (evalNumInput.value && String(evalNumInput.value) !== String(currentConfigDefaults.eval_num_samples ?? "")) {
      args.push("--eval-num-samples", evalNumInput.value);
    }
    if (evalStepsInput.value && String(evalStepsInput.value) !== String(currentConfigDefaults.eval_sampling_steps ?? "")) {
      args.push("--eval-sampling-steps", evalStepsInput.value);
    }
    if (evalSeedInput.value && String(evalSeedInput.value) !== String(currentConfigDefaults.eval_seed ?? "")) {
      args.push("--eval-seed", evalSeedInput.value);
    }
    if (snrInput.value && String(snrInput.value) !== String(currentConfigDefaults.snr_ratio ?? "")) {
      args.push("--snr-ratio", snrInput.value);
    }
    if (spectralSelect.value && spectralSelect.value !== String(currentConfigDefaults.spectral_operator_mode ?? "")) {
      args.push("--spectral-operator-mode", spectralSelect.value);
    }
    if (logLevelSelect.value) args.push("--log-level", logLevelSelect.value);
    launchCommand.textContent = args.join(" ");
  }

  async function launchTraining() {
    const normalize = (value, defaultValue) => {
      if (value === "" || value === null || value === undefined) return null;
      if (defaultValue === undefined || defaultValue === null) return value;
      return String(value) === String(defaultValue) ? null : value;
    };
    const payload = {
      config: configSelect.value,
      run_id: runIdInput.value || null,
      seed: normalize(seedInput.value, currentConfigDefaults.seed),
      train_steps: normalize(trainStepsInput.value, currentConfigDefaults.train_steps),
      checkpoint_every: normalize(ckptEveryInput.value, currentConfigDefaults.checkpoint_every),
      eval_every: normalize(evalEveryInput.value, currentConfigDefaults.eval_every),
      eval_num_samples: normalize(evalNumInput.value, currentConfigDefaults.eval_num_samples),
      eval_sampling_steps: normalize(evalStepsInput.value, currentConfigDefaults.eval_sampling_steps),
      eval_seed: normalize(evalSeedInput.value, currentConfigDefaults.eval_seed),
      snr_ratio: normalize(snrInput.value, currentConfigDefaults.snr_ratio),
      spectral_operator_mode: normalize(spectralSelect.value, currentConfigDefaults.spectral_operator_mode),
      log_level: logLevelSelect.value || "WARNING",
    };
    launchBtn.disabled = true;
    launchStatus.textContent = "Launching...";
    const resp = await fetch("/api/launch", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.error) {
      launchStatus.textContent = `Error: ${data.error}`;
    } else {
      launchStatus.textContent = `Started: ${data.run_dir}`;
      artifactRunDir.textContent = data.run_dir || "n/a";
      artifactLog.textContent = data.log_path || "n/a";
      if (data.run_dir) {
        artifactInspectCli.textContent = `python scripts/debug/inspect_step.py ${data.run_dir}`;
      } else {
        artifactInspectCli.textContent = "n/a";
      }
      setStage(true, true, false);
    }
    launchBtn.disabled = false;
    await loadRuns();
    if (data.run_dir) {
      runSelect.value = data.run_dir;
      await refreshAll();
    }
  }

  async function refreshAll() {
    const runDir = runSelect.value;
    if (!runDir) return;
    setStatus("Refreshing...");
    const limit = limitInput.value || 500;
    const [infoResp, metricsResp, evalResp, logResp] = await Promise.all([
      fetch(`/api/run_info?run_dir=${encodeURIComponent(runDir)}`),
      fetch(`/api/metrics?run_dir=${encodeURIComponent(runDir)}&limit=${limit}`),
      fetch(`/api/evals?run_dir=${encodeURIComponent(runDir)}`),
      fetch(`/api/log_tail?run_dir=${encodeURIComponent(runDir)}&lines=200`),
    ]);
    const info = await infoResp.json();
    const metrics = await metricsResp.json();
    const evals = await evalResp.json();
    const logs = await logResp.json();

    runSummary.innerHTML = `
      <div>Model: ${info.model}</div>
      <div>Dataset: ${info.dataset}</div>
      <div>Train steps: ${info.train_steps}</div>
      <div>Last step: ${info.last_step ?? "n/a"}</div>
      <div>Last loss: ${info.last_loss ?? "n/a"}</div>
      <div>SNR ratio: ${info.snr_ratio}</div>
      <div>Spectral mode: ${info.spectral_operator_mode}</div>
    `;
    const lastStep = info.last_step ?? 0;
    const totalSteps = info.train_steps ?? 0;
    if (totalSteps > 0 && lastStep) {
      const pct = Math.min(100, Math.round((lastStep / totalSteps) * 100));
      progressPill.textContent = `Progress: ${lastStep}/${totalSteps} (${pct}%)`;
    } else if (lastStep) {
      progressPill.textContent = `Progress: ${lastStep} steps`;
    } else {
      progressPill.textContent = "Progress: n/a";
    }
    setStage(true, true, false);
    artifactRunDir.textContent = runDir || "n/a";
    artifactLog.textContent = `${runDir}/logs/train.log`;
    artifactInspectCli.textContent = `python scripts/debug/inspect_step.py ${runDir}`;

    const rows = metrics.rows || [];
    const getSeries = (key) => rows.map((r) => r[key]).filter((v) => v !== null && v !== undefined);
    drawLine(charts.loss, getSeries("loss"), "#1f4e79");
    drawLine(charts.mae, getSeries("mae"), "#2b8a3e");
    drawLine(charts.grad_norm, getSeries("grad_norm"), "#a64b2a");
    drawLine(charts.snr_rel, getSeries("snr_rel"), "#6a4c93");

    evalGrid.innerHTML = "";
    let latestEvalGridPath = null;
    (evals.artifacts || []).forEach((artifact) => {
      const fig = document.createElement("figure");
      const cap = document.createElement("figcaption");
      cap.textContent = artifact.label;
      const img = document.createElement("img");
      img.src = `/file?path=${encodeURIComponent(artifact.path)}`;
      fig.appendChild(cap);
      fig.appendChild(img);
      evalGrid.appendChild(fig);
      if (!latestEvalGridPath && artifact.path.includes("samples/grid.png")) {
        latestEvalGridPath = artifact.path;
      }
    });
    if (latestEvalGridPath) {
      artifactEvalGrid.textContent = latestEvalGridPath;
      openEvalGrid.href = `/file?path=${encodeURIComponent(latestEvalGridPath)}`;
      setStage(true, true, true);
    } else {
      artifactEvalGrid.textContent = "n/a";
      openEvalGrid.href = "#";
    }

    logBox.textContent = (logs.lines || []).join("\\n");
    setStatus("Ready.");
  }

  refreshBtn.addEventListener("click", refreshAll);
  runSelect.addEventListener("change", refreshAll);
  configSelect.addEventListener("change", loadConfigDefaults);
  [
    runIdInput, seedInput, trainStepsInput, ckptEveryInput, evalEveryInput, evalNumInput,
    evalStepsInput, evalSeedInput, snrInput, spectralSelect, logLevelSelect,
  ].forEach((el) => el.addEventListener("input", updateCommandPreview));
  launchBtn.addEventListener("click", launchTraining);
  autoBtn.addEventListener("click", () => {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
      autoBtn.textContent = "Auto refresh: off";
    } else {
      autoTimer = setInterval(refreshAll, 5000);
      autoBtn.textContent = "Auto refresh: on";
    }
  });

  loadRuns().catch((err) => setStatus(`Error: ${err}`));
  loadConfigs().catch((err) => launchStatus.textContent = `Error: ${err}`);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/configs":
            configs = [_relpath(p) for p in _discover_configs()]
            self._send_json({"configs": configs})
            return
        if parsed.path == "/api/config_summary":
            params = parse_qs(parsed.query)
            config_path = params.get("config", [None])[0]
            if not config_path:
                self._send_json({"error": "Missing config"}, status=400)
                return
            try:
                summary = _config_summary(_safe_path(config_path))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(summary)
            return
        if parsed.path == "/api/runs":
            runs = [_relpath(p) for p in _discover_runs()]
            self._send_json({"runs": runs})
            return
        if parsed.path == "/api/run_info":
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [None])[0]
            if not run_dir:
                self._send_json({"error": "Missing run_dir"}, status=400)
                return
            try:
                info = _run_info(_safe_path(run_dir))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(info)
            return
        if parsed.path == "/api/metrics":
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [None])[0]
            if not run_dir:
                self._send_json({"error": "Missing run_dir"}, status=400)
                return
            try:
                limit = int(params.get("limit", ["500"])[0])
            except ValueError:
                limit = 500
            try:
                run_path = _safe_path(run_dir)
                history = run_path / "diagnostics" / "training_history.csv"
                rows = [_coerce_row(r) for r in _read_csv_tail(history, limit=limit)]
                self._send_json({"rows": rows})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/evals":
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [None])[0]
            if not run_dir:
                self._send_json({"error": "Missing run_dir"}, status=400)
                return
            try:
                payload = _latest_eval(_safe_path(run_dir))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(payload)
            return
        if parsed.path == "/api/log_tail":
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [None])[0]
            if not run_dir:
                self._send_json({"error": "Missing run_dir"}, status=400)
                return
            try:
                lines = int(params.get("lines", ["200"])[0])
            except ValueError:
                lines = 200
            try:
                run_path = _safe_path(run_dir)
                log_path = run_path / "logs" / "train.log"
                if not log_path.exists():
                    self._send_json({"lines": []})
                    return
                content = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                self._send_json({"lines": content[-lines:]})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/file":
            params = parse_qs(parsed.query)
            path_str = params.get("path", [None])[0]
            if not path_str:
                self._send_json({"error": "Missing path"}, status=400)
                return
            try:
                file_path = _safe_path(path_str)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            if not file_path.exists():
                self._send_json({"error": "File not found"}, status=404)
                return
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self._send_bytes(file_path.read_bytes(), content_type)
            return
        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/launch":
            self._send_json({"error": "Not found"}, status=404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
            return
        config = payload.get("config")
        if not config:
            self._send_json({"error": "Missing config."}, status=400)
            return
        try:
            config_path = _safe_path(config)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        if not config_path.exists():
            self._send_json({"error": "Config not found."}, status=404)
            return
        run_id = payload.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd = [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(ROOT / "runs"),
            "--run-id",
            str(run_id),
            "--log-level",
            payload.get("log_level", "WARNING"),
        ]
        run_root = ROOT / "runs" / run_id
        logs_dir = run_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        train_log = logs_dir / "train.log"
        if not train_log.exists():
            train_log.write_text("", encoding="utf-8")
        optional_args = {
            "seed": "--seed",
            "train_steps": "--train-steps",
            "checkpoint_every": "--checkpoint-every",
            "eval_every": "--eval-every",
            "eval_num_samples": "--eval-num-samples",
            "eval_sampling_steps": "--eval-sampling-steps",
            "eval_seed": "--eval-seed",
            "snr_ratio": "--snr-ratio",
            "spectral_operator_mode": "--spectral-operator-mode",
        }
        for key, flag in optional_args.items():
            value = payload.get(key)
            if value not in (None, "", "null"):
                cmd.extend([flag, str(value)])
        try:
            subprocess.Popen(cmd, cwd=str(ROOT))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        run_dir = f"runs/{run_id}"
        log_path = str(train_log)
        self._send_json(
            {"status": "started", "run_dir": run_dir, "command": " ".join(cmd), "log_path": log_path}
        )


def main() -> int:
    host = "127.0.0.1"
    port = 8790
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Training monitor running at http://{host}:{port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
