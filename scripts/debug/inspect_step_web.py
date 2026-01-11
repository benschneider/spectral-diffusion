#!/usr/bin/env python3
"""
Lightweight web UI for inspect_step.py (stdlib only).

Usage:
  python scripts/debug/inspect_step_web.py
  open http://127.0.0.1:8787
"""

from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli.common import load_config  # noqa: E402
from scripts.debug.inspect_step import inspect_step  # noqa: E402


RUN_ROOTS = [ROOT / "runs", ROOT / "results"]


def _discover_runs() -> List[Path]:
    runs: List[Path] = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for config_path in root.rglob("config.yaml"):
            run_dir = config_path.parent
            ckpt_dir = run_dir / "checkpoints"
            if any(ckpt_dir.glob("checkpoint_step_*.pt")):
                runs.append(run_dir)
    return sorted(set(runs), key=lambda p: str(p).lower())


def _relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _run_info(run_dir: Path) -> Dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    diffusion_cfg = config.get("diffusion", {}) or {}
    num_timesteps = int(diffusion_cfg.get("num_timesteps", 1000))
    ckpt_steps = []
    for ckpt in (run_dir / "checkpoints").glob("checkpoint_step_*.pt"):
        try:
            step = int(ckpt.stem.split("_")[-1])
        except (ValueError, IndexError):
            continue
        ckpt_steps.append(step)
    ckpt_steps = sorted(set(ckpt_steps))
    return {
        "run_dir": _relpath(run_dir),
        "num_timesteps": num_timesteps,
        "checkpoint_steps": ckpt_steps,
    }


def _safe_path(path_str: str) -> Path:
    candidate = (ROOT / path_str).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError("Path escapes repo root.")
    return candidate


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Diffusion Step Inspector</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #ffffff;
      --ink: #1f1f1f;
      --muted: #5a5a5a;
      --accent: #2b6cb0;
      --line: #ded8cd;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(180deg, #f4f1ea 0%, #f8f6f1 100%);
      color: var(--ink);
    }
    header {
      padding: 24px 28px 12px 28px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-weight: 600;
      font-size: 22px;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      padding: 16px 24px 32px 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.06);
    }
    label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      display: block;
      margin-top: 14px;
    }
    select, input[type="number"], input[type="text"] {
      width: 100%;
      padding: 8px 10px;
      margin-top: 6px;
      border-radius: 8px;
      border: 1px solid var(--line);
      font-size: 14px;
    }
    input[type="range"] {
      width: 100%;
      margin-top: 6px;
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
    button:disabled {
      opacity: 0.6;
      cursor: wait;
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
    figure {
      margin: 0;
      background: #faf7f2;
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
    pre {
      background: #111;
      color: #d7d7d7;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
      font-size: 12px;
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
    <h1>Diffusion Step Inspector</h1>
    <p>Pick a run, checkpoint, and diffusion timestep. The inspector renders x0, xt, pred x0, and eps grids plus a summary.</p>
  </header>
  <div class="layout">
    <div class="panel">
      <label for="runSelect">Run directory</label>
      <select id="runSelect"></select>

      <label for="ckptSelect">Checkpoint step</label>
      <select id="ckptSelect"></select>

      <label for="timestepRange">Timestep</label>
      <input id="timestepRange" type="range" min="0" max="999" value="500" />
      <input id="timestepInput" type="number" min="0" max="999" value="500" />

      <label for="numInput">Num samples</label>
      <input id="numInput" type="number" min="1" max="64" value="16" />

      <label for="seedInput">Seed</label>
      <input id="seedInput" type="number" min="0" value="0" />

      <button id="inspectBtn">Inspect</button>
      <div class="status" id="status">Idle.</div>
    </div>
    <div class="panel">
      <div class="grid" id="imageGrid"></div>
      <h3>Summary</h3>
      <pre id="summaryBox">{}</pre>
    </div>
  </div>
<script>
  const runSelect = document.getElementById("runSelect");
  const ckptSelect = document.getElementById("ckptSelect");
  const timestepRange = document.getElementById("timestepRange");
  const timestepInput = document.getElementById("timestepInput");
  const numInput = document.getElementById("numInput");
  const seedInput = document.getElementById("seedInput");
  const inspectBtn = document.getElementById("inspectBtn");
  const statusEl = document.getElementById("status");
  const imageGrid = document.getElementById("imageGrid");
  const summaryBox = document.getElementById("summaryBox");

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function syncTimestepInputs(value) {
    timestepRange.value = value;
    timestepInput.value = value;
  }

  timestepRange.addEventListener("input", () => syncTimestepInputs(timestepRange.value));
  timestepInput.addEventListener("input", () => syncTimestepInputs(timestepInput.value));

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
      await loadRunInfo(data.runs[0]);
    }
  }

  async function loadRunInfo(runDir) {
    const resp = await fetch(`/api/run_info?run_dir=${encodeURIComponent(runDir)}`);
    const info = await resp.json();
    ckptSelect.innerHTML = "";
    const latestOpt = document.createElement("option");
    latestOpt.value = "latest";
    latestOpt.textContent = "latest";
    ckptSelect.appendChild(latestOpt);
    info.checkpoint_steps.forEach((step) => {
      const opt = document.createElement("option");
      opt.value = step;
      opt.textContent = step;
      ckptSelect.appendChild(opt);
    });
    const maxT = Math.max(0, (info.num_timesteps || 1000) - 1);
    timestepRange.max = maxT;
    timestepInput.max = maxT;
    syncTimestepInputs(Math.floor(maxT / 2));
  }

  runSelect.addEventListener("change", async () => {
    await loadRunInfo(runSelect.value);
  });

  inspectBtn.addEventListener("click", async () => {
    const runDir = runSelect.value;
    if (!runDir) return;
    inspectBtn.disabled = true;
    setStatus("Running inspection...");
    const params = new URLSearchParams({
      run_dir: runDir,
      ckpt_step: ckptSelect.value,
      timestep: timestepInput.value,
      num: numInput.value,
      seed: seedInput.value,
    });
    try {
      const resp = await fetch(`/api/inspect?${params.toString()}`);
      const data = await resp.json();
      if (data.error) {
        setStatus(`Error: ${data.error}`);
        inspectBtn.disabled = false;
        return;
      }
      summaryBox.textContent = JSON.stringify(data, null, 2);
      imageGrid.innerHTML = "";
      const labels = {
        x0: "x0 (clean)",
        xt: "xt (noised)",
        pred_x0: "pred x0",
        eps_true: "eps true",
        eps_pred: "eps pred",
      };
      Object.keys(labels).forEach((key) => {
        const entry = data.outputs[key];
        if (!entry) return;
        const fig = document.createElement("figure");
        const cap = document.createElement("figcaption");
        cap.textContent = labels[key];
        const img = document.createElement("img");
        img.src = entry.url;
        fig.appendChild(cap);
        fig.appendChild(img);
        imageGrid.appendChild(fig);
      });
      setStatus("Done.");
    } catch (err) {
      setStatus(`Error: ${err}`);
    } finally {
      inspectBtn.disabled = false;
    }
  });

  loadRuns().catch((err) => setStatus(`Error: ${err}`));
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
            except Exception as exc:  # pragma: no cover - handler safety
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(info)
            return
        if parsed.path == "/api/inspect":
            params = parse_qs(parsed.query)
            run_dir = params.get("run_dir", [None])[0]
            if not run_dir:
                self._send_json({"error": "Missing run_dir"}, status=400)
                return
            ckpt_step_raw = params.get("ckpt_step", ["latest"])[0]
            ckpt_step: Optional[int]
            if ckpt_step_raw in ("latest", "", None):
                ckpt_step = None
            else:
                try:
                    ckpt_step = int(ckpt_step_raw)
                except ValueError:
                    self._send_json({"error": f"Invalid ckpt_step {ckpt_step_raw}"}, status=400)
                    return
            try:
                timestep = int(params.get("timestep", ["0"])[0])
                num = int(params.get("num", ["16"])[0])
                seed = int(params.get("seed", ["0"])[0])
            except ValueError:
                self._send_json({"error": "Invalid numeric parameters."}, status=400)
                return
            try:
                summary = inspect_step(
                    run_dir=_safe_path(run_dir),
                    ckpt_step=ckpt_step,
                    timestep=timestep,
                    num=num,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - handler safety
                self._send_json({"error": str(exc)}, status=400)
                return
            outputs = {}
            for key, path in summary.get("outputs", {}).items():
                rel = _relpath(Path(path))
                outputs[key] = {"path": rel, "url": f"/file?path={rel}"}
            summary["outputs"] = outputs
            self._send_json(summary)
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


def main() -> int:
    host = "127.0.0.1"
    port = 8787
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Inspect step web UI running at http://{host}:{port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
