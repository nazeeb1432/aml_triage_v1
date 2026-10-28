"""Check that this machine can run the project, and say how to fix what's missing.

    python3 scripts/doctor.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = True


def check(name: str, passed: bool, fix: str = "") -> None:
    global ok
    print(f"  {'OK  ' if passed else 'FIX '} {name}" + ("" if passed else f"\n        -> {fix}"))
    ok &= passed


print(f"Machine: {platform.system()} {platform.machine()}, Python {platform.python_version()} at {sys.executable}\n")

print("Tools")
check("Python 3.11 or newer", sys.version_info >= (3, 11),
      "install Python 3.12 (python.org or `brew install python@3.12`), then run: make setup PYTHON=python3.12")
node = shutil.which("node")
node_ver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip() if node else ""
check(f"Node.js 18+ for the console {node_ver}", bool(node) and int(node_ver.lstrip("v").split(".")[0] or 0) >= 18,
      "install Node.js 20 or 22 (nodejs.org or `brew install node`)")
if platform.system() == "Darwin":
    libomp = any(Path(p).exists() for p in ("/opt/homebrew/opt/libomp/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib"))
    check("libomp (LightGBM needs it on macOS)", libomp, "brew install libomp")

print("\nProject")
venv = ROOT / ".venv" / "bin" / "python"
check("virtual environment (.venv)", venv.exists(), "make setup")
if venv.exists():
    r = subprocess.run([str(venv), "-c", "import aml_triage, lightgbm, numba, polars, fastapi; print('ok')"],
                       capture_output=True, text=True)
    check("Python packages import (incl. LightGBM)", r.stdout.strip() == "ok",
          (r.stderr.strip().splitlines() or ["unknown error"])[-1] + "  (on macOS this is usually `brew install libomp`)")
    g = subprocess.run([str(venv), "-c", "import ctgan, torch"], capture_output=True, text=True)
    check("GAN extras (ctgan, torch) - optional", g.returncode == 0, "make setup installs them; without them the CTGAN arm is skipped")
check("console packages (ui/node_modules)", (ROOT / "ui" / "node_modules").exists(), "make setup  (or: cd ui && npm install)")
check("console built (ui/dist)", (ROOT / "ui" / "dist" / "index.html").exists(), "make ui-build")

print("\nData")
amlsim = ROOT / "data" / "raw" / "amlsim" / "transactions.csv"
check("AMLSim files in data/raw/amlsim/", amlsim.exists(),
      "copy transactions.csv, alerts.csv, accounts.csv there, or run: make get-amlsim")
runs = sorted(p.parent.name for p in (ROOT / "reports").glob("*/scored_alerts.parquet"))
check(f"a scored run the console can show {runs if runs else ''}", bool(runs),
      "make run-amlsim-fast (~15 min) or make run-amlsim (~45 min)")

print("\nAll set." if ok else "\nFix the items marked FIX, then run this again.")
sys.exit(0 if ok else 1)
