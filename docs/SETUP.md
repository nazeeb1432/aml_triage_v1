# Setting up on your Mac

Tested from a clean copy of the repo: setup took ~6 minutes (mostly downloading PyTorch for the GAN), the quick
AMLSim run ~15 minutes, the full one ~45 minutes on 2 cores. An Apple-silicon Mac is usually faster.

## 1. Prerequisites (once)

```bash
brew install libomp python@3.12 node     # LightGBM needs libomp on macOS; Node runs the console build
```
Already have Python 3.11+ (e.g. from Anaconda) and Node 18+? Then you only need `brew install libomp`.

## 2. Get the project

Unzip `aml-triage.zip` somewhere you keep code, then:
```bash
cd aml-triage
git init && git add -A && git commit -m "Initial import"    # so Claude Code and you can track changes
```
To put it on GitHub: create an empty repo (e.g. `aml-triage`), then
`git remote add origin https://github.com/<you>/aml-triage.git && git push -u origin main`.

## 3. Install

```bash
make setup                    # Python venv + packages (incl. CTGAN/PyTorch) + console packages
# if your default python3 is older than 3.11:
make setup PYTHON=python3.12
make ui-build                 # build the analyst console
```

## 4. Add the data

Copy your three AMLSim files into `data/raw/amlsim/`:
```bash
mkdir -p data/raw/amlsim
cp ~/Downloads/{transactions,alerts,accounts}.csv data/raw/amlsim/    # adjust the source path
```
(or `make get-amlsim` to download them with kagglehub)

## 5. Check

```bash
make doctor      # lists anything missing and the command that fixes it
make test        # 16 tests, ~1-3 minutes
```

## 6. Run and look

```bash
make run-amlsim-fast          # ~15 min: small tuning budget, results within a point of the full run
make serve                    # open http://127.0.0.1:8000
```
For the reportable numbers: `make run-amlsim` (~45 min), then `make serve-amlsim`.
The synthetic world: `make run` (~25 min). The console shows the newest run unless you pass `--run <name>`.

Results land in `reports/<run>/`: `ablation.md` (what each feature family and the GAN add), `backtest.md`
(robustness), `metrics.json` (everything the console reads).

## 7. Working with Claude Code

Open the folder in VS Code (or run `claude` in it). Claude Code reads `CLAUDE.md` automatically, which points it at
the rules, the plan and the progress log. Useful first prompts:
- "Read docs/PROGRESS.md and tell me what's next."
- "Use the run-pipeline skill on AMLSim and summarise the results."
- "Use the leakage-audit skill on my last change."
- "Ask the reviewer agent to check this branch."

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Library not loaded: libomp.dylib` | `brew install libomp`, then `make setup` again |
| `requires a different Python` during setup | `make setup PYTHON=python3.12` |
| CTGAN step very slow or torch won't install | `.venv/bin/pip install -e ".[dev]"` (skips the GAN arm) |
| Console says "no scored run" | run a pipeline first (`make run-amlsim-fast`) |
| Port 8000 busy | `.venv/bin/aml-triage serve --port 8010` |
