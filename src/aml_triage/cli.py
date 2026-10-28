"""Command line entry point.

    aml-triage run                         # all stages, default synthetic config
    aml-triage run --stages features train # rerun some stages
    aml-triage run --profile configs/ibm_hi_small.yaml
    aml-triage run --set training.n_trials=5 --set data.synthetic.n_accounts=3000
    aml-triage serve                       # API on :8000 (serves the UI build too, if present)
"""

from __future__ import annotations

import argparse

from aml_triage.config import load_config

ALL = ["ingest", "alerts", "features", "train", "score", "backtest"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aml-triage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/default.yaml")
    r.add_argument("--profile", default=None)
    r.add_argument("--stages", nargs="*", default=ALL, choices=ALL)
    r.add_argument("--set", action="append", default=[], help="override, e.g. training.n_trials=5")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--run", default=None, help="reports/<run> to serve (default: newest)")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from aml_triage.pipeline import run_all

        cfg = load_config(args.config, args.profile, args.set)
        run_all(cfg, args.stages)
    else:
        import os

        import uvicorn

        if args.run:
            os.environ["AML_TRIAGE_RUN"] = args.run
        uvicorn.run("aml_triage.api.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
