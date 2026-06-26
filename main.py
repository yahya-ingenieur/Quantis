"""
main.py — Quantis pipeline orchestrator.

Runs the project end to end: load data -> train -> evaluate -> save outputs.
Each stage runs as its own process (so per-module import order is correct and a
failure in one stage is isolated), mirroring the documented per-module commands.

Usage:
    python main.py                 # core pipeline: data -> train -> evaluate -> report
    python main.py --all           # core pipeline + optional experiments
    python main.py --only evaluate # re-run a single stage
    python main.py --from train    # resume from a stage onward

Optional experiments (--all): multi-seed robustness, feature ablation, and the
sentiment with/without experiment. The live news-sentiment comparison and the
fetch/score steps need network access (and a NEWSAPI_KEY for the live parts), so
they are left out of the default run and invoked separately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (name, command, description)
CORE_STAGES = [
    ("data", [PY, "src/data.py"], "Download OHLCV + build the cleaned dataset"),
    ("train", [PY, "-m", "src.train"], "Train LSTM/GRU, baselines, clip + HP experiments"),
    ("evaluate", [PY, "-m", "src.evaluate"], "Metrics, VaR, backtest (with costs), 11 plots"),
    ("report", [PY, "-m", "src.report_plots"], "Extra report/presentation figures"),
]
OPTIONAL_STAGES = [
    ("robustness", [PY, "-m", "src.robustness"], "Multi-seed LSTM vs GRU (mean ± std)"),
    ("ablation", [PY, "-m", "src.feature_ablation"], "Feature-engineering ablation"),
    ("sentiment", [PY, "-m", "src.sentiment_experiment"], "Sentiment with-vs-without"),
]


def run_stage(name: str, cmd: list[str], desc: str) -> None:
    print(f"\n=== [{name}] {desc} ===", flush=True)
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[{name}] FAILED (exit {result.returncode}) — stopping.", flush=True)
        sys.exit(result.returncode)
    print(f"[{name}] done in {time.time() - start:.0f}s", flush=True)


def main() -> None:
    names = [s[0] for s in CORE_STAGES]
    ap = argparse.ArgumentParser(description="Run the Quantis pipeline end to end.")
    ap.add_argument("--all", action="store_true",
                    help="also run optional experiments (robustness, ablation, sentiment)")
    ap.add_argument("--only", choices=names, help="run only this core stage")
    ap.add_argument("--from", dest="start", choices=names, help="start from this core stage onward")
    args = ap.parse_args()

    if args.only:
        stages = [s for s in CORE_STAGES if s[0] == args.only]
    elif args.start:
        idx = names.index(args.start)
        stages = CORE_STAGES[idx:]
    else:
        stages = list(CORE_STAGES)

    print("Quantis pipeline — stages:", ", ".join(s[0] for s in stages)
          + (" (+ optional)" if args.all and not args.only else ""))
    overall = time.time()
    for name, cmd, desc in stages:
        run_stage(name, cmd, desc)
    if args.all and not args.only:
        for name, cmd, desc in OPTIONAL_STAGES:
            run_stage(name, cmd, desc)

    print(f"\nPipeline complete in {time.time() - overall:.0f}s. "
          "Outputs in outputs/ (plots, JSON) and checkpoints/.")


if __name__ == "__main__":
    main()
