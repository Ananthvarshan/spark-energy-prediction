"""
============================================================
PHASE V MASTER RUNNER  --  run_phase5.py
============================================================

Reproduces every Phase V result with one command:

    python run_phase5.py                     # everything (~6 h)
    python run_phase5.py --skip task12       # tests only, from stored runs
    python run_phase5.py --only task11 figures
    python run_phase5.py --neural-seeds 3    # a cheaper Task 12A

STEPS
-----
  task12   repeated runs: forecasting models over seeds, the
           labeller over ten seeds, the decision layer over ten
           seeds                                        (~5 h)
  task11   significance: paired Wilcoxon over windows with Holm and
           Benjamini-Hochberg adjustment, Friedman + Nemenyi over
           seeds, day-block bootstrap of the decision layer,
           one-week block bootstrap of the state-time totals, and
           the cross-machine tests with their power    (~25 min)
  figures  Phase V figures                              (~3 min)

PREREQUISITES
-------------
Phase IV's labelled parquet files (`outputs/phase4/labelled/`) and,
for the cross-machine tests, its Task 9B transfer matrix.  The
forecasting tests read Phase II's labelling by default, because they
have to be comparable with the Task 5 table they extend.

WHAT PHASE V DOES NOT DO
------------------------
It does not re-run Phases I-IV.  Every test here consumes stored
outputs, so a claim in the paper and the interval around it come
from the same numbers.
============================================================
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.common import (            # noqa: E402
    DEFAULT_MACHINE, MACHINES, ensure_dir, info, section,
)

PHASE5_DIR = "outputs/phase5"
STEPS = ["task12", "task11", "figures"]


def _run(name: str, fn, **kwargs):
    section(f"RUNNING {name}")
    t0 = time.time()
    try:
        fn(**kwargs)
        dt = time.time() - t0
        info(f"{name} finished in {dt/60:.1f} min")
        return {"step": name, "status": "ok", "minutes": round(dt / 60, 2)}
    except Exception as exc:
        dt = time.time() - t0
        info(f"{name} FAILED after {dt/60:.1f} min: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return {"step": name, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "minutes": round(dt / 60, 2)}


def main():
    ap = argparse.ArgumentParser(description="Phase V master runner")
    ap.add_argument("--machine", default=DEFAULT_MACHINE, choices=list(MACHINES))
    ap.add_argument("--only", nargs="*", default=None, help=f"steps: {STEPS}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--neural-seeds", type=int, default=None)
    ap.add_argument("--label-seeds", type=int, default=10)
    ap.add_argument("--decision-seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    steps = [s for s in (args.only or STEPS) if s not in args.skip]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        print(f"Unknown step(s): {unknown}. Choose from {STEPS}")
        sys.exit(1)

    ensure_dir(f"{PHASE5_DIR}/logs")
    section(f"PHASE V  --  machine {args.machine}  --  steps {steps}")

    report, t_all = [], time.time()

    if "task12" in steps:
        from experiments.task12_multiple_runs import main as t12
        report.append(_run("task12", t12, machine=args.machine,
                           neural_seeds=args.neural_seeds,
                           label_seeds=args.label_seeds,
                           decision_seed_count=args.decision_seeds))

    if "task11" in steps:
        from experiments.task11_significance import main as t11
        report.append(_run("task11", t11, machine=args.machine,
                           n_boot=args.n_boot))

    if "figures" in steps:
        from experiments.make_figures_phase5 import main as figs
        report.append(_run("figures", figs, machine=args.machine))

    section("PHASE V SUMMARY")
    for r in report:
        mark = "OK  " if r["status"] == "ok" else "FAIL"
        line = f"  [{mark}] {r['step']:<8} {r['minutes']:>7.1f} min"
        if r["status"] != "ok":
            line += f"   {r['error']}"
        print(line)
    print(f"\n  total {(time.time()-t_all)/60:.1f} min")
    print(f"  results under {PHASE5_DIR}/")

    if any(r["status"] != "ok" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
