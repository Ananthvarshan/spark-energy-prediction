"""
============================================================
PHASE III MASTER RUNNER  --  run_phase3.py
============================================================

Reproduces every Phase III result with one command:

    python run_phase3.py                      # everything, pelletizer-I
    python run_phase3.py --machine pelletizer-II
    python run_phase3.py --only task6         # a subset
    python run_phase3.py --skip task7

STEPS
-----
  task6    optimisation-based decision layer: measured restart
           coefficients, remaining-idle-duration forecaster,
           chance-constrained sequential policy, and the policy
           comparison against the plant's own status quo
  task7    break-even attainability curve, the economic plane,
           tariff-invariance check, constraint sensitivity, payback
  figures  Phase III result figures

PREREQUISITE
------------
The labelled CSV must exist:

    outputs/imdeld_labelled/<machine>_labelled.csv

Produce it with `python validate_gmm.py <machine>` if it is missing.

RUNTIME
-------
On a CPU-only machine the whole phase takes about 4.5 min: task6
~2.1 min, task7 ~2.2 min, figures ~0.2 min.  Both tasks reload the
5.5-million-row record and refit the same models through the shared
`task6.prepare`; there is no cache between processes by design, so
each task stays independently reproducible.
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
from experiments.task6_optimization import PHASE3_DIR   # noqa: E402

STEPS = ["task6", "task7", "figures"]


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
    ap = argparse.ArgumentParser(description="Phase III master runner")
    ap.add_argument("--machine", default=DEFAULT_MACHINE, choices=list(MACHINES))
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"run only these steps: {STEPS}")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    steps = [s for s in (args.only or STEPS) if s not in args.skip]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        print(f"Unknown step(s): {unknown}. Choose from {STEPS}")
        sys.exit(1)

    labelled = MACHINES[args.machine]["labelled"]
    if not os.path.exists(labelled):
        print(f"\nLabelled CSV not found: {labelled}")
        print(f"Run first:  python validate_gmm.py {args.machine}\n")
        sys.exit(1)

    ensure_dir(f"{PHASE3_DIR}/logs")
    section(f"PHASE III  --  machine {args.machine}  --  steps {steps}")

    report = []
    t_all = time.time()

    if "task6" in steps:
        from experiments.task6_optimization import main as t6
        report.append(_run("task6", t6, machine=args.machine, seed=args.seed))

    if "task7" in steps:
        from experiments.task7_sensitivity import main as t7
        report.append(_run("task7", t7, machine=args.machine, seed=args.seed))

    if "figures" in steps:
        from experiments.make_figures_phase3 import main as figs
        report.append(_run("figures", figs, machine=args.machine))

    section("PHASE III SUMMARY")
    for r in report:
        mark = "OK  " if r["status"] == "ok" else "FAIL"
        line = f"  [{mark}] {r['step']:<8} {r['minutes']:>6.1f} min"
        if r["status"] != "ok":
            line += f"   {r['error']}"
        print(line)
    print(f"\n  total {(time.time()-t_all)/60:.1f} min")
    print(f"  results under {PHASE3_DIR}/")

    if any(r["status"] != "ok" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
