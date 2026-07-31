"""
============================================================
PHASE II MASTER RUNNER  --  run_phase2.py
============================================================

Reproduces every Phase II result with one command:

    python run_phase2.py                      # everything, pelletizer-I
    python run_phase2.py --machine pelletizer-II
    python run_phase2.py --only task4 task4b  # a subset
    python run_phase2.py --skip task5         # everything but the long one
    python run_phase2.py --seeds 20           # Phase V-style repetition

STEPS
-----
  task3   feature engineering + importance (MI / XGBoost gain / SHAP)
  task4   state-detection comparison (5 methods + min-dwell variant)
  task4b  sticky-prior strength sweep
  task4c  flicker diagnosis + minimum-dwell decode
  task5   forecasting baselines (5 neural + XGBoost)
  figures result figures

PREREQUISITE
------------
The labelled CSV must exist:

    outputs/imdeld_labelled/<machine>_labelled.csv

Produce it with `python validate_gmm.py <machine>` if it is missing.

RUNTIME
-------
On a CPU-only machine, task3 ~2 min, task4 ~6 min, task4b ~7 min,
task4c ~3 min, task5 ~1.5-2 h at the default single seed.  task5
dominates; use --skip task5 for a fast pass over everything else.
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
    DEFAULT_MACHINE, MACHINES, PHASE2_DIR, ensure_dir, info, section,
)

STEPS = ["task3", "task4", "task4b", "task4c", "task5", "figures"]


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
    ap = argparse.ArgumentParser(description="Phase II master runner")
    ap.add_argument("--machine", default=DEFAULT_MACHINE,
                    choices=list(MACHINES))
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"run only these steps: {STEPS}")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="skip these steps")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repetitions per forecasting model (Phase V uses 20-30)")
    ap.add_argument("--decimate", type=int, default=10,
                    help="task5 decimation factor (1 Hz / this)")
    ap.add_argument("--epochs", type=int, default=30)
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

    ensure_dir(f"{PHASE2_DIR}/logs")
    section(f"PHASE II  --  machine {args.machine}  --  steps {steps}")

    report = []
    t_all = time.time()

    if "task3" in steps:
        from experiments.task3_feature_importance import main as t3
        report.append(_run("task3", t3, machine=args.machine))

    if "task4" in steps:
        from experiments.task4_state_detection import main as t4
        report.append(_run("task4", t4, machine=args.machine))

    if "task4b" in steps:
        from experiments.task4_sticky_prior_sweep import main as t4b
        report.append(_run("task4b", t4b, machine=args.machine))

    if "task4c" in steps:
        from experiments.task4_flicker_diagnosis import main as t4c
        report.append(_run("task4c", t4c, machine=args.machine))

    if "task5" in steps:
        from experiments.task5_forecasting_baselines import main as t5
        report.append(_run("task5", t5, machine=args.machine,
                           seeds=args.seeds, decimate_factor=args.decimate,
                           epochs=args.epochs))

    if "figures" in steps:
        from experiments.make_figures import main as figs
        report.append(_run("figures", figs, machine=args.machine))

    section("PHASE II SUMMARY")
    for r in report:
        mark = "OK  " if r["status"] == "ok" else "FAIL"
        line = f"  [{mark}] {r['step']:<8} {r['minutes']:>6.1f} min"
        if r["status"] != "ok":
            line += f"   {r['error']}"
        print(line)
    print(f"\n  total {(time.time()-t_all)/60:.1f} min")
    print(f"  results under {PHASE2_DIR}/")

    if any(r["status"] != "ok" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
