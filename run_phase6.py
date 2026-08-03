"""
============================================================
PHASE VI MASTER RUNNER  --  run_phase6.py
============================================================

Reproduces every Phase VI result with one command:

    python run_phase6.py                      # everything (~12 min)
    python run_phase6.py --only task13c task13e figures
    python run_phase6.py --skip task13a       # the long one

Timings are measured on this machine (CPU only) and every step is
deterministic: a re-run reproduces the stored attributions exactly,
which is what makes the numbers in the paper checkable.

STEPS
-----
  task13a  Shapley attribution of both XGBoost forecasters: the
           decision layer's remaining-duration model on BOTH
           labellings, its chance-constraint classifier, and the
           window state forecaster refitted under Task 12A's
           protocol                                        (~8 min)
  task13b  Shapley attribution of the state layer, through a
           gradient-boosted surrogate on the minimum-dwell labels,
           for two target definitions                     (~1.5 min)
  task13c  transition matrices -- learned, realised, and realised
           after the minimum-dwell constraint -- for all eight
           IMDELD machines                                 (~1 min)
  task13e  the component ablation table, assembled from stored
           Phase II-V outputs                              (~5 s)
  figures  twelve figures: seven for Phase VI's own results and
           five publication re-renders of Phase V's        (~2 min)

PREREQUISITES
-------------
Phase IV's labelled parquet files and `task8/*/label_info.json`;
Phase V's Task 11 and Task 12 outputs; Phase II's Task 3 and Task 4
tables.  Nothing here refits a model that an earlier phase already
fitted, with one stated exception: the window forecaster, which
Task 5 and Task 12 store as predictions rather than as a booster.
Task 13A refits it at seed 0 and REFUSES TO CONTINUE unless it
reproduces the stored seed-0 metrics exactly.

WHAT PHASE VI DOES NOT DO
-------------------------
It does not revise any Phase I-V number.  Where an attribution
contradicts an earlier claim -- and one does, the 94.5% calendar
gain -- the correction is recorded in `paper/PHASE6_NOTES.md` for
the phase that owns it, in the same way Phase V recorded its own.
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
    DEFAULT_MACHINE, MACHINES, IMDELD_MACHINES, ensure_dir, info, section,
)

PHASE6_DIR = "outputs/phase6"
STEPS = ["task13a", "task13b", "task13c", "task13e", "figures"]


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
    ap = argparse.ArgumentParser(description="Phase VI master runner")
    ap.add_argument("--machine", default=DEFAULT_MACHINE, choices=list(MACHINES))
    ap.add_argument("--only", nargs="*", default=None, help=f"steps: {STEPS}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--label-sources", nargs="*", default=["auto", "phase4"],
                    help="labellings to attribute in Task 13A")
    ap.add_argument("--machines", nargs="*", default=None,
                    help="machines for Task 13C (default: all eight)")
    ap.add_argument("--scenario", default="S2_moderate",
                    help="economic scenario for the Task 13E decision block")
    args = ap.parse_args()

    steps = [s for s in (args.only or STEPS) if s not in args.skip]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        print(f"Unknown step(s): {unknown}. Choose from {STEPS}")
        sys.exit(1)

    ensure_dir(f"{PHASE6_DIR}/logs")
    section(f"PHASE VI  --  machine {args.machine}  --  steps {steps}")

    report, t_all = [], time.time()

    if "task13a" in steps:
        from experiments.task13a_forecaster_shap import main as t13a
        report.append(_run("task13a", t13a, machine=args.machine,
                           label_sources=args.label_sources))

    if "task13b" in steps:
        from experiments.task13b_state_shap import main as t13b
        report.append(_run("task13b", t13b, machine=args.machine))

    if "task13c" in steps:
        from experiments.task13c_transitions import main as t13c
        report.append(_run("task13c", t13c,
                           machines=args.machines or IMDELD_MACHINES))

    if "task13e" in steps:
        from experiments.task13e_ablation import main as t13e
        report.append(_run("task13e", t13e, machine=args.machine,
                           scenario=args.scenario))

    if "figures" in steps:
        from experiments.make_figures_phase6 import main as figs
        report.append(_run("figures", figs, machine=args.machine))

    section("PHASE VI SUMMARY")
    for r in report:
        mark = "OK  " if r["status"] == "ok" else "FAIL"
        line = f"  [{mark}] {r['step']:<8} {r['minutes']:>7.1f} min"
        if r["status"] != "ok":
            line += f"   {r['error']}"
        print(line)
    print(f"\n  total {(time.time()-t_all)/60:.1f} min")
    print(f"  results under {PHASE6_DIR}/")

    if any(r["status"] != "ok" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
