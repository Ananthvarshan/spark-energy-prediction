"""
============================================================
PHASE IV MASTER RUNNER  --  run_phase4.py
============================================================

Reproduces every Phase IV result with one command:

    python run_phase4.py                       # everything
    python run_phase4.py --only task8
    python run_phase4.py --skip task9b         # the long one
    python run_phase4.py --machines pelletizer-I pelletizer-II

STEPS
-----
  label     label all eight IMDELD machines with the one uniform
            procedure in src/labelling.py  (~25 min)
  task8     per-machine state identification, difficulty analysis
            and decision-layer head-room                (~20 min)
  task9a    zero-shot transfer of the decision forecaster,
            8 x 8 x {raw, scaled}                       (~30 min)
  task9b    zero-shot transfer of the Seq2Seq LSTM, 8 trainings
            and a 8 x 8 x {source, target scaler} matrix  (~2 h)
  task10    cross-dataset validation on the SPARK 2024 records,
            with the PV negative control                (~25 min)
  figures   Phase IV figures                            (~1 min)

PREREQUISITES
-------------
Raw CSVs under data/Appliances/ and the two SPARK archives under
data/.  Nothing from Phases I-III is required except for the
optional continuity check against
outputs/imdeld_labelled/pelletizer-I_labelled.csv, which is skipped
if that file is absent.

RUNTIME AND MEMORY
------------------
About three and a half hours end to end on a CPU-only machine, of
which task9b is two.  Every step processes one record at a time and
frees it before the next, because eight 5.5-million-row records do
not fit in memory together; the steps are therefore run
sequentially by design and should not be parallelised without
revisiting that.
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
    IMDELD_MACHINES, SPARK_DATASETS, PHASE4_DIR, MACHINES,
    ensure_dir, info, section,
)

STEPS = ["label", "task8", "task9a", "task9b", "task10", "figures"]


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
    ap = argparse.ArgumentParser(description="Phase IV master runner")
    ap.add_argument("--machines", nargs="*", default=None,
                    help=f"subset of {IMDELD_MACHINES}")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help=f"subset of {list(SPARK_DATASETS)}")
    ap.add_argument("--only", nargs="*", default=None, help=f"steps: {STEPS}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30,
                    help="task9b training epochs")
    args = ap.parse_args()

    steps = [s for s in (args.only or STEPS) if s not in args.skip]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        print(f"Unknown step(s): {unknown}. Choose from {STEPS}")
        sys.exit(1)

    machines = args.machines or IMDELD_MACHINES
    missing = [m for m in machines if not os.path.exists(MACHINES[m]["raw"])]
    if missing and "label" in steps:
        print(f"\nRaw CSV missing for: {missing}\n")
        sys.exit(1)

    ensure_dir(f"{PHASE4_DIR}/logs")
    section(f"PHASE IV  --  machines {machines}  --  steps {steps}")

    report = []
    t_all = time.time()

    if "label" in steps:
        from experiments.task8_multi_machine import label_all
        report.append(_run("label", label_all, machines=machines,
                           force=args.relabel, seed=args.seed))

    if "task8" in steps:
        from experiments.task8_multi_machine import main as t8
        report.append(_run("task8", t8, machines=machines,
                           steps=("characterise", "decision"), seed=args.seed))

    if "task9a" in steps:
        from experiments.task9_cross_machine import main as t9
        report.append(_run("task9a", t9, machines=machines, parts=("a",)))

    if "task9b" in steps:
        from experiments.task9_cross_machine import main as t9
        report.append(_run("task9b", t9, machines=machines, parts=("b",),
                           epochs=args.epochs))

    if "task10" in steps:
        from experiments.task10_cross_dataset import main as t10
        report.append(_run("task10", t10, datasets=args.datasets,
                           relabel=args.relabel, seed=args.seed))

    if "figures" in steps:
        from experiments.make_figures_phase4 import main as figs
        report.append(_run("figures", figs))

    section("PHASE IV SUMMARY")
    for r in report:
        mark = "OK  " if r["status"] == "ok" else "FAIL"
        line = f"  [{mark}] {r['step']:<8} {r['minutes']:>7.1f} min"
        if r["status"] != "ok":
            line += f"   {r['error']}"
        print(line)
    print(f"\n  total {(time.time()-t_all)/60:.1f} min")
    print(f"  results under {PHASE4_DIR}/")

    if any(r["status"] != "ok" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
