"""
============================================================
TASK 13C -- HMM TRANSITION STRUCTURE
experiments/task13c_transitions.py
============================================================

WHAT THIS ANSWERS
-----------------
The paper claims the temporal model learns a physically plausible
state graph: states persist, a machine reaches load through an
intermediate state, and it does not jump from de-energised to peak
load in one second.  Until now that claim has been made in prose.
Task 13C extracts the matrices, puts the four states in a canonical
order, and reports what they actually contain -- including where the
claim does not hold.

THREE MATRICES PER MACHINE, AND THEY ARE NOT THE SAME OBJECT
------------------------------------------------------------
  learned    the HMM's estimated transition matrix (`transmat_`).
             A model parameter.  What Viterbi decoding is run
             against.
  realised   transition frequencies COUNTED from the decoded state
             sequence before the minimum-dwell constraint
             (`state_raw`).  What the model actually produced.
  final      the same count after the minimum-dwell constraint
             (`state`).  What every downstream stage consumes.

Reporting only the first would describe a prior; reporting only the
last would describe a post-processed artefact.  The interesting
result is the difference between them, and specifically what the
dwell constraint removes.

TRANSITIONS ARE NEVER COUNTED ACROSS AN ACQUISITION GAP
-------------------------------------------------------
The records contain 66-1082 contiguous fragments.  A pair of
readings either side of a three-day gap is not a transition, and
counting it would manufacture exactly the implausible jumps this
task exists to look for.

EXPECTED DWELL
--------------
For a self-transition probability p and a sample interval dt, the
geometric dwell has mean dt / (1 - p).  Reporting it converts the
fourth-decimal differences between the diagonal entries -- which are
unreadable -- into seconds, which are checkable against the
machine's duty cycle.

USAGE
-----
    python -m experiments.task13c_transitions
    python -m experiments.task13c_transitions --machines pelletizer-I
============================================================
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                       # noqa: E402
    IMDELD_MACHINES, MACHINES, PHASE4_DIR, STATE_ORDER,
    ensure_dir, save_json, info, section,
)

PHASE6_DIR = "outputs/phase6"

# Transitions the state definition says a driven machine cannot make in one
# sample: it cannot go from de-energised to a loaded state, or back, without
# passing through the energised-idle band.  Stated here so the check is a
# prediction rather than an observation dressed up as one.
FORBIDDEN = [("OFF", "WORKING"), ("OFF", "PEAK_LOAD"),
             ("WORKING", "OFF"), ("PEAK_LOAD", "OFF")]


def load_label_info(machine: str) -> dict:
    """The labeller's stored parameters for one machine (Phase IV, Task 8)."""
    path = f"{PHASE4_DIR}/task8/{machine}/label_info.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No label_info for {machine}: {path}\n"
            f"Run:  python run_phase4.py --only task8")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def canonical_learned(info_d: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Reindex the learned transition matrix from cluster ids to state names,
    in ascending-power order.

    The HMM's component indices are arbitrary -- they come out of the GMM's
    initialisation -- so a matrix printed in component order is not
    comparable between machines and not readable at all.  `cluster_to_state`
    is the mapping the labeller already committed to when it named the states.
    """
    trans = np.asarray(info_d["transmat"], dtype=float)
    c2s = {int(k): v for k, v in info_d["cluster_to_state"].items()}
    present = [s for s in STATE_ORDER if s in set(c2s.values())]
    order = [next(c for c, s in c2s.items() if s == name) for name in present]
    return pd.DataFrame(trans[np.ix_(order, order)],
                        index=present, columns=present), present


def count_transitions(states: np.ndarray, segments: np.ndarray,
                      names: list[str]) -> pd.DataFrame:
    """
    Row-stochastic transition frequencies counted from a decoded sequence,
    skipping every pair that straddles a segment boundary.
    """
    code = {s: i for i, s in enumerate(names)}
    idx = np.array([code.get(s, -1) for s in states])
    a, b = idx[:-1], idx[1:]
    same_segment = segments[:-1] == segments[1:]
    ok = same_segment & (a >= 0) & (b >= 0)

    counts = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(counts, (a[ok], b[ok]), 1)
    row = counts.sum(axis=1, keepdims=True)
    freq = np.divide(counts, np.maximum(row, 1), dtype=float)
    return pd.DataFrame(freq, index=names, columns=names), \
        pd.DataFrame(counts, index=names, columns=names)


def expected_dwell(matrix: pd.DataFrame, dt: float) -> pd.Series:
    """Geometric mean dwell in seconds implied by each self-transition."""
    p = np.clip(np.diag(matrix.to_numpy()), 0.0, 1 - 1e-12)
    return pd.Series(dt / (1.0 - p), index=matrix.index, name="expected_dwell_s")


def summarise(matrix: pd.DataFrame, counts: pd.DataFrame | None,
              dt: float) -> dict:
    """
    The four things the paper says about a transition matrix, as numbers.

    `forbidden` reports the transitions the state definition rules out, with
    their observed counts where they are available.  A frequency of 1e-4 on a
    5.5-million-row record is 550 events, not a rounding artefact, so the
    count is what makes the entry interpretable.
    """
    names = list(matrix.index)
    diag = pd.Series(np.diag(matrix.to_numpy()), index=names)
    forbidden = {}
    for a, b in FORBIDDEN:
        if a in names and b in names:
            entry = {"probability": float(matrix.loc[a, b])}
            if counts is not None:
                entry["count"] = int(counts.loc[a, b])
            forbidden[f"{a}->{b}"] = entry
    return {
        "states": names,
        "self_transition": diag.to_dict(),
        "min_self_transition": float(diag.min()),
        "expected_dwell_s": expected_dwell(matrix, dt).to_dict(),
        "forbidden": forbidden,
        "max_forbidden_probability": (max(v["probability"] for v in forbidden.values())
                                      if forbidden else float("nan")),
        "matrix": matrix.round(8).to_dict(orient="index"),
    }


def run_machine(machine: str, out_dir: str) -> dict:
    """Extract all three matrices for one machine."""
    section(f"TASK 13C -- {MACHINES[machine]['name']}")
    info_d = load_label_info(machine)
    dt = float(info_d["sample_interval_s"])

    learned, names = canonical_learned(info_d)
    info(f"k = {info_d['k']}, states {names}, dt = {dt:.2f}s, "
         f"mean self-transition {np.mean(np.diag(learned.to_numpy())):.6f}")

    out = {
        "machine": machine,
        "machine_name": MACHINES[machine]["name"],
        "sample_interval_s": dt,
        "min_dwell_s": info_d.get("min_dwell_s"),
        "hmm_converged": info_d.get("hmm_converged"),
        "hmm_iterations": info_d.get("hmm_iterations"),
        "rows_relabelled_by_min_dwell_pct":
            info_d.get("rows_relabelled_by_min_dwell_pct"),
        "learned": summarise(learned, None, dt),
    }
    learned.to_csv(f"{out_dir}/transmat_learned_{machine}.csv")

    # Realised counts, from the labelled record.  Only three columns are read,
    # which is what keeps this affordable on a 5.5-million-row parquet.
    path = MACHINES[machine]["labelled_phase4"]
    if os.path.exists(path):
        df = pd.read_parquet(path, columns=["state", "state_raw", "segment_id"])
        seg = df["segment_id"].to_numpy()
        for key, col in (("realised_pre_min_dwell", "state_raw"),
                         ("realised_final", "state")):
            freq, counts = count_transitions(
                df[col].astype(str).to_numpy(), seg, names)
            out[key] = summarise(freq, counts, dt)
            freq.to_csv(f"{out_dir}/transmat_{key}_{machine}.csv")
        info(f"realised over {len(df):,} rows in "
             f"{int(pd.Series(seg).nunique())} contiguous segments")
        del df
    else:
        info(f"no labelled parquet at {path}; realised matrices skipped")

    if "realised_final" in out:
        f_pre = out["realised_pre_min_dwell"]["max_forbidden_probability"]
        f_fin = out["realised_final"]["max_forbidden_probability"]
        info(f"largest forbidden transition: learned "
             f"{out['learned']['max_forbidden_probability']:.2e}, realised "
             f"{f_pre:.2e} -> {f_fin:.2e} after the dwell constraint")
    return out


def cross_machine_table(results: dict) -> pd.DataFrame:
    """One row per machine: what the transition structure looks like plant-wide."""
    rows = []
    for machine, r in results.items():
        fin = r.get("realised_final") or r["learned"]
        dwell = fin["expected_dwell_s"]
        rows.append({
            "machine": machine,
            "min_self_transition_learned": r["learned"]["min_self_transition"],
            "min_self_transition_realised": fin["min_self_transition"],
            "expected_dwell_STANDBY_s": dwell.get("STANDBY", float("nan")),
            "expected_dwell_OFF_s": dwell.get("OFF", float("nan")),
            "max_forbidden_learned": r["learned"]["max_forbidden_probability"],
            "max_forbidden_realised": fin["max_forbidden_probability"],
            "rows_relabelled_pct": r.get("rows_relabelled_by_min_dwell_pct"),
        })
    return pd.DataFrame(rows).set_index("machine")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machines: list[str] | None = None) -> dict:
    section("PHASE VI / TASK 13C -- HMM TRANSITION STRUCTURE")
    out_dir = ensure_dir(f"{PHASE6_DIR}/task13c")
    machines = machines or IMDELD_MACHINES

    results = {}
    for m in machines:
        try:
            results[m] = run_machine(m, out_dir)
        except FileNotFoundError as exc:
            info(f"{m}: skipped -- {exc}")

    tbl = cross_machine_table(results)
    tbl.to_csv(f"{out_dir}/transition_summary.csv")
    print("\n  Transition structure across the plant:")
    print(tbl.round(6).to_string())

    save_json(results, f"{out_dir}/task13c_results.json")
    section("TASK 13C COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase VI Task 13C -- transitions")
    ap.add_argument("--machines", nargs="*", default=None)
    a = ap.parse_args()
    main(a.machines)
