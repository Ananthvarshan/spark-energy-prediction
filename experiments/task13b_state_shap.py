"""
============================================================
TASK 13B -- SHAP FOR THE STATE CLASSIFIER
experiments/task13b_state_shap.py
============================================================

WHAT THIS ANSWERS
-----------------
The state layer is a GMM-HMM: it assigns states, but it does not
say which measured quantity carried the decision.  Task 13B fits a
gradient-boosted SURROGATE on the state labels and attributes it,
so that the partition can be described in terms of the physical
channels a reader can check.

The claim being tested is the one the whole framework rests on.
Methods 1 and 2 of the five-method proof assert that an energised
but unloaded induction motor is identified by its POWER FACTOR and
its magnetising-current ratio, not by its power magnitude.  Phase II
established that on the Phase I-III export; this task re-establishes
it on the minimum-dwell labelling Phase IV produced and Phase V
adopted, and reports whether the ranking moved.

WHAT THE SURROGATE'S ACCURACY IS AND IS NOT EVIDENCE OF
-------------------------------------------------------
The surrogate reaches ~99.9% agreement with its targets.  Its
targets ARE the GMM-HMM's own labels, so that number measures only
that a tree can re-express a mixture model's decision boundary --
which was never in doubt.  It is NOT evidence that the states are
correct; nothing in this task validates them, and Phase II's notes
already record this.  What survives the caveat is the RANKING: given
that the state model made these assignments, these are the channels
that carry them.  Every statement in the paper drawn from this task
is of that form.

TWO TARGETS, AND ONLY THE SECOND IS INFORMATIVE
-----------------------------------------------
  multiclass              all four states.  Dominated by the OFF /
                          not-OFF split, which power magnitude alone
                          resolves; reported for completeness.
  standby_vs_productive   STANDBY against {WORKING, PEAK_LOAD} on
                          ENERGISED rows only.  This is the hard
                          split, the one the physics argument is
                          about, and the one the paper quotes.

USAGE
-----
    python -m experiments.task13b_state_shap
    python -m experiments.task13b_state_shap --label-sources phase4
============================================================
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                       # noqa: E402
    DEFAULT_MACHINE, PHASE2_DIR,
    load_labelled, ensure_dir, save_json, info, section,
)
from experiments.task3_feature_importance import (     # noqa: E402
    run_target, N_SAMPLE_MULTICLASS, N_SAMPLE_STANDBY,
)
from src.features import all_feature_names             # noqa: E402

PHASE6_DIR = "outputs/phase6"

# Channels the physics argument names in advance.  Listing them here rather
# than reading them off the result is what makes the check confirmatory: the
# prediction is registered before the ranking is looked at.
PHYSICS_PREDICTED = ["power_factor", "q_p_ratio", "load_factor",
                     "pf_roll_mean"]


def _importance_series(path: str, column: str = "shap_mean_abs") -> pd.Series | None:
    """Read one importance column out of a stored Task 3 / Task 13B table."""
    if not os.path.exists(path):
        return None
    tbl = pd.read_csv(path, index_col=0)
    return tbl[column] if column in tbl.columns else None


def compare_rankings(new: pd.Series, old: pd.Series) -> dict:
    """
    Spearman agreement and top-5 overlap between two feature rankings.

    Used to ask whether re-labelling under the minimum-dwell constraint moved
    the explanation.  A high correlation is the interesting outcome here: it
    would mean the physics reading of the state layer does not depend on which
    of the two labellings the paper quotes, which is what Phase V's correction
    4 needs in order to leave the explainability claim standing.
    """
    from scipy.stats import spearmanr

    common = [f for f in new.index if f in old.index]
    rho, p = spearmanr(new[common].to_numpy(), old[common].to_numpy())
    top_new = list(new[common].sort_values(ascending=False).head(5).index)
    top_old = list(old[common].sort_values(ascending=False).head(5).index)
    return {
        "spearman": float(rho), "p_value": float(p), "n_common": len(common),
        "top5_new": top_new, "top5_previous": top_old,
        "top5_overlap": len(set(top_new) & set(top_old)),
    }


def physics_verdict(tbl_path: str) -> dict:
    """
    Score the pre-registered prediction: do the motor-physics channels take
    the top of the STANDBY-vs-productive ranking?

    Reported as the rank each named channel actually reached and the share of
    total attribution they carry between them, so a partial result reads as a
    partial result rather than as a pass or a fail.
    """
    tbl = pd.read_csv(tbl_path, index_col=0)
    share = tbl["shap_mean_abs"] / max(tbl["shap_mean_abs"].sum(), 1e-12)
    ranks = share.rank(ascending=False, method="min").astype(int)
    present = [f for f in PHYSICS_PREDICTED if f in share.index]
    return {
        "predicted_channels": PHYSICS_PREDICTED,
        "rank_of_each": {f: int(ranks[f]) for f in present},
        "share_of_each": {f: float(share[f]) for f in present},
        "combined_share": float(share[present].sum()),
        "best_rank": int(min(ranks[f] for f in present)) if present else None,
        "n_in_top5": int(sum(ranks[f] <= 5 for f in present)),
    }


def run_source(machine: str, source: str, out_root: str,
               roll_window_s: float = 300.0, seed: int = 42) -> dict:
    """Fit and attribute the surrogate on one labelling of one machine."""
    section(f"TASK 13B -- state surrogate, labels '{source}'")
    out_dir = ensure_dir(f"{out_root}/{source}")

    df, meta = load_labelled(machine, drop_spikes=True, add_features=True,
                             roll_window_s=roll_window_s, source=source)
    feature_names = [f for f in all_feature_names() if f in df.columns]
    info(f"{len(feature_names)} engineered features, {len(df):,} clean rows")

    out: dict = {"label_source": meta["label_source"],
                 "labelled_path": meta["labelled_path"],
                 "n_rows": int(len(df)), "targets": {}}

    out["targets"]["multiclass"] = run_target(
        df, feature_names, "multiclass", "state",
        N_SAMPLE_MULTICLASS, out_dir, random_state=seed)

    energised = df[df["state"] != "OFF"].copy()
    energised["standby_vs_productive"] = np.where(
        energised["state"] == "STANDBY", "STANDBY", "PRODUCTIVE")
    info(f"energised subset: {len(energised):,} rows (OFF excluded)")
    out["targets"]["standby_vs_productive"] = run_target(
        energised, feature_names, "standby_vs_productive",
        "standby_vs_productive", N_SAMPLE_STANDBY, out_dir, random_state=seed)

    out["physics_prediction"] = physics_verdict(
        f"{out_dir}/importance_standby_vs_productive.csv")
    info(f"physics channels take ranks "
         f"{out['physics_prediction']['rank_of_each']} and carry "
         f"{out['physics_prediction']['combined_share']*100:.1f}% of the "
         f"attribution")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machine: str = DEFAULT_MACHINE,
    label_sources: list[str] | None = None,
    roll_window_s: float = 300.0,
    seed: int = 42,
) -> dict:
    section("PHASE VI / TASK 13B -- SHAP FOR THE STATE CLASSIFIER")
    out_root = ensure_dir(f"{PHASE6_DIR}/task13b/{machine}")
    label_sources = label_sources or ["phase4"]

    results: dict = {"machine": machine, "seed": seed,
                     "caveat": ("the surrogate's targets are the GMM-HMM's own "
                                "labels; its accuracy measures re-expressibility, "
                                "not correctness. Only the ranking is a result."),
                     "sources": {}}

    for src in label_sources:
        results["sources"][src] = run_source(machine, src, out_root,
                                             roll_window_s, seed)

    # Did the minimum-dwell re-labelling move the explanation?  Compared
    # against Phase II's stored Task 3 table, which was computed on the
    # Phase I-III export under the identical estimator.
    phase2_path = (f"{PHASE2_DIR}/task3/{machine}/"
                   f"importance_standby_vs_productive.csv")
    old = _importance_series(phase2_path)
    for src in results["sources"]:
        new = _importance_series(
            f"{out_root}/{src}/importance_standby_vs_productive.csv")
        if old is not None and new is not None:
            cmp = compare_rankings(new, old)
            results["sources"][src]["vs_phase2_task3"] = cmp
            info(f"[{src}] agreement with Phase II Task 3: "
                 f"Spearman {cmp['spearman']:.3f}, "
                 f"top-5 overlap {cmp['top5_overlap']}/5")
        else:
            results["sources"][src]["vs_phase2_task3"] = None

    save_json(results, f"{out_root}/task13b_results.json")
    section("TASK 13B COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase VI Task 13B -- state SHAP")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--label-sources", nargs="*", default=None,
                    choices=["auto", "reference", "phase4"])
    ap.add_argument("--roll-window-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.machine, a.label_sources, a.roll_window_s, a.seed)
