"""
============================================================
TASK 3 -- FEATURE ENGINEERING + IMPORTANCE ANALYSIS
experiments/task3_feature_importance.py
============================================================

PURPOSE
-------
Phase II / Task 3.  Build the engineered feature set (src/features.py)
and quantify which features actually carry the state signal, using
three independent importance measures:

  1. Mutual information       model-free, captures non-monotone
                              dependence (Kraskov et al., 2004 estimator
                              as implemented in scikit-learn)
  2. Gradient-boosted gain    XGBoost total gain per split
  3. SHAP values              Shapley attribution on the same XGBoost
                              surrogate (Lundberg & Lee, 2017)

Agreement across all three is what makes the conclusion defensible;
any single measure has known failure modes (MI ignores redundancy,
gain is biased toward high-cardinality features, SHAP inherits the
surrogate's own biases).

TWO TARGETS
-----------
  multiclass : all states, i.e. "what drives state assignment overall"
  standby    : STANDBY vs {WORKING, PEAK_LOAD}, restricted to ENERGISED
               rows only (OFF excluded)

The second target is the one that matters.  Separating OFF from
everything else is trivial -- power magnitude alone does it.  The hard
and paper-relevant question is whether STANDBY can be separated from a
genuinely loaded state, which is exactly the limitation recorded in
Section 11.1 of the project's research report.  If power factor and
Q/P ratio dominate this target, the physics argument behind Methods 1
and 2 of the five-method proof is empirically confirmed rather than
merely asserted.

NOTE ON WHAT "IMPORTANCE" MEANS HERE
------------------------------------
The targets are the pipeline's own GMM-HMM labels, not ground truth.
These results therefore say which features EXPLAIN the proposed
method's decisions -- they are an explainability result, not a
validation one.  Validation is Task 4's job.

USAGE
-----
    python -m experiments.task3_feature_importance
    python -m experiments.task3_feature_importance --machine pelletizer-I
============================================================
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (            # noqa: E402
    DEFAULT_MACHINE, PHASE2_DIR, PRODUCTIVE_STATES,
    load_labelled, ensure_dir, save_json, info, section,
)
from src.features import all_feature_names, group_of   # noqa: E402


# ── Configuration ─────────────────────────────────────────────────────────────

# Rows used for the importance analysis.  The full pelletizer-I record is
# 4.4M clean rows; every estimator here is O(n log n) or worse and none of
# them needs that many samples to rank 24 features stably.  Sampling is
# stratified by state so the 5.7%-prevalence STANDBY class is not swamped.
N_SAMPLE_MULTICLASS = 300_000
N_SAMPLE_STANDBY = 200_000

# Mutual information is by far the most expensive estimator (k-nearest-
# neighbour based), so it gets its own smaller subsample.
N_SAMPLE_MI = 40_000
MI_NEIGHBORS = 3

# SHAP is evaluated on a subsample of the test split; TreeExplainer is exact
# for trees but still O(n x trees x depth).
N_SHAP = 20_000

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    tree_method="hist",
)

TEST_FRACTION = 0.25     # chronological holdout


# ── Helpers ───────────────────────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n_total: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample up to `n_total` rows with equal allocation per class, falling
    back to whatever a rare class can supply.

    Equal allocation (rather than proportional) is deliberate: with
    STANDBY at 5.7% prevalence, a proportional sample would give the
    importance estimators almost no STANDBY rows to learn from, and every
    measure would then simply rank the features that separate OFF.
    """
    rng = np.random.default_rng(random_state)
    classes = df[label_col].unique()
    per_class = max(1, n_total // len(classes))

    picks = []
    for c in classes:
        pos = np.flatnonzero((df[label_col] == c).to_numpy())
        take = min(per_class, len(pos))
        picks.append(rng.choice(pos, size=take, replace=False))
    idx = np.sort(np.concatenate(picks))
    out = df.iloc[idx].copy()
    info(f"stratified sample: {len(out):,} rows "
         f"({dict(out[label_col].value_counts())})")
    return out


def chronological_holdout(
    sub: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Positional train/test masks split by TIME, not at random.

    A random split would be badly optimistic here: the rolling-statistic
    features share a 300-second window between neighbouring rows, so a
    randomly held-out row is almost always adjacent to a training row that
    contains most of its information.
    """
    order = np.argsort(sub["timestamp"].to_numpy())
    cut = int(len(order) * (1 - test_fraction))
    train_pos = np.sort(order[:cut])
    test_pos = np.sort(order[cut:])
    return train_pos, test_pos


def mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_sample: int = N_SAMPLE_MI,
    random_state: int = 42,
) -> pd.Series:
    """Mutual information between each feature and the label."""
    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.default_rng(random_state)
    if len(X) > n_sample:
        sel = rng.choice(len(X), size=n_sample, replace=False)
        X, y = X[sel], y[sel]

    t0 = time.time()
    mi = mutual_info_classif(
        X, y, discrete_features=False,
        n_neighbors=MI_NEIGHBORS, random_state=random_state,
    )
    info(f"mutual information on {len(X):,} rows took {time.time()-t0:.1f}s")
    return pd.Series(mi, index=feature_names, name="mutual_information")


def fit_surrogate(
    X_tr, y_tr, X_te, y_te, feature_names, n_classes, label_names,
):
    """Fit the XGBoost surrogate and report its holdout performance."""
    import xgboost as xgb
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    params = dict(XGB_PARAMS)
    if n_classes > 2:
        params.update(objective="multi:softprob", num_class=n_classes)
    else:
        params.update(objective="binary:logistic")

    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=False)
    pred = model.predict(X_te)

    perf = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "fit_seconds": round(time.time() - t0, 1),
        "report": classification_report(
            y_te, pred, target_names=label_names, zero_division=0, digits=3
        ),
    }
    info(f"surrogate holdout accuracy {perf['accuracy']*100:.2f}%  "
         f"macro-F1 {perf['macro_f1']:.3f}  ({perf['fit_seconds']}s)")
    print(perf["report"])
    return model, perf


def shap_importance(model, X, feature_names, n_sample=N_SHAP, random_state=42):
    """
    Mean |SHAP| per feature.

    For multiclass models shap returns one attribution per class; we take
    the mean absolute value across classes, which is the standard global
    summary and is what shap.summary_plot displays.
    """
    import shap

    rng = np.random.default_rng(random_state)
    if len(X) > n_sample:
        X = X[rng.choice(len(X), size=n_sample, replace=False)]

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    arr = np.asarray(values)
    if arr.ndim == 3:                       # (n, features, classes) or (classes, n, features)
        axis = 2 if arr.shape[2] <= arr.shape[0] else 0
        arr = np.abs(arr).mean(axis=axis) if axis == 2 else np.abs(arr).mean(axis=0)
    mean_abs = np.abs(arr).mean(axis=0)

    info(f"SHAP on {len(X):,} rows took {time.time()-t0:.1f}s")
    return pd.Series(mean_abs, index=feature_names, name="shap_mean_abs"), values, X


def rank_table(
    mi: pd.Series,
    gain: pd.Series,
    shap_s: pd.Series,
) -> pd.DataFrame:
    """
    Combine the three measures into one ranked table.

    Each measure is normalised to sum to 1 so they are comparable, then
    ranked; `mean_rank` is the consensus ordering used for reporting.
    """
    def _norm(s):
        tot = s.sum()
        return s / tot if tot > 0 else s

    tbl = pd.DataFrame({
        "mutual_information": _norm(mi),
        "xgb_gain": _norm(gain),
        "shap_mean_abs": _norm(shap_s),
    })
    for c in tbl.columns:
        tbl[f"rank_{c}"] = tbl[c].rank(ascending=False, method="min").astype(int)
    tbl["mean_rank"] = tbl[[c for c in tbl.columns if c.startswith("rank_")]].mean(axis=1)
    tbl["group"] = [group_of(f) for f in tbl.index]
    return tbl.sort_values("mean_rank")


def run_target(
    df: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    label_col: str,
    n_sample: int,
    out_dir: str,
    random_state: int = 42,
) -> dict:
    """Run all three importance measures for one target definition."""
    section(f"TASK 3 -- importance for target '{target_name}'")

    sub = stratified_sample(df, label_col, n_sample, random_state)
    label_names = sorted(sub[label_col].unique().tolist())
    code = {n: i for i, n in enumerate(label_names)}
    y = sub[label_col].map(code).to_numpy(int)
    X = sub[feature_names].to_numpy(np.float32)

    train_pos, test_pos = chronological_holdout(sub)
    info(f"chronological holdout: {len(train_pos):,} train / {len(test_pos):,} test")

    mi = mutual_information(X[train_pos], y[train_pos], feature_names,
                            random_state=random_state)

    model, perf = fit_surrogate(
        X[train_pos], y[train_pos], X[test_pos], y[test_pos],
        feature_names, len(label_names), label_names,
    )
    gain = pd.Series(model.feature_importances_, index=feature_names, name="xgb_gain")

    shap_s, shap_raw, shap_X = shap_importance(
        model, X[test_pos], feature_names, random_state=random_state
    )

    tbl = rank_table(mi, gain, shap_s)

    grp = (tbl.groupby("group")[["mutual_information", "xgb_gain", "shap_mean_abs"]]
              .sum().sort_values("shap_mean_abs", ascending=False))

    ensure_dir(out_dir)
    tbl.to_csv(f"{out_dir}/importance_{target_name}.csv")
    grp.to_csv(f"{out_dir}/importance_by_group_{target_name}.csv")
    np.savez_compressed(
        f"{out_dir}/shap_raw_{target_name}.npz",
        shap_values=np.asarray(shap_raw, dtype=np.float32),
        X=shap_X.astype(np.float32),
        feature_names=np.array(feature_names),
        class_names=np.array(label_names),
    )
    info(f"wrote {out_dir}/importance_{target_name}.csv")

    print(f"\n  Top 10 features for '{target_name}' (consensus rank):")
    print(tbl.head(10)[["group", "mutual_information", "xgb_gain",
                        "shap_mean_abs", "mean_rank"]].round(4).to_string())
    print(f"\n  Importance summed by feature group:")
    print(grp.round(4).to_string())

    return {
        "target": target_name,
        "classes": label_names,
        "n_rows": int(len(sub)),
        "surrogate": {k: v for k, v in perf.items() if k != "report"},
        "surrogate_report": perf["report"],
        "top10_consensus": tbl.head(10).index.tolist(),
        "by_group": grp.to_dict(orient="index"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = DEFAULT_MACHINE, roll_window_s: float = 300.0):
    section("PHASE II / TASK 3 -- FEATURE ENGINEERING + IMPORTANCE")
    out_dir = ensure_dir(f"{PHASE2_DIR}/task3/{machine}")

    df, meta = load_labelled(machine, drop_spikes=True, add_features=True,
                             roll_window_s=roll_window_s)

    feature_names = [f for f in all_feature_names() if f in df.columns]
    info(f"{len(feature_names)} engineered features available")

    results = {"meta": meta, "features": feature_names, "targets": {}}

    # -- Target 1: all states ---------------------------------------------
    results["targets"]["multiclass"] = run_target(
        df, feature_names, "multiclass", "state",
        N_SAMPLE_MULTICLASS, out_dir,
    )

    # -- Target 2: STANDBY vs productive, energised rows only -------------
    energised = df[df["state"] != "OFF"].copy()
    energised["standby_vs_productive"] = np.where(
        energised["state"] == "STANDBY", "STANDBY", "PRODUCTIVE"
    )
    info(f"energised subset: {len(energised):,} rows "
         f"(OFF excluded -- see module docstring)")
    results["targets"]["standby_vs_productive"] = run_target(
        energised, feature_names, "standby_vs_productive",
        "standby_vs_productive", N_SAMPLE_STANDBY, out_dir,
    )

    # -- Descriptive statistics per state, for the paper's feature table --
    desc = (df.groupby("state")[feature_names]
              .agg(["mean", "std"])
              .round(4))
    desc.to_csv(f"{out_dir}/feature_stats_by_state.csv")
    info(f"wrote {out_dir}/feature_stats_by_state.csv")

    results["config"] = {
        "roll_window_s": roll_window_s,
        "n_sample_multiclass": N_SAMPLE_MULTICLASS,
        "n_sample_standby": N_SAMPLE_STANDBY,
        "n_sample_mi": N_SAMPLE_MI,
        "n_shap": N_SHAP,
        "xgb_params": XGB_PARAMS,
        "test_fraction": TEST_FRACTION,
    }
    save_json(results, f"{out_dir}/task3_results.json")
    section("TASK 3 COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase II Task 3 -- feature importance")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--roll-window-s", type=float, default=300.0)
    a = ap.parse_args()
    main(a.machine, a.roll_window_s)
