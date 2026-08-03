"""
============================================================
TASK 13A -- SHAP FOR THE XGBOOST FORECASTERS
experiments/task13a_forecaster_shap.py
============================================================

WHAT THIS ANSWERS
-----------------
Phase V demoted the Seq2Seq LSTM and promoted a gradient-boosted
tree to the paper's forecasting component (Phase V, consequent
correction 2).  Phase IV separately reported that the DECISION
layer's forecaster draws 94.5% of its split gain from four calendar
features.  Both statements are global gain scores, which are known
to be biased toward high-cardinality splits and say nothing about
individual predictions.  Task 13A replaces them with Shapley
attributions (Lundberg & Lee, 2017), which are per-prediction,
additive, and consistent.

THERE ARE TWO XGBOOST FORECASTERS IN THIS FRAMEWORK, NOT ONE
------------------------------------------------------------
They answer different questions on different supports and must be
explained separately; conflating them is the main way this result
could be got wrong.

  A1  decision-layer duration forecaster  (Task 6)
      "given the machine has been idle for `elapsed` seconds, how
      much longer will it stay idle?"  Ten features, one row per
      decision epoch, consumed by the optimisation layer.  This is
      the model whose 94.5% calendar gain Phase IV reported.

      Its chance-constraint companion -- the classifier for
      P(remaining >= min_off) -- is attributed alongside it, because
      the constraint is an ablation row in Task 13E and its
      contribution cannot be discussed without knowing what it keys
      on.

  A2  window state forecaster              (Task 5 / Task 12A)
      "given 600 s of history, what state is the machine in at each
      of the next 300 s?"  217 design columns summarising 24
      engineered channels, consumed by the STANDBY-seconds metric.
      This is the model Phase V ranked first of seven.

WHAT AN ATTRIBUTION HERE IS AND IS NOT
--------------------------------------
Both targets are derived from the pipeline's own GMM-HMM labels, so
these results explain what the FORECASTER learned, not whether the
underlying state assignment is correct.  That distinction is carried
through the whole of Phase VI and is stated again in the notes.

USAGE
-----
    python -m experiments.task13a_forecaster_shap
    python -m experiments.task13a_forecaster_shap --only decision
    python -m experiments.task13a_forecaster_shap --label-sources phase4
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

from experiments.common import (                       # noqa: E402
    DEFAULT_MACHINE, ensure_dir, save_json, info, section,
)

PHASE6_DIR = "outputs/phase6"

# Rows drawn for the Shapley computation.  TreeExplainer is exact for trees,
# so this controls the sampling error of the GLOBAL summary only; 20k rows put
# the standard error on a mean |SHAP| well below the gap between adjacent
# features in every ranking reported here.
N_SHAP_DECISION = 30_000
N_SHAP_WINDOW = 15_000

# Tolerance for the Shapley additivity identity, expressed RELATIVE to the
# scale of the model output.  See `tree_shap` for why an absolute tolerance is
# the wrong instrument on a model whose outputs are durations in seconds.
ADDITIVITY_RTOL = 1e-3

# Feature groups for the decision forecaster.  The split is the one Phase IV's
# claim is about: what the plant's clock says, against what this particular
# machine did.
DECISION_GROUPS = {
    "elapsed_log": "elapsed idle time",
    "onset_hour": "factory calendar",
    "onset_dow": "factory calendar",
    "is_weekend": "factory calendar",
    "cur_hour_sin": "factory calendar",
    "cur_hour_cos": "factory calendar",
    "cur_is_open": "factory calendar",
    "cur_is_weekend": "factory calendar",
    "prev_productive_log": "machine history",
    "prev_productive_power_w": "machine history",
}


# ── Shared Shapley machinery ──────────────────────────────────────────────────

def tree_shap(model, X: np.ndarray, feature_names: list[str],
              n_sample: int, seed: int = 0):
    """
    Exact tree Shapley values on a subsample, returned with the rows they
    were computed on.

    Returns (mean_abs, shap_values, X_used, diagnostics).  `shap_values`
    keeps its native shape -- (n, f) for a regressor or binary classifier,
    (n, f, c) for a multiclass one -- because the beeswarm figure needs the
    signed per-row values and only the summary is collapsed over classes.

    ADDITIVITY IS CHECKED HERE RATHER THAN BY THE LIBRARY, AND THE REASON
    MATTERS.  `shap`'s built-in assertion compares the reconstruction against
    the model output at an ABSOLUTE tolerance of 1e-2.  The decision
    forecaster predicts remaining idle time in SECONDS, so its outputs run to
    1e4 and the tree traversal accumulates in float32: a relative error of a
    few parts in 10^4 -- numerically irrelevant, and far below the resolution
    of any claim made from these attributions -- trips an absolute threshold
    written for models whose outputs are probabilities.  The check is
    therefore performed on the RELATIVE error and the measured value is
    recorded in the results JSON, so the tolerance that was actually met is
    auditable rather than silently disabled.
    """
    import shap

    rng = np.random.default_rng(seed)
    if len(X) > n_sample:
        X = X[rng.choice(len(X), size=n_sample, replace=False)]

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X, check_additivity=False))
    info(f"SHAP on {len(X):,} x {X.shape[1]} took {time.time() - t0:.1f}s "
         f"-> array{values.shape}")

    diag = _check_additivity(model, X, values, explainer.expected_value)
    info(f"additivity: max relative error {diag['max_relative_error']:.2e} "
         f"(max absolute {diag['max_absolute_error']:.3g} on outputs of "
         f"scale {diag['output_scale']:.3g})")
    if diag["max_relative_error"] > ADDITIVITY_RTOL:
        raise RuntimeError(
            f"SHAP additivity violated: relative error "
            f"{diag['max_relative_error']:.2e} > {ADDITIVITY_RTOL:.0e}")

    collapsed = np.abs(values)
    while collapsed.ndim > 2:                 # average over the class axis
        collapsed = collapsed.mean(axis=-1)
    mean_abs = pd.Series(collapsed.mean(axis=0), index=feature_names,
                         name="shap_mean_abs")
    return mean_abs, values, X, diag


def _check_additivity(model, X, values, expected_value) -> dict:
    """
    Shapley values must satisfy  f(x) = E[f] + sum_j phi_j(x).

    Compared against the model's RAW MARGIN, which is the quantity tree SHAP
    decomposes -- for a classifier that is the log-odds / softmax input, not
    the probability, and comparing against the probability would report a
    violation where there is none.
    """
    margin = model.predict(X, output_margin=True)
    recon = np.asarray(expected_value) + values.sum(axis=1)
    margin = np.asarray(margin, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if recon.shape != margin.shape:
        recon = recon.reshape(margin.shape)

    err = np.abs(recon - margin)
    scale = max(float(np.abs(margin).max()), 1e-12)
    return {
        "max_absolute_error": float(err.max()),
        "max_relative_error": float(err.max() / scale),
        "mean_absolute_error": float(err.mean()),
        "output_scale": scale,
        "n_rows_checked": int(len(X)),
    }


def attribution_table(mean_abs: pd.Series, gain: pd.Series) -> pd.DataFrame:
    """
    Put SHAP beside the gain score it is meant to replace.

    Both are normalised to a share of their own total, so `shap_share` and
    `gain_share` are directly comparable and their disagreement is readable
    as a number rather than as two rankings the reader has to align by eye.
    """
    tbl = pd.DataFrame({"shap_mean_abs": mean_abs, "xgb_gain": gain})
    tbl["shap_share"] = tbl["shap_mean_abs"] / max(tbl["shap_mean_abs"].sum(), 1e-12)
    tbl["gain_share"] = tbl["xgb_gain"] / max(tbl["xgb_gain"].sum(), 1e-12)
    tbl["rank_shap"] = tbl["shap_share"].rank(ascending=False, method="min").astype(int)
    tbl["rank_gain"] = tbl["gain_share"].rank(ascending=False, method="min").astype(int)
    tbl["rank_shift"] = tbl["rank_gain"] - tbl["rank_shap"]
    return tbl.sort_values("shap_share", ascending=False)


def rank_agreement(a: pd.Series, b: pd.Series) -> dict:
    """Spearman and top-5 overlap between two importance orderings."""
    from scipy.stats import spearmanr

    common = [f for f in a.index if f in b.index]
    if len(common) < 3:
        return {"spearman": float("nan"), "n_common": len(common)}
    rho, p = spearmanr(a[common].to_numpy(), b[common].to_numpy())
    top_a = set(a[common].sort_values(ascending=False).head(5).index)
    top_b = set(b[common].sort_values(ascending=False).head(5).index)
    return {"spearman": float(rho), "p_value": float(p),
            "n_common": len(common), "top5_overlap": len(top_a & top_b)}


# ── A1: the decision-layer duration forecaster ────────────────────────────────

def run_decision_forecaster(
    machine: str,
    source: str,
    out_dir: str,
    seed: int = 0,
    n_shap: int = N_SHAP_DECISION,
) -> dict:
    """
    Attribute the remaining-idle-duration regressor and its feasibility
    classifier, on one labelling.

    BOTH LABELLINGS ARE RUN BY DEFAULT, AND THAT IS NOT REDUNDANCY.
    Phase IV's 94.5% figure was measured on the Phase I-III export; Phase V
    then established that the minimum-dwell labelling is the one the paper
    should quote, and it merges 3,224 episodes into 1,312.  Whether the
    calendar dominance survives that re-partitioning is exactly the question,
    and it cannot be answered by running only the labelling that produced the
    original claim.
    """
    from experiments.task6_optimization import prepare, FEATURE_NAMES

    section(f"TASK 13A-1 -- decision forecaster, labels '{source}'")
    prep = prepare(machine, seed=seed, source=source)

    model = prep["forecaster_models"]["mean"]
    Xte, yte = prep["Xte"], prep["yte"]
    info(f"{len(prep['ep_train']):,} training episodes -> "
         f"{len(prep['Xtr']):,} decision epochs; "
         f"{len(prep['ep_test']):,} test episodes -> {len(Xte):,} epochs")

    gain = pd.Series(model.feature_importances_, index=FEATURE_NAMES,
                     name="xgb_gain")
    mean_abs, values, X_used, diag = tree_shap(model, Xte, FEATURE_NAMES,
                                               n_shap, seed)
    tbl = attribution_table(mean_abs, gain)
    tbl["group"] = [DECISION_GROUPS.get(f, "unknown") for f in tbl.index]

    grp = tbl.groupby("group")[["shap_share", "gain_share"]].sum()
    grp = grp.sort_values("shap_share", ascending=False)

    # The feasibility classifier -- the chance constraint's left-hand side.
    feas = prep["feasibility_model"]
    feas_out = None
    if feas is not None:
        f_gain = pd.Series(feas.feature_importances_, index=FEATURE_NAMES,
                           name="xgb_gain")
        f_mean, f_values, f_X, f_diag = tree_shap(feas, Xte, FEATURE_NAMES,
                                                  n_shap, seed + 1)
        f_tbl = attribution_table(f_mean, f_gain)
        f_tbl["group"] = [DECISION_GROUPS.get(f, "unknown") for f in f_tbl.index]
        f_tbl.to_csv(f"{out_dir}/shap_feasibility_{source}.csv")
        np.savez_compressed(
            f"{out_dir}/shap_raw_feasibility_{source}.npz",
            shap_values=f_values.astype(np.float32), X=f_X.astype(np.float32),
            feature_names=np.array(FEATURE_NAMES))
        feas_out = {
            "top5": f_tbl.head(5).index.tolist(),
            "by_group": f_tbl.groupby("group")["shap_share"].sum().to_dict(),
            "agreement_with_duration_model": rank_agreement(mean_abs, f_mean),
            "additivity": f_diag,
        }
        info("feasibility classifier top-5: " + ", ".join(feas_out["top5"]))

    tbl.to_csv(f"{out_dir}/shap_decision_forecaster_{source}.csv")
    grp.to_csv(f"{out_dir}/shap_decision_by_group_{source}.csv")
    np.savez_compressed(
        f"{out_dir}/shap_raw_decision_{source}.npz",
        shap_values=values.astype(np.float32), X=X_used.astype(np.float32),
        feature_names=np.array(FEATURE_NAMES))
    info(f"wrote {out_dir}/shap_decision_forecaster_{source}.csv")

    print(f"\n  Decision forecaster attribution ({source} labels):")
    print(tbl[["shap_share", "gain_share", "rank_shap", "rank_gain",
               "group"]].round(4).to_string())
    print("\n  By group:")
    print(grp.round(4).to_string())

    calendar_shap = float(grp.loc["factory calendar", "shap_share"]) \
        if "factory calendar" in grp.index else float("nan")
    calendar_gain = float(grp.loc["factory calendar", "gain_share"]) \
        if "factory calendar" in grp.index else float("nan")

    return {
        "label_source": source,
        "n_episodes_train": int(len(prep["ep_train"])),
        "n_episodes_test": int(len(prep["ep_test"])),
        "n_epochs_train": int(len(prep["Xtr"])),
        "n_epochs_test": int(len(Xte)),
        "n_shap_rows": int(len(X_used)),
        "top5_shap": tbl.head(5).index.tolist(),
        "top5_gain": gain.sort_values(ascending=False).head(5).index.tolist(),
        "calendar_share_shap": calendar_shap,
        "calendar_share_gain": calendar_gain,
        "by_group": grp.to_dict(orient="index"),
        "shap_vs_gain_agreement": rank_agreement(mean_abs, gain),
        "additivity": diag,
        "feasibility": feas_out,
        "forecaster_mae_s": float(np.mean(np.abs(
            np.clip(model.predict(Xte), 0.0, None) - yte))),
    }


# ── A2: the window state forecaster ───────────────────────────────────────────

def _phase5_seed0_reference(machine: str) -> dict | None:
    """Seed-0 metrics of the XGBoost window model as Phase V's Task 12A stored them."""
    import json

    path = (f"outputs/phase5/task12/{machine}/forecasting/forecasting_seeds.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        runs = json.load(fh)["results"].get("xgboost", {}).get("runs", [])
    for r in runs:
        if r.get("seed") == 0:
            return {k: r[k] for k in ("f1_STANDBY", "standby_mae_s",
                                      "accuracy", "macro_f1")}
    return None



def run_window_forecaster(
    machine: str,
    out_dir: str,
    seed: int = 0,
    n_shap: int = N_SHAP_WINDOW,
) -> dict:
    """
    Attribute the Task 5 tree over the same windows the six sequence models
    were scored on.

    The model is REFITTED here rather than reloaded, because Task 5 and Task
    12 store predictions and not boosters.

    THE PROTOCOL IS TASK 12A'S, NOT TASK 5'S DEFAULT.  The two differ in one
    constant -- Task 5's module default decimates by 5, Task 12A by 10 -- and
    Phase V's forecasting table, the one this attribution is meant to
    explain, is the Task 12A run.  Attributing a model fitted at a different
    sampling rate would explain a model that appears in no table.  Seed 0
    therefore has to reproduce the stored seed-0 metrics exactly, and the
    reproduction is asserted below rather than assumed.
    """
    from sklearn.utils.class_weight import compute_class_weight
    from src.features import all_feature_names, group_of
    from experiments.common import load_labelled
    from experiments.task5_forecasting_baselines import (
        TRAIN_PCT, VAL_PCT,
        decimate, make_windows, standardise, train_xgboost,
        xgb_design_feature_names, standby_seconds, sequence_metrics,
        duration_metrics,
    )
    from experiments.task12_multiple_runs import (
        DECIMATE, LOOKBACK_S, HORIZON_S, STRIDE_S,
    )

    section("TASK 13A-2 -- window state forecaster (Task 12A protocol)")

    df, meta = load_labelled(machine, drop_spikes=True, add_features=True)
    feature_cols = [f for f in all_feature_names() if f in df.columns]
    state_names = sorted(df["state"].unique().tolist())
    encoder = {s: i for i, s in enumerate(state_names)}
    df["state_id"] = df["state"].map(encoder).astype(np.int16)
    standby_id = encoder["STANDBY"]

    df = decimate(df, DECIMATE)
    step_s = meta["sample_interval_s"] * DECIMATE
    lb = max(2, int(round(LOOKBACK_S / step_s)))
    hz = max(1, int(round(HORIZON_S / step_s)))
    st = max(1, int(round(STRIDE_S / step_s)))

    n = len(df)
    cut1, cut2 = int(n * TRAIN_PCT), int(n * (TRAIN_PCT + VAL_PCT))
    parts = [df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]]
    W = [make_windows(p.reset_index(drop=True), feature_cols, lb, hz, st)
         for p in parts]
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = W
    (X_tr, X_va, X_te), _ = standardise(X_tr, X_va, X_te)
    info(f"{len(X_tr):,} train / {len(X_va):,} val / {len(X_te):,} test windows")

    flat = y_tr.ravel()
    present = np.unique(flat)
    class_weight = np.ones(len(state_names), dtype=np.float32)
    class_weight[present] = compute_class_weight("balanced", classes=present,
                                                 y=flat)

    cap: dict = {}
    y_pred, extra = train_xgboost(X_tr, y_tr, X_va, y_va, X_te,
                                  len(state_names), class_weight, seed,
                                  out_dir, capture=cap)

    # Reproduction check against the stored Phase V run, not an assumption.
    m = sequence_metrics(y_te, y_pred, state_names)
    m.update(duration_metrics(standby_seconds(y_te, standby_id, step_s),
                              standby_seconds(y_pred, standby_id, step_s)))
    ref = _phase5_seed0_reference(machine)
    info(f"refit STANDBY F1 {m['f1_STANDBY']:.4f}  MAE {m['standby_mae_s']:.3f}s"
         + (f"  (Phase V seed 0: {ref['f1_STANDBY']:.4f} / "
            f"{ref['standby_mae_s']:.3f} s)" if ref else "  (no stored run)"))
    if ref and seed == 0:
        drift = abs(m["f1_STANDBY"] - ref["f1_STANDBY"])
        if drift > 1e-6:
            raise RuntimeError(
                f"the refitted window forecaster does not reproduce the Phase V "
                f"seed-0 run (STANDBY F1 {m['f1_STANDBY']:.6f} against stored "
                f"{ref['f1_STANDBY']:.6f}); the attribution below would explain "
                f"a model that appears in no table")

    design_names = xgb_design_feature_names(feature_cols)
    model = cap["model"]
    assert cap["design_test"].shape[1] == len(design_names), \
        "design matrix width does not match the generated column names"

    gain = pd.Series(model.feature_importances_, index=design_names,
                     name="xgb_gain")
    mean_abs, values, X_used, diag = tree_shap(model, cap["design_test"],
                                               design_names, n_shap, seed)
    tbl = attribution_table(mean_abs, gain)

    # Two aggregations.  By CHANNEL answers "which measured quantity matters";
    # by SUMMARY answers "does the tree use the level, the trend or the
    # variability of it", which is the question the rolling-statistics feature
    # group was added to Phase II to test.
    def _channel(col: str) -> str:
        return col.split("|")[0] if "|" in col else col

    def _summary(col: str) -> str:
        if "|" not in col:
            return "horizon index"
        suffix = col.split("|")[1]
        if suffix == "last":
            return "final observed value"
        return "block mean" if suffix.endswith("_mean") else "block std"

    tbl["channel"] = [_channel(c) for c in tbl.index]
    tbl["summary"] = [_summary(c) for c in tbl.index]
    tbl["group"] = [group_of(c) if c != "horizon_step" else "horizon"
                    for c in tbl["channel"]]

    by_channel = (tbl.groupby("channel")[["shap_share", "gain_share"]].sum()
                     .sort_values("shap_share", ascending=False))
    by_group = (tbl.groupby("group")[["shap_share", "gain_share"]].sum()
                   .sort_values("shap_share", ascending=False))
    by_summary = (tbl.groupby("summary")[["shap_share", "gain_share"]].sum()
                     .sort_values("shap_share", ascending=False))

    tbl.to_csv(f"{out_dir}/shap_window_forecaster.csv")
    by_channel.to_csv(f"{out_dir}/shap_window_by_channel.csv")
    by_group.to_csv(f"{out_dir}/shap_window_by_group.csv")
    by_summary.to_csv(f"{out_dir}/shap_window_by_summary.csv")
    np.savez_compressed(
        f"{out_dir}/shap_raw_window.npz",
        shap_values=values.astype(np.float32), X=X_used.astype(np.float32),
        feature_names=np.array(design_names),
        class_names=np.array(state_names))

    print("\n  Window forecaster, top 15 design columns:")
    print(tbl.head(15)[["shap_share", "gain_share", "channel", "summary",
                        "group"]].round(4).to_string())
    print("\n  By engineered channel (top 10):")
    print(by_channel.head(10).round(4).to_string())
    print("\n  By feature group:")
    print(by_group.round(4).to_string())

    calendar_channels = [c for c in by_channel.index
                         if group_of(c) == "temporal"]
    return {
        "n_windows_train": int(len(X_tr)),
        "n_windows_test": int(len(X_te)),
        "n_design_columns": len(design_names),
        "n_shap_rows": int(len(X_used)),
        "state_names": state_names,
        "refit_metrics": {k: m[k] for k in
                          ("accuracy", "macro_f1", "f1_STANDBY",
                           "standby_mae_s", "standby_r2")},
        "phase5_seed0_reference": ref,
        "top10_shap": tbl.head(10).index.tolist(),
        "by_channel": by_channel.head(12).to_dict(orient="index"),
        "by_group": by_group.to_dict(orient="index"),
        "by_summary": by_summary.to_dict(orient="index"),
        "calendar_share_shap": float(by_channel.loc[calendar_channels,
                                                    "shap_share"].sum()),
        "shap_vs_gain_agreement": rank_agreement(mean_abs, gain),
        "additivity": diag,
        "train_seconds": extra["train_seconds"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machine: str = DEFAULT_MACHINE,
    only: list[str] | None = None,
    label_sources: list[str] | None = None,
    seed: int = 0,
) -> dict:
    section("PHASE VI / TASK 13A -- SHAP FOR THE XGBOOST FORECASTERS")
    out_dir = ensure_dir(f"{PHASE6_DIR}/task13a/{machine}")
    only = only or ["decision", "window"]
    label_sources = label_sources or ["auto", "phase4"]

    # Merge into whatever a previous run left, so that `--only window` does
    # not silently delete the decision-forecaster half of the result file.
    results: dict = {"machine": machine, "seed": seed,
                     "decision_forecaster": {}, "window_forecaster": None}
    prev_path = f"{out_dir}/task13a_results.json"
    if os.path.exists(prev_path):
        import json
        with open(prev_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("seed") == seed and prev.get("machine") == machine:
            results.update({k: v for k, v in prev.items() if v not in (None, {})})
            results["decision_forecaster"] = dict(
                prev.get("decision_forecaster") or {})
            info(f"merging into {prev_path}")

    if "decision" in only:
        for src in label_sources:
            results["decision_forecaster"][src] = run_decision_forecaster(
                machine, src, out_dir, seed=seed)

    if "window" in only:
        results["window_forecaster"] = run_window_forecaster(
            machine, out_dir, seed=seed)

    save_json(results, f"{out_dir}/task13a_results.json")
    section("TASK 13A COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase VI Task 13A -- forecaster SHAP")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["decision", "window"])
    ap.add_argument("--label-sources", nargs="*", default=None,
                    choices=["auto", "reference", "phase4"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.machine, a.only, a.label_sources, a.seed)
