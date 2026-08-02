"""
============================================================
TASK 5 -- FORECASTING BASELINE COMPARISON
experiments/task5_forecasting_baselines.py
============================================================

PURPOSE
-------
Phase II / Task 5.  Benchmark the proposed Seq2Seq LSTM against
five alternatives under one protocol: same windows, same features,
same split, same class weighting, same metrics.

WHAT IS BEING FORECAST
----------------------
Given `lookback` seconds of history, predict the state at every
step of the next `horizon` seconds.  From that sequence we derive
the quantity the decision layer actually consumes: the number of
STANDBY seconds inside the horizon.  Both are reported --
sequence-level classification metrics, and duration-level
regression metrics -- because a model can be good at one and bad
at the other.  A model that scatters STANDBY predictions uniformly
gets decent per-step F1 while producing a useless duration
estimate.

DECIMATION
----------
The record is 1 Hz over 155 days (4.4M clean rows).  At full rate a
600 s lookback is 600 steps and the windowed tensor for a single
model exceeds available memory, making a six-model x multi-seed
comparison impossible.  The series is therefore decimated by
`--decimate` (default 5, i.e. 0.2 Hz) before windowing.

This is a modelling decision, not a shortcut, and it is a defensible
one: the target quantity is a standby interval measured in minutes,
so 0.2 Hz is still an order of magnitude finer than the phenomenon.
The engineered rolling-statistic features (300 s window) are computed
BEFORE decimation and therefore still summarise every original
sample, acting as an anti-aliasing filter.  The decimation factor is
recorded in the results JSON and is identical for all six models.

SPLIT
-----
Chronological 70/15/15 on rows, then windows are built within each
split independently so that no window straddles a split boundary.
Random splitting would be badly leaky: adjacent windows overlap by
`lookback - stride` steps.

CLASS WEIGHTING
---------------
STANDBY is ~6% of rows.  All six models receive the same balanced
per-step sample weights derived from the training split, so no model
is advantaged by a different imbalance treatment.

USAGE
-----
    python -m experiments.task5_forecasting_baselines
    python -m experiments.task5_forecasting_baselines --models gru tcn
    python -m experiments.task5_forecasting_baselines --seeds 3 --epochs 40
============================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from experiments.common import (            # noqa: E402
    DEFAULT_MACHINE, PHASE2_DIR,
    load_labelled, ensure_dir, save_json, info, section,
)
from src.features import all_feature_names   # noqa: E402


# ── Configuration ─────────────────────────────────────────────────────────────

DECIMATE = 5                  # 1 Hz -> 0.2 Hz
LOOKBACK_S = 600.0            # 10 minutes of history
HORIZON_S = 300.0             # 5 minutes ahead
STRIDE_S = 120.0              # window step

TRAIN_PCT, VAL_PCT = 0.70, 0.15

EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 6

# XGBoost consumes a flattened window summary rather than the raw sequence.
XGB_SUBBLOCKS = 4             # lookback split into this many summary blocks
XGB_MAX_TRAIN_ROWS = 600_000  # cap on the (window x horizon-step) expansion
XGB_PARAMS = dict(
    n_estimators=400, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=-1, tree_method="hist",
)

ALL_MODELS = ["persistence", "seq2seq_lstm", "vanilla_lstm", "gru", "tcn",
              "transformer", "xgboost"]

# Models with no fitted parameters, so a seed changes nothing and one run is
# the whole distribution.  Kept explicit so `--seeds 25` does not silently
# report a standard deviation of zero as if it had been measured.
DETERMINISTIC_MODELS = {"persistence"}


# ── Windowing ─────────────────────────────────────────────────────────────────

def decimate(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Keep every `factor`-th row within each contiguous segment."""
    if factor <= 1:
        return df.reset_index(drop=True)
    keep = df.groupby("segment_id", sort=False).cumcount() % factor == 0
    out = df[keep].reset_index(drop=True)
    info(f"decimated by {factor}x: {len(df):,} -> {len(out):,} rows")
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback_rows: int,
    horizon_rows: int,
    stride_rows: int,
    target_col: str = "state_id",
    segment_col: str = "segment_id",
    return_last_state: bool = False,
):
    """
    Sliding windows that never cross a segment boundary.

    Returns X (n, lookback, n_features) float32 and y (n, horizon) int16, and
    with `return_last_state=True` also the state at the FINAL LOOKBACK STEP
    (n,) int16.

    That third output exists for the persistence baseline: "the machine stays
    in the state it is in" is the forecast any controller can make without a
    model, and it is the reference the trained models have to beat.  It is
    taken from the last observed step, so it uses no information the models do
    not also have.
    """
    X_parts, y_parts, last_parts = [], [], []
    feats_all = df[feature_cols].to_numpy(np.float32)
    targ_all = df[target_col].to_numpy(np.int16)
    seg = df[segment_col].to_numpy()

    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(df)]

    span = lookback_rows + horizon_rows
    for s, e in zip(starts, ends):
        if e - s < span:
            continue
        f = feats_all[s:e]
        t = targ_all[s:e]
        n = len(f)
        idx = np.arange(0, n - span + 1, stride_rows)
        if len(idx) == 0:
            continue
        # Strided views -> one contiguous copy per segment.
        xi = idx[:, None] + np.arange(lookback_rows)[None, :]
        yi = idx[:, None] + lookback_rows + np.arange(horizon_rows)[None, :]
        X_parts.append(f[xi])
        y_parts.append(t[yi])
        last_parts.append(t[idx + lookback_rows - 1])

    if not X_parts:
        raise ValueError("No windows produced -- check lookback/horizon/stride.")
    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int16)
    if return_last_state:
        return X, y, np.concatenate(last_parts).astype(np.int16)
    return X, y


def standardise(X_tr, X_va, X_te):
    """
    Standardise per feature using TRAINING statistics only.

    Fitting the scaler on all splits would leak test-set distribution
    information into training, which for a non-stationary industrial
    signal is a real effect, not a formality.
    """
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0
    return [((X - mu) / sd).astype(np.float32) for X in (X_tr, X_va, X_te)], (mu, sd)


# ── Metrics ───────────────────────────────────────────────────────────────────

def standby_seconds(y: np.ndarray, standby_id: int, step_s: float) -> np.ndarray:
    """STANDBY seconds inside each window's horizon."""
    return (y == standby_id).sum(axis=1).astype(float) * step_s


def duration_metrics(true_s: np.ndarray, pred_s: np.ndarray) -> dict:
    """
    MAE / RMSE / MAPE / sMAPE on predicted STANDBY seconds per window.

    MAPE is undefined when the true duration is zero, which happens for
    most windows (the machine is usually not in standby).  It is therefore
    computed only over windows with a non-zero true duration, and the
    count of such windows is reported alongside so the figure can be read
    honestly.  sMAPE is reported over ALL windows as the complete-coverage
    alternative; it is symmetric and finite whenever true+pred > 0.
    """
    err = pred_s - true_s
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    nz = true_s > 0
    mape = (float(np.mean(np.abs(err[nz]) / true_s[nz]) * 100.0)
            if nz.any() else float("nan"))

    denom = np.abs(true_s) + np.abs(pred_s)
    ok = denom > 0
    smape = (float(np.mean(2.0 * np.abs(err[ok]) / denom[ok]) * 100.0)
             if ok.any() else float("nan"))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_s - true_s.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "standby_mae_s": mae,
        "standby_rmse_s": rmse,
        "standby_mape_pct": mape,
        "standby_smape_pct": smape,
        "standby_r2": r2,
        "n_windows": int(len(true_s)),
        "n_windows_nonzero_standby": int(nz.sum()),
        "mean_true_standby_s": float(true_s.mean()),
        "mean_pred_standby_s": float(pred_s.mean()),
    }


def sequence_metrics(y_true, y_pred, state_names) -> dict:
    """Per-step classification metrics over the flattened horizon."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    t, p = y_true.ravel(), y_pred.ravel()
    prec, rec, f1, sup = precision_recall_fscore_support(
        t, p, labels=np.arange(len(state_names)), zero_division=0)
    out = {
        "accuracy": float((t == p).mean()),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(t, p, average="weighted", zero_division=0)),
    }
    for i, name in enumerate(state_names):
        out[f"f1_{name}"] = float(f1[i])
        out[f"precision_{name}"] = float(prec[i])
        out[f"recall_{name}"] = float(rec[i])
        out[f"support_{name}"] = int(sup[i])
    return out


# ── Training drivers ──────────────────────────────────────────────────────────

def train_neural(
    key, X_tr, y_tr, X_va, y_va, X_te, n_states, w_tr, w_va,
    epochs, batch_size, patience, seed, out_dir,
):
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from experiments.baseline_models import NEURAL_BUILDERS, count_params

    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()

    model = NEURAL_BUILDERS[key](
        n_features=X_tr.shape[2], lookback=X_tr.shape[1],
        horizon=y_tr.shape[1], n_states=n_states,
    )
    n_params = count_params(model)
    info(f"{key}: {n_params:,} trainable parameters")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=max(2, patience // 2), verbose=1),
    ]

    t0 = time.time()
    hist = model.fit(
        X_tr, y_tr, sample_weight=w_tr,
        validation_data=(X_va, y_va, w_va),
        epochs=epochs, batch_size=batch_size,
        callbacks=cbs, verbose=2, shuffle=True,
    )
    train_s = time.time() - t0

    t0 = time.time()
    probs = model.predict(X_te, batch_size=256, verbose=0)
    infer_s = time.time() - t0
    y_pred = probs.argmax(axis=-1).astype(np.int16)

    pd.DataFrame(hist.history).to_csv(
        f"{out_dir}/history_{key}_seed{seed}.csv", index=False)

    return y_pred, {
        "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "inference_seconds_per_window_ms": round(infer_s / max(len(X_te), 1) * 1000, 4),
        "epochs_run": len(hist.history["loss"]),
        "best_val_loss": float(np.min(hist.history["val_loss"])),
    }


def _xgb_featurise(X: np.ndarray, n_blocks: int = XGB_SUBBLOCKS) -> np.ndarray:
    """
    Collapse a (n, lookback, n_features) window tensor into a flat design
    matrix for the tree baseline.

    A tree cannot consume a sequence, so the standard sliding-window
    treatment is to summarise it.  Each window is split into `n_blocks`
    equal sub-blocks; per feature we take each sub-block's mean and std,
    plus the final observed value.  Handing XGBoost all 120x24 raw cells
    instead would give it 2,880 mostly-redundant columns and would make
    the comparison a statement about feature engineering rather than
    about the model class.
    """
    n, T, F = X.shape
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    parts = [X[:, -1, :]]
    for a, b in zip(edges[:-1], edges[1:]):
        seg = X[:, a:b, :]
        parts.append(seg.mean(axis=1))
        parts.append(seg.std(axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_xgboost(
    X_tr, y_tr, X_va, y_va, X_te, n_states, class_weight, seed, out_dir,
):
    """
    Direct multi-horizon tree baseline: the horizon step index is an input
    feature, so one model serves all H steps rather than training H
    separate models.
    """
    import xgboost as xgb

    F_tr, F_va, F_te = (_xgb_featurise(X) for X in (X_tr, X_va, X_te))
    H = y_tr.shape[1]
    rng = np.random.default_rng(seed)

    def expand(F, y, cap=None):
        n = len(F)
        wi = np.repeat(np.arange(n), H)
        si = np.tile(np.arange(H), n)
        if cap is not None and len(wi) > cap:
            sel = rng.choice(len(wi), size=cap, replace=False)
            wi, si = wi[sel], si[sel]
        Xd = np.column_stack([F[wi], si.astype(np.float32)])
        yd = y[wi, si]
        return Xd, yd

    Xd_tr, yd_tr = expand(F_tr, y_tr, XGB_MAX_TRAIN_ROWS)
    Xd_va, yd_va = expand(F_va, y_va, XGB_MAX_TRAIN_ROWS // 4)
    info(f"xgboost design matrix: {Xd_tr.shape[0]:,} x {Xd_tr.shape[1]} "
         f"(from {len(F_tr):,} windows x {H} horizon steps)")

    w_tr = class_weight[yd_tr]

    t0 = time.time()
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=n_states,
        random_state=seed, early_stopping_rounds=25, **XGB_PARAMS,
    )
    model.fit(Xd_tr, yd_tr, sample_weight=w_tr,
              eval_set=[(Xd_va, yd_va)], verbose=False)
    train_s = time.time() - t0

    # Predict the FULL test expansion (no capping at inference).
    n_te = len(F_te)
    wi = np.repeat(np.arange(n_te), H)
    si = np.tile(np.arange(H), n_te)
    t0 = time.time()
    pred = model.predict(np.column_stack([F_te[wi], si.astype(np.float32)]))
    infer_s = time.time() - t0
    y_pred = pred.reshape(n_te, H).astype(np.int16)

    return y_pred, {
        "n_params": int(model.get_booster().num_boosted_rounds()),
        "n_design_features": int(Xd_tr.shape[1]),
        "train_seconds": round(train_s, 1),
        "inference_seconds_per_window_ms": round(infer_s / max(n_te, 1) * 1000, 4),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machine=DEFAULT_MACHINE, models=None, seeds=1, decimate_factor=DECIMATE,
    lookback_s=LOOKBACK_S, horizon_s=HORIZON_S, stride_s=STRIDE_S,
    epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
    append=False,
):
    from sklearn.utils.class_weight import compute_class_weight
    from experiments.baseline_models import MODEL_INFO

    section("PHASE II / TASK 5 -- FORECASTING BASELINE COMPARISON")
    out_dir = ensure_dir(f"{PHASE2_DIR}/task5/{machine}")
    models = models or ALL_MODELS

    df, meta = load_labelled(machine, drop_spikes=True, add_features=True)
    feature_cols = [f for f in all_feature_names() if f in df.columns]

    state_names = sorted(df["state"].unique().tolist())
    encoder = {s: i for i, s in enumerate(state_names)}
    df["state_id"] = df["state"].map(encoder).astype(np.int16)
    standby_id = encoder["STANDBY"]
    info(f"states {encoder}")

    # -- Decimate, then window ---------------------------------------------
    df = decimate(df, decimate_factor)
    step_s = meta["sample_interval_s"] * decimate_factor
    lb = max(2, int(round(lookback_s / step_s)))
    hz = max(1, int(round(horizon_s / step_s)))
    st = max(1, int(round(stride_s / step_s)))
    info(f"step {step_s:.1f}s | lookback {lb} rows ({lookback_s:.0f}s) | "
         f"horizon {hz} rows ({horizon_s:.0f}s) | stride {st} rows")

    n = len(df)
    cut1, cut2 = int(n * TRAIN_PCT), int(n * (TRAIN_PCT + VAL_PCT))
    splits = {"train": df.iloc[:cut1], "val": df.iloc[cut1:cut2],
              "test": df.iloc[cut2:]}
    info(f"chronological row split: train {cut1:,} | val {cut2-cut1:,} | "
         f"test {n-cut2:,}")
    info(f"train ends {splits['train']['timestamp'].max()} | "
         f"test starts {splits['test']['timestamp'].min()}")

    W = {}
    for name, part in splits.items():
        Xw, yw, lw = make_windows(part.reset_index(drop=True), feature_cols,
                                  lb, hz, st, return_last_state=True)
        W[name] = (Xw, yw, lw)
        info(f"{name}: {len(Xw):,} windows, X{Xw.shape} "
             f"({Xw.nbytes/1e6:.0f} MB)")

    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, l_te) = (
        W["train"], W["val"], W["test"])
    (X_tr, X_va, X_te), (mu, sd) = standardise(X_tr, X_va, X_te)

    # -- Shared class weights ----------------------------------------------
    flat = y_tr.ravel()
    present = np.unique(flat)
    cw_present = compute_class_weight("balanced", classes=present, y=flat)
    class_weight = np.ones(len(state_names), dtype=np.float32)
    class_weight[present] = cw_present
    info("balanced class weights: " +
         ", ".join(f"{state_names[i]}={class_weight[i]:.3f}"
                   for i in range(len(state_names))))
    w_tr = class_weight[y_tr].astype(np.float32)
    w_va = class_weight[y_va].astype(np.float32)

    true_s = standby_seconds(y_te, standby_id, step_s)
    info(f"test windows: mean true STANDBY {true_s.mean():.1f}s of "
         f"{hz*step_s:.0f}s horizon; {(true_s>0).mean()*100:.1f}% have any")

    # -- Run every model, every seed ---------------------------------------
    results: dict = {}
    if append:
        # Merge into whatever a previous run left, so that adding one model
        # does not silently reduce the comparison table to that model.
        prev_path = f"{out_dir}/task5_results.json"
        if os.path.exists(prev_path):
            with open(prev_path, encoding="utf-8") as fh:
                results = json.load(fh).get("results", {})
            info(f"appending to {len(results)} existing model result(s)")

    for key in models:
        name, citation = MODEL_INFO[key]
        per_seed = []
        n_seeds = 1 if key in DETERMINISTIC_MODELS else seeds
        for seed in range(n_seeds):
            section(f"{name}  [{citation}]  seed {seed}")
            t0 = time.time()
            try:
                if key == "persistence":
                    # No training: the horizon repeats the last observed state.
                    y_pred = np.repeat(l_te[:, None], y_te.shape[1],
                                       axis=1).astype(np.int16)
                    extra = {"n_params": 0, "train_seconds": 0.0,
                             "inference_seconds_per_window_ms": 0.0,
                             "note": "no fitted parameters; deterministic"}
                elif key == "xgboost":
                    y_pred, extra = train_xgboost(
                        X_tr, y_tr, X_va, y_va, X_te, len(state_names),
                        class_weight, seed, out_dir)
                else:
                    y_pred, extra = train_neural(
                        key, X_tr, y_tr, X_va, y_va, X_te, len(state_names),
                        w_tr, w_va, epochs, batch_size, patience, seed, out_dir)
            except Exception as exc:
                info(f"FAILED: {type(exc).__name__}: {exc}")
                per_seed.append({"seed": seed,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue

            m = sequence_metrics(y_te, y_pred, state_names)
            m.update(duration_metrics(true_s,
                                      standby_seconds(y_pred, standby_id, step_s)))
            m.update(extra)
            m["seed"] = seed
            m["total_seconds"] = round(time.time() - t0, 1)
            per_seed.append(m)

            info(f"acc {m['accuracy']*100:.2f}%  macroF1 {m['macro_f1']:.3f}  "
                 f"STANDBY-F1 {m.get('f1_STANDBY', float('nan')):.3f}  "
                 f"MAE {m['standby_mae_s']:.1f}s  RMSE {m['standby_rmse_s']:.1f}s")

            if seed == 0:
                np.savez_compressed(f"{out_dir}/preds_{key}.npz",
                                    y_pred=y_pred, y_true=y_te)

        ok = [s for s in per_seed if "error" not in s]
        agg = {}
        if ok:
            keys = [k for k in ok[0] if isinstance(ok[0][k], (int, float))]
            for k in keys:
                vals = [s[k] for s in ok]
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
        results[key] = {"model": name, "citation": citation,
                        "seeds": per_seed, "aggregate": agg}
        _dump_table(results, state_names, out_dir)      # checkpoint after each

    # -- Final table --------------------------------------------------------
    section("TASK 5 -- COMPARISON TABLE")
    tbl = _dump_table(results, state_names, out_dir)
    print(tbl.to_string())

    cfg = {
            "decimate": decimate_factor, "step_seconds": step_s,
            "lookback_s": lookback_s, "horizon_s": horizon_s,
            "stride_s": stride_s, "lookback_rows": lb, "horizon_rows": hz,
            "stride_rows": st, "train_pct": TRAIN_PCT, "val_pct": VAL_PCT,
            "epochs": epochs, "batch_size": batch_size, "patience": patience,
            "seeds": seeds, "feature_cols": feature_cols,
            "state_encoder": encoder,
            "class_weights": class_weight.tolist(),
            "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
            "n_test_windows": int(len(X_te)),
            "xgb_params": XGB_PARAMS, "xgb_subblocks": XGB_SUBBLOCKS,
    }
    payload = {"meta": meta, "config": cfg, "results": results}
    if append and os.path.exists(f"{out_dir}/task5_results.json"):
        # An appended run must not overwrite the config the EARLIER models were
        # trained under -- `epochs` and `seeds` differ per invocation, and
        # silently restamping them would misdescribe results already in the
        # file.  The original config is kept and this run's is recorded beside
        # it, with the window geometry checked for compatibility.
        with open(f"{out_dir}/task5_results.json", encoding="utf-8") as fh:
            prev = json.load(fh)
        prev_cfg = prev.get("config", {})
        for k in ("decimate", "lookback_rows", "horizon_rows", "stride_rows",
                  "n_test_windows"):
            if k in prev_cfg and prev_cfg[k] != cfg.get(k):
                raise ValueError(
                    f"cannot append: {k} differs ({prev_cfg[k]} vs {cfg.get(k)}). "
                    f"The appended model would be scored on different windows.")
        appended = prev_cfg.get("appended_runs", [])
        appended.append({"models": list(models), "seeds": seeds,
                         "epochs": epochs})
        prev_cfg["appended_runs"] = appended
        payload = {"meta": prev.get("meta", meta), "config": prev_cfg,
                   "results": results}
    save_json(payload, f"{out_dir}/task5_results.json")
    section("TASK 5 COMPLETE")
    return results


def _dump_table(results, state_names, out_dir) -> pd.DataFrame:
    """Build and persist the comparison table from whatever has finished."""
    rows = []
    for key, r in results.items():
        a = r.get("aggregate", {})
        if not a:
            rows.append({"key": key, "model": r["model"], "status": "FAILED"})
            continue
        row = {"key": key, "model": r["model"], "citation": r["citation"],
               "n_seeds": len([s for s in r["seeds"] if "error" not in s])}
        for m in ("accuracy", "macro_f1", "f1_STANDBY", "standby_mae_s",
                  "standby_rmse_s", "standby_mape_pct", "standby_smape_pct",
                  "standby_r2", "n_params", "train_seconds",
                  "inference_seconds_per_window_ms"):
            if f"{m}_mean" in a:
                row[m] = a[f"{m}_mean"]
                if a.get(f"{m}_std", 0) > 0:
                    row[f"{m}_std"] = a[f"{m}_std"]
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("key")
    tbl.to_csv(f"{out_dir}/comparison_table.csv")
    return tbl.round(4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase II Task 5 -- forecasting baselines")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--models", nargs="*", default=None, help=f"subset of {ALL_MODELS}")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--append", action="store_true",
                    help="merge into the existing results JSON instead of replacing it")
    ap.add_argument("--decimate", type=int, default=DECIMATE)
    ap.add_argument("--lookback-s", type=float, default=LOOKBACK_S)
    ap.add_argument("--horizon-s", type=float, default=HORIZON_S)
    ap.add_argument("--stride-s", type=float, default=STRIDE_S)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    a = ap.parse_args()
    main(a.machine, a.models, a.seeds, a.decimate, a.lookback_s, a.horizon_s,
         a.stride_s, a.epochs, a.batch_size, a.patience, append=a.append)
