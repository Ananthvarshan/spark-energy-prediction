"""
=============================================================
THRESHOLD ACCURACY TEST
experiments/threshold_accuracy_test.py
=============================================================

PURPOSE
-------
Answers three questions:

  1. What features does XGBoost use to make its prediction?
  2. How accurately can XGBoost predict whether an idle episode will
     last LONGER than a given threshold? (10, 20, 30 ... 120 minutes)
  3. How does that compare to the LSTM / GRU / Seq2Seq / XGBoost
     state-forecasting models at the same thresholds?

The two "XGBoost" objects here are DIFFERENT:
  - "XGBoost-Decision" = the idle-duration forecaster in Task 6.
    It directly predicts how many seconds remain in an idle episode.
    We can turn that into a binary classifier at any threshold.
  - "XGBoost-State" = the state-sequence forecaster in Task 12 (Phase 5).
    It predicts which state (OFF/STANDBY/WORKING/PEAK) each row will
    be in, up to 5 minutes ahead. We count how often it says STANDBY
    when the machine really is still in standby at each horizon.

OUTPUT
------
Prints a formatted table + saves results to:
    outputs/threshold_test/threshold_accuracy_results.csv

Run from repo root:
    python -m experiments.threshold_accuracy_test
=============================================================
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── paths ─────────────────────────────────────────────────────────────────────
PHASE3_DIR   = "outputs/phase3/task6/pelletizer-I"
PHASE5_DIR   = "outputs/phase5/task12/pelletizer-I/forecasting"
OUT_DIR      = "outputs/threshold_test"
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds for XGBoost-Decision (10 to 120 min)
THRESHOLDS_MIN = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
THRESHOLDS_S   = [t * 60 for t in THRESHOLDS_MIN]

# Separate thresholds for state models (they only reach 5 min = 300s)
# We test them at 1, 2, 3, 4, 5 minutes
STATE_THRESHOLDS_MIN = [1, 2, 3, 4, 5]
STATE_THRESHOLDS_S   = [t * 60 for t in STATE_THRESHOLDS_MIN]

# State index for STANDBY (from state encoder used in Task 12)
# Check forecasting_seeds.json to confirm
STANDBY_IDX  = None   # resolved below

SEPARATOR = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — XGBoost-Decision features + how it works
# ══════════════════════════════════════════════════════════════════════════════

def show_xgboost_features() -> None:
    section("PART 1 — What does XGBoost-Decision look at?")

    features = [
        ("elapsed_log",           "How long has the machine been idle RIGHT NOW? (log scale)"),
        ("onset_hour",            "What hour of the day did the idle period START?"),
        ("onset_dow",             "What day of the week did it start? (0=Mon … 6=Sun)"),
        ("is_weekend",            "Was the start of the idle period on a weekend?"),
        ("cur_hour_sin",          "Sine of the CURRENT hour (cyclical encoding)"),
        ("cur_hour_cos",          "Cosine of the CURRENT hour (cyclical encoding)"),
        ("cur_is_open",           "Is the factory CURRENTLY open? (1=yes, 0=closed/weekend)"),
        ("cur_is_weekend",        "Is it CURRENTLY the weekend?"),
        ("prev_productive_log",   "How many seconds was the machine working BEFORE it stopped? (log scale)"),
        ("prev_productive_power_w","Average power (Watts) during that last working run"),
    ]

    print("\n  XGBoost-Decision sees 10 logical features at every 60-second decision point:\n")
    print(f"  {'Feature':<28} {'What it represents'}")
    print(f"  {'-'*28} {'-'*42}")
    for name, desc in features:
        print(f"  {name:<28} {desc}")

    print("\n  --> NO raw 1-second waveforms. Only factory calendar + episode summary.")
    print("  --> It re-evaluates EVERY 60 seconds while the machine stays idle.")

    # Load importance from results
    results_path = f"{PHASE3_DIR}/task6_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            res = json.load(f)
        imp = res["forecaster"]["mean"]["importance_gain"]
        print("\n  Feature importances (how much each feature contributed to accuracy):\n")
        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
        for name, val in sorted_imp:
            bar = "#" * int(val * 50)
            print(f"  {name:<28} {val:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — XGBoost-Decision: binary classification accuracy at each threshold
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_xgboost_decision() -> pd.DataFrame:
    section("PART 2 — XGBoost-Decision: accuracy at each threshold (10–120 min)")

    npz = np.load(f"{PHASE3_DIR}/forecast_test.npz")
    actual_s  = npz["actual_remaining_s"]    # true remaining idle seconds
    pred_s    = npz["pred_mean_s"]           # XGBoost predicted remaining seconds
    p_feasible = npz["p_feasible"]           # P(remaining >= min_off_s) from feasibility model

    print(f"\n  Test set: {len(actual_s):,} decision epochs")
    print(f"  Actual remaining time — mean: {actual_s.mean()/60:.1f} min, "
          f"median: {np.median(actual_s)/60:.1f} min, "
          f"max: {actual_s.max()/3600:.1f} h")

    rows = []
    print(f"\n  {'Threshold':>10} {'Base rate':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'Accuracy':>10} {'AUC':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for t_min, t_s in zip(THRESHOLDS_MIN, THRESHOLDS_S):
        y_true = (actual_s >= t_s).astype(int)
        y_pred = (pred_s   >= t_s).astype(int)

        base_rate = y_true.mean()
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy  = (tp + tn) / len(y_true)

        # AUC via rank correlation (no sklearn dependency)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, pred_s) if y_true.sum() > 0 and y_true.sum() < len(y_true) else float("nan")
        except Exception:
            auc = float("nan")

        print(f"  {t_min:>8} min {base_rate:>10.1%} {precision:>10.1%} "
              f"{recall:>10.1%} {f1:>10.1%} {accuracy:>10.1%} {auc:>8.3f}")

        rows.append({
            "model": "XGBoost-Decision",
            "threshold_min": t_min,
            "base_rate_pct": round(base_rate * 100, 1),
            "precision_pct": round(precision * 100, 1),
            "recall_pct":    round(recall * 100, 1),
            "f1_pct":        round(f1 * 100, 1),
            "accuracy_pct":  round(accuracy * 100, 1),
            "auc":           round(auc, 3) if not np.isnan(auc) else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — LSTM / deep learning state-forecasters at each threshold
#
# These models predict the state SEQUENCE for the next 5 minutes (30 steps of
# 10-second intervals).  We adapt them to the threshold question by asking:
# "Does this model correctly predict that the machine is still in STANDBY at
#  the step that corresponds to the given threshold?"
#
# Threshold 10 min = step 6 (60s = step 6 @ 10s per step)
# But the horizon is only 5 min = 300 s = 30 steps of 10 s.
# So for thresholds > 5 min we cannot use the state forecasters directly.
# Instead we report the STANDBY F1 at each available horizon step.
# ══════════════════════════════════════════════════════════════════════════════

def load_state_index() -> int:
    """Find the integer index that encodes STANDBY in the preds arrays."""
    seeds_path = f"{PHASE5_DIR}/forecasting_seeds.json"
    if not os.path.exists(seeds_path):
        return 1  # fallback assumption
    with open(seeds_path) as f:
        seeds = json.load(f)
    # The encoder maps state_name → int; look for STANDBY
    encoder = seeds.get("state_encoder", {})
    if "STANDBY" in encoder:
        return int(encoder["STANDBY"])
    # Try first seed
    for seed_data in seeds.values():
        if isinstance(seed_data, dict) and "state_encoder" in seed_data:
            enc = seed_data["state_encoder"]
            if "STANDBY" in enc:
                return int(enc["STANDBY"])
    return 1   # common default


def evaluate_state_model(model_name: str, npz_path: str,
                         standby_idx: int) -> pd.DataFrame:
    """
    Evaluate a state-sequence forecasting model at 1, 2, 3, 4, 5 minute horizons.

    The .npz files contain:
      y_true  : (n_windows, horizon_steps)  — true state IDs
      y_pred  : (seeds, n_windows, horizon_steps) OR (n_windows, horizon_steps)
      step_s  : scalar — seconds per step
    """
    if not os.path.exists(npz_path):
        return pd.DataFrame()

    npz  = np.load(npz_path)
    keys = list(npz.keys())

    y_true_key = next((k for k in keys if "y_true" in k.lower()), None)
    y_pred_key = next((k for k in keys if "y_pred" in k.lower()), None)

    if y_true_key is None or y_pred_key is None:
        if "y_true" in keys and "y_pred" in keys:
            y_true_key, y_pred_key = "y_true", "y_pred"
        else:
            print(f"  [{model_name}] WARNING: cannot find y_true/y_pred. Keys: {keys}")
            return pd.DataFrame()

    y_true = npz[y_true_key]
    y_pred = npz[y_pred_key]

    # Median over seeds if 3D
    if y_pred.ndim == 3:
        y_pred = np.round(np.median(y_pred, axis=0)).astype(int)
    if y_true.ndim == 3:
        y_true = y_true[0]

    # Read actual step_s from the file (default 10s)
    step_s = float(npz["step_s"]) if "step_s" in keys else 10.0

    n_windows, horizon_steps = y_true.shape
    horizon_s = horizon_steps * step_s

    rows = []
    for t_min in STATE_THRESHOLDS_MIN:
        t_s = t_min * 60

        if t_s > horizon_s:
            rows.append({
                "model": model_name,
                "threshold_min": t_min,
                "base_rate_pct": None,
                "precision_pct": None,
                "recall_pct": None,
                "f1_pct": None,
                "accuracy_pct": None,
                "note": f"beyond {int(horizon_s//60)}-min horizon",
            })
            continue

        step_idx = min(max(int(round(t_s / step_s)) - 1, 0), horizon_steps - 1)

        yt = y_true[:, step_idx]
        yp = y_pred[:, step_idx]

        tp = int(((yp == standby_idx) & (yt == standby_idx)).sum())
        fp = int(((yp == standby_idx) & (yt != standby_idx)).sum())
        tn = int(((yp != standby_idx) & (yt != standby_idx)).sum())
        fn = int(((yp != standby_idx) & (yt == standby_idx)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy  = (tp + tn) / len(yt)
        base_rate = (yt == standby_idx).mean()

        rows.append({
            "model": model_name,
            "threshold_min": t_min,
            "base_rate_pct": round(base_rate * 100, 1),
            "precision_pct": round(precision * 100, 1),
            "recall_pct":    round(recall * 100, 1),
            "f1_pct":        round(f1 * 100, 1),
            "accuracy_pct":  round(accuracy * 100, 1),
            "note": f"step {step_idx+1}/{horizon_steps} @ {step_s:.0f}s",
        })

    return pd.DataFrame(rows)



def evaluate_all_state_models(standby_idx: int) -> pd.DataFrame:
    section("PART 3 — LSTM / GRU / Seq2Seq / XGBoost-State: STANDBY accuracy per threshold")

    print(f"\n  STANDBY class index = {standby_idx}")
    print(f"  Horizon = 5 minutes (300 s, 30 steps of 10 s)")
    print(f"  Thresholds > 5 min are BEYOND the model horizon --> marked as N/A\n")

    models_to_test = {
        "Vanilla-LSTM":   f"{PHASE5_DIR}/preds_vanilla_lstm.npz",
        "Seq2Seq-LSTM":   f"{PHASE5_DIR}/preds_seq2seq_lstm.npz",
        "GRU":            f"{PHASE5_DIR}/preds_gru.npz",
        "Transformer":    f"{PHASE5_DIR}/preds_transformer.npz",
        "TCN":            f"{PHASE5_DIR}/preds_tcn.npz",
        "Persistence":    f"{PHASE5_DIR}/preds_persistence.npz",
        "XGBoost-State":  f"{PHASE5_DIR}/preds_xgboost.npz",
    }

    all_dfs = []
    for model_name, path in models_to_test.items():
        if not os.path.exists(path):
            print(f"  [{model_name}] file not found — skipping")
            continue
        df = evaluate_state_model(model_name, path, standby_idx)
        if df.empty:
            print(f"  [{model_name}] no data produced — skipping")
            continue
        all_dfs.append(df)

        print(f"  -- {model_name} --")
        print(f"  {'Threshold':>10} {'Base rate':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1':>10} {'Accuracy':>10}  Note")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  ----")
        for _, row in df.iterrows():
            if row.get("precision_pct") is None:
                print(f"  {int(row['threshold_min']):>8} min {'':>10} {'N/A':>10} "
                      f"{'':>10} {'':>10} {'':>10}  {row.get('note','')}")
            else:
                print(f"  {int(row['threshold_min']):>8} min "
                      f"{row['base_rate_pct']:>9.1f}% "
                      f"{row['precision_pct']:>9.1f}% "
                      f"{row['recall_pct']:>9.1f}% "
                      f"{row['f1_pct']:>9.1f}% "
                      f"{row['accuracy_pct']:>9.1f}%  {row.get('note','')}")
        print()

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Head-to-head summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(xgb_df: pd.DataFrame, state_df: pd.DataFrame) -> None:
    section("PART 4 — Head-to-head: XGBoost-Decision vs best LSTM at each threshold")

    print(f"\n  {'Threshold':>10}  {'XGB-Decision F1':>16}  {'Best-LSTM F1':>14}  "
          f"{'Best LSTM model':>18}  {'Winner':>10}")
    print(f"  {'-'*10}  {'-'*16}  {'-'*14}  {'-'*18}  {'-'*10}")

    for t_min in THRESHOLDS_MIN:
        # XGBoost-Decision F1
        xgb_row = xgb_df[xgb_df["threshold_min"] == t_min]
        xgb_f1 = float(xgb_row["f1_pct"].values[0]) if not xgb_row.empty else float("nan")

        # Best state-model F1 at this threshold (exclude N/A rows)
        state_rows = state_df[
            (state_df["threshold_min"] == t_min) &
            (state_df["f1_pct"].notna())
        ]
        if state_rows.empty:
            best_lstm_f1   = float("nan")
            best_lstm_name = "N/A (beyond horizon)"
        else:
            best_idx       = state_rows["f1_pct"].idxmax()
            best_lstm_f1   = float(state_rows.loc[best_idx, "f1_pct"])
            best_lstm_name = state_rows.loc[best_idx, "model"]

        if np.isnan(xgb_f1) and np.isnan(best_lstm_f1):
            winner = "—"
        elif np.isnan(best_lstm_f1):
            winner = "XGB-Decision"
        elif np.isnan(xgb_f1):
            winner = best_lstm_name
        elif xgb_f1 > best_lstm_f1:
            winner = "XGB-Decision [WIN]"
        elif best_lstm_f1 > xgb_f1:
            winner = f"{best_lstm_name} [WIN]"
        else:
            winner = "Tie"

        xgb_str  = f"{xgb_f1:.1f}%" if not np.isnan(xgb_f1) else "—"
        lstm_str = f"{best_lstm_f1:.1f}%" if not np.isnan(best_lstm_f1) else "N/A"

        print(f"  {t_min:>8} min  {xgb_str:>16}  {lstm_str:>14}  "
              f"{best_lstm_name:>18}  {winner:>10}")

    print("\n  KEY INSIGHT:")
    print("  * State models (LSTM/GRU/etc.) can ONLY predict up to 5 minutes ahead.")
    print("    For thresholds > 5 min they have NO prediction capability at all.")
    print("  * XGBoost-Decision can predict for ANY threshold up to 24 hours.")
    print("  * That is why XGBoost replaced LSTM in the final system architecture.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{'='*72}")
    print("  THRESHOLD ACCURACY TEST")
    print("  Comparing XGBoost-Decision vs LSTM/GRU/Seq2Seq state forecasters")
    print(f"{'='*72}")

    # Part 1: feature explanation
    show_xgboost_features()

    # Part 2: XGBoost-Decision accuracy
    xgb_df = evaluate_xgboost_decision()

    # Part 3: state models
    global STANDBY_IDX
    STANDBY_IDX = load_state_index()
    state_df = evaluate_all_state_models(STANDBY_IDX)

    # Part 4: head-to-head
    if not xgb_df.empty:
        print_summary(xgb_df, state_df)

    # Save results
    out_path = f"{OUT_DIR}/threshold_accuracy_results.csv"
    if not xgb_df.empty or not state_df.empty:
        combined = pd.concat([xgb_df, state_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"\n  Results saved --> {out_path}")

    section("DONE")


if __name__ == "__main__":
    main()
