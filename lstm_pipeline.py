"""
============================================================
LSTM PIPELINE  --  lstm_pipeline.py
============================================================

PURPOSE
-------
End-to-end orchestration of Steps 5–10 of the architecture:

  Step 5  : Load GMM-labelled CSV + frozen state mapping
  Step 6  : Build enriched labeled dataset (LSTM-ready features)
  Step 7  : Window, split, train the encoder-decoder LSTM
  Step 8  : Evaluate on held-out test set
  Step 9  : Demo decision logic on the test predictions
  Step 10 : Action layer (manual mode by default)

USAGE
-----
    python lstm_pipeline.py                    # default: pelletizer-I
    python lstm_pipeline.py pelletizer-II
    python lstm_pipeline.py --machine pelletizer-I --mode automatic
    python lstm_pipeline.py --help

PREREQUISITES
-------------
  1. Run validate_gmm.py first to generate the labelled CSV:
         python validate_gmm.py pelletizer-I
  2. The labelled CSV must exist at:
         outputs/imdeld_labelled/<machine>_labelled.csv

OUTPUTS
-------
  outputs/models/<machine>/
    best_model.keras         — saved Keras model (best val_loss)
    training_history.csv     — loss/accuracy per epoch
    test_evaluation.txt      — per-class F1 + STANDBY duration MAE
    state_map.json           — cluster→state mapping (frozen at GMM fit time)
    state_encoder.json       — state_name→int encoding for LSTM targets
    training_config.json     — all hyperparameters for reproducibility

  outputs/action_log/
    recommendations.csv      — every recommendation with confidence + savings
    outcomes.csv             — (filled later) actual vs predicted outcomes
============================================================
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.state_mapping import (
    build_state_encoder,
    save_state_encoder,
    load_state_encoder,
    refine_standby_with_pf,
)
from src.lstm_data import (
    add_lstm_features,
    compute_sample_interval,
    make_windows,
    chronological_split,
    compute_class_weights,
)
from src.lstm_model import build_lstm_model, train_model, evaluate_model
from src.lstm_inference import (
    predict_future_states,
    extract_predicted_standby_seconds,
    should_recommend_shutdown,
    compute_break_even_duration_s,
)
from src.action_layer import EnergyAction


# ============================================================
# MACHINE CONFIGURATION
# ============================================================

MACHINES = {
    "pelletizer-I": {
        "labelled":       "outputs/imdeld_labelled/pelletizer-I_labelled.csv",
        "name":           "Pelletizer I (IMDELD)",
        "factory_tz":     "America/Sao_Paulo",
        "close_hour":     17,
        "open_hour":      22,
        # Physics parameters for decision logic (Step 9)
        # standby_power_w: use the GMM cluster mean for STANDBY (inverse-scaled)
        # restart_energy_wh: machine-specific — update from spec sheet
        "standby_power_w":      3700.0,   # ~3.7 kW typical Pelletizer STANDBY draw
        "restart_energy_wh":    0.5,      # estimated restart energy cost in Wh
        "electricity_price":    0.15,     # USD per kWh
    },
    "pelletizer-II": {
        "labelled":       "outputs/imdeld_labelled/pelletizer-II_labelled.csv",
        "name":           "Pelletizer II (IMDELD)",
        "factory_tz":     "America/Sao_Paulo",
        "close_hour":     17,
        "open_hour":      22,
        "standby_power_w":      3700.0,
        "restart_energy_wh":    0.5,
        "electricity_price":    0.15,
    },
    "dpc-I": {
        "labelled":       "outputs/imdeld_labelled/dpc-I_labelled.csv",
        "name":           "Double-Pole Contactor I (IMDELD)",
        "factory_tz":     "America/Sao_Paulo",
        "close_hour":     17,
        "open_hour":      22,
        "standby_power_w":      500.0,
        "restart_energy_wh":    0.05,
        "electricity_price":    0.15,
    },
    "dpc-II": {
        "labelled":       "outputs/imdeld_labelled/dpc-II_labelled.csv",
        "name":           "Double-Pole Contactor II (IMDELD)",
        "factory_tz":     "America/Sao_Paulo",
        "close_hour":     17,
        "open_hour":      22,
        "standby_power_w":      500.0,
        "restart_energy_wh":    0.05,
        "electricity_price":    0.15,
    },
}

DEFAULT_MACHINE = "pelletizer-I"

# ── LSTM Hyperparameters ──────────────────────────────────────────────────────
# Lookback/horizon specified in REAL SECONDS — converted to rows at runtime
# using the data's own sampling interval.
# pelletizer-I is ~1s sampling → lookback 600 rows = 10 min past context
#                                  horizon 300 rows = 5 min ahead prediction
LOOKBACK_SECONDS = 600    # past context fed to encoder (10 minutes)
HORIZON_SECONDS  = 300    # prediction window (5 minutes)
STRIDE_SECONDS   = 30     # step between window starts (1 window every 30s)

TRAIN_PCT   = 0.70
VAL_PCT     = 0.15
# test = remaining 0.15

EPOCHS      = 50
BATCH_SIZE  = 64
PATIENCE    = 7

# Feature columns used as LSTM input (raw electrical + engineered)
# 'state_id' is the TARGET, not an input feature
BASE_FEATURE_COLS = [
    "active_power",
    "reactive_power",
    "apparent_power",
    "current",
    "voltage",
    "power_factor",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_factory_open",
    "dwell_seconds_so_far",
]


# ============================================================
# HELPERS
# ============================================================

def _print_section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _save_training_config(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[pipeline] Training config saved → {path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(machine_key: str, action_mode: str = "manual"):
    if machine_key not in MACHINES:
        print(f"Unknown machine '{machine_key}'. Choose from: {list(MACHINES.keys())}")
        sys.exit(1)

    cfg          = MACHINES[machine_key]
    machine_name = cfg["name"]
    output_dir   = f"outputs/models/{machine_key}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  LSTM ENERGY PIPELINE")
    print(f"  Machine : {machine_name}")
    print(f"  Mode    : {action_mode}")
    print(f"  Output  : {output_dir}/")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Load GMM-labelled CSV
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 5 — Load Labelled Dataset")

    labelled_path = cfg["labelled"]
    if not os.path.exists(labelled_path):
        print(f"\n[pipeline] ERROR: Labelled CSV not found: {labelled_path}")
        print(f"  Run first: python validate_gmm.py {machine_key}")
        sys.exit(1)

    print(f"[pipeline] Loading: {labelled_path}")
    df = pd.read_csv(labelled_path)
    print(f"[pipeline] Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Remove spike rows from training — they carry no real signal
    if "is_spike" in df.columns:
        n_spikes = int(df["is_spike"].sum())
        df = df[~df["is_spike"]].reset_index(drop=True)
        print(f"[pipeline] Removed {n_spikes:,} spike rows — {len(df):,} clean rows remain")

    # Verify state column exists
    if "state" not in df.columns:
        print("[pipeline] ERROR: 'state' column not found in labelled CSV.")
        print("  Re-run validate_gmm.py to regenerate the labelled CSV.")
        sys.exit(1)

    print(f"\n[pipeline] State distribution:")
    for state, count in df["state"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {state:<12}: {count:>10,} rows  ({pct:.1f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Build LSTM-ready labeled dataset
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 6 — Feature Engineering + State Encoding")

    # Derive sampling interval
    sample_interval_s = compute_sample_interval(df)

    # Add LSTM features
    df = add_lstm_features(
        df,
        state_col="state",
        factory_tz=cfg["factory_tz"],
        schedule_close_hour=cfg["close_hour"],
        schedule_open_hour=cfg["open_hour"],
        sample_interval_s=sample_interval_s,
    )

    # Optional: flag ambiguous STANDBY rows
    if "power_factor" in df.columns:
        df = refine_standby_with_pf(df, state_col="state", pf_col="power_factor")

    # Build state encoder
    unique_states      = sorted(df["state"].unique())
    encoder, decoder   = build_state_encoder(unique_states)
    df["state_id"]     = df["state"].map(encoder).astype(int)

    encoder_path = os.path.join(output_dir, "state_encoder.json")
    save_state_encoder(encoder, encoder_path)

    print(f"\n[pipeline] State encoding: {encoder}")

    # Select feature columns (only those present in the DataFrame)
    feature_cols = [c for c in BASE_FEATURE_COLS if c in df.columns]
    print(f"\n[pipeline] Feature columns ({len(feature_cols)}): {feature_cols}")

    n_states   = len(encoder)
    n_features = len(feature_cols)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7a: Chronological Split
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 7a — Chronological Train/Val/Test Split")

    train_df, val_df, test_df = chronological_split(
        df, train_pct=TRAIN_PCT, val_pct=VAL_PCT
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7b: Windowing
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 7b — Sliding Window Generation")

    print("[pipeline] Building training windows ...")
    X_train, y_train = make_windows(
        train_df, feature_cols,
        lookback_s=LOOKBACK_SECONDS,
        horizon_s=HORIZON_SECONDS,
        sample_interval_s=sample_interval_s,
        target_col="state_id",
        stride_s=STRIDE_SECONDS,
    )

    print("[pipeline] Building validation windows ...")
    X_val, y_val = make_windows(
        val_df, feature_cols,
        lookback_s=LOOKBACK_SECONDS,
        horizon_s=HORIZON_SECONDS,
        sample_interval_s=sample_interval_s,
        target_col="state_id",
        stride_s=STRIDE_SECONDS,
    )

    print("[pipeline] Building test windows ...")
    X_test, y_test = make_windows(
        test_df, feature_cols,
        lookback_s=LOOKBACK_SECONDS,
        horizon_s=HORIZON_SECONDS,
        sample_interval_s=sample_interval_s,
        target_col="state_id",
        stride_s=STRIDE_SECONDS,
    )

    lookback_rows = X_train.shape[1]
    horizon_rows  = y_train.shape[1]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7c: Class weights
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 7c — Class Weights (Imbalance Correction)")

    sample_weight = compute_class_weights(y_train, n_classes=n_states)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7d: Build + Train LSTM
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 7d — Build + Train Encoder-Decoder LSTM")

    model = build_lstm_model(
        n_features=n_features,
        lookback=lookback_rows,
        horizon=horizon_rows,
        n_states=n_states,
    )

    # Save training config for reproducibility
    training_config = {
        "machine":            machine_key,
        "n_features":         n_features,
        "feature_cols":       feature_cols,
        "n_states":           n_states,
        "state_encoder":      encoder,
        "lookback_seconds":   LOOKBACK_SECONDS,
        "horizon_seconds":    HORIZON_SECONDS,
        "stride_seconds":     STRIDE_SECONDS,
        "lookback_rows":      lookback_rows,
        "horizon_rows":       horizon_rows,
        "sample_interval_s":  sample_interval_s,
        "train_pct":          TRAIN_PCT,
        "val_pct":            VAL_PCT,
        "epochs":             EPOCHS,
        "batch_size":         BATCH_SIZE,
        "patience":           PATIENCE,
        "n_train_windows":    len(X_train),
        "n_val_windows":      len(X_val),
        "n_test_windows":     len(X_test),
    }
    _save_training_config(
        training_config,
        os.path.join(output_dir, "training_config.json")
    )

    history = train_model(
        model,
        X_train, y_train,
        X_val,   y_val,
        sample_weight=sample_weight,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        output_dir=output_dir,
        patience=PATIENCE,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Evaluate on test set
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 8 — Test Set Evaluation")

    metrics = evaluate_model(
        model, X_test, y_test,
        state_decoder=decoder,
        sample_interval_s=sample_interval_s,
        output_dir=output_dir,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Decision logic demo (on first 100 test windows)
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 9 — Decision Logic Demo (first 100 test windows)")

    standby_power_w    = cfg["standby_power_w"]
    restart_energy_wh  = cfg["restart_energy_wh"]
    elec_price         = cfg["electricity_price"]

    break_even_s = compute_break_even_duration_s(standby_power_w, restart_energy_wh)
    print(
        f"[pipeline] Break-even STANDBY duration: {break_even_s:.1f}s "
        f"({break_even_s/60:.2f} min)\n"
        f"  (based on {standby_power_w:.0f}W standby draw, "
        f"{restart_energy_wh:.3f}Wh restart cost)"
    )

    action_layer = EnergyAction(
        mode=action_mode,
        output_dir="outputs/action_log",
    )

    n_demo       = min(100, len(X_test))
    n_recommend  = 0
    total_savings = 0.0

    for i in range(n_demo):
        window = X_test[i : i + 1]   # (1, lookback, n_features)
        state_names, probs = predict_future_states(model, window, decoder)

        predicted_stby_s, confidence = extract_predicted_standby_seconds(
            state_names, probs, sample_interval_s, confidence_threshold=0.0
        )

        recommend, savings = should_recommend_shutdown(
            predicted_standby_seconds=predicted_stby_s,
            standby_power_w=standby_power_w,
            restart_energy_wh=restart_energy_wh,
            electricity_price_per_kwh=elec_price,
        )

        if recommend:
            n_recommend += 1
            total_savings += savings
            action_layer.execute(
                machine_id=machine_key,
                action="SHUTDOWN_RECOMMENDED",
                confidence=confidence,
                savings_estimate=savings,
                predicted_standby_s=predicted_stby_s,
                notes=f"test window {i}",
            )

    print(
        f"\n[pipeline] Demo summary ({n_demo} test windows):\n"
        f"  Shutdown recommendations: {n_recommend} / {n_demo}\n"
        f"  Estimated total savings  : ${total_savings:.4f}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: Final summary
    # ─────────────────────────────────────────────────────────────────────────
    _print_section("STEP 10 — Pipeline Complete")

    print(
        f"\n[pipeline] ✅ All steps complete for {machine_name}\n"
        f"\nOutputs:\n"
        f"  Model checkpoint  : {output_dir}/best_model.keras\n"
        f"  State encoder     : {output_dir}/state_encoder.json\n"
        f"  Training config   : {output_dir}/training_config.json\n"
        f"  Test evaluation   : {output_dir}/test_evaluation.txt\n"
        f"  Training history  : {output_dir}/training_history.csv\n"
        f"  Action log        : outputs/action_log/recommendations.csv\n"
        f"\nTest accuracy     : {metrics['accuracy']*100:.2f}%"
    )
    if metrics.get("standby_duration_mae_s") is not None:
        print(
            f"STANDBY duration MAE: {metrics['standby_duration_mae_s']:.1f}s "
            f"({metrics['standby_duration_mae_s']/60:.2f} min)"
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSTM pipeline for GMM-based industrial energy state prediction"
    )
    parser.add_argument(
        "machine",
        nargs="?",
        default=DEFAULT_MACHINE,
        choices=list(MACHINES.keys()),
        help=f"Machine to run (default: {DEFAULT_MACHINE})",
    )
    parser.add_argument(
        "--machine", dest="machine_flag",
        default=None,
        choices=list(MACHINES.keys()),
        help="Alternative way to specify machine (--machine pelletizer-II)",
    )
    parser.add_argument(
        "--mode",
        default="manual",
        choices=["manual", "automatic"],
        help="Action mode: 'manual' (log only) or 'automatic' (PLC signal if confident)",
    )

    args = parser.parse_args()
    machine_key = args.machine_flag or args.machine

    run_pipeline(machine_key, action_mode=args.mode)
