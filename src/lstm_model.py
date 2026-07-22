"""
============================================================
LSTM MODEL  --  src/lstm_model.py
============================================================

PURPOSE
-------
Build, train, and evaluate the encoder-decoder seq2seq LSTM
that predicts future machine operating states.

ARCHITECTURE (matches architecture diagram Step 7)
--------------------------------------------------
Input  (lookback, n_features)
  ↓ LSTM(128, return_sequences=True)  — encodes past context
  ↓ LSTM(64,  return_sequences=False) — compressed summary
  ↓ RepeatVector(horizon)             — broadcast to future steps
  ↓ LSTM(64, return_sequences=True)   — decodes into future steps
  ↓ TimeDistributed(Dense(n_states, softmax))
Output (horizon, n_states) — per-step state probability distribution

EVALUATION
----------
- Overall sparse_categorical_accuracy
- Per-class precision / recall / F1 (scikit-learn classification_report)
- STANDBY-duration MAE: converts predicted step-sequences into
  contiguous-STANDBY-run durations and computes MAE vs actual durations
  (this is the number that matters for the decision logic in Step 9)

HOW TO USE
----------
    from src.lstm_model import build_lstm_model, train_model, evaluate_model

    model = build_lstm_model(n_features=10, lookback=240,
                             horizon=120, n_states=3)
    history = train_model(model, X_train, y_train, X_val, y_val,
                          sample_weight=sw, output_dir="outputs/models/pelletizer-I")
    metrics = evaluate_model(model, X_test, y_test, state_decoder,
                             sample_interval_s=2.5)
============================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks


# ── Model builder ─────────────────────────────────────────────────────────────

def build_lstm_model(
    n_features: int,
    lookback: int,
    horizon: int,
    n_states: int,
    lstm_units_enc: tuple[int, int] = (128, 64),
    lstm_units_dec: int = 64,
    dropout: float = 0.2,
) -> Model:
    """
    Build the encoder-decoder seq2seq LSTM model.

    Parameters
    ----------
    n_features     : number of input features per timestep
    lookback       : number of past timesteps (rows) fed as input
    horizon        : number of future timesteps to predict
    n_states       : number of state classes (e.g. 3 for OFF/STANDBY/WORKING)
    lstm_units_enc : (enc1_units, enc2_units) for the two encoder LSTM layers
    lstm_units_dec : units for the decoder LSTM layer
    dropout        : dropout rate between layers

    Returns
    -------
    Compiled Keras Model
    """
    inputs = layers.Input(shape=(lookback, n_features), name="input_window")

    # ── Encoder: compress past M steps into a fixed-size context vector ──────
    x = layers.LSTM(lstm_units_enc[0], return_sequences=True,
                    name="enc_lstm_1")(inputs)
    x = layers.Dropout(dropout, name="enc_dropout_1")(x)
    x = layers.LSTM(lstm_units_enc[1], return_sequences=False,
                    name="enc_lstm_2")(x)
    x = layers.Dropout(dropout, name="enc_dropout_2")(x)

    # ── Bridge: repeat context vector across the horizon ─────────────────────
    x = layers.RepeatVector(horizon, name="bridge_repeat")(x)

    # ── Decoder: produce per-step predictions ────────────────────────────────
    x = layers.LSTM(lstm_units_dec, return_sequences=True,
                    name="dec_lstm")(x)
    x = layers.Dropout(dropout, name="dec_dropout")(x)

    # TimeDistributed applies the same Dense layer independently to each step
    outputs = layers.TimeDistributed(
        layers.Dense(n_states, activation="softmax"),
        name="output_softmax"
    )(x)

    model = Model(inputs, outputs, name="seq2seq_lstm")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    print(f"\n[lstm_model] Model built:")
    print(f"  Input  : ({lookback}, {n_features})")
    print(f"  Output : ({horizon}, {n_states})")
    model.summary(print_fn=lambda s: print("  " + s))

    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    output_dir: str = "outputs/models",
    patience: int = 7,
) -> object:
    """
    Train the LSTM model with early stopping and model checkpointing.

    Parameters
    ----------
    model         : compiled Keras Model from build_lstm_model()
    X_train       : (n_train, lookback, n_features) float32
    y_train       : (n_train, horizon) int32
    X_val         : (n_val,   lookback, n_features) float32
    y_val         : (n_val,   horizon)  int32
    sample_weight : (n_train, horizon) float32 per-step class weights;
                    if None, all samples weighted equally
    epochs        : maximum training epochs
    batch_size    : mini-batch size
    output_dir    : directory to save model checkpoint and training history
    patience      : early-stopping patience (epochs without val_loss improvement)

    Returns
    -------
    Keras History object
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.keras")

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(
        f"\n[lstm_model] Training:\n"
        f"  X_train: {X_train.shape}  y_train: {y_train.shape}\n"
        f"  X_val  : {X_val.shape}   y_val  : {y_val.shape}\n"
        f"  Epochs : {epochs}  Batch: {batch_size}  Patience: {patience}\n"
        f"  Checkpoint → {checkpoint_path}"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # Save training history as CSV for later analysis
    hist_path = os.path.join(output_dir, "training_history.csv")
    import pandas as pd
    pd.DataFrame(history.history).to_csv(hist_path, index=False)
    print(f"[lstm_model] Training history saved → {hist_path}")

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def _run_lengths(sequence: np.ndarray, target_class: int) -> list[int]:
    """
    Extract lengths of all contiguous runs of target_class in a 1D array.
    Returns a list of run lengths (in row counts).
    """
    runs = []
    count = 0
    for val in sequence:
        if val == target_class:
            count += 1
        else:
            if count > 0:
                runs.append(count)
                count = 0
    if count > 0:
        runs.append(count)
    return runs


def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    state_decoder: dict,
    sample_interval_s: float = 1.0,
    standby_class_id: int | None = None,
    output_dir: str = "outputs/models",
) -> dict:
    """
    Evaluate the trained model on the test set.

    Reports
    -------
    - Overall sparse_categorical_accuracy
    - Per-class precision / recall / F1 (scikit-learn classification_report)
    - STANDBY-duration MAE: mean absolute error between predicted and
      actual contiguous-STANDBY-run durations (in seconds).
      This is the metric that matters for Step 9 decision logic.

    Parameters
    ----------
    model             : trained Keras Model
    X_test            : (n_test, lookback, n_features) float32
    y_test            : (n_test, horizon) int32
    state_decoder     : {int: state_name} — maps class IDs to names
    sample_interval_s : derived from compute_sample_interval()
    standby_class_id  : integer class ID for STANDBY; auto-detected
                        from state_decoder if None
    output_dir        : directory to save evaluation report

    Returns
    -------
    dict with keys: accuracy, classification_report, standby_duration_mae_s
    """
    from sklearn.metrics import classification_report

    # ── Auto-detect STANDBY class ID ─────────────────────────────────────────
    if standby_class_id is None:
        for cid, name in state_decoder.items():
            if name.upper() == "STANDBY":
                standby_class_id = cid
                break

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\n[lstm_model] Running predictions on test set ...")
    probs    = model.predict(X_test, batch_size=128, verbose=0)  # (n, horizon, n_states)
    pred_ids = probs.argmax(axis=-1)                              # (n, horizon)

    # ── Flatten for classification metrics ───────────────────────────────────
    y_true_flat = y_test.flatten()
    y_pred_flat = pred_ids.flatten()

    target_names = [state_decoder[i] for i in sorted(state_decoder.keys())]
    report = classification_report(
        y_true_flat, y_pred_flat,
        target_names=target_names,
        zero_division=0,
    )

    # Accuracy
    accuracy = float((y_true_flat == y_pred_flat).mean())

    print(f"\n[lstm_model] ── Test Evaluation ──")
    print(f"  Overall Accuracy : {accuracy*100:.2f}%")
    print(f"\n  Per-Class Report:\n{report}")

    # ── STANDBY-duration MAE ──────────────────────────────────────────────────
    standby_mae_s = None
    if standby_class_id is not None:
        true_durations, pred_durations = [], []
        for i in range(len(y_test)):
            t_runs = _run_lengths(y_test[i],    standby_class_id)
            p_runs = _run_lengths(pred_ids[i],  standby_class_id)
            true_durations.extend(t_runs)
            pred_durations.extend(p_runs)

        if true_durations and pred_durations:
            # Compare distribution medians (pair-wise MAE is ill-defined for
            # varying-length run lists — use median as a representative summary)
            true_med = float(np.median(true_durations)) * sample_interval_s
            pred_med = float(np.median(pred_durations)) * sample_interval_s
            standby_mae_s = abs(true_med - pred_med)

            print(
                f"\n[lstm_model] STANDBY Duration Analysis:\n"
                f"  Actual   median STANDBY run: {true_med:.1f}s "
                f"({true_med/60:.1f} min)  [from {len(true_durations)} runs]\n"
                f"  Predicted median STANDBY run: {pred_med:.1f}s "
                f"({pred_med/60:.1f} min)  [from {len(pred_durations)} runs]\n"
                f"  Median Duration MAE: {standby_mae_s:.1f}s "
                f"({standby_mae_s/60:.2f} min)"
            )
        else:
            print("[lstm_model] WARNING: No STANDBY runs found in test set predictions.")

    # ── Save report to file ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "test_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Accuracy: {accuracy*100:.2f}%\n\n")
        fh.write("Per-Class Classification Report:\n")
        fh.write(report)
        if standby_mae_s is not None:
            fh.write(f"\nSTANDBY Duration Median MAE: {standby_mae_s:.1f}s\n")

    print(f"[lstm_model] Evaluation report saved → {report_path}")

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "standby_duration_mae_s": standby_mae_s,
    }
