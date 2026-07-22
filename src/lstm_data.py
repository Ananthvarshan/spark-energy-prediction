"""
============================================================
LSTM DATA PREPARATION  --  src/lstm_data.py
============================================================

PURPOSE
-------
Prepare a GMM-labelled DataFrame for LSTM training:
  1. Enrich with cyclical time features + dwell time
  2. Build sliding windows (X, y) from the time series
  3. Chronological train/val/test split (NO shuffling)
  4. Compute class weights for imbalanced state distributions

All windowing is specified in REAL TIME (seconds), not
arbitrary row counts — the actual lookback/horizon in rows
is derived from the data's own sampling interval.

HOW TO USE
----------
    from src.lstm_data import (
        add_lstm_features,
        compute_sample_interval,
        make_windows,
        chronological_split,
        compute_class_weights,
    )

    df = add_lstm_features(df, factory_tz="America/Sao_Paulo",
                           schedule_close_hour=17,
                           schedule_open_hour=22)
    interval_s = compute_sample_interval(df)
    X, y = make_windows(df, feature_cols, lookback_s=600, horizon_s=300,
                        sample_interval_s=interval_s)
    train_df, val_df, test_df = chronological_split(df)
    sample_weight = compute_class_weights(y_train)
============================================================
"""

import warnings
import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
GAP_MULTIPLE    = 10.0   # interval > GAP_MULTIPLE × median → data gap, no window crosses it
MIN_ROWS_WINDOW = 10     # minimum rows in data for windowing to be attempted


# ── Feature engineering ───────────────────────────────────────────────────────

def add_lstm_features(
    df: pd.DataFrame,
    state_col: str = "state",
    timestamp_col: str = "timestamp",
    factory_tz: str = "America/Sao_Paulo",
    schedule_close_hour: int = 17,
    schedule_open_hour: int = 22,
    sample_interval_s: float | None = None,
) -> pd.DataFrame:
    """
    Add LSTM-ready feature columns to a GMM-labelled DataFrame.

    New columns added
    -----------------
    hour_sin, hour_cos      : cyclical hour-of-day encoding
    dow_sin, dow_cos        : cyclical day-of-week encoding
    is_weekend              : bool — Saturday or Sunday
    is_factory_open         : bool — within factory operating hours
    dwell_seconds_so_far    : seconds the machine has been in the
                              current state as of each row (resets
                              at every state transition)
    power_factor            : active_power / apparent_power (if not
                              already present and both columns exist)

    Parameters
    ----------
    df                : labelled DataFrame (from validate_gmm.py export)
    state_col         : column containing state labels
    timestamp_col     : column containing UTC-aware timestamps
    factory_tz        : local timezone string for schedule checks
    schedule_close_hour : hour (local) when factory closes (5 PM → 17)
    schedule_open_hour  : hour (local) when factory opens next (10 PM → 22)
    sample_interval_s   : known sampling interval in seconds; if None,
                          computed from the timestamp column.

    Returns
    -------
    DataFrame with added columns (copy, original unchanged).
    """
    df = df.copy()

    # ── Timestamp handling ────────────────────────────────────────────────────
    if timestamp_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")

        # Convert to local timezone for schedule checks
        ts_local = df[timestamp_col].dt.tz_convert(factory_tz)

        # ── Cyclical time features ────────────────────────────────────────────
        hour = ts_local.dt.hour + ts_local.dt.minute / 60.0
        dow  = ts_local.dt.dayofweek   # 0=Monday … 6=Sunday

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)

        # ── Weekend + factory schedule ────────────────────────────────────────
        df["is_weekend"] = dow.isin([5, 6])

        # Factory open window: handles the case where close > open
        # (i.e., 17:00–22:00 means CLOSED, so open = NOT in that window)
        h_int = ts_local.dt.hour
        if schedule_close_hour < schedule_open_hour:
            # Simple: closed between close_hour and open_hour on same day
            in_closed_window = (h_int >= schedule_close_hour) & (h_int < schedule_open_hour)
        else:
            # Wraps midnight: closed between close_hour and midnight + midnight to open_hour
            in_closed_window = (h_int >= schedule_close_hour) | (h_int < schedule_open_hour)

        df["is_factory_open"] = ~in_closed_window & ~df["is_weekend"]

    else:
        # No timestamp column — fill with neutral values
        df["hour_sin"]        = 0.0
        df["hour_cos"]        = 1.0
        df["dow_sin"]         = 0.0
        df["dow_cos"]         = 1.0
        df["is_weekend"]      = False
        df["is_factory_open"] = True

    # ── Power factor (if missing) ─────────────────────────────────────────────
    if "power_factor" not in df.columns:
        if "active_power" in df.columns and "apparent_power" in df.columns:
            df["power_factor"] = (
                df["active_power"].abs()
                / df["apparent_power"].replace(0, float("nan"))
            ).clip(0.0, 1.0)

    # ── Dwell time so far ─────────────────────────────────────────────────────
    if state_col in df.columns:
        # Group consecutive rows with the same state into runs
        state_change = df[state_col].ne(df[state_col].shift()).cumsum()
        row_counts   = df.groupby(state_change).cumcount()   # 0, 1, 2 … within each run

        if sample_interval_s is None and timestamp_col in df.columns:
            sample_interval_s = compute_sample_interval(df, timestamp_col)
        if sample_interval_s is None:
            sample_interval_s = 1.0   # safest fallback: 1 second per row

        df["dwell_seconds_so_far"] = row_counts * sample_interval_s
    else:
        df["dwell_seconds_so_far"] = 0.0

    return df


# ── Sampling interval ─────────────────────────────────────────────────────────

def compute_sample_interval(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> float:
    """
    Derive the actual median sampling interval in seconds from the timestamp column.

    Uses the median (not the mean) so that large gaps from sensor downtime
    don't inflate the estimate.

    Returns
    -------
    float — median sampling interval in seconds
    """
    if timestamp_col not in df.columns:
        warnings.warn(
            f"[lstm_data] '{timestamp_col}' column not found — "
            "defaulting to 1.0 second sampling interval."
        )
        return 1.0

    ts = pd.to_datetime(df[timestamp_col], utc=True).sort_values()
    diffs = ts.diff().dt.total_seconds().dropna()
    pos   = diffs[diffs > 0]

    if len(pos) == 0:
        return 1.0

    interval = float(pos.median())
    print(f"[lstm_data] Derived sampling interval: {interval:.3f}s "
          f"(from {len(pos):,} consecutive timestamp pairs)")
    return interval


# ── Windowing ─────────────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback_s: float,
    horizon_s: float,
    sample_interval_s: float,
    target_col: str = "state_id",
    stride_s: float | None = None,
    timestamp_col: str = "timestamp",
    gap_multiple: float = GAP_MULTIPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y) arrays for LSTM training.

    Windows are specified in REAL TIME (seconds) and converted to
    row counts using sample_interval_s.  No window is allowed to
    cross a detected data gap (> gap_multiple × median interval).

    Parameters
    ----------
    df               : enriched DataFrame (after add_lstm_features)
    feature_cols     : list of column names to use as LSTM input features
    lookback_s       : past window size in seconds (input to LSTM)
    horizon_s        : future window size in seconds (prediction target)
    sample_interval_s: derived from compute_sample_interval()
    target_col       : integer-encoded state column (default 'state_id')
    stride_s         : step between window starts in seconds; defaults to
                       sample_interval_s (stride = 1 row = densest sampling)
    timestamp_col    : used for gap detection
    gap_multiple     : intervals > this × median are treated as data gaps

    Returns
    -------
    (X, y)
      X : float32 array of shape (n_windows, lookback_rows, n_features)
      y : int32   array of shape (n_windows, horizon_rows)
    """
    lookback_rows = max(1, int(round(lookback_s  / sample_interval_s)))
    horizon_rows  = max(1, int(round(horizon_s   / sample_interval_s)))
    stride_rows   = max(1, int(round((stride_s or sample_interval_s) / sample_interval_s)))

    print(
        f"[lstm_data] Window config: "
        f"lookback={lookback_s:.0f}s ({lookback_rows} rows), "
        f"horizon={horizon_s:.0f}s ({horizon_rows} rows), "
        f"stride={stride_rows} rows"
    )

    n = len(df)
    if n < lookback_rows + horizon_rows + MIN_ROWS_WINDOW:
        raise ValueError(
            f"[lstm_data] DataFrame has only {n} rows — need at least "
            f"{lookback_rows + horizon_rows + MIN_ROWS_WINDOW} for a single window."
        )

    # ── Gap mask: True at row i means the INTERVAL before row i is a gap ────
    gap_mask = np.zeros(n, dtype=bool)
    if timestamp_col in df.columns:
        ts    = pd.to_datetime(df[timestamp_col], utc=True).values
        diffs = np.diff(ts.astype("datetime64[ns]")).astype(float) / 1e9  # → seconds
        cap   = gap_multiple * sample_interval_s
        for i, d in enumerate(diffs):
            if d > cap:
                gap_mask[i + 1] = True

    features = df[feature_cols].values.astype(np.float32)
    targets  = df[target_col].values.astype(np.int32)

    X_list, y_list = [], []
    skipped_gap = 0

    for start in range(0, n - lookback_rows - horizon_rows, stride_rows):
        end_x = start + lookback_rows
        end_y = end_x  + horizon_rows

        # Skip if any gap falls inside this window
        if gap_mask[start : end_y].any():
            skipped_gap += 1
            continue

        X_list.append(features[start : end_x])
        y_list.append(targets[end_x : end_y])

    if not X_list:
        raise ValueError(
            "[lstm_data] No valid windows produced — "
            "check lookback/horizon settings or data gaps."
        )

    X = np.stack(X_list, axis=0)   # (n_windows, lookback_rows, n_features)
    y = np.stack(y_list, axis=0)   # (n_windows, horizon_rows)

    print(
        f"[lstm_data] Windows produced: {len(X_list):,} "
        f"(skipped {skipped_gap:,} gap-crossing windows)"
    )
    print(f"[lstm_data] X shape: {X.shape}  y shape: {y.shape}")

    return X, y


# ── Chronological split ───────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float   = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame into contiguous train / val / test slices.

    NEVER shuffles — shuffling leaks future information into training on
    time-series data (a window adjacent to a test sample shares most rows).

    Parameters
    ----------
    df        : chronologically ordered DataFrame
    train_pct : fraction of rows for training (default 0.70)
    val_pct   : fraction of rows for validation (default 0.15)
                test gets the remaining (1 - train_pct - val_pct)

    Returns
    -------
    (train_df, val_df, test_df) — non-overlapping contiguous slices.

    Raises
    ------
    AssertionError if any overlap is detected between splits.
    """
    assert 0 < train_pct < 1, "train_pct must be in (0, 1)"
    assert 0 < val_pct   < 1, "val_pct must be in (0, 1)"
    assert train_pct + val_pct < 1, "train_pct + val_pct must be < 1"

    n         = len(df)
    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print(
        f"\n[lstm_data] Chronological split:\n"
        f"  Train : rows 0–{train_end-1:,}         ({len(train_df):,} rows, {train_pct*100:.0f}%)\n"
        f"  Val   : rows {train_end:,}–{val_end-1:,}  ({len(val_df):,} rows, {val_pct*100:.0f}%)\n"
        f"  Test  : rows {val_end:,}–{n-1:,}        ({len(test_df):,} rows, "
        f"{(1-train_pct-val_pct)*100:.0f}%)"
    )

    # ── Verify no timestamp overlap ───────────────────────────────────────────
    if timestamp_col in df.columns:
        try:
            train_max = pd.to_datetime(train_df[timestamp_col], utc=True).max()
            val_min   = pd.to_datetime(val_df[timestamp_col],   utc=True).min()
            val_max   = pd.to_datetime(val_df[timestamp_col],   utc=True).max()
            test_min  = pd.to_datetime(test_df[timestamp_col],  utc=True).min()

            assert train_max <= val_min, (
                f"Timestamp overlap: train ends {train_max}, val starts {val_min}"
            )
            assert val_max <= test_min, (
                f"Timestamp overlap: val ends {val_max}, test starts {test_min}"
            )
            print(
                f"[lstm_data] Timestamp continuity verified:\n"
                f"  Train ends : {train_max}\n"
                f"  Val starts : {val_min}\n"
                f"  Val ends   : {val_max}\n"
                f"  Test starts: {test_min}"
            )
        except Exception as e:
            print(f"[lstm_data] WARNING: Timestamp overlap check failed: {e}")

    return train_df, val_df, test_df


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    y_train: np.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Compute per-sample class weights for imbalanced state distributions.

    Keras does not support class_weight with 3D y (TimeDistributed targets),
    so this returns a sample_weight array of the same shape as y_train,
    where each element is the weight of its class.

    Parameters
    ----------
    y_train   : int32 array of shape (n_windows, horizon_rows)
    n_classes : number of classes; inferred from y_train if None

    Returns
    -------
    sample_weight : float32 array of shape (n_windows, horizon_rows)
    """
    from sklearn.utils.class_weight import compute_class_weight

    flat = y_train.flatten()
    classes = np.arange(n_classes) if n_classes else np.unique(flat)

    weights = compute_class_weight("balanced", classes=classes, y=flat)
    weight_map = dict(zip(classes.tolist(), weights.tolist()))

    print(f"\n[lstm_data] Class weights (balanced):")
    for cls, w in weight_map.items():
        print(f"  class {cls}: {w:.4f}")

    sample_weight = np.vectorize(weight_map.get)(y_train).astype(np.float32)
    return sample_weight
