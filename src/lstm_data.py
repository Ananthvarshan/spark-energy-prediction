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
        add_standby_band_feature,
        smooth_short_standby_labels,
        compute_sample_interval,
        make_windows,
        chronological_split,
        compute_class_weights,
    )

    df = add_lstm_features(df, factory_tz="America/Sao_Paulo",
                           schedule_close_hour=17,
                           schedule_open_hour=22)
    df = smooth_short_standby_labels(df, min_dwell_s=30)  # Fix flicker labels
    df = add_standby_band_feature(df)                     # Physics-anchored hint
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


# ── STANDBY label smoother ────────────────────────────────────────────────────

def smooth_short_standby_labels(
    df: pd.DataFrame,
    state_col: str = "state",
    min_dwell_s: float = 30.0,
    sample_interval_s: float | None = None,
    timestamp_col: str = "timestamp",
    standby_name: str = "STANDBY",
) -> pd.DataFrame:
    """
    Remove STANDBY micro-flicker labels shorter than min_dwell_s seconds.

    The GMM+Viterbi pipeline often produces very short STANDBY bursts
    (median=2s from Method 4 results) between WORKING segments. These
    are labelling artefacts — real STANDBY events last minutes, not
    seconds. Training the LSTM on flicker labels teaches it that
    'STANDBY = WORKING with noise', which is why STANDBY F1 is low.

    This function:
    1. Identifies all contiguous STANDBY runs in state_col.
    2. Any run shorter than min_dwell_s is re-labelled to the
       majority state of its immediate neighbours (typically WORKING).
    3. Logs how many rows were smoothed for full traceability.

    Parameters
    ----------
    df              : DataFrame with a state label column.
    state_col       : name of the state column to smooth.
    min_dwell_s     : minimum valid STANDBY dwell in seconds (default 30s).
                      Runs shorter than this are merged into neighbours.
    sample_interval_s : seconds per row; computed from timestamp if None.
    timestamp_col   : used for interval computation only.
    standby_name    : exact string for STANDBY in state_col (default 'STANDBY').

    Returns
    -------
    DataFrame with smoothed state labels. Original column is overwritten;
    the old labels are preserved in 'state_pre_smooth' for auditability.
    """
    df = df.copy()
    df["state_pre_smooth"] = df[state_col].copy()  # preserve originals

    if sample_interval_s is None:
        if timestamp_col in df.columns:
            sample_interval_s = compute_sample_interval(df, timestamp_col)
        else:
            sample_interval_s = 1.0

    min_dwell_rows = max(1, int(round(min_dwell_s / sample_interval_s)))

    states      = df[state_col].values.copy()
    n           = len(states)
    n_smoothed  = 0
    n_runs      = 0

    # Walk through runs
    i = 0
    while i < n:
        if states[i] != standby_name:
            i += 1
            continue

        # Found start of a STANDBY run
        j = i
        while j < n and states[j] == standby_name:
            j += 1
        run_len = j - i  # [i, j) are all STANDBY
        n_runs += 1

        if run_len < min_dwell_rows:
            # Determine replacement label from immediate neighbours
            left_label  = states[i - 1] if i > 0 else None
            right_label = states[j]     if j < n else None

            # Prefer right neighbour (what comes AFTER standby = more stable)
            if right_label is not None and right_label != standby_name:
                replacement = right_label
            elif left_label is not None and left_label != standby_name:
                replacement = left_label
            else:
                replacement = standby_name  # surrounded by STANDBY — leave it

            if replacement != standby_name:
                states[i:j] = replacement
                n_smoothed += run_len

        i = j  # jump past the run we just processed

    df[state_col] = states

    # Recompute state_id if present
    if "state_id" in df.columns:
        unique_states = sorted(df["state_pre_smooth"].unique())  # keep encoder stable
        encoder = {s: idx for idx, s in enumerate(unique_states)}
        df["state_id"] = df[state_col].map(encoder).astype(int)

    removed_pct = n_smoothed / max(len(df), 1) * 100
    print(
        f"\n[lstm_data] STANDBY label smoothing (min_dwell={min_dwell_s:.0f}s, "
        f"{min_dwell_rows} rows):\n"
        f"  Total STANDBY runs found  : {n_runs:,}\n"
        f"  Rows re-labelled          : {n_smoothed:,} ({removed_pct:.1f}% of dataset)\n"
        f"  Original labels preserved : df['state_pre_smooth']"
    )


    # Post-smooth state distribution
    print(f"  Post-smooth state distribution:")
    for state, count in pd.Series(states).value_counts().items():
        print(f"    {state:<12}: {count:>10,} rows  ({count/len(df)*100:.1f}%)")

    return df


# ── STANDBY band feature ──────────────────────────────────────────────────────

def add_standby_band_feature(
    df: pd.DataFrame,
    power_col: str = "active_power",
    pf_col: str = "power_factor",
    standby_power_low_w: float = 1000.0,
    standby_power_high_w: float = 40000.0,
    standby_pf_max: float = 0.35,
) -> pd.DataFrame:
    """
    Add a physics-anchored 'standby_power_band' boolean feature.

    STANDBY on a large industrial machine occupies a specific region
    of the (active_power, power_factor) plane:
      - Active power between low_w and high_w  (non-zero but sub-WORKING)
      - Power factor below standby_pf_max      (no real mechanical load)

    This feature gives the LSTM a direct, physics-derived hint about
    the STANDBY operating zone that is much more stable than the raw
    power value (which has std=38,884W in STANDBY).

    Parameters
    ----------
    df                    : enriched DataFrame.
    power_col             : active power column name.
    pf_col                : power factor column name.
    standby_power_low_w   : lower power bound for STANDBY band (default 1 kW).
    standby_power_high_w  : upper power bound for STANDBY band (default 40 kW).
    standby_pf_max        : PF ceiling for STANDBY (default 0.35).

    Returns
    -------
    DataFrame with added 'standby_power_band' float column (0.0 or 1.0).
    """
    df = df.copy()

    has_power = power_col in df.columns
    has_pf    = pf_col    in df.columns

    if has_power and has_pf:
        band = (
            (df[power_col] >= standby_power_low_w) &
            (df[power_col] <  standby_power_high_w) &
            (df[pf_col]    <  standby_pf_max)
        )
        df["standby_power_band"] = band.astype(np.float32)
        n_in_band = int(band.sum())
        print(
            f"\n[lstm_data] Standby power-band feature:"
            f" {n_in_band:,} rows in STANDBY band "
            f"({n_in_band/len(df)*100:.1f}%)"
            f" [P∈[{standby_power_low_w:.0f}W, {standby_power_high_w:.0f}W] & PF<{standby_pf_max}]"
        )
    elif has_power:
        # PF not available — use power-only band
        band = (
            (df[power_col] >= standby_power_low_w) &
            (df[power_col] <  standby_power_high_w)
        )
        df["standby_power_band"] = band.astype(np.float32)
        print(
            f"[lstm_data] Standby power-band feature (power-only, no PF column):"
            f" {int(band.sum()):,} rows in band."
        )
    else:
        df["standby_power_band"] = 0.0
        print("[lstm_data] WARNING: Cannot compute standby_power_band — neither power nor PF column found.")

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
