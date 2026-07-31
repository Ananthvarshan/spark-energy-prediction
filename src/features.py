"""
============================================================
FEATURE ENGINEERING  --  src/features.py
============================================================

PURPOSE
-------
Phase II / Task 3.  Derive the engineered feature set used by
BOTH the state-detection comparison (Task 4) and the
forecasting baselines (Task 5), so that every model in the
paper is fed exactly the same inputs.

All features are derivable from the five raw IMDELD electrical
columns plus the timestamp -- no extra instrumentation.

FEATURE GROUPS
--------------
Group 1 -- Electrical derived
    power_factor    P / S            motor loading (Fitzgerald et al., Ch. 6-7)
    q_p_ratio       Q / max(P, eps)  magnetising vs useful current
    load_factor     I / I_rated      fraction of rated current drawn
    s_residual      (S - sqrt(P^2+Q^2)) / S   power-triangle closure error

Group 2 -- Rolling statistics (window given in SECONDS)
    p_roll_mean, p_roll_std, p_roll_range
    i_roll_mean, i_roll_std
    pf_roll_mean, pf_roll_std
    p_delta         first difference of active power

Group 3 -- Temporal / schedule
    hour_of_day, day_of_week, is_weekend
    hour_sin, hour_cos, dow_sin, dow_cos
    is_working_hours, is_factory_open

WHY THESE
---------
Group 1 encodes induction-motor physics directly: an unloaded
motor draws mostly magnetising (reactive) current, so it sits
at low power_factor and high q_p_ratio regardless of its
absolute power draw.  This is what separates STANDBY from a
lightly-loaded WORKING state, which power magnitude alone
cannot do -- the central conceptual limitation recorded in
Section 11.1 of the project's own research report.

Group 2 encodes the fact that STANDBY is *quiet*: an idling
motor's power trace has low local variance, while a machine
under real load is continuously modulated by process
throughput.

Group 3 encodes the factory's shift schedule, which is the one
piece of genuinely external operational information available.

GAP SAFETY
----------
Rolling statistics are computed WITHIN contiguous segments.
The pelletizer-I record contains a ~30-day acquisition gap; a
naive rolling window would smear pre-gap values across it.
Every rolling column is therefore grouped by `segment_id`
(see `add_segment_ids`).

HOW TO USE
----------
    from src.features import add_engineered_features, FEATURE_GROUPS

    df = add_engineered_features(
        df,
        factory_tz="America/Sao_Paulo",
        schedule_close_hour=17,
        schedule_open_hour=22,
        roll_window_s=300.0,
    )
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# A sample interval greater than GAP_MULTIPLE x the median marks the start of
# a new contiguous segment.  Matches src/lstm_data.GAP_MULTIPLE.
GAP_MULTIPLE = 10.0

# Rated current is estimated as a high percentile of the non-spike current
# rather than the maximum: the raw maximum is set by measurement spikes
# (19.5% of rows are MAD-flagged on pelletizer-I), which would make
# load_factor meaningless.
RATED_CURRENT_PCTILE = 99.5

# Small positive floor used when dividing by active power, which is ~0 in OFF.
_EPS_W = 1.0    # 1 W


# Feature names grouped for reporting.  Task 3's importance analysis reports
# results per group as well as per feature.
FEATURE_GROUPS: dict[str, list[str]] = {
    "raw_electrical": [
        "active_power", "reactive_power", "apparent_power",
        "current", "voltage",
    ],
    "derived_electrical": [
        "power_factor", "q_p_ratio", "load_factor", "s_residual",
    ],
    "rolling_statistics": [
        "p_roll_mean", "p_roll_std", "p_roll_range",
        "i_roll_mean", "i_roll_std",
        "pf_roll_mean", "pf_roll_std",
        "p_delta",
    ],
    "temporal": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_working_hours", "is_factory_open",
    ],
}


def all_feature_names() -> list[str]:
    """Flat list of every engineered feature, in group order."""
    return [f for group in FEATURE_GROUPS.values() for f in group]


def group_of(feature: str) -> str:
    """Return the group name a feature belongs to ('unknown' if unlisted)."""
    for name, feats in FEATURE_GROUPS.items():
        if feature in feats:
            return name
    return "unknown"


# ── Segment identification ────────────────────────────────────────────────────

def compute_sample_interval_s(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> float:
    """Median inter-sample interval in seconds (robust to gaps and spikes)."""
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    dt = ts.diff().dt.total_seconds().dropna()
    if dt.empty:
        return 1.0
    med = float(dt.median())
    return med if med > 0 else 1.0


def add_segment_ids(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    sample_interval_s: float | None = None,
    gap_multiple: float = GAP_MULTIPLE,
) -> pd.DataFrame:
    """
    Add a `segment_id` column: an integer that increments at every
    acquisition gap.  Rows within one segment are contiguous in time at
    the nominal sampling rate, so rolling windows and sequence models may
    safely operate within a segment but never across one.
    """
    df = df.copy()
    if timestamp_col not in df.columns:
        df["segment_id"] = 0
        return df

    if sample_interval_s is None:
        sample_interval_s = compute_sample_interval_s(df, timestamp_col)

    ts = pd.to_datetime(df[timestamp_col], utc=True)
    dt = ts.diff().dt.total_seconds()
    is_gap = (dt > gap_multiple * sample_interval_s).fillna(False)
    df["segment_id"] = is_gap.cumsum().astype(int)
    return df


# ── Group 1: electrical derived ───────────────────────────────────────────────

def add_derived_electrical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Power factor, Q/P ratio, load factor, and power-triangle residual.

    power_factor  = |P| / S, clipped to [0, 1].  Rows with S == 0 (the
                    machine is OFF, no current flows) get PF = 0, matching
                    the convention used in validate_gmm.py.

    q_p_ratio     = |Q| / max(P, 1 W).  Unbounded above by construction --
                    at P -> 0 with Q > 0 the ratio diverges -- so it is
                    log1p-compressed.  The compressed form is what
                    downstream models see; it is monotone in the raw ratio
                    and therefore preserves the ordering that matters.

    load_factor   = I / I_rated, with I_rated the 99.5th percentile of
                    current (see RATED_CURRENT_PCTILE).

    s_residual    = (S - sqrt(P^2 + Q^2)) / S.  For a correctly measuring
                    three-channel meter this is ~0 by definition; a
                    non-zero value indicates sensor disagreement.  This is
                    Method 3 of the five-method proof, exposed as a feature
                    so that models can learn to distrust bad rows.
    """
    df = df.copy()

    P = df["active_power"].astype(float)
    Q = df["reactive_power"].astype(float) if "reactive_power" in df else None
    S = df["apparent_power"].astype(float) if "apparent_power" in df else None
    I = df["current"].astype(float) if "current" in df else None

    # -- power factor ------------------------------------------------------
    if S is not None:
        df["power_factor"] = (
            P.abs() / S.replace(0, np.nan)
        ).fillna(0.0).clip(0.0, 1.0)
    elif "power_factor" not in df.columns:
        df["power_factor"] = 0.0

    # -- Q/P ratio (log1p-compressed; see docstring) -----------------------
    if Q is not None:
        raw_ratio = Q.abs() / P.clip(lower=_EPS_W)
        df["q_p_ratio"] = np.log1p(raw_ratio)
    else:
        df["q_p_ratio"] = 0.0

    # -- load factor -------------------------------------------------------
    if I is not None:
        if "is_spike" in df.columns:
            clean_I = I[~df["is_spike"].astype(bool)]
        else:
            clean_I = I
        i_rated = float(np.nanpercentile(clean_I, RATED_CURRENT_PCTILE)) if len(clean_I) else 0.0
        df["load_factor"] = (I / i_rated) if i_rated > 0 else 0.0
    else:
        df["load_factor"] = 0.0

    # -- power-triangle residual ------------------------------------------
    if Q is not None and S is not None:
        s_calc = np.sqrt(P.pow(2) + Q.pow(2))
        df["s_residual"] = (
            (S - s_calc) / S.replace(0, np.nan)
        ).fillna(0.0).clip(-1.0, 1.0)
    else:
        df["s_residual"] = 0.0

    return df


# ── Group 2: rolling statistics ───────────────────────────────────────────────

def add_rolling_statistics(
    df: pd.DataFrame,
    roll_window_s: float = 300.0,
    sample_interval_s: float | None = None,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """
    Rolling mean / std / range of power, current, and power factor over a
    trailing window of `roll_window_s` seconds, plus the first difference
    of active power.

    Windows are TRAILING (not centred): a centred window would let a model
    see the future, which invalidates any forecasting result computed on
    top of these features.

    Rolling statistics are computed independently within each
    `segment_id`, so no window spans an acquisition gap.
    """
    df = df.copy()
    if sample_interval_s is None:
        sample_interval_s = compute_sample_interval_s(df)

    win = max(2, int(round(roll_window_s / sample_interval_s)))

    if segment_col not in df.columns:
        df = add_segment_ids(df, sample_interval_s=sample_interval_s)

    grp = df.groupby(segment_col, sort=False)

    def _roll(col: str, how: str) -> pd.Series:
        r = grp[col].rolling(win, min_periods=1)
        out = getattr(r, how)()
        # groupby().rolling() returns a MultiIndex (segment, original_index)
        return out.reset_index(level=0, drop=True).astype(float)

    df["p_roll_mean"] = _roll("active_power", "mean")
    df["p_roll_std"] = _roll("active_power", "std").fillna(0.0)
    p_max = _roll("active_power", "max")
    p_min = _roll("active_power", "min")
    df["p_roll_range"] = (p_max - p_min).fillna(0.0)

    if "current" in df.columns:
        df["i_roll_mean"] = _roll("current", "mean")
        df["i_roll_std"] = _roll("current", "std").fillna(0.0)
    else:
        df["i_roll_mean"] = 0.0
        df["i_roll_std"] = 0.0

    df["pf_roll_mean"] = _roll("power_factor", "mean")
    df["pf_roll_std"] = _roll("power_factor", "std").fillna(0.0)

    df["p_delta"] = grp["active_power"].diff().fillna(0.0).astype(float)

    return df


# ── Group 3: temporal / schedule ──────────────────────────────────────────────

def add_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    factory_tz: str = "America/Sao_Paulo",
    schedule_close_hour: int = 17,
    schedule_open_hour: int = 22,
    working_start_hour: int = 8,
    working_end_hour: int = 17,
) -> pd.DataFrame:
    """
    Cyclical hour-of-day and day-of-week encodings plus schedule flags.

    Hour and weekday are encoded as (sin, cos) pairs rather than integers
    so that hour 23 and hour 0 are adjacent in feature space; a raw integer
    encoding places them maximally far apart.

    `is_factory_open` follows the IMDELD facility's published schedule:
    the plant halts every weekday between `schedule_close_hour` and
    `schedule_open_hour` (a tariff-avoidance shutdown) and does not run at
    weekends.  This is the same definition used in validate_gmm.py's T10
    schedule proof, kept identical so the two agree.
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        df["hour_of_day"] = 0
        df["day_of_week"] = 0
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0
        df["is_weekend"] = False
        df["is_working_hours"] = True
        df["is_factory_open"] = True
        return df

    ts = pd.to_datetime(df[timestamp_col], utc=True)
    ts_local = ts.dt.tz_convert(factory_tz)

    hour_f = ts_local.dt.hour + ts_local.dt.minute / 60.0
    hour_i = ts_local.dt.hour
    dow = ts_local.dt.dayofweek           # 0 = Monday

    df["hour_of_day"] = hour_i.astype(int)
    df["day_of_week"] = dow.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour_f / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_f / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    df["is_weekend"] = dow.isin([5, 6])
    df["is_working_hours"] = (
        (hour_i >= working_start_hour) & (hour_i < working_end_hour)
    )

    if schedule_close_hour < schedule_open_hour:
        in_closed = (hour_i >= schedule_close_hour) & (hour_i < schedule_open_hour)
    else:
        in_closed = (hour_i >= schedule_close_hour) | (hour_i < schedule_open_hour)
    df["is_factory_open"] = (~in_closed) & (~df["is_weekend"])

    return df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def add_engineered_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    factory_tz: str = "America/Sao_Paulo",
    schedule_close_hour: int = 17,
    schedule_open_hour: int = 22,
    roll_window_s: float = 300.0,
    sample_interval_s: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply all three feature groups in dependency order and return a copy.

    Order matters: derived electrical features must exist before the
    rolling statistics that summarise them, and segment ids must exist
    before any rolling window is taken.
    """
    if sample_interval_s is None:
        sample_interval_s = compute_sample_interval_s(df, timestamp_col)

    df = add_segment_ids(df, timestamp_col, sample_interval_s)
    n_seg = int(df["segment_id"].nunique())

    df = add_derived_electrical(df)
    df = add_rolling_statistics(
        df,
        roll_window_s=roll_window_s,
        sample_interval_s=sample_interval_s,
        segment_col="segment_id",
    )
    df = add_temporal_features(
        df,
        timestamp_col=timestamp_col,
        factory_tz=factory_tz,
        schedule_close_hour=schedule_close_hour,
        schedule_open_hour=schedule_open_hour,
    )

    # Boolean -> float so every downstream model sees a numeric matrix.
    for col in ("is_weekend", "is_working_hours", "is_factory_open"):
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Guard against inf leaking in from any division.
    feats = [c for c in all_feature_names() if c in df.columns]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if verbose:
        win = max(2, int(round(roll_window_s / sample_interval_s)))
        print(
            f"[features] sample interval {sample_interval_s:.3f}s | "
            f"{n_seg} contiguous segment(s) | "
            f"rolling window {roll_window_s:.0f}s ({win} rows)"
        )
        for name, cols in FEATURE_GROUPS.items():
            present = [c for c in cols if c in df.columns]
            print(f"[features]   {name:<20} {len(present):>2} features: "
                  f"{', '.join(present)}")

    return df
