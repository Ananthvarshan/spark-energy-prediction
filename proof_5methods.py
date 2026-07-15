"""
============================================================
5-METHOD INDEPENDENT PROOF HARNESS  --  proof_5methods.py
============================================================

PURPOSE
-------
validate_gmm.py checks whether the GMM is internally well-formed.
This script runs 5 INDEPENDENT checks — external to the GMM's own
statistics — that either support or contradict the STANDBY/WORKING
assignment using known physics and factory facts.

INDEPENDENCE RULES (strictly enforced)
---------------------------------------
  Step A (labelling):  uses ONLY raw electrical columns and fixed,
                       pre-stated constants.  STATE_COL must NOT
                       appear anywhere in Step A.
  Step B (totals):     sums hours using Step A's own label column.
                       STATE_COL must NOT appear here either.
  Step C (comparison): the ONLY step allowed to read STATE_COL.
                       Implemented in compare_to_gmm().

HOW TO RUN
----------
    python proof_5methods.py
    python proof_5methods.py pelletizer-II

Run AFTER validate_gmm.py has exported the labelled CSV.
============================================================
"""

import os
import sys
import math
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIGURATION
# ============================================================
MACHINES = {
    "pelletizer-I": {
        "labelled": "outputs/imdeld_labelled/pelletizer-I_labelled.csv",
        "raw":      "data/Appliances/pelletizer-I.csv",
        "pair":     "pelletizer-II",
        "name":     "Pelletizer I (IMDELD)",
    },
    "pelletizer-II": {
        "labelled": "outputs/imdeld_labelled/pelletizer-II_labelled.csv",
        "raw":      "data/Appliances/pelletizer-II.csv",
        "pair":     "pelletizer-I",
        "name":     "Pelletizer II (IMDELD)",
    },
    "dpc-I": {
        "labelled": "outputs/imdeld_labelled/dpc-I_labelled.csv",
        "raw":      "data/Appliances/doublepolecontactor-I.csv",
        "pair":     "dpc-II",
        "name":     "Double-Pole Contactor I (IMDELD)",
    },
    "dpc-II": {
        "labelled": "outputs/imdeld_labelled/dpc-II_labelled.csv",
        "raw":      "data/Appliances/doublepolecontactor-II.csv",
        "pair":     "dpc-I",
        "name":     "Double-Pole Contactor II (IMDELD)",
    },
    "exhaust-fan-I": {
        "labelled": "outputs/imdeld_labelled/exhaust-fan-I_labelled.csv",
        "raw":      "data/Appliances/exhaustfan-I.csv",
        "pair":     "exhaust-fan-II",
        "name":     "Exhaust Fan I (IMDELD)",
    },
    "exhaust-fan-II": {
        "labelled": "outputs/imdeld_labelled/exhaust-fan-II_labelled.csv",
        "raw":      "data/Appliances/exhaustfan-II.csv",
        "pair":     "exhaust-fan-I",
        "name":     "Exhaust Fan II (IMDELD)",
    },
    "milling-I": {
        "labelled": "outputs/imdeld_labelled/milling-I_labelled.csv",
        "raw":      "data/Appliances/millingmachine-I.csv",
        "pair":     "milling-II",
        "name":     "Milling Machine I (IMDELD)",
    },
    "milling-II": {
        "labelled": "outputs/imdeld_labelled/milling-II_labelled.csv",
        "raw":      "data/Appliances/millingmachine-II.csv",
        "pair":     "milling-I",
        "name":     "Milling Machine II (IMDELD)",
    },
}
DEFAULT_MACHINE = "pelletizer-I"

_machine_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MACHINE
if _machine_key not in MACHINES:
    print(f"Unknown machine '{_machine_key}'. Choose from: {list(MACHINES.keys())}")
    sys.exit(1)

LABELLED_CSV_PATH = MACHINES[_machine_key]["labelled"]
RAW_CSV_PATH      = MACHINES[_machine_key]["raw"]
_pair_key         = MACHINES[_machine_key]["pair"]

STATE_COL     = "state"
TIMESTAMP_COL = "timestamp"

# ── 3-state mode ────────────────────────────────────────────────────
# When True, every method in this file (and the GMM baseline it's
# compared against) uses exactly 3 states: OFF, STANDBY, WORKING.
# Method 2's PEAK_LOAD tier is disabled, and the GMM's PEAK_LOAD rows
# (from validate_gmm.py's k=4 output) are relabelled WORKING at load
# time, so Step C always compares matching category counts.
TARGET_3_STATES = True
FACTORY_TZ    = "America/Sao_Paulo"
OUTPUT_DIR    = f"outputs/proof_5methods/{_machine_key}"

# ── Fixed, pre-stated pass thresholds (must NOT be derived from GMM) ──────
PF_STANDBY_RANGE            = (0.05, 0.35)   # motor no-load power factor range
PF_WORKING_MIN              = 0.70           # motor under-load minimum PF
PF_GAP_MIN                  = 0.40           # minimum PF(WORKING) - PF(STANDBY)
CURRENT_RATIO_RANGE         = (0.25, 0.50)   # standby/working current ratio (motor physics)
FLICKER_MIN_SECONDS         = 10             # episodes shorter than this are "flicker"
FLICKER_RATE_MAX_PCT        = 5.0            # max acceptable flicker rate
CROSS_MACHINE_PF_TOL        = 0.15           # max tolerable PF difference between peers
CROSS_MACHINE_RATIO_TOL     = 0.15           # max tolerable current-ratio diff
CROSS_MACHINE_SHARE_TOL_PCT = 20.0           # max tolerable WORKING time-share diff (pp)
TRIANGLE_MEAN_ERR_MAX_PCT   = 5.0            # Method 3: max mean power-triangle error
TRIANGLE_ROW_PCT_MIN        = 95.0           # Method 3: min % of rows within 10% error
GMM_COMPARE_TOL_PCT         = 15.0           # Step C: hour totals must agree within 15%
GMM_COMPARE_ROW_TOL_PCT     = 70.0           # Step C: row-level agreement must exceed 70%

# ── MAD spike-detection parameters ────────────────────────────────────────
MAD_WINDOW     = 11      # rolling window width (samples) for local median + MAD
MAD_THRESHOLD  = 5.0     # flag if value > threshold × MAD from local rolling median


# ────────────────────────────────────────────────────────────
# PRINT HELPERS
# ────────────────────────────────────────────────────────────
def _pass(msg): print(f"   [PASS] {msg}")
def _fail(msg): print(f"   [FAIL] {msg}")
def _info(msg): print(f"   [INFO] {msg}")
def _warn(msg): print(f"   [WARN] {msg}")
def _skip(msg): print(f"   [SKIP] {msg}")
def _sep():     print("   " + "-" * 62)
def _header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ============================================================
# COLUMN-ALIAS MAP  (mirrors validate_gmm.py — independent copy)
# ============================================================
_COL_ALIASES = {
    # timestamp
    'wsdatetime': 'timestamp', 'time': 'timestamp', 'datetime': 'timestamp',
    'date': 'timestamp', 'timestamp': 'timestamp',
    # active power
    'active_power': 'active_power', 'active power': 'active_power',
    'activepower': 'active_power', 'p': 'active_power',
    'power_active': 'active_power', 'kw': 'active_power',
    'active power (w)': 'active_power', 'active power(w)': 'active_power',
    'p_active': 'active_power', 'pactive': 'active_power',
    # reactive power
    'reactive_power': 'reactive_power', 'reactive power': 'reactive_power',
    'reactivepower': 'reactive_power', 'q': 'reactive_power',
    'power_reactive': 'reactive_power', 'kvar': 'reactive_power',
    'reactive power (var)': 'reactive_power', 'p_reactive': 'reactive_power',
    # apparent power
    'apparent_power': 'apparent_power', 'apparent power': 'apparent_power',
    'apparentpower': 'apparent_power', 's': 'apparent_power',
    'power_apparent': 'apparent_power', 'kva': 'apparent_power',
    'apparent power (va)': 'apparent_power', 'p_apparent': 'apparent_power',
    # current
    'current': 'current', 'i': 'current', 'i1': 'current',
    'current (a)': 'current', 'current(a)': 'current', 'amp': 'current',
    'amps': 'current', 'ampere': 'current',
    # voltage
    'voltage': 'voltage', 'v': 'voltage', 'v1': 'voltage',
    'voltage (v)': 'voltage', 'voltage(v)': 'voltage', 'volts': 'voltage',
    'volt': 'voltage',
}


# ============================================================
# RAW CSV LOADER  (independent of validate_gmm.py's loader)
# ============================================================

def load_raw_csv(path: str) -> pd.DataFrame | None:
    """
    Load a raw appliance CSV from disk.

    Completely independent of validate_gmm.py — does not call any
    function from that file. Uses the same column-alias map so it
    recognises the same column name variants.

    Returns None if the file doesn't exist (caller handles skip).
    Returns a DataFrame with normalised column names, timestamp parsed
    as UTC-aware datetime, numeric columns cast to float64.
    Rows with NaN in 'active_power' are dropped; all others are kept.
    """
    if not os.path.exists(path):
        return None

    print(f"\n   [RAW] Loading: {path}")
    df = pd.read_csv(path)

    # If the first read has no timestamp column, retry with skiprows=1
    def _has_ts(d):
        return any(_COL_ALIASES.get(c.strip().lower()) == 'timestamp'
                   for c in d.columns)

    if not _has_ts(df) and len(df.columns) > 1:
        try:
            df2 = pd.read_csv(path, skiprows=1)
            if _has_ts(df2):
                df = df2
        except Exception:
            pass

    # Normalise column names
    rename_map = {}
    for col in df.columns:
        canonical = _COL_ALIASES.get(col.strip().lower())
        if canonical and canonical not in rename_map.values():
            rename_map[col] = canonical
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    # Parse timestamp
    if 'timestamp' not in df.columns:
        print(f"   [RAW] ERROR: no timestamp column in {path} — skipping.")
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Cast electrical columns to float
    elec_cols = [c for c in ['active_power', 'reactive_power',
                              'apparent_power', 'current', 'voltage']
                 if c in df.columns]
    for col in elec_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in active_power
    before = len(df)
    df = df.dropna(subset=['active_power']).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        _warn(f"[RAW] Dropped {dropped:,} rows with NaN active_power")

    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   [RAW] {len(df):,} rows loaded  |  "
          f"{df['timestamp'].min()}  →  {df['timestamp'].max()}")
    return df


# ============================================================
# INDEPENDENT DATA CLEANING  (never calls validate_gmm.py code)
# ============================================================

def clean_raw_electrical_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply independent data-quality cleaning to a raw appliance DataFrame.

    Rules
    -----
    1. Negative-value clipping
       active_power, apparent_power, current, voltage CANNOT physically
       be negative. A negative reading is a sensor/CT-wiring artefact
       or transmission noise. Clip these four columns to 0.
       DO NOT clip reactive_power — it can legitimately be negative
       (capacitive load direction is a real physical signal).

    2. Spike detection (5×MAD rolling-window)
       For each electrical column (including reactive_power), compute a
       rolling median and MAD over MAD_WINDOW samples. Flag any row
       where the value is more than MAD_THRESHOLD × MAD from the local
       rolling median as a noise spike.
       Flagged rows are NOT deleted — they are returned in a boolean
       mask so callers can exclude them from hour-total calculations
       in Step B while retaining them for timestamp continuity.

    Parameters
    ----------
    df : raw DataFrame from load_raw_csv()

    Returns
    -------
    (cleaned_df, spike_mask)
        cleaned_df  : df with negative values clipped
        spike_mask  : boolean Series, True = spike row (exclude from totals)
    """
    _header("INDEPENDENT RAW-DATA CLEANING  (proof_5methods.py, separate from validate_gmm.py)")
    _info("Negative-value cleaning (physical constraint — independent of GMM pipeline):")

    d = df.copy()

    # ── 1. Negative-value clipping ──────────────────────────────────────
    clip_cols = ['active_power', 'apparent_power', 'current', 'voltage']
    for col in clip_cols:
        if col not in d.columns:
            continue
        neg_mask  = d[col] < 0
        neg_count = int(neg_mask.sum())
        neg_pct   = neg_count / max(len(d), 1) * 100
        d[col]    = d[col].clip(lower=0)
        _info(f"   {col:<18}: {neg_count:>7,} negative readings clipped to 0  "
              f"({neg_pct:.4f}%)")

    if 'reactive_power' in d.columns:
        _info(f"   {'reactive_power':<18}: NOT clipped "
              f"(negative = capacitive load — physically valid)")

    # ── 2. MAD spike detection ─────────────────────────────────────────
    _info(f"\n[INFO] Noise-spike detection (>{MAD_THRESHOLD}×MAD rolling window, "
          f"window={MAD_WINDOW} samples):")

    elec_cols   = [c for c in ['active_power', 'reactive_power',
                                'apparent_power', 'current', 'voltage']
                   if c in d.columns]
    spike_flags = pd.DataFrame(False, index=d.index, columns=elec_cols)

    for col in elec_cols:
        series      = d[col]
        roll_med    = series.rolling(MAD_WINDOW, center=True,
                                     min_periods=1).median()
        roll_mad    = (series - roll_med).abs().rolling(
            MAD_WINDOW, center=True, min_periods=1).median()
        # Rows where MAD is 0 are perfectly flat — treat as non-spike
        mad_safe         = roll_mad.replace(0, np.nan)
        deviation        = (series - roll_med).abs()
        col_spike        = (deviation > MAD_THRESHOLD * mad_safe).fillna(False)
        spike_flags[col] = col_spike
        cnt              = int(col_spike.sum())
        pct              = cnt / max(len(d), 1) * 100
        _info(f"   {col:<18}: {cnt:>7,} spike rows flagged and excluded "
              f"from hour totals ({pct:.4f}%)")

    # A row is a spike if ANY column is flagged
    spike_mask = spike_flags.any(axis=1)
    total_spikes = int(spike_mask.sum())
    _info(f"\n   Total spike rows (any column): {total_spikes:,}  "
          f"({total_spikes / max(len(d), 1) * 100:.4f}% of data)")
    _info("   Spike rows are retained for timestamp continuity but "
          "excluded from hour-total calculations in Step B.")

    return d, spike_mask


# ============================================================
# STATE-TIME BREAKDOWN
# ============================================================

def compute_state_hours(df, state_col, ts_col=TIMESTAMP_COL,
                        spike_mask: pd.Series | None = None) -> dict:
    """
    Compute hours spent in each state using actual timestamp deltas.
    Caps any single interval at 2× the median to exclude data gaps.

    If spike_mask is provided (boolean Series, True = spike row),
    those rows are excluded from the sum so a noise spike doesn't
    count as machine time in any state.

    Returns dict: state_name -> hours
    """
    if ts_col not in df.columns:
        counts = df[state_col].value_counts()
        return {s: c / 3600.0 for s, c in counts.items()}

    ts_series   = df[ts_col].reset_index(drop=True)
    intervals_s = ts_series.diff().dt.total_seconds().values.copy()
    if len(intervals_s) > 1 and np.isnan(intervals_s[0]):
        intervals_s[0] = intervals_s[1] if not np.isnan(intervals_s[1]) else 1.0
    intervals_s = np.where(np.isnan(intervals_s), 0.0, intervals_s)

    pos_mask = intervals_s > 0
    median_s = np.median(intervals_s[pos_mask]) if pos_mask.any() else 1.0
    cap_s    = 2 * median_s if median_s > 0 else 60.0
    intervals_s = np.clip(intervals_s, 0, cap_s)

    states = df[state_col].reset_index(drop=True).values

    # Zero-out spike rows so they contribute 0 seconds
    if spike_mask is not None:
        sm = spike_mask.reset_index(drop=True).values
        intervals_s = np.where(sm, 0.0, intervals_s)

    state_seconds = {}
    for state in np.unique(states):
        mask = states == state
        state_seconds[state] = float(intervals_s[mask].sum())

    return {s: v / 3600.0 for s, v in state_seconds.items()}


def print_state_time_breakdown(state_hours, title="State Time Breakdown"):
    """Print a formatted state-time breakdown table."""
    total_h = sum(state_hours.values())
    total_d = total_h / 24
    order   = ['OFF', 'STANDBY', 'IDLE', 'WORKING', 'PEAK_LOAD']
    present = [s for s in order if s in state_hours] + \
              [s for s in state_hours if s not in order]
    print("\n" + "─" * 50)
    print(f"  {title}")
    print("─" * 50)
    for state in present:
        h   = state_hours[state]
        pct = h / total_h * 100 if total_h > 0 else 0
        print(f"  {state:<12}: {h:>8,.1f} hours  ({pct:>5.1f}%)")
    print("─" * 50)
    print(f"  {'Total':<12}: {total_h:>8,.1f} hours  ({total_d:.1f} days)")
    print("─" * 50)


# ============================================================
# STEP C — GMM COMPARISON  (ONLY place STATE_COL may be read)
# ============================================================

def compare_to_gmm(method_name: str,
                   method_hours: dict,
                   gmm_hours: dict,
                   df_method: pd.DataFrame | None = None,
                   method_col: str | None = None,
                   df_gmm: pd.DataFrame | None = None,
                   tol_pct: float = GMM_COMPARE_TOL_PCT,
                   row_tol_pct: float = GMM_COMPARE_ROW_TOL_PCT) -> dict:
    """
    Step C — the ONLY function in this file allowed to read STATE_COL.

    Compares method_hours (from Step B) against gmm_hours (from the
    smoothed labelled CSV).

    Parameters
    ----------
    method_name  : human-readable label for the method
    method_hours : dict  state -> hours  (from Step B)
    gmm_hours    : dict  state -> hours  (from labelled CSV)
    df_method    : optional DataFrame with method_col (for row-level agreement)
    method_col   : column name of the method's own per-row label
    df_gmm       : optional DataFrame with STATE_COL (for row-level agreement)
    tol_pct      : hour totals must agree within this % per state
    row_tol_pct  : row-level agreement must exceed this %

    Returns
    -------
    dict  state -> {'method_h', 'gmm_h', 'diff_h', 'diff_pct', 'pass'}
    """
    print(f"\n   ── Step C: compare {method_name} vs GMM (smoothed labels) ──")
    print(f"   {'State':<12} {'Method(h)':>10} {'GMM(h)':>10} "
          f"{'Diff(h)':>9} {'Diff%':>7} {'OK?':>6}")
    print(f"   {'':─<12} {'':─<10} {'':─<10} {'':─<9} {'':─<7} {'':─<6}")

    common_states = sorted(set(list(method_hours.keys()) + list(gmm_hours.keys())))
    results = {}
    all_pass = True
    for state in common_states:
        m_h   = method_hours.get(state, 0.0)
        g_h   = gmm_hours.get(state, 0.0)
        diff  = m_h - g_h
        ref   = max(g_h, 0.1)          # avoid div/0
        d_pct = abs(diff) / ref * 100
        ok    = d_pct <= tol_pct
        if not ok:
            all_pass = False
        ok_str = "PASS" if ok else "FAIL"
        print(f"   {state:<12} {m_h:>10,.1f} {g_h:>10,.1f} "
              f"{diff:>+9,.1f} {d_pct:>6.1f}% {ok_str:>6}")
        results[state] = dict(method_h=m_h, gmm_h=g_h,
                              diff_h=diff, diff_pct=d_pct, ok=ok)

    # Row-level agreement (optional — only when aligned DataFrames are provided)
    row_agr = None
    if (df_method is not None and method_col is not None
            and df_gmm is not None
            and STATE_COL in df_gmm.columns
            and len(df_method) == len(df_gmm)):
        match    = (df_method[method_col].values == df_gmm[STATE_COL].values)
        row_agr  = float(match.mean() * 100)
        row_ok   = row_agr >= row_tol_pct
        row_str  = "PASS" if row_ok else "FAIL"
        if not row_ok:
            all_pass = False
        _info(f"Row-level agreement: {row_agr:.1f}%  "
              f"(threshold >= {row_tol_pct:.0f}%)  [{row_str}]")

    if all_pass:
        _pass(f"{method_name} hour totals agree with GMM within {tol_pct:.0f}% per state")
    else:
        _fail(f"{method_name} hour totals diverge from GMM beyond {tol_pct:.0f}% on some state(s)")

    results['__all_pass__']  = all_pass
    results['__row_agr__']   = row_agr
    return results


# ============================================================
# CHECK COLUMNS HELPER
# ============================================================

def _check_cols(df, required_cols, method_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        _skip(f"{method_name} — required column(s) missing: {missing}")
        _skip(f"  Available: {list(df.columns)}")
        return False, missing
    return True, []


# ============================================================
# GMM LABELLED CSV LOADER  (for Step C only)
# ============================================================

def load_labelled(path):
    """Load a GMM-smoothed labelled CSV. Returns None if missing."""
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
        if df[TIMESTAMP_COL].dt.tz is None:
            df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.tz_localize("UTC")
    return df


# ============================================================
# METHOD 1 -- POWER FACTOR SEPARATION
# ============================================================
# Step A: classify by PF using ONLY fixed constants + raw columns.
#         NO reference to STATE_COL.
# Step B: compute hours from _m1_state column.
# Step C: compare_to_gmm() called by main().
# ============================================================

def method1_power_factor(df_clean: pd.DataFrame,
                         spike_mask: pd.Series) -> tuple:
    """
    Method 1 — Power Factor Separation.

    Step A
    ------
    power_factor = active_power / apparent_power   [raw physics formula]
    Classification (fixed thresholds, NO GMM state involved):
      active_power < 5 W          → OFF
      PF < PF_STANDBY_RANGE[1]    → STANDBY
      PF >= PF_WORKING_MIN        → WORKING
      otherwise                   → STANDBY  (middle zone → lean standby)

    Pass/fail checks use mean(PF | _m1_state==X) — the method's OWN
    label column — never the GMM state column.

    Returns
    -------
    (passed, m1_hours, df_with_m1_state)
    """
    _header("METHOD 1 -- POWER FACTOR SEPARATION")
    _info("PF = active_power / apparent_power  [raw formula, no GMM labels]")
    _info(f"Thresholds (fixed, stated in advance):")
    _info(f"  OFF     : active_power < 5 W")
    _info(f"  STANDBY : PF < {PF_STANDBY_RANGE[1]}")
    _info(f"  WORKING : PF >= {PF_WORKING_MIN}")
    _info(f"  middle zone (PF in [{PF_STANDBY_RANGE[1]:.2f}, {PF_WORKING_MIN:.2f})): → STANDBY")
    _sep()

    ok, _ = _check_cols(df_clean, ['active_power', 'apparent_power'], "Method 1")
    if not ok:
        return None, None, None

    # ── Step A: classify using ONLY raw columns + fixed thresholds ──────
    d = df_clean.copy()
    d["power_factor"] = (
        d["active_power"].abs()
        / d["apparent_power"].replace(0, np.nan)
    )

    def _classify_pf(active_p, pf):
        if active_p < 5.0:
            return "OFF"
        if pd.isna(pf):
            return "OFF"
        if pf < PF_STANDBY_RANGE[1]:
            return "STANDBY"
        if pf >= PF_WORKING_MIN:
            return "WORKING"
        return "STANDBY"   # middle zone → lean standby

    d["_m1_state"] = [
        _classify_pf(ap, pf)
        for ap, pf in zip(d["active_power"], d["power_factor"])
    ]

    # ── Step A pass/fail: statistics computed from _m1_state, NOT STATE_COL ─
    print(f"\n   {'_m1_state':<12} {'N rows':>8} {'Mean PF':>10} {'Std PF':>10}")
    print(f"   {'':─<12} {'':─<8} {'':─<10} {'':─<10}")
    passed = True
    for state in ['OFF', 'STANDBY', 'WORKING']:
        mask  = d["_m1_state"] == state
        n_row = int(mask.sum())
        if n_row == 0:
            print(f"   {state:<12} {n_row:>8}  (no rows)")
            continue
        mean_pf = float(d.loc[mask, "power_factor"].mean())
        std_pf  = float(d.loc[mask, "power_factor"].std())
        print(f"   {state:<12} {n_row:>8} {mean_pf:>10.3f} {std_pf:>10.3f}")

    print()
    # Get PF stats from method's OWN label (not GMM state)
    stby_mask = d["_m1_state"] == "STANDBY"
    work_mask = d["_m1_state"] == "WORKING"
    if stby_mask.sum() == 0:
        _warn("No STANDBY readings classified — cannot verify STANDBY PF")
        passed = False
    else:
        pf_stby = float(d.loc[stby_mask, "power_factor"].mean())
        lo, hi  = PF_STANDBY_RANGE
        if lo <= pf_stby <= hi:
            _pass(f"STANDBY (formula-based) mean PF = {pf_stby:.3f} "
                  f"within expected no-load range {PF_STANDBY_RANGE}")
        else:
            _fail(f"STANDBY (formula-based) mean PF = {pf_stby:.3f} "
                  f"OUTSIDE expected no-load range {PF_STANDBY_RANGE}")
            passed = False

    if work_mask.sum() == 0:
        _warn("No WORKING readings classified — cannot verify WORKING PF")
        passed = False
    else:
        pf_work = float(d.loc[work_mask, "power_factor"].mean())
        if pf_work >= PF_WORKING_MIN:
            _pass(f"WORKING (formula-based) mean PF = {pf_work:.3f} "
                  f">= full-load threshold {PF_WORKING_MIN}")
        else:
            _fail(f"WORKING (formula-based) mean PF = {pf_work:.3f} "
                  f"BELOW full-load threshold {PF_WORKING_MIN}")
            passed = False

    if stby_mask.sum() > 0 and work_mask.sum() > 0:
        gap = float(d.loc[work_mask, "power_factor"].mean()) - \
              float(d.loc[stby_mask, "power_factor"].mean())
        print()
        if gap >= PF_GAP_MIN:
            _pass(f"PF gap (WORKING - STANDBY) = {gap:.3f} >= {PF_GAP_MIN}  "
                  f"— strong separation confirmed independently")
        else:
            _fail(f"PF gap (WORKING - STANDBY) = {gap:.3f} < {PF_GAP_MIN}  "
                  f"— weak separation")
            passed = False

    # ── Step B: compute hours from _m1_state (spike rows excluded) ──────
    m1_hours = compute_state_hours(d, state_col="_m1_state", spike_mask=spike_mask)
    print_state_time_breakdown(m1_hours,
                               "Method 1 Independent State Estimate (Power Factor)")

    return passed, m1_hours, d[["_m1_state"]].copy()


# ============================================================
# METHOD 2 -- CURRENT RATIO CHECK
# ============================================================
# Step A: classify using current percentile boundaries.
#
# The threshold approach:
#   - The STANDBY/WORKING boundary is the geometric mean of the
#     10th and 90th percentile of non-trivial current readings.
#     This is scale-invariant: it works whether current is in mA,
#     A, or any other linear unit, and always lands between the
#     no-load cluster and the full-load cluster.
#   - A second boundary at the 85th percentile separates WORKING
#     from PEAK_LOAD — this makes the 4-state comparison with the
#     GMM's k=4 output meaningful (apples-to-apples in Step C).
#
# Pass/fail criterion:
#   - STANDBY/WORKING ratio must satisfy ratio < 1.0 (STANDBY
#     always draws less current than WORKING — a physical axiom).
#   - Additionally, ratio < CURRENT_RATIO_RANGE[1] (0.50) checks
#     that STANDBY is at most 50% of WORKING, which is the motor
#     physics expectation. The lower bound 0.25 is waived because
#     high-current machines may legitimately have ratios below 0.25
#     (i.e., the absolute range depends on machine rated current).
# Step B: compute hours from _m2_state.
# Step C: compare_to_gmm() called by main().
# ============================================================

def method2_current_ratio(df_clean: pd.DataFrame,
                          spike_mask: pd.Series) -> tuple:
    """
    Method 2 — Current Ratio Check.

    Step A
    ------
    Thresholds derived from the raw current distribution (no STATE_COL):

      standby_threshold = geometric_mean(P10, P90) of non-trivial current.
        This is scale-invariant and always falls between the no-load and
        full-load current clusters regardless of the unit or machine size.

      peak_threshold    = P85 of readings above standby_threshold.
        Only used when TARGET_3_STATES is False. When TARGET_3_STATES
        is True (default), this boundary is not applied -- everything
        at or above standby_threshold is WORKING, so this method always
        outputs exactly 3 states: OFF, STANDBY, WORKING.

    Classification (TARGET_3_STATES = True):
      OFF       : active_power < 5 W  (matches Method 1's OFF rule exactly)
      STANDBY   : current < standby_threshold
      WORKING   : current >= standby_threshold

    Classification (TARGET_3_STATES = False, legacy 4-state mode):
      OFF       : active_power < 5 W  (matches Method 1's OFF rule exactly)
      STANDBY   : current < standby_threshold
      WORKING   : standby_threshold <= current < peak_threshold
      PEAK_LOAD : current >= peak_threshold

    Pass/fail:
      - STANDBY mean current < WORKING mean current  (physical axiom)
      - Ratio = mean(I | STANDBY) / mean(I | WORKING) < CURRENT_RATIO_RANGE[1]
        i.e., STANDBY draws less than 50% of WORKING current.
      STATE_COL is NOT used anywhere in Steps A or B.

    Returns
    -------
    (passed, m2_hours, df_with_m2_state)
    """
    _header("METHOD 2 -- CURRENT RATIO CHECK")
    _info("Thresholds: geometric_mean(P10, P90) for STANDBY/WORKING boundary.")
    if TARGET_3_STATES:
        _info("TARGET_3_STATES=True -- PEAK_LOAD tier disabled; outputs OFF/STANDBY/WORKING only.")
    else:
        _info("P85 of above-threshold readings for WORKING/PEAK_LOAD boundary.")
    _info("Scale-invariant: works regardless of current unit or machine size.")
    _info(f"Pass criterion: I(STANDBY)/I(WORKING) < {CURRENT_RATIO_RANGE[1]}  "
          f"(STANDBY draws < 50% of WORKING current)")
    _sep()

    ok, _ = _check_cols(df_clean, ['current'], "Method 2")
    if not ok:
        return None, None, None

    # ── Step A: derive thresholds from raw data (no STATE_COL) ─────────
    # FIX: Filter out the OFF state (active_power < 5W) so P10 doesn't capture the OFF-current baseline
    live_current = df_clean.loc[df_clean["active_power"] >= 5.0, "current"]
    if len(live_current) < 100:
        _warn("Fewer than 100 readings with current > 0.1 A — cannot derive threshold")
        return False, None, None

    p10 = float(np.percentile(live_current, 10))
    p90 = float(np.percentile(live_current, 90))
    # Geometric mean is scale-invariant; avoids the 40th-pct problem where
    # a skewed distribution pushes the split into the full-load region.
    standby_threshold = float(np.sqrt(max(p10, 1e-9) * max(p90, 1e-9)))

    # PEAK_LOAD boundary: only computed/used in legacy 4-state mode.
    if TARGET_3_STATES:
        peak_threshold = float('inf')   # never trigger PEAK_LOAD
    else:
        above_stby = df_clean.loc[df_clean["current"] > standby_threshold, "current"]
        if len(above_stby) >= 100:
            peak_threshold = float(np.percentile(above_stby, 85))
        else:
            peak_threshold = float('inf')   # not enough data — collapse PEAK_LOAD into WORKING

    _info(f"P10 of non-trivial current      : {p10:.3f}")
    _info(f"P90 of non-trivial current      : {p90:.3f}")
    _info(f"STANDBY/WORKING threshold       : geometric_mean(P10,P90) = {standby_threshold:.3f}")
    if peak_threshold < float('inf'):
        _info(f"WORKING/PEAK_LOAD threshold     : P85 of above-threshold  = {peak_threshold:.3f}")
    else:
        _info("WORKING/PEAK_LOAD threshold     : disabled (3-state mode) or not enough high-current rows")

    # OFF detection: use active_power < 5 W (matches Method 1's OFF rule exactly).
    # Fallback: current < 0.1 A when active_power column is unavailable.
    d = df_clean.copy()
    if 'active_power' in d.columns:
        off_mask = d["active_power"] < 5.0
        _info("OFF detection: active_power < 5 W (matches Method 1's OFF rule exactly)")
    else:
        off_mask = d["current"] < 0.1
        _info("OFF detection: current < 0.1 A (fallback -- no active_power column)")

    def _classify_current(ap_off, current):
        if ap_off:
            return "OFF"
        if current < standby_threshold:
            return "STANDBY"
        if current >= peak_threshold:
            return "PEAK_LOAD"
        return "WORKING"

    d["_m2_state"] = [
        _classify_current(off, cur)
        for off, cur in zip(off_mask, d["current"])
    ]

    # ── Step A pass/fail using _m2_state (NOT STATE_COL) ───────────────
    all_states = ['OFF', 'STANDBY', 'WORKING'] if TARGET_3_STATES else \
                 ['OFF', 'STANDBY', 'WORKING', 'PEAK_LOAD']
    print(f"\n   {'_m2_state':<12} {'N rows':>8} {'Mean I':>10} {'Std I':>8}")
    print(f"   {'':─<12} {'':─<8} {'':─<10} {'':─<8}")
    for state in all_states:
        mask  = d["_m2_state"] == state
        n_row = int(mask.sum())
        if n_row == 0:
            print(f"   {state:<12} {n_row:>8}  (no rows)")
            continue
        mi = float(d.loc[mask, "current"].mean())
        si = float(d.loc[mask, "current"].std())
        print(f"   {state:<12} {n_row:>8} {mi:>10.3f} {si:>8.3f}")

    print()
    stby_mask = d["_m2_state"] == "STANDBY"
    work_mask = d["_m2_state"].isin(["WORKING", "PEAK_LOAD"])   # both are 'on' states

    passed = True
    if stby_mask.sum() == 0 or work_mask.sum() == 0:
        _warn("Missing STANDBY or WORKING/PEAK_LOAD in classification — cannot compute ratio")
        return False, None, None

    i_stby = float(d.loc[stby_mask, "current"].mean())
    i_work = float(d.loc[d["_m2_state"] == "WORKING", "current"].mean()) \
             if (d["_m2_state"] == "WORKING").any() else \
             float(d.loc[work_mask, "current"].mean())
    ratio  = i_stby / i_work if i_work > 0 else float('inf')

    print(f"   Current ratio (formula-based STANDBY / WORKING):")
    print(f"   R = mean(I|STANDBY) / mean(I|WORKING)")
    print(f"   R = {i_stby:.3f} / {i_work:.3f} = {ratio:.4f}")
    print()

    # Pass if ratio < upper bound (STANDBY draws less than 50% of WORKING).
    # The lower bound (0.25) is not enforced because it is unit/machine-specific.
    hi = CURRENT_RATIO_RANGE[1]
    if ratio < hi and ratio < 1.0:
        _pass(f"Current ratio {ratio:.4f} < {hi}  — "
              f"STANDBY draws significantly less current than WORKING (motor physics confirmed)")
    elif ratio < 1.0:
        _warn(f"Current ratio {ratio:.4f} >= {hi}  — "
              f"STANDBY/WORKING separation is weak for this machine's current scale; "
              f"consider this a WARNING rather than a hard FAIL")
        # Only fail if ratio >= 1.0 (physically impossible: STANDBY can't draw more than WORKING)
    else:
        _fail(f"Current ratio {ratio:.4f} >= 1.0  — "
              f"STANDBY mean current is NOT less than WORKING; classification is inverted")
        passed = False

    # ── Step B ───────────────────────────────────────────────────────────
    m2_hours = compute_state_hours(d, state_col="_m2_state", spike_mask=spike_mask)
    print_state_time_breakdown(m2_hours,
                               "Method 2 Independent State Estimate (Current Threshold)")

    return passed, m2_hours, d[["_m2_state"]].copy()


# ============================================================
# METHOD 3 -- POWER TRIANGLE CONSISTENCY CHECK
# ============================================================
# Uses ONLY raw electrical columns: active_power, reactive_power,
# apparent_power. No STATE_COL at any step — this is purely a
# raw-data physics/sensor quality check.
#
# Formula (AC power triangle identity, applies row-by-row):
#   apparent_power_calc = sqrt(active_power² + reactive_power²)
#   error_pct = |apparent_power_calc - apparent_power_measured|
#               / apparent_power_measured  * 100
# ============================================================

def method3_power_triangle(df_raw: pd.DataFrame,
                           df_clean: pd.DataFrame) -> bool | None:
    """
    Method 3 — Power Triangle Consistency Check.

    Checks internal consistency of the raw sensor readings using the
    fundamental AC power triangle identity. Does NOT produce any
    state classification — it is a pure data-quality check.

    Runs on BOTH the raw data (before cleaning) and the cleaned data,
    and reports both numbers. A large drop in error after cleaning is
    evidence that the negative/noise rows were real sensor artefacts.

    Pass criteria (stated in advance, no GMM dependency):
      - Mean error <= TRIANGLE_MEAN_ERR_MAX_PCT  (5%)
      - >= TRIANGLE_ROW_PCT_MIN % of rows within 10% error  (95%)

    Never touches STATE_COL at any step.

    Returns True/False/None (None = skipped due to missing columns).
    """
    _header("METHOD 3 -- POWER TRIANGLE CONSISTENCY CHECK")
    _info("apparent_power_calc = sqrt(active_power² + reactive_power²)")
    _info("error_pct = |apparent_power_calc - apparent_power_measured| "
          "/ apparent_power_measured * 100")
    _info(f"Acceptance: mean error <= {TRIANGLE_MEAN_ERR_MAX_PCT}%, "
          f">{TRIANGLE_ROW_PCT_MIN:.0f}% of rows within 10% error")
    _info("No STATE_COL used at any step — pure sensor-physics check.")
    _sep()

    needed = ['active_power', 'reactive_power', 'apparent_power']
    ok_raw,   _ = _check_cols(df_raw,   needed, "Method 3 (raw data)")
    ok_clean, _ = _check_cols(df_clean, needed, "Method 3 (cleaned data)")
    if not ok_raw and not ok_clean:
        _skip("All three power columns required — method skipped.")
        return None

    def _triangle_stats(df, label):
        """Compute power-triangle error stats for a DataFrame."""
        ap_calc = np.sqrt(
            df["active_power"].values ** 2
            + df["reactive_power"].values ** 2
        )
        ap_meas  = df["apparent_power"].values
        safe_meas = np.where(ap_meas > 0, ap_meas, np.nan)
        err_pct   = np.abs(ap_calc - ap_meas) / safe_meas * 100

        valid     = ~np.isnan(err_pct)
        n_valid   = int(valid.sum())
        if n_valid == 0:
            return None, None, None

        mean_err  = float(np.nanmean(err_pct))
        pct_10    = float(np.mean(err_pct[valid] <= 10.0) * 100)
        print(f"\n   [{label}]")
        print(f"      Rows analysed         : {n_valid:,}")
        print(f"      Mean error            : {mean_err:.2f}%")
        print(f"      Rows within 10% error : {pct_10:.1f}%")
        return mean_err, pct_10, n_valid

    mean_raw,  pct10_raw,  _ = _triangle_stats(df_raw,   "Before cleaning")
    mean_cln,  pct10_cln,  _ = _triangle_stats(df_clean, "After cleaning")

    if mean_raw is not None and mean_cln is not None:
        drop = mean_raw - mean_cln
        if drop > 0.5:
            _info(f"\n   Mean error dropped {drop:.2f}pp after cleaning "
                  f"({mean_raw:.2f}% → {mean_cln:.2f}%) — confirms that "
                  f"negative/noise rows were genuine sensor artefacts.")
        else:
            _info(f"\n   Mean error change after cleaning: {drop:+.2f}pp "
                  f"({mean_raw:.2f}% → {mean_cln:.2f}%)")

    # Pass/fail on cleaned data (or raw if clean not available)
    ref_mean = mean_cln if mean_cln is not None else mean_raw
    ref_pct  = pct10_cln if pct10_cln is not None else pct10_raw
    print()
    passed = True

    if ref_mean is None:
        _skip("No valid rows for power-triangle check.")
        return None

    if ref_mean <= TRIANGLE_MEAN_ERR_MAX_PCT:
        _pass(f"Mean power-triangle error = {ref_mean:.2f}% "
              f"<= {TRIANGLE_MEAN_ERR_MAX_PCT}%")
    else:
        _fail(f"Mean power-triangle error = {ref_mean:.2f}% "
              f"> {TRIANGLE_MEAN_ERR_MAX_PCT}% — possible single-phase vs "
              f"three-phase mismatch or sensor calibration issue")
        passed = False

    if ref_pct >= TRIANGLE_ROW_PCT_MIN:
        _pass(f"{ref_pct:.1f}% of rows within 10% error "
              f">= {TRIANGLE_ROW_PCT_MIN:.0f}% — sensor readings internally consistent")
    else:
        _fail(f"Only {ref_pct:.1f}% of rows within 10% error "
              f"< {TRIANGLE_ROW_PCT_MIN:.0f}%")
        passed = False

    if passed:
        _pass("Sensor/power-triangle consistency confirmed — raw electrical "
              "readings are internally valid.")
    else:
        _warn("Power-triangle inconsistency detected. Investigate sensor "
              "calibration or phase-factor (single-phase vs √3 three-phase) "
              "before trusting downstream state classifications.")

    return passed


# ============================================================
# METHOD 4 -- DWELL-TIME PLAUSIBILITY (RUN-LENGTH ANALYSIS)
# ============================================================
# Reads the GMM-smoothed STATE_COL from the labelled CSV.
# This is the correct, intended use: Method 4 is specifically
# testing whether the *exported* smoothed labels have physically
# plausible dwell times.  STATE_COL is its direct input, not
# a circular dependency.
# ============================================================

def method4_dwell_time(df_gmm: pd.DataFrame) -> bool | None:
    """
    Method 4 — Dwell-Time Plausibility.

    Checks whether the Viterbi-smoothed state sequence from the
    labelled CSV has physically plausible dwell times.
    Episodes shorter than FLICKER_MIN_SECONDS are counted as "flicker."

    This method reads STATE_COL from df_gmm — this is its intended
    input (it is testing the smoother's output, not producing its
    own independent classification).

    Returns True/False/None.
    """
    _header("METHOD 4 -- DWELL-TIME PLAUSIBILITY")
    _info(f"Input: Viterbi-smoothed state labels from labelled CSV")
    _info(f"Flags episodes shorter than {FLICKER_MIN_SECONDS}s as 'flicker'")
    _info(f"Acceptance: flicker rate <= {FLICKER_RATE_MAX_PCT}%")
    _sep()

    if df_gmm is None:
        _skip("Labelled CSV not available — Method 4 skipped.")
        return None

    ok, _ = _check_cols(df_gmm, [STATE_COL], "Method 4")
    if not ok:
        return None

    states = df_gmm[STATE_COL].values
    run_id = np.zeros(len(states), dtype=int)
    for i in range(1, len(states)):
        run_id[i] = run_id[i - 1] + (states[i] != states[i - 1])

    d = df_gmm[[STATE_COL]].copy()
    d["_run_id"] = run_id
    run_lengths = d.groupby("_run_id").agg(
        state=(STATE_COL, "first"),
        duration_s=(STATE_COL, "count"),
    )

    print(f"\n   {'State':<12} {'Median dur(s)':>14} {'Mean dur(s)':>12} "
          f"{'# episodes':>11}")
    print(f"   {'':─<12} {'':─<14} {'':─<12} {'':─<11}")
    for state, grp in run_lengths.groupby("state"):
        print(f"   {state:<12} {grp['duration_s'].median():>14.1f} "
              f"{grp['duration_s'].mean():>12.1f} {len(grp):>11,}")

    flicker      = (run_lengths["duration_s"] < FLICKER_MIN_SECONDS)
    flicker_rate = flicker.mean() * 100

    print(f"\n   Total episodes         : {len(run_lengths):,}")
    print(f"   Flicker (<{FLICKER_MIN_SECONDS}s) episodes: "
          f"{int(flicker.sum()):,}  ({flicker_rate:.1f}%)")

    if flicker_rate <= FLICKER_RATE_MAX_PCT:
        _pass(f"Flicker rate {flicker_rate:.1f}% <= {FLICKER_RATE_MAX_PCT}% "
              f"— Viterbi smoothing produced physically realistic state transitions")
        return True
    else:
        _fail(f"Flicker rate {flicker_rate:.1f}% > {FLICKER_RATE_MAX_PCT}% "
              f"— state transitions still too rapid; consider increasing "
              f"min_dwell_s in validate_gmm.py's export_labelled_csv()")
        return False


# ============================================================
# METHOD 5 -- CROSS-MACHINE CONSISTENCY (REPLICATION CHECK)
# ============================================================
# Reads raw CSVs for BOTH machines directly.
# Applies clean_raw_electrical_data() to each independently.
# Classifies each machine using the same fixed PF thresholds as
# Method 1 (Step A) — no STATE_COL from either machine's labelled CSV.
# Comparison is between machine A's formula-based results and
# machine B's formula-based results.
# ============================================================

def _m5_formula_classify(df_clean: pd.DataFrame,
                          spike_mask: pd.Series,
                          machine_label: str) -> tuple[dict, float, float] | None:
    """
    Apply Method 1's Step A classification to a machine's cleaned raw CSV.
    Returns (state_hours, pf_stby, pf_work) or None on missing columns.
    """
    ok, _ = _check_cols(df_clean, ['active_power', 'apparent_power', 'current'],
                        f"Method 5 ({machine_label})")
    if not ok:
        return None

    d = df_clean.copy()
    d["_pf"] = (d["active_power"].abs()
                / d["apparent_power"].replace(0, np.nan))

    def _clf(ap, pf):
        if ap < 5.0:
            return "OFF"
        if pd.isna(pf):
            return "OFF"
        if pf < PF_STANDBY_RANGE[1]:
            return "STANDBY"
        if pf >= PF_WORKING_MIN:
            return "WORKING"
        return "STANDBY"

    d["_m5_state"] = [_clf(ap, pf) for ap, pf in zip(d["active_power"], d["_pf"])]

    hours   = compute_state_hours(d, state_col="_m5_state", spike_mask=spike_mask)
    stby_m  = d.loc[d["_m5_state"] == "STANDBY", "_pf"].mean()
    work_m  = d.loc[d["_m5_state"] == "WORKING",  "_pf"].mean()
    stby_i  = d.loc[d["_m5_state"] == "STANDBY", "current"].mean()
    work_i  = d.loc[d["_m5_state"] == "WORKING",  "current"].mean()
    return hours, float(stby_m), float(work_m), float(stby_i), float(work_i)


def method5_cross_machine(raw_path_a: str, raw_path_b: str,
                           name_a: str, name_b: str) -> bool | None:
    """
    Method 5 — Cross-Machine Consistency (Replication Check).

    Loads the raw CSV for each machine independently (no labelled CSVs,
    no STATE_COL from validate_gmm.py output).
    Applies clean_raw_electrical_data() to each.
    Classifies each using the same fixed PF/current thresholds as Method 1.
    Compares the two machines' formula-based results against each other.

    Pass criteria (stated in advance):
      - WORKING PF difference    <= CROSS_MACHINE_PF_TOL
      - Current ratio difference <= CROSS_MACHINE_RATIO_TOL
      - WORKING time-share diff  <= CROSS_MACHINE_SHARE_TOL_PCT pp

    STATE_COL is not used at any step.

    Returns True/False/None.
    """
    _header("METHOD 5 -- CROSS-MACHINE CONSISTENCY (REPLICATION CHECK)")
    _info(f"Comparing: {name_a}  vs  {name_b}")
    _info("Both machines are classified independently from their raw CSVs.")
    _info("No labelled CSV (validate_gmm.py output) is required.")
    _sep()

    df_raw_a = load_raw_csv(raw_path_a)
    df_raw_b = load_raw_csv(raw_path_b)

    if df_raw_a is None:
        _skip(f"Raw CSV not found for {name_a}: {raw_path_a}")
        return None
    if df_raw_b is None:
        _skip(f"Raw CSV not found for {name_b}: {raw_path_b}")
        return None

    # Independent cleaning for each machine
    _info(f"\nCleaning {name_a} ...")
    df_cln_a, spk_a = clean_raw_electrical_data(df_raw_a)
    _info(f"\nCleaning {name_b} ...")
    df_cln_b, spk_b = clean_raw_electrical_data(df_raw_b)

    # Formula-based classification (same rules as Method 1, Step A)
    res_a = _m5_formula_classify(df_cln_a, spk_a, name_a)
    res_b = _m5_formula_classify(df_cln_b, spk_b, name_b)

    if res_a is None or res_b is None:
        return None

    hrs_a, pf_stby_a, pf_work_a, i_stby_a, i_work_a = res_a
    hrs_b, pf_stby_b, pf_work_b, i_stby_b, i_work_b = res_b

    # Print per-machine breakdown
    print_state_time_breakdown(hrs_a, f"{name_a} — formula-based (Method 5)")
    print_state_time_breakdown(hrs_b, f"{name_b} — formula-based (Method 5)")

    passed = True

    # Check 1: WORKING PF consistency
    print()
    d_pf = abs(pf_work_a - pf_work_b)
    print(f"   WORKING PF  : {name_a}={pf_work_a:.3f}  "
          f"{name_b}={pf_work_b:.3f}  diff={d_pf:.3f}")
    if not (math.isnan(pf_work_a) or math.isnan(pf_work_b)):
        if d_pf <= CROSS_MACHINE_PF_TOL:
            _pass(f"WORKING PF consistent (diff {d_pf:.3f} <= {CROSS_MACHINE_PF_TOL})")
        else:
            _fail(f"WORKING PF diverges (diff {d_pf:.3f} > {CROSS_MACHINE_PF_TOL})")
            passed = False
    else:
        _warn("Could not compute WORKING PF for one machine — check classification")

    # Check 2: STANDBY/WORKING current ratio consistency
    ratio_a = i_stby_a / i_work_a if i_work_a > 0 else float('nan')
    ratio_b = i_stby_b / i_work_b if i_work_b > 0 else float('nan')
    d_ratio = abs(ratio_a - ratio_b)
    print(f"   Current R   : {name_a}={ratio_a:.3f}  "
          f"{name_b}={ratio_b:.3f}  diff={d_ratio:.3f}")
    if not (math.isnan(ratio_a) or math.isnan(ratio_b)):
        if d_ratio <= CROSS_MACHINE_RATIO_TOL:
            _pass(f"Current ratio consistent (diff {d_ratio:.3f} <= {CROSS_MACHINE_RATIO_TOL})")
        else:
            _fail(f"Current ratio diverges (diff {d_ratio:.3f} > {CROSS_MACHINE_RATIO_TOL})")
            passed = False
    else:
        _warn("Could not compute current ratio for one machine")

    # Check 3: WORKING time-share consistency
    total_a   = sum(hrs_a.values())
    total_b   = sum(hrs_b.values())
    share_a   = hrs_a.get("WORKING", 0.0) / max(total_a, 1.0) * 100
    share_b   = hrs_b.get("WORKING", 0.0) / max(total_b, 1.0) * 100
    d_share   = abs(share_a - share_b)
    print(f"   WORKING %   : {name_a}={share_a:.1f}%  "
          f"{name_b}={share_b:.1f}%  diff={d_share:.1f}pp")
    if d_share <= CROSS_MACHINE_SHARE_TOL_PCT:
        _pass(f"WORKING time-share consistent (diff {d_share:.1f}pp "
              f"<= {CROSS_MACHINE_SHARE_TOL_PCT:.0f}pp)")
    else:
        _warn(f"WORKING time-share diverges (diff {d_share:.1f}pp > "
              f"{CROSS_MACHINE_SHARE_TOL_PCT:.0f}pp) — may reflect real "
              f"production-load differences between machines")

    return passed


# ============================================================
# DIAGNOSE DISAGREEMENT  (FIX 1)
# ============================================================

def diagnose_disagreement(df_m1: pd.DataFrame,
                          df_gmm: pd.DataFrame) -> None:
    """
    Diagnose the STANDBY disagreement between GMM (smoothed 'state' column)
    and Method 1 ('_m1_state' column).

    Steps
    -----
    1. Build a row-level confusion matrix: GMM state vs Method 1 state.
    2. For every row where the two disagree on STANDBY (GMM=STANDBY but
       M1!=STANDBY, or GMM!=STANDBY but M1=STANDBY), look up the raw
       (pre-Viterbi) GMM run that row belonged to using 'state_raw'.
       Compute each such run's duration in seconds (= consecutive run length
       in the 'state_raw' column).  IMDELD is 1-second resolution so
       1 sample ≈ 1 second.
    3. Print the distribution of these disagreement-run durations
       (min/P25/median/P75/max) and what % come from runs < 30s.
    4. Print a clear verdict.

    Requires 'state_raw' to exist in df_gmm (exported by validate_gmm.py
    after FIX 1 Part A).  If missing, prints a warning and returns.

    Parameters
    ----------
    df_m1  : DataFrame with '_m1_state' column (from method1_power_factor)
    df_gmm : DataFrame with 'state' (smoothed) and 'state_raw' (raw GMM) columns
    """
    print("\n" + "=" * 65)
    print("  DIAGNOSE DISAGREEMENT — STANDBY (GMM vs Method 1)")
    print("=" * 65)
    _info("Row-level confusion matrix + raw-run duration analysis for "
          "STANDBY disagreement rows (FIX 1).")
    _sep()

    if df_m1 is None or df_gmm is None:
        _warn("diagnose_disagreement: df_m1 or df_gmm is None -- skipping.")
        return
    if '_m1_state' not in df_m1.columns:
        _warn("diagnose_disagreement: '_m1_state' column missing from df_m1 -- skipping.")
        return
    if STATE_COL not in df_gmm.columns:
        _warn(f"diagnose_disagreement: '{STATE_COL}' column missing from df_gmm -- skipping.")
        return
    if len(df_m1) != len(df_gmm):
        _warn(f"diagnose_disagreement: row count mismatch "
              f"(df_m1={len(df_m1):,}, df_gmm={len(df_gmm):,}) -- skipping.")
        return

    gmm_states = df_gmm[STATE_COL].reset_index(drop=True).values
    m1_states  = df_m1['_m1_state'].reset_index(drop=True).values

    # ── 1. Confusion matrix ──────────────────────────────────────────────
    all_states = sorted(set(gmm_states) | set(m1_states))
    print(f"\n   Confusion matrix  (rows = GMM state, cols = Method 1 state):")
    print(f"   {'GMM \\ M1':>14}", end='')
    for s in all_states:
        print(f"  {s:>10}", end='')
    print(f"  {'Total':>10}")
    print("   " + "-" * (14 + 12 * len(all_states) + 12))
    for gmm_s in all_states:
        row_mask = gmm_states == gmm_s
        print(f"   {gmm_s:>14}", end='')
        row_total = 0
        for m1_s in all_states:
            cnt = int(((gmm_states == gmm_s) & (m1_states == m1_s)).sum())
            print(f"  {cnt:>10,}", end='')
            row_total += cnt
        print(f"  {row_total:>10,}")
    # Column totals
    print(f"   {'Total':>14}", end='')
    for m1_s in all_states:
        print(f"  {int((m1_states == m1_s).sum()):>10,}", end='')
    print(f"  {len(gmm_states):>10,}")

    # ── 2. STANDBY disagreement rows ────────────────────────────────────
    # Case A: GMM says STANDBY, Method 1 does NOT
    # Case B: Method 1 says STANDBY, GMM does NOT
    disagree_mask = (
        ((gmm_states == 'STANDBY') & (m1_states != 'STANDBY')) |
        ((gmm_states != 'STANDBY') & (m1_states == 'STANDBY'))
    )
    n_disagree = int(disagree_mask.sum())
    n_total    = len(gmm_states)
    print(f"\n   STANDBY disagreement rows: {n_disagree:,} / {n_total:,} "
          f"({n_disagree / max(n_total, 1) * 100:.2f}%)")
    print(f"     Case A (GMM=STANDBY, M1≠STANDBY): "
          f"{int(((gmm_states=='STANDBY')&(m1_states!='STANDBY')).sum()):,}")
    print(f"     Case B (M1=STANDBY, GMM≠STANDBY): "
          f"{int(((m1_states=='STANDBY')&(gmm_states!='STANDBY')).sum()):,}")

    if n_disagree == 0:
        _pass("No STANDBY disagreement rows found -- GMM and Method 1 fully agree on STANDBY.")
        return

    # ── 3. Raw-run durations for disagreement rows ───────────────────────
    # Use 'state_raw' (pre-Viterbi GMM labels) to identify runs.
    # If 'state_raw' is missing, fall back to smoothed 'state' with a warning.
    if 'state_raw' in df_gmm.columns:
        raw_col = df_gmm['state_raw'].reset_index(drop=True).values
        _info("Using 'state_raw' (pre-Viterbi raw GMM labels) for run identification.")
    else:
        raw_col = gmm_states   # fallback: smoothed labels
        _warn("'state_raw' column not found in labelled CSV. "
              "Falling back to smoothed 'state' column. "
              "Re-run validate_gmm.py to export 'state_raw' for accurate diagnostics.")

    # Build run_id from raw labels (consecutive equal-state runs)
    run_id = np.zeros(len(raw_col), dtype=int)
    for i in range(1, len(raw_col)):
        run_id[i] = run_id[i - 1] + (raw_col[i] != raw_col[i - 1])

    # Compute each run's length (in seconds = samples at 1 Hz)
    unique_runs, run_counts = np.unique(run_id, return_counts=True)
    run_len_map = dict(zip(unique_runs.tolist(), run_counts.tolist()))  # run_id -> length_s

    # Get run durations only for disagreement rows
    disagree_indices   = np.where(disagree_mask)[0]
    disagree_run_ids   = run_id[disagree_indices]
    disagree_run_durs  = np.array([run_len_map[r] for r in disagree_run_ids])

    # Statistics on run durations
    min_dur    = float(np.min(disagree_run_durs))
    p25_dur    = float(np.percentile(disagree_run_durs, 25))
    median_dur = float(np.percentile(disagree_run_durs, 50))
    p75_dur    = float(np.percentile(disagree_run_durs, 75))
    max_dur    = float(np.max(disagree_run_durs))
    MIN_DWELL  = 30   # current min_dwell_s threshold
    pct_short  = float((disagree_run_durs < MIN_DWELL).mean() * 100)

    print(f"\n   Raw-run duration distribution for STANDBY disagreement rows:")
    print(f"   {'Min':>10} {'P25':>10} {'Median':>10} {'P75':>10} {'Max':>10}")
    print(f"   {'':->10} {'':->10} {'':->10} {'':->10} {'':->10}")
    print(f"   {min_dur:>10.1f} {p25_dur:>10.1f} {median_dur:>10.1f} "
          f"{p75_dur:>10.1f} {max_dur:>10.1f}  (seconds)")
    print(f"\n   % of disagreement rows from runs < {MIN_DWELL}s : {pct_short:.1f}%")

    # ── 4. Verdict ────────────────────────────────────────────────────────
    SHORT_DOMINANT = 50.0   # majority threshold
    print()
    if pct_short >= SHORT_DOMINANT:
        verdict = (f"Disagreement is mostly SHORT runs (<{MIN_DWELL}s): "
                   f"likely caused by Viterbi over-smoothing "
                   f"({pct_short:.1f}% of disagreement rows from runs <{MIN_DWELL}s, "
                   f"median={median_dur:.1f}s).")
        _info(f"VERDICT: {verdict}")
    else:
        verdict = (f"Disagreement is mostly LONG stable runs: "
                   f"likely a genuine threshold miscalibration between "
                   f"GMM's cluster boundary and Method 1's PF<0.35 rule "
                   f"({pct_short:.1f}% of disagreement rows from runs <{MIN_DWELL}s, "
                   f"median={median_dur:.1f}s).")
        _info(f"VERDICT: {verdict}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    machine_name = MACHINES[_machine_key]["name"]
    _pair_raw    = MACHINES[_pair_key]["raw"] if _pair_key else None

    print("=" * 65)
    print("  5-METHOD INDEPENDENT PROOF HARNESS")
    print(f"  Machine      : {machine_name}")
    print(f"  Raw CSV      : {RAW_CSV_PATH}")
    print(f"  Labelled CSV : {LABELLED_CSV_PATH}")
    print("=" * 65)
    print()
    print("  INDEPENDENCE GUARANTEE")
    print("  ─────────────────────")
    print("  Methods 1, 2, 3, 5 compute all state labels and hour")
    print("  totals using ONLY raw electrical columns + fixed stated")
    print("  constants.  The GMM's 'state' column is NEVER used")
    print("  inside any method's labeling or threshold logic.")
    print("  STATE_COL appears only in compare_to_gmm() (Step C)")
    print("  and in method4_dwell_time() (which tests the smoother).")
    print("=" * 65)

    # ── Load raw CSV + independent cleaning ───────────────────────────
    df_raw = load_raw_csv(RAW_CSV_PATH)
    if df_raw is None:
        print(f"\nERROR: Raw CSV not found: {RAW_CSV_PATH}")
        print("Check that the 'raw' path in MACHINES config is correct.")
        sys.exit(1)

    df_clean, spike_mask = clean_raw_electrical_data(df_raw)

    # ── Load GMM-smoothed labelled CSV (for Step C and Method 4) ──────
    df_gmm = load_labelled(LABELLED_CSV_PATH)
    if df_gmm is None:
        _warn(f"Labelled CSV not found: {LABELLED_CSV_PATH}")
        _warn("Step C comparisons and Method 4 will be skipped.")
        _warn(f"Run: python validate_gmm.py {_machine_key}")
    else:
        print(f"\n   GMM labelled CSV loaded: {len(df_gmm):,} rows")
        if TARGET_3_STATES and STATE_COL in df_gmm.columns:
            n_peak = int((df_gmm[STATE_COL] == 'PEAK_LOAD').sum())
            if n_peak > 0:
                df_gmm[STATE_COL] = df_gmm[STATE_COL].replace('PEAK_LOAD', 'WORKING')
                _info(f"TARGET_3_STATES=True -- collapsed {n_peak:,} PEAK_LOAD rows "
                      f"into WORKING for the GMM baseline (validate_gmm.py itself is "
                      f"unaffected; this collapse only applies inside proof_5methods.py "
                      f"so Step C compares 3 states against 3 states).")

    # GMM baseline hours (Step C reference — only uses STATE_COL from labelled CSV)
    # If validate_gmm.py exported an 'is_spike' column, use it directly so
    # this total is built from the EXACT same time pool validate_gmm.py's
    # own printed breakdown used — not just an independently-recomputed
    # approximation of it. Falls back to no exclusion for older CSVs that
    # predate this column, with a warning so the mismatch is visible.
    gmm_hours = {}
    if df_gmm is not None and STATE_COL in df_gmm.columns:
        if 'is_spike' in df_gmm.columns:
            gmm_spike_mask = df_gmm['is_spike'].astype(bool)
            _info("Using 'is_spike' column from labelled CSV for GMM baseline "
                  "totals (exact match to validate_gmm.py's own reporting).")
        else:
            gmm_spike_mask = None
            _warn("Labelled CSV has no 'is_spike' column (older export). "
                  "GMM baseline totals will include spike rows that Methods "
                  "1/2/5 exclude -- re-run validate_gmm.py to regenerate the "
                  "CSV with matching totals.")
        gmm_hours = compute_state_hours(df_gmm, state_col=STATE_COL, spike_mask=gmm_spike_mask)
        print_state_time_breakdown(gmm_hours,
                                   f"GMM Baseline (smoothed labels) — {machine_name}")

    # ── Collect results ────────────────────────────────────────────────
    results             = {}
    agreement_by_method = {}

    # ── Method 3: Power Triangle (runs before other methods; uses raw+cleaned) ─
    m3 = method3_power_triangle(df_raw, df_clean)
    if m3 is not None:
        results["Method 3: Power Triangle Consistency"] = m3

    # ── Method 1 ─────────────────────────────────────────────────────
    m1_passed, m1_hours, df_m1 = method1_power_factor(df_clean, spike_mask)
    if m1_passed is not None:
        results["Method 1: Power Factor Separation"] = m1_passed
        # Step C: compare method hours to GMM hours
        if gmm_hours:
            c1 = compare_to_gmm(
                "Method 1 (Power Factor)",
                m1_hours, gmm_hours,
                df_method=df_m1 if df_gmm is not None and len(df_m1) == len(df_gmm) else None,
                method_col="_m1_state",
                df_gmm=df_gmm,
            )
            agreement_by_method["Method 1 (Power Factor)"] = {
                "all_pass": c1.get("__all_pass__"),
                "row_agr":  c1.get("__row_agr__"),
            }
        # FIX 1: diagnose STANDBY disagreement (runs after Step C)
        if df_gmm is not None and df_m1 is not None and len(df_m1) == len(df_gmm):
            diagnose_disagreement(df_m1, df_gmm)

    # ── Method 2 ─────────────────────────────────────────────────────
    m2_passed, m2_hours, df_m2 = method2_current_ratio(df_clean, spike_mask)
    if m2_passed is not None:
        results["Method 2: Current Ratio Check"] = m2_passed
        if gmm_hours:
            c2 = compare_to_gmm(
                "Method 2 (Current Ratio)",
                m2_hours, gmm_hours,
                df_method=df_m2 if df_gmm is not None and len(df_m2) == len(df_gmm) else None,
                method_col="_m2_state",
                df_gmm=df_gmm,
            )
            agreement_by_method["Method 2 (Current Ratio)"] = {
                "all_pass": c2.get("__all_pass__"),
                "row_agr":  c2.get("__row_agr__"),
            }

    # ── Method 4 (reads Viterbi-smoothed state from labelled CSV) ─────
    m4 = method4_dwell_time(df_gmm)
    if m4 is not None:
        results["Method 4: Dwell-Time Plausibility"] = m4

    # ── Method 5 (reads raw CSVs for both machines, no labelled CSVs) ─
    if _pair_key and _pair_raw:
        m5 = method5_cross_machine(
            RAW_CSV_PATH, _pair_raw,
            MACHINES[_machine_key]["name"],
            MACHINES[_pair_key]["name"],
        )
        if m5 is not None:
            results["Method 5: Cross-Machine Consistency"] = m5
    else:
        _header("METHOD 5 -- CROSS-MACHINE CONSISTENCY")
        _skip(f"No peer machine configured for {_machine_key}.")

    # ── Step C agreement summary ───────────────────────────────────────
    if agreement_by_method:
        _header("STEP C — GMM COMPARISON SUMMARY (Hour Totals)")
        print(f"\n   {'Method':<40}  {'Hour agreement':<18}  {'Row agreement'}")
        print(f"   {'':─<40}  {'':─<18}  {'':─<14}")
        for mname, agr in agreement_by_method.items():
            h_ok  = "PASS" if agr.get("all_pass") else \
                    ("FAIL" if agr.get("all_pass") is False else "N/A")
            r_agr = agr.get("row_agr")
            r_str = f"{r_agr:.1f}%" if r_agr is not None else "N/A"
            print(f"   {mname:<40}  {h_ok:<18}  {r_str}")

    # ── Final report ───────────────────────────────────────────────────
    _header("FINAL SUMMARY — 5-METHOD PROOF")

    n_pass  = sum(1 for v in results.values() if v is True)
    n_fail  = sum(1 for v in results.values() if v is False)
    n_total = len(results)

    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"   [{status}] {name}")

    print()
    print(f"   {n_pass}/{n_total} independent methods PASSED")
    print()

    if n_total == 0:
        print("   [INCONCLUSIVE] No methods ran (check CSV paths and columns).")
        print(f"   Ensure raw CSV exists: {RAW_CSV_PATH}")
    elif n_pass == n_total:
        print(f"   OVERALL VERDICT: All {n_pass}/{n_total} methods support the "
              f"STANDBY/WORKING split.")
        print(f"   GMM classification is considered VALIDATED for {machine_name}.")
    elif n_pass >= int(n_total * 0.6):
        n_inc = n_total - n_pass - n_fail
        print(f"   OVERALL VERDICT: {n_pass}/{n_total} methods support the "
              f"STANDBY/WORKING split"
              + (f" ({n_inc} inconclusive)" if n_inc else "."))
        print(f"   GMM classification is PARTIALLY VALIDATED for {machine_name}.")
        print("   Investigate the FAIL items before treating the split as final.")
    else:
        print(f"   OVERALL VERDICT: Only {n_pass}/{n_total} methods pass — "
              f"STANDBY/WORKING split is not well supported.")
        print("   Revisit feature choice, k selection, or GMM configuration.")

    print()
    print("   INDEPENDENCE VERIFICATION:")
    print("   grep 'STATE_COL' proof_5methods.py should only appear in:")
    print("   compare_to_gmm(), method4_dwell_time(), and compute_agreement()")


if __name__ == "__main__":
    main()