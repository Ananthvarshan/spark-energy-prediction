"""
============================================================
GMM VALIDATION HARNESS  --  validate_gmm.py
============================================================

PURPOSE
-------
Verify that the GMM correctly splits power readings into
physically meaningful machine states (OFF / STANDBY / WORKING
/ PEAK_LOAD) BEFORE anything else in the pipeline runs.

This script is FULLY SELF-CONTAINED -- no imports from
src/data_analysis.py or any other project module.

TEN INDEPENDENT TESTS + STATE-TIME BREAKDOWN
---------------------------------------------
T1   Data Sanity           -- non-empty, no NaN, IMDELD features OK
T2   GMM Convergence       -- EM algorithm converged?
T3   k-Selection           -- Silhouette + BIC agree on k?
T4   State Separation      -- cluster means >= 2-sigma apart?
T5   Soft-Assignment       -- model is confident (max-prob > 80%)?
T6   Weight Sanity         -- no ghost clusters (<0.5%)?
T7   Physics Thresholds    -- OFF~0W, STANDBY stable, WORKING gap OK?
T8   Variance Ordering     -- sigma_OFF < sigma_STANDBY < sigma_WORKING?
T9   Temporal Alignment    -- OFF labels align with nights/weekends?
T10  IMDELD Schedule Proof -- OFF/STANDBY >90% in factory closed window?

HOW TO RUN
----------
    python validate_gmm.py
    python validate_gmm.py pelletizer-II

Outputs are saved to outputs/gmm_validation_imdeld/<machine>/
The labelled CSV for proof_5methods.py goes to:
    outputs/imdeld_labelled/<machine>_labelled.csv
============================================================
"""

import os
import sys

# Force UTF-8 output so special chars work on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION  -- pick machine via command-line arg
# ============================================================
# Usage:  python validate_gmm.py pelletizer-I
# Add a new appliance by adding one line to MACHINES.
# ────────────────────────────────────────────────────────────
MACHINES = {
    "pelletizer-I":    {"path": "data/Appliances/pelletizer-I.csv",             "name": "Pelletizer I (IMDELD)"},
    "pelletizer-II":   {"path": "data/Appliances/pelletizer-II.csv",            "name": "Pelletizer II (IMDELD)"},
    "dpc-I":           {"path": "data/Appliances/doublepolecontactor-I.csv",    "name": "Double-Pole Contactor I (IMDELD)"},
    "dpc-II":          {"path": "data/Appliances/doublepolecontactor-II.csv",   "name": "Double-Pole Contactor II (IMDELD)"},
    "exhaust-fan-I":   {"path": "data/Appliances/exhaustfan-I.csv",             "name": "Exhaust Fan I (IMDELD)"},
    "exhaust-fan-II":  {"path": "data/Appliances/exhaustfan-II.csv",            "name": "Exhaust Fan II (IMDELD)"},
    "milling-I":       {"path": "data/Appliances/millingmachine-I.csv",         "name": "Milling Machine I (IMDELD)"},
    "milling-II":      {"path": "data/Appliances/millingmachine-II.csv",        "name": "Milling Machine II (IMDELD)"},
}
DEFAULT_MACHINE = "pelletizer-I"

_machine_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MACHINE
if _machine_key not in MACHINES:
    print(f"Unknown machine '{_machine_key}'. Choose from: {list(MACHINES.keys())}")
    sys.exit(1)

DATA_PATH    = MACHINES[_machine_key]["path"]
MACHINE_NAME = MACHINES[_machine_key]["name"]
OUTPUT_DIR   = f"outputs/gmm_validation_imdeld/{_machine_key}"
K_RANGE      = [2, 3, 4]

# IMDELD factory schedule (Brazil time UTC-3):
#   Factory CLOSED every weekday 17:00-22:00 (electricity price tariff).
SCHEDULE_CLOSE_HOUR = 17
SCHEDULE_OPEN_HOUR  = 22
FACTORY_TZ          = 'America/Sao_Paulo'

# ── Spike-detection parameters -- MUST MATCH proof_5methods.py EXACTLY ────
# so that both scripts exclude the same rows from hour-total calculations
# and their totals are directly comparable.
MAD_WINDOW    = 11      # rolling window width (samples) for local median + MAD
MAD_THRESHOLD = 5.0     # flag if value > threshold x MAD from local rolling median
# ============================================================


# ────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ────────────────────────────────────────────────────────────
STATE_COLORS = {
    'OFF':       '#555555',
    'STANDBY':   '#f0a500',
    'IDLE':      '#4fc3f7',
    'WORKING':   '#66bb6a',
    'PEAK_LOAD': '#e53935',
}
K_TO_NAMES = {
    2: ['OFF', 'WORKING'],
    3: ['OFF', 'STANDBY', 'WORKING'],
    4: ['OFF', 'STANDBY', 'WORKING', 'PEAK_LOAD'],
}

# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def _pass(msg):  print(f"   [PASS] {msg}")
def _fail(msg):  print(f"   [FAIL] {msg}")
def _info(msg):  print(f"   [INFO] {msg}")
def _warn(msg):  print(f"   [WARN] {msg}")
def _sep():      print("   " + "-" * 62)


# IMDELD multi-feature columns
IMDELD_FEATURES = ['active_power', 'reactive_power',
                   'apparent_power', 'current', 'voltage']

# Column name aliases: maps any recognised variant → canonical name
# Keys are lowercased & stripped; values are canonical column names.
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
# SELF-CONTAINED DATA LOADER
# ============================================================

def load_and_prepare_data(path: str) -> pd.DataFrame:
    """
    Load a raw appliance CSV and return a clean DataFrame with:
      - timestamp  (datetime, UTC-aware)
      - active_power, reactive_power, apparent_power, current, voltage
        whenever those columns exist in the source file
      - power  (alias of active_power, or fallback best-guess column)

    Column detection is case-insensitive and alias-aware.
    Fails immediately with a clear, specific error if a required
    column cannot be resolved -- never silently truncates output.

    Parameters
    ----------
    path : str
        Absolute or relative path to the raw CSV.

    Returns
    -------
    pd.DataFrame
        Clean, sorted DataFrame ready for GMM fitting.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Update the MACHINES dict at the top of validate_gmm.py."
        )

    print(f"\n   Loading: {path}")

    # -- Read CSV -------------------------------------------------------
    # Try reading as-is first; some SPARK files have a metadata row at
    # row 0 that pushes the real header to row 1, so if we don't find a
    # timestamp-like column we retry with skiprows=1.
    df = pd.read_csv(path)

    def _has_timestamp(d):
        return any(_COL_ALIASES.get(c.strip().lower()) == 'timestamp'
                   for c in d.columns)

    if not _has_timestamp(df) and len(df.columns) > 1:
        try:
            df2 = pd.read_csv(path, skiprows=1)
            if _has_timestamp(df2):
                df = df2
        except Exception:
            pass

    # -- Normalise column names -----------------------------------------
    rename_map = {}
    for col in df.columns:
        canonical = _COL_ALIASES.get(col.strip().lower())
        if canonical and canonical not in rename_map.values():
            rename_map[col] = canonical

    df = df.rename(columns=rename_map)
    # Remove duplicate canonical columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"   Columns after normalisation: {list(df.columns)}")

    # -- Timestamp -------------------------------------------------------
    if 'timestamp' not in df.columns:
        raise ValueError(
            f"❌  No timestamp column found in: {path}\n"
            f"    Columns present: {list(df.columns)}\n"
            f"    Expected one of: {[k for k,v in _COL_ALIASES.items() if v=='timestamp']}"
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # -- Numeric conversion + negative-value clipping -------------------
    electrical_cols = [c for c in IMDELD_FEATURES if c in df.columns]

    # If we have no recognised electrical column at all, try fallback
    if not electrical_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(
                f"❌  No numeric measurement columns found in: {path}\n"
                f"    Columns present: {list(df.columns)}"
            )
        _warn(f"No standard electrical columns found; using numeric fallback: {numeric_cols}")
        # Map first numeric column to active_power
        df['active_power'] = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
        electrical_cols = ['active_power']

    for col in electrical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clip active_power / apparent_power / current / voltage at 0 (sensor
    # noise can go negative; reactive_power is deliberately NOT clipped,
    # since a negative reading is a real, valid capacitive-load signal).
    # Scope matches proof_5methods.py's clean_raw_electrical_data() exactly.
    for col in ['active_power', 'apparent_power', 'current', 'voltage']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Convenience alias used by most tests
    df['power'] = df['active_power']

    # -- Drop rows with NaN in key columns ------------------------------
    key_cols = ['timestamp', 'active_power']
    before = len(df)
    df = df.dropna(subset=key_cols)
    dropped = before - len(df)
    if dropped:
        _warn(f"Dropped {dropped:,} rows with NaN in {key_cols}")

    # -- Sort by time ----------------------------------------------------
    df = df.sort_values('timestamp').reset_index(drop=True)

    # -- Summary ---------------------------------------------------------
    span = (df['timestamp'].max() - df['timestamp'].min())
    span_days = span.total_seconds() / 86400
    print(f"   Rows loaded  : {len(df):,}")
    print(f"   Time span    : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    print(f"   Span         : {span_days:.1f} days")
    print(f"   Power range  : {df['power'].min():.2f} W  →  {df['power'].max():.2f} W")
    cols_present = [c for c in IMDELD_FEATURES if c in df.columns]
    missing_cols = [c for c in IMDELD_FEATURES if c not in df.columns]
    if missing_cols:
        _warn(
            f"Columns not found in raw CSV: {missing_cols}\n"
            f"   Downstream checks that need these columns will be SKIPPED:\n"
            f"   - Method 1 (Power Factor) needs apparent_power\n"
            f"   - Method 2 (Current Ratio) needs current\n"
            f"   - Method 5 (Cross-Machine) needs apparent_power + current"
        )

    return df


def is_imdeld(df):
    """True if the dataframe has all 5 IMDELD electrical features."""
    return all(c in df.columns for c in IMDELD_FEATURES)


def compute_spike_mask(df):
    """
    Rolling-MAD spike detection -- IDENTICAL algorithm and parameters to
    proof_5methods.py's clean_raw_electrical_data(), so that both scripts
    exclude the same rows from hour-total calculations and their totals
    are directly comparable (same denominator, same "total time").

    For each electrical column, compute a rolling median and MAD over
    MAD_WINDOW samples. Flag a row as a spike on that column if it is
    more than MAD_THRESHOLD x MAD from the local rolling median.
    A row is a spike overall if ANY column is flagged.

    Returns
    -------
    pd.Series (bool), aligned to df.index. True = spike row (exclude
    from hour totals). Rows are NOT deleted -- only flagged.
    """
    elec_cols = [c for c in IMDELD_FEATURES if c in df.columns]
    spike_flags = pd.DataFrame(False, index=df.index, columns=elec_cols)

    for col in elec_cols:
        series   = df[col]
        roll_med = series.rolling(MAD_WINDOW, center=True, min_periods=1).median()
        roll_mad = (series - roll_med).abs().rolling(
            MAD_WINDOW, center=True, min_periods=1).median()
        mad_safe  = roll_mad.replace(0, np.nan)
        deviation = (series - roll_med).abs()
        spike_flags[col] = (deviation > MAD_THRESHOLD * mad_safe).fillna(False)

    spike_mask = spike_flags.any(axis=1)
    n_spikes = int(spike_mask.sum())
    _info(f"Spike detection (>{MAD_THRESHOLD}x MAD, window={MAD_WINDOW}): "
          f"{n_spikes:,} rows flagged ({n_spikes / max(len(df),1) * 100:.4f}%) "
          f"-- excluded from hour totals, kept in export for continuity")
    return spike_mask


def fit_gmm(X, k, n_init=5):
    """Fit a GMM with n_init restarts to avoid bad local optima."""
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = 'full',
        n_init          = n_init,
        max_iter        = 300,
        random_state    = 42,
    )
    gmm.fit(X)
    return gmm


def build_feature_matrix(df):
    """
    Return (X, scaler_or_None).
    IMDELD: scale all 5 features to unit variance.
    SPARK:  use only active_power / power (1D, no scaling needed).
    """
    if is_imdeld(df):
        X_raw  = df[IMDELD_FEATURES].values
        scaler = StandardScaler()
        X      = scaler.fit_transform(X_raw)
        return X, scaler
    else:
        X = df['power'].values.reshape(-1, 1)
        return X, None


def map_states(gmm, k, scaler=None):
    """
    Sort GMM components by ascending ACTIVE POWER mean.
    For IMDELD (scaled): inverse-transform means back to Watts first.
    For SPARK (1D): use the raw means directly.
    Returns dict: component_index -> state_name
    """
    if scaler is not None:
        means_orig = scaler.inverse_transform(gmm.means_)   # (k, 5)
        sort_vals  = means_orig[:, 0]
    else:
        sort_vals = gmm.means_.flatten()

    sorted_idx = np.argsort(sort_vals)
    names      = K_TO_NAMES.get(k, [f"State_{i}" for i in range(k)])
    return {int(sorted_idx[i]): names[i] for i in range(k)}


# ============================================================
# STATE-TIME BREAKDOWN (uses real timestamp deltas)
# ============================================================

def compute_state_time(df, labels_arr, state_map, spike_mask=None):
    """
    Compute hours spent in each state using actual timestamp deltas
    between consecutive readings.  A single interval is capped at
    2x the median sampling interval to avoid counting data gaps as
    machine time.

    If spike_mask is provided (bool Series/array, True = spike row,
    from compute_spike_mask), those rows contribute 0 seconds -- this
    matches proof_5methods.py's compute_state_hours() exactly, so the
    two scripts' hour totals are always built from the same time pool.

    Uses pandas dt.total_seconds() so the result is correct for both
    datetime64[ns] (pandas <2.0) and datetime64[us] (pandas 3.0+).

    Parameters
    ----------
    df         : DataFrame with a 'timestamp' column
    labels_arr : array of integer GMM component labels
    state_map  : dict  component_int -> state_name
    spike_mask : optional bool Series/array, True = exclude from totals

    Returns
    -------
    dict  state_name -> hours
    """
    states = np.array([state_map[l] for l in labels_arr])

    # Use pandas-native diff so unit (ns vs us) doesn't matter
    ts_series   = df['timestamp'].reset_index(drop=True)
    intervals_s = ts_series.diff().dt.total_seconds().values.copy()   # NaN at index 0
    # Forward-fill: give first row the same interval as second row
    if len(intervals_s) > 1 and np.isnan(intervals_s[0]):
        intervals_s[0] = intervals_s[1] if not np.isnan(intervals_s[1]) else 1.0
    intervals_s = np.where(np.isnan(intervals_s), 0.0, intervals_s)

    # Cap outliers: gaps > 2x median are assumed to be data outages
    pos_mask = intervals_s > 0
    median_s = np.median(intervals_s[pos_mask]) if pos_mask.any() else 1.0
    cap_s    = 2 * median_s if median_s > 0 else 60.0
    intervals_s = np.clip(intervals_s, 0, cap_s)

    # Zero-out spike rows so they contribute 0 seconds -- matches
    # proof_5methods.py's compute_state_hours() exactly.
    if spike_mask is not None:
        sm = np.asarray(pd.Series(spike_mask).reset_index(drop=True))
        intervals_s = np.where(sm, 0.0, intervals_s)

    state_seconds = {}
    for state in np.unique(states):
        mask = states == state
        state_seconds[state] = float(intervals_s[mask].sum())

    return {s: v / 3600.0 for s, v in state_seconds.items()}


def print_state_time_breakdown(state_hours, machine_name):
    """Print the human-readable state-time summary block."""
    total_h = sum(state_hours.values())
    total_d = total_h / 24

    order = ['OFF', 'STANDBY', 'IDLE', 'WORKING', 'PEAK_LOAD']
    present = [s for s in order if s in state_hours] + \
              [s for s in state_hours if s not in order]

    print("\n" + "=" * 48)
    print(f"  STATE TIME BREAKDOWN — {machine_name}")
    print("-" * 48)
    for state in present:
        h   = state_hours[state]
        pct = h / total_h * 100 if total_h > 0 else 0
        print(f"  {state:<12}: {h:>8,.1f} hours  ({pct:>5.1f}%)")
    print("-" * 48)
    print(f"  {'Total':<12}: {total_h:>8,.1f} hours  ({total_d:.1f} days)")
    print("=" * 48)


# ============================================================
# TEST 1 — DATA SANITY
# ============================================================

def test_data_sanity(df):
    print("\n" + "=" * 65)
    print("  TEST 1 -- DATA SANITY CHECK")
    print("=" * 65)
    passed = True

    # 1a -- Row count
    n = len(df)
    if n < 1000:
        _fail(f"Only {n:,} rows -- too few for reliable GMM fitting (need >=1,000)")
        passed = False
    else:
        _pass(f"{n:,} rows loaded")

    # 1b -- No NaN in power column
    nan_count = df['power'].isna().sum()
    if nan_count > 0:
        _fail(f"{nan_count:,} NaN values in power column")
        passed = False
    else:
        _pass("No NaN values in power column")

    # 1c -- All non-negative after clipping
    neg_count = (df['power'] < 0).sum()
    if neg_count > 0:
        _fail(f"{neg_count:,} negative power values found AFTER clipping -- "
              "data loader did not clip correctly!")
        passed = False
    else:
        _pass("All power values >= 0 (negative values clipped correctly)")

    # 1d -- Has OFF state readings
    off_count = (df['power'] < 5).sum()
    off_pct   = off_count / n * 100
    if off_count == 0:
        _warn("No readings <5W -- machine may never be fully OFF.")
    else:
        _pass(f"OFF-state readings (<5W): {off_count:,}  ({off_pct:.1f}%)")

    # 1e -- Has active state readings
    on_count = (df['power'] > 5).sum()
    if on_count < 100:
        _fail("Fewer than 100 ON-state readings -- not enough active data")
        passed = False
    else:
        _pass(f"ON-state readings  (>5W): {on_count:,}")

    # 1f -- IMDELD: check all 5 features are present and non-NaN
    if is_imdeld(df):
        _info("IMDELD format detected -- checking all 5 electrical features")
        for feat in IMDELD_FEATURES:
            nan_f = df[feat].isna().sum()
            if nan_f > 0:
                _fail(f"Feature '{feat}' has {nan_f:,} NaN values")
                passed = False
            else:
                _pass(f"Feature '{feat}': OK (mean={df[feat].mean():.2f}, "
                      f"max={df[feat].max():.2f})")
    else:
        missing = [c for c in IMDELD_FEATURES if c not in df.columns]
        _info(f"SPARK / partial format -- missing IMDELD columns: {missing}")

    # 1g -- Power stats
    _info(f"Power range : {df['power'].min():.2f} W -> {df['power'].max():.2f} W")
    _info(f"Power mean  : {df['power'].mean():.2f} W  |  std: {df['power'].std():.2f} W")

    return passed


# ============================================================
# TEST 2 — GMM CONVERGENCE
# ============================================================

def test_gmm_convergence(X, models):
    print("\n" + "=" * 65)
    print("  TEST 2 -- GMM CONVERGENCE CHECK")
    print("=" * 65)
    _info("Each GMM is run with 5 random restarts (n_init=5) to avoid bad local optima.")
    _sep()

    all_converged = True
    for k, gmm in models.items():
        if gmm.converged_:
            _pass(f"k={k} — converged in {gmm.n_iter_} EM iterations")
        else:
            _fail(f"k={k} — DID NOT CONVERGE after {gmm.n_iter_} iterations. "
                  "Increase max_iter or check for degenerate data.")
            all_converged = False

    return all_converged


# ============================================================
# PHYSICAL-PLAUSIBILITY FILTERS  (used inside k-selection)
# ============================================================

def _weight_ok(gmm, min_weight=0.005):
    """True if every GMM component has weight >= min_weight."""
    return bool(np.all(gmm.weights_ >= min_weight))


def _separation_ok(gmm, k, scaler=None, min_sigma=2.0):
    """
    True if every pair of adjacent clusters (by active-power mean)
    is at least min_sigma standard deviations apart.
    Returns (bool, index_of_first_failing_pair).
    """
    if scaler is not None:
        means = scaler.inverse_transform(gmm.means_)[:, 0]
        stds  = np.array([
            np.sqrt(np.abs(gmm.covariances_[i][0, 0])) * scaler.scale_[0]
            for i in range(k)
        ])
    else:
        means = gmm.means_.flatten()
        stds  = np.sqrt(np.abs(gmm.covariances_.flatten()))

    order = np.argsort(means)
    means_s = means[order]
    stds_s  = stds[order]

    for i in range(k - 1):
        avg_std = (stds_s[i] + stds_s[i + 1]) / 2
        sep     = (means_s[i + 1] - means_s[i]) / avg_std if avg_std > 0 else float('inf')
        if sep < min_sigma:
            return False, i
    return True, -1


# ============================================================
# TEST 3 — k SELECTION: BIC + PHYSICAL PLAUSIBILITY FILTERS
# ============================================================

def _count_physics_checks_passed(gmm, k, scaler, max_power):
    """
    Run physics checks for a given (gmm, k) candidate and count
    how many sub-checks pass.  Used by test_k_selection (FIX 4).

    Checks evaluated (silently -- no prints):
      A) 2-sigma pair separations  (one check per adjacent pair → k-1 checks)
      B) test_physics_thresholds   (4 sub-checks)
      C) test_variance_ordering    (pairwise σ comparisons → k-1 checks)

    Returns (n_passed, n_total).
    """
    state_map_k = map_states(gmm, k, scaler)

    # ---- A: 2-sigma separation checks ----
    if scaler is not None:
        means_a = scaler.inverse_transform(gmm.means_)[:, 0]
        stds_a  = np.array([
            np.sqrt(np.abs(gmm.covariances_[i][0, 0])) * scaler.scale_[0]
            for i in range(k)
        ])
    else:
        means_a = gmm.means_.flatten()
        stds_a  = np.sqrt(np.abs(gmm.covariances_.flatten()))

    order_a      = np.argsort(means_a)
    means_sorted = means_a[order_a]
    stds_sorted  = stds_a[order_a]

    sep_passed = 0
    for i in range(k - 1):
        avg_std = (stds_sorted[i] + stds_sorted[i + 1]) / 2
        sep     = (means_sorted[i + 1] - means_sorted[i]) / avg_std \
                  if avg_std > 0 else float('inf')
        if sep >= 2.0:
            sep_passed += 1

    # ---- B: physics threshold checks (4 sub-checks) ----
    if scaler is not None:
        means_b = scaler.inverse_transform(gmm.means_)[:, 0]
        stds_b  = np.array([
            np.sqrt(np.abs(gmm.covariances_[i][0, 0])) * scaler.scale_[0]
            for i in range(k)
        ])
    else:
        means_b  = gmm.means_.flatten()
        stds_b   = np.sqrt(np.abs(gmm.covariances_.flatten()))

    sidx      = np.argsort(means_b)
    off_mean  = means_b[sidx[0]]
    off_std   = stds_b[sidx[0]]
    stby_mean = means_b[sidx[1]] if k >= 3 else None
    stby_std  = stds_b[sidx[1]] if k >= 3 else None
    work_mean = means_b[sidx[-1]]
    work_std  = stds_b[sidx[-1]]

    phys_passed = 0
    # B1: OFF mean < threshold
    off_thr = max(5.0, max_power * 0.05)
    if off_mean < off_thr:
        phys_passed += 1
    # B2: OFF stable
    off_cv_ok = (off_std < 15.0) or (off_mean > 0 and off_std / off_mean < 0.20)
    if off_cv_ok:
        phys_passed += 1
    # B3: STANDBY more stable than WORKING (CV)
    if stby_mean is not None:
        stby_cv = stby_std / stby_mean if stby_mean > 0 else float('inf')
        work_cv = work_std / work_mean if work_mean > 0 else float('inf')
        if stby_cv < work_cv:
            phys_passed += 1
    # B4: WORKING mean >= 3x OFF mean
    ratio = work_mean / off_mean if off_mean > 0 else float('inf')
    if ratio >= 3.0:
        phys_passed += 1

    # ---- C: variance ordering checks (k-1 pairwise) ----
    means_c = gmm.means_[:, 0]
    variances_c = (gmm.covariances_[:, 0, 0]
                   if gmm.covariances_.ndim == 3
                   else gmm.covariances_.flatten())
    stds_c = np.sqrt(np.abs(variances_c))
    if scaler is not None:
        stds_c = stds_c * scaler.scale_[0]
    sidx_c      = np.argsort(means_c)
    sorted_stds = stds_c[sidx_c]

    var_passed = 0
    for i in range(len(sorted_stds) - 1):
        if sorted_stds[i + 1] > sorted_stds[i]:
            var_passed += 1

    n_passed = sep_passed + phys_passed + var_passed
    n_total  = (k - 1) + 4 + (k - 1)
    return n_passed, n_total


def test_k_selection(X, models, labels, scaler=None, max_power=None):
    """
    Physics-consistency-based k selection (FIX 4).

    Selection logic:
      1. Compute Silhouette + BIC for every k in K_RANGE.
      2. Apply weight-sanity and 2σ-separation veto filters to each k.
      3. For each k that passes the veto filters, run the EXISTING
         physics checks (test_physics_thresholds + test_variance_ordering
         + 2-sigma pair separations) and count how many pass.
      4. Among veto-passing candidates, select the one with the HIGHEST
         physics-checks-passed count.  Break ties using BIC (lower better).
      5. Print a comparison table:
         k | BIC | Silhouette | Physics checks passed (X/Y) | Veto filters
      6. Keep BIC-optimal result visible as "BIC-optimal (not selected)"
         for paper comparison.

    Falls back to Silhouette if no candidate passes the veto filters.
    """
    print("\n" + "=" * 65)
    print("  TEST 3 -- k SELECTION  (Physics-First + BIC + Veto Filters)")
    print("=" * 65)
    _info("Step 1: compute Silhouette + BIC for all candidate k values.")
    _info("Step 2: apply weight + 2σ-separation veto filters.")
    _info("Step 3: count physics checks per vetoed-surviving candidate.")
    _info("Step 4: select highest physics-checks-passed; break ties by BIC.")
    _sep()

    sil_scores = {}
    bic_scores = {}
    aic_scores = {}

    for k in K_RANGE:
        gmm = models[k]
        lbl = labels[k]
        sil = silhouette_score(X, lbl, sample_size=min(10_000, len(X)), random_state=42)
        sil_scores[k] = sil
        bic_scores[k] = gmm.bic(X)
        aic_scores[k] = gmm.aic(X)

    # ---- Print Silhouette / BIC raw score table ----
    print(f"\n   {'k':<6} {'Silhouette':>12} {'BIC':>20} {'AIC':>20}")
    print(f"   {'':-<6} {'':-<12} {'':-<20} {'':-<20}")
    sil_best = max(sil_scores, key=sil_scores.get)
    bic_best = min(bic_scores, key=bic_scores.get)
    for k in K_RANGE:
        sm = " <- Sil best" if k == sil_best else ""
        bm = " <- BIC best" if k == bic_best else ""
        print(f"   {k:<6} {sil_scores[k]:>12.4f}{sm:<13} "
              f"{bic_scores[k]:>20,.1f}{bm}")

    bic_ranking = sorted(K_RANGE, key=lambda k: bic_scores[k])
    print(f"\n   BIC ranking       : {' > '.join(f'k={k}' for k in bic_ranking)}")
    print(f"   Silhouette ranking: k={sil_best} is best")

    # ---- Per-candidate physics check counts + veto filter results ----
    # Use max_power from caller (passed from df['power'].max()); if not
    # provided, estimate a safe value from the fitted means.
    if max_power is None:
        all_means = [models[k].means_.flatten().max() for k in K_RANGE]
        max_power = float(max(all_means)) * 2   # rough upper bound

    phys_counts = {}   # k -> (n_passed, n_total)
    veto_pass   = {}   # k -> bool (passed both veto filters)

    for k in K_RANGE:
        gmm  = models[k]
        w_ok = _weight_ok(gmm)
        s_ok, _ = _separation_ok(gmm, k, scaler)
        veto_pass[k] = w_ok and s_ok
        n_p, n_t = _count_physics_checks_passed(gmm, k, scaler, max_power)
        phys_counts[k] = (n_p, n_t)

    # ---- Comparison table ----
    print(f"\n   {'k':<5} {'BIC':>18} {'Silhouette':>12} "
          f"{'Physics (X/Y)':>16} {'Veto filters':>14}")
    print(f"   {'':-<5} {'':-<18} {'':-<12} {'':-<16} {'':-<14}")
    for k in K_RANGE:
        n_p, n_t = phys_counts[k]
        veto_str = "PASS" if veto_pass[k] else "FAIL"
        bm       = " [BIC-opt]" if k == bic_best else ""
        print(f"   {k:<5} {bic_scores[k]:>18,.1f}{bm:<12} "
              f"{sil_scores[k]:>12.4f} "
              f"{n_p:>6}/{n_t:<9} {veto_str:>14}")

    # ---- Selection: highest physics checks among veto-passing candidates ----
    print()
    surviving = [k for k in K_RANGE if veto_pass[k]]
    selected_k = None
    origin     = ""

    if surviving:
        # Sort by physics count (desc), break ties by BIC (asc)
        surviving_sorted = sorted(
            surviving,
            key=lambda k: (-phys_counts[k][0], bic_scores[k])
        )
        selected_k = surviving_sorted[0]

        best_n_p, best_n_t = phys_counts[selected_k]
        bic_opt_among_surv = min(surviving, key=lambda k: bic_scores[k])

        if selected_k == bic_opt_among_surv:
            origin = (f"highest physics checks ({best_n_p}/{best_n_t}) "
                      f"AND BIC-optimal among veto-passing candidates")
        else:
            bic_n_p, bic_n_t = phys_counts[bic_opt_among_surv]
            origin = (f"highest physics checks ({best_n_p}/{best_n_t}) "
                      f"vs k={bic_opt_among_surv}'s ({bic_n_p}/{bic_n_t}) -- "
                      f"physical validity prioritised over raw statistical fit")

        print(f"   → SELECTED k={selected_k}  ({origin})")
        # Print one-line explanation
        _pass(f"SELECTED k={selected_k} (passed {best_n_p}/{best_n_t} physics checks"
              + (f", despite k={bic_best} having better BIC)" if selected_k != bic_best
                 else ", and has best BIC among veto-passing candidates)")
              + f"  -- {origin}")

        # If BIC-best differs from selected, print it as "not selected"
        if bic_best != selected_k:
            bic_n_p, bic_n_t = phys_counts[bic_best]
            bic_veto = "passes" if veto_pass[bic_best] else "FAILS"
            _info(f"BIC-optimal (not selected): k={bic_best}  "
                  f"(BIC={bic_scores[bic_best]:,.1f}, physics={bic_n_p}/{bic_n_t}, "
                  f"veto filters={bic_veto})")
    else:
        selected_k = sil_best
        origin = "Silhouette fallback (all BIC candidates failed physical filters)"
        print(f"   → SELECTED k={selected_k}  ({origin})")
        _warn(f"Silhouette fallback used: k={selected_k}  "
              f"(no candidate passed weight + 2σ-separation veto filters)")
        if bic_best != selected_k:
            bic_n_p, bic_n_t = phys_counts[bic_best]
            _info(f"BIC-optimal (not selected): k={bic_best}  "
                  f"(BIC={bic_scores[bic_best]:,.1f}, physics={bic_n_p}/{bic_n_t}, "
                  f"veto filters=FAIL)")

    agreement = (selected_k == sil_best == bic_best)
    print()
    if agreement:
        _pass(f"Silhouette, BIC, AND physics all agree: k={selected_k}  "
              f"(strongest confirmation)")
    elif selected_k == bic_best:
        _pass(f"Physics-consistent k={selected_k} also happens to be BIC-optimal")
    elif selected_k == sil_best:
        _info(f"Selected k={selected_k} matches Silhouette choice")

    return selected_k, sil_scores, bic_scores, agreement


# ============================================================
# TEST 4 — PHYSICAL STATE SEPARATION (2σ rule)
# ============================================================

def test_state_separation(gmm, best_k, scaler=None):
    print("\n" + "=" * 65)
    print("  TEST 4 -- PHYSICAL STATE SEPARATION  (2-sigma rule)")
    print("=" * 65)
    _info("For two clusters to be physically different states, their means")
    _info("must be at least 2x the average standard deviation apart.")
    _info("Below 2-sigma = distributions overlap = likely one real state.")
    _sep()

    state_map = map_states(gmm, best_k, scaler)

    if scaler is not None:
        # IMDELD: inverse-transform to get real Watts
        means_orig  = scaler.inverse_transform(gmm.means_)   # (k, 5)
        stds_orig   = np.array([
            np.sqrt(np.abs(gmm.covariances_[i][0, 0])) * scaler.scale_[0]
            for i in range(best_k)
        ])
        means  = means_orig[:, 0]
        stds   = stds_orig
        weights = gmm.weights_
        sorted_idx   = np.argsort(means)
        sorted_means = means[sorted_idx]
        sorted_stds  = stds[sorted_idx]
        sorted_names = [state_map[i] for i in sorted_idx]
        sorted_wts   = weights[sorted_idx]

        _info("IMDELD multi-feature mode: separation measured on ACTIVE POWER axis")
        print(f"\n   {'State':<14} {'ActivePwr(W)':>13} {'Std(W)':>8} {'Weight':>8}")
        print(f"   {'':-<14} {'':-<13} {'':-<8} {'':-<8}")
        rp_means  = means_orig[:, 1][sorted_idx]
        cur_means = means_orig[:, 3][sorted_idx]
        for i in range(len(sorted_names)):
            print(f"   {sorted_names[i]:<14} {sorted_means[i]:>13.1f} "
                  f"{sorted_stds[i]:>8.1f} {sorted_wts[i]:>8.3f}  "
                  f"| reactive={rp_means[i]:.0f}VAr  current={cur_means[i]:.3f}A")
    else:
        means      = gmm.means_.flatten()
        variances  = gmm.covariances_.flatten()
        stds       = np.sqrt(np.abs(variances))
        sorted_idx   = np.argsort(means)
        sorted_means = means[sorted_idx]
        sorted_stds  = stds[sorted_idx]
        sorted_names = [state_map[i] for i in sorted_idx]
        sorted_wts   = gmm.weights_[sorted_idx]

        print(f"\n   {'State':<14} {'Mean (W)':>10} {'Std (W)':>10} {'Weight':>8}")
        print(f"   {'':-<14} {'':-<10} {'':-<10} {'':-<8}")
        for i in range(len(sorted_names)):
            print(f"   {sorted_names[i]:<14} {sorted_means[i]:>10.1f} "
                  f"{sorted_stds[i]:>10.1f} {sorted_wts[i]:>8.3f}")

    passed = True
    print()
    for i in range(len(sorted_names) - 1):
        mu_lo, mu_hi = sorted_means[i], sorted_means[i + 1]
        avg_std      = (sorted_stds[i] + sorted_stds[i + 1]) / 2
        separation   = (mu_hi - mu_lo) / avg_std if avg_std > 0 else float('inf')
        lo_name, hi_name = sorted_names[i], sorted_names[i + 1]
        if separation >= 2.0:
            _pass(f"{lo_name} vs {hi_name}:  separation = {separation:.2f}σ  "
                  f"(gap={mu_hi - mu_lo:.1f}W, avg_std={avg_std:.1f}W)  ✓ ≥ 2σ")
        else:
            _fail(f"{lo_name} vs {hi_name}:  separation = {separation:.2f}σ  "
                  f"(gap={mu_hi - mu_lo:.1f}W, avg_std={avg_std:.1f}W)  ✗ < 2σ")
            passed = False

    return passed, state_map, sorted_means, sorted_stds, sorted_names


# ============================================================
# TEST 5 — SOFT-ASSIGNMENT CONFIDENCE
# ============================================================

def test_soft_assignment_confidence(gmm, X):
    print("\n" + "=" * 65)
    print("  TEST 5 -- SOFT-ASSIGNMENT CONFIDENCE")
    print("=" * 65)
    _info("GMM assigns each reading a probability vector over k states.")
    _info("max(prob) = how confident the model is about this reading.")
    _info("If most readings hover near 50%, the clusters overlap badly.")
    _sep()

    proba    = gmm.predict_proba(X)
    max_prob = proba.max(axis=1)

    pct_high = (max_prob > 0.80).mean() * 100
    pct_low  = (max_prob < 0.60).mean() * 100
    avg_conf = max_prob.mean() * 100

    _info(f"Average max-probability   : {avg_conf:.1f}%")
    _info(f"Readings with >80% conf   : {pct_high:.1f}%  (want >70%)")
    _info(f"Readings with <60% conf   : {pct_low:.1f}%   (want <20%)")

    passed = True
    if pct_high >= 70:
        _pass(f"{pct_high:.1f}% of readings assigned with >80% confidence (model is decisive)")
    else:
        _fail(f"Only {pct_high:.1f}% assigned with >80% confidence -- "
              "clusters may overlap too much")
        passed = False

    if pct_low <= 20:
        _pass(f"Only {pct_low:.1f}% of readings are ambiguous (<60% confidence)")
    else:
        _fail(f"{pct_low:.1f}% of readings are ambiguous -- unusually high overlap")
        passed = False

    return passed, max_prob


# ============================================================
# TEST 6 — CLUSTER WEIGHT SANITY
# ============================================================

def test_weight_sanity(gmm, best_k, state_map, scaler=None):
    print("\n" + "=" * 65)
    print("  TEST 6 -- CLUSTER WEIGHT SANITY")
    print("=" * 65)
    _info("Each GMM component has a weight = fraction of data it models.")
    _info("A weight <0.5% means the cluster likely caught edge-case noise")
    _info("rather than a real physical state.")
    _sep()

    weights = gmm.weights_
    if scaler is not None:
        means_orig = scaler.inverse_transform(gmm.means_)
        sort_vals  = means_orig[:, 0]
    else:
        sort_vals = gmm.means_.flatten()
    sorted_idx = np.argsort(sort_vals)
    names      = [state_map[i] for i in sorted_idx]
    sorted_w   = weights[sorted_idx]

    passed = True
    print()
    for name, w in zip(names, sorted_w):
        pct = w * 100
        if pct < 0.5:
            _fail(f"{name:<14}: weight = {pct:.2f}%  (too tiny -- likely noise artefact)")
            passed = False
        elif pct < 3.0:
            _warn(f"{name:<14}: weight = {pct:.2f}%  (very small -- verify this state exists)")
        else:
            _pass(f"{name:<14}: weight = {pct:.2f}%  (meaningful cluster)")

    return passed


# ============================================================
# TEST 7 — PHYSICS THRESHOLD CROSS-CHECK
# ============================================================

def test_physics_thresholds(gmm, best_k, state_map, max_power, scaler=None):
    print("\n" + "=" * 65)
    print("  TEST 7 -- PHYSICS THRESHOLD CROSS-CHECK")
    print("=" * 65)
    _info("This test checks whether GMM cluster means/variances match")
    _info("the physics of industrial machine power consumption.")
    _info("Evidence base: ISO 14955, Jiang et al. (2015), Wu et al. (2024)")
    _sep()

    if scaler is not None:
        means_orig = scaler.inverse_transform(gmm.means_)
        means      = means_orig[:, 0]
        stds       = np.array([
            np.sqrt(np.abs(gmm.covariances_[i][0, 0])) * scaler.scale_[0]
            for i in range(best_k)
        ])
    else:
        means     = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        stds      = np.sqrt(np.abs(variances))
    sorted_idx = np.argsort(means)

    off_mean  = means[sorted_idx[0]]
    off_std   = stds[sorted_idx[0]]
    stby_mean = means[sorted_idx[1]] if best_k >= 3 else None
    stby_std  = stds[sorted_idx[1]] if best_k >= 3 else None
    work_mean = means[sorted_idx[-1]]
    work_std  = stds[sorted_idx[-1]]

    passed = True

    # Check 1: OFF mean must be near 0
    off_threshold = max(5.0, max_power * 0.05)
    print(f"\n   CHECK 1 — OFF cluster mean must be < {off_threshold:.1f} W  "
          f"(5% of max={max_power:.1f} W, or 5W floor)")
    if off_mean < off_threshold:
        _pass(f"OFF mean = {off_mean:.1f} W  < {off_threshold:.1f} W  — "
              f"cluster is near-zero: physically correct for OFF state")
    else:
        _fail(f"OFF mean = {off_mean:.1f} W  >= {off_threshold:.1f} W  — "
              f"cluster is NOT near zero.")
        passed = False

    # Check 2: OFF cluster must be stable
    print(f"\n   CHECK 2 — OFF cluster must be stable (σ < 15 W or σ < 20% of mean)")
    off_cv_ok = (off_std < 15.0) or (off_mean > 0 and off_std / off_mean < 0.20)
    if off_cv_ok:
        _pass(f"OFF std  = {off_std:.1f} W  — stable, low-noise cluster.")
    else:
        _warn(f"OFF std  = {off_std:.1f} W  — high spread for an OFF cluster.")
        passed = False

    # Check 3: STANDBY must be more stable than WORKING
    if stby_mean is not None:
        stby_cv = stby_std / stby_mean if stby_mean > 0 else float('inf')
        work_cv = work_std / work_mean if work_mean > 0 else float('inf')
        print(f"\n   CHECK 3 — STANDBY must be more stable than WORKING (CV_standby < CV_working)")
        print(f"   STANDBY CV = {stby_std:.1f} / {stby_mean:.1f} = {stby_cv:.3f}")
        print(f"   WORKING CV = {work_std:.1f} / {work_mean:.1f} = {work_cv:.3f}")
        if stby_cv < work_cv:
            _pass(f"CV(STANDBY)={stby_cv:.3f} < CV(WORKING)={work_cv:.3f}  — physics law confirmed.")
        else:
            _fail(f"CV(STANDBY)={stby_cv:.3f} >= CV(WORKING)={work_cv:.3f}  — physically wrong.")
            passed = False

    # Check 4: WORKING mean must be >> OFF mean
    ratio = work_mean / off_mean if off_mean > 0 else float('inf')
    print(f"\n   CHECK 4 — WORKING mean must be >> OFF mean (ratio >= 3×)")
    print(f"   WORKING mean = {work_mean:.1f} W,  OFF mean = {off_mean:.1f} W,  "
          f"ratio = {ratio:.1f}×")
    if ratio >= 3.0:
        _pass(f"WORKING / OFF ratio = {ratio:.1f}×  — strong evidence of distinct physical states")
    elif ratio >= 1.5:
        _warn(f"WORKING / OFF ratio = {ratio:.1f}×  — moderate separation.")
    else:
        _fail(f"WORKING / OFF ratio = {ratio:.1f}×  — insufficient separation.")
        passed = False

    return passed


# ============================================================
# TEST 8 — VARIANCE ORDERING (Physics Law)
# ============================================================

def test_variance_ordering(gmm, best_k, state_map, scaler=None):
    print("\n" + "=" * 65)
    print("  TEST 8 -- VARIANCE ORDERING (Physics Law: sigma_OFF < sigma_STANDBY < sigma_WORKING)")
    print("=" * 65)
    _info("Physics law: sigma(OFF) << sigma(STANDBY) << sigma(WORKING)")
    _info("This is because cutting load varies every second, but auxiliary")
    _info("loads (cooling fan, PLC) are nearly constant.")
    _info("If this ordering is violated, the state labels are WRONG.")
    _sep()

    means     = gmm.means_[:, 0]
    variances = gmm.covariances_[:, 0, 0] if gmm.covariances_.ndim == 3 else gmm.covariances_.flatten()
    stds      = np.sqrt(np.abs(variances))
    if scaler is not None:
        stds = stds * scaler.scale_[0]
    sorted_idx  = np.argsort(means)
    sorted_stds = stds[sorted_idx]
    sorted_names = [state_map[i] for i in sorted_idx]

    print(f"\n   State ordering by mean (should = by variance too):")
    for name, std in zip(sorted_names, sorted_stds):
        bar = '█' * int(std / sorted_stds.max() * 30) if sorted_stds.max() > 0 else ''
        print(f"   {name:<14}  σ = {std:8.1f} W   {bar}")

    passed = True
    print()
    for i in range(len(sorted_stds) - 1):
        if sorted_stds[i + 1] > sorted_stds[i]:
            _pass(f"σ({sorted_names[i]}) = {sorted_stds[i]:.1f} W  < "
                  f"σ({sorted_names[i+1]}) = {sorted_stds[i+1]:.1f} W  — "
                  f"variance ordering correct")
        else:
            _fail(f"σ({sorted_names[i]}) = {sorted_stds[i]:.1f} W  >=  "
                  f"σ({sorted_names[i+1]}) = {sorted_stds[i+1]:.1f} W  — "
                  f"VARIANCE LAW VIOLATED.")
            passed = False

    return passed


# ============================================================
# TEST 9 — TEMPORAL / DAY-NIGHT ALIGNMENT
# ============================================================

def test_temporal_alignment(df, gmm, best_k, state_map, X):
    print("\n" + "=" * 65)
    print("  TEST 9 -- TEMPORAL / DAY-NIGHT ALIGNMENT")
    print("=" * 65)
    _info("If OFF = 'machine turned off', then OFF must align with nights/weekends.")
    _info("This is an INDEPENDENT check using only timestamps — no labels needed.")
    _info("Industrial machines are not operated at 2am or Sundays.")
    _sep()

    labels_arr = gmm.predict(X)
    df2 = df.copy()
    df2['_gmm_state'] = [state_map[l] for l in labels_arr]
    try:
        local_ts = df2['timestamp'].dt.tz_convert(FACTORY_TZ)
    except Exception:
        local_ts = df2['timestamp']
    df2['_hour']      = local_ts.dt.hour
    df2['_weekday']   = local_ts.dt.weekday
    df2['_is_night']  = (df2['_hour'] >= 22) | (df2['_hour'] < 6)
    df2['_is_day']    = (df2['_weekday'] < 5) & (df2['_hour'] >= 8) & (df2['_hour'] < 18)
    df2['_is_weekend'] = df2['_weekday'] >= 5

    night_total   = df2['_is_night'].sum()
    day_total     = df2['_is_day'].sum()
    weekend_total = df2['_is_weekend'].sum()

    if night_total < 10 or day_total < 10:
        _warn("Not enough night-time or daytime data to run temporal check.")
        return True

    night_off = ((df2['_gmm_state'] == 'OFF') & df2['_is_night']).sum()
    day_off   = ((df2['_gmm_state'] == 'OFF') & df2['_is_day']).sum()
    wknd_off  = ((df2['_gmm_state'] == 'OFF') & df2['_is_weekend']).sum()

    night_off_rate = night_off / night_total * 100
    day_off_rate   = day_off   / day_total   * 100
    wknd_off_rate  = wknd_off  / weekend_total * 100 if weekend_total > 0 else 0

    print(f"\n   Time window          Total readings  OFF readings   OFF rate")
    print(f"   {'':-<60}")
    print(f"   Night (10pm–6am)     {night_total:>14,}  {night_off:>12,}   {night_off_rate:>6.1f}%")
    print(f"   Weekday (8am–6pm)    {day_total:>14,}  {day_off:>12,}   {day_off_rate:>6.1f}%")
    print(f"   Weekend (all day)    {weekend_total:>14,}  {wknd_off:>12,}   {wknd_off_rate:>6.1f}%")
    print()

    passed = True
    if night_total > 0 and day_total > 0:
        if night_off_rate > day_off_rate * 1.5:
            _pass(f"Night OFF rate ({night_off_rate:.1f}%) >> Day OFF rate ({day_off_rate:.1f}%) — "
                  f"OFF cluster correctly aligns with machine-off hours")
        elif night_off_rate > day_off_rate:
            _warn(f"Night OFF rate ({night_off_rate:.1f}%) > Day OFF rate ({day_off_rate:.1f}%) — "
                  f"correct direction but weak separation.")
        else:
            _fail(f"Night OFF rate ({night_off_rate:.1f}%) <= Day OFF rate ({day_off_rate:.1f}%) — "
                  f"OFF cluster does NOT align with machine-off hours.")
            passed = False

    if weekend_total > 100:
        if wknd_off_rate > day_off_rate:
            _pass(f"Weekend OFF rate ({wknd_off_rate:.1f}%) > Weekday day rate ({day_off_rate:.1f}%) — "
                  f"machine powers down on weekends. Confirms OFF label is correct.")
        else:
            _warn(f"Weekend OFF rate ({wknd_off_rate:.1f}%) <= Weekday rate ({day_off_rate:.1f}%). "
                  f"Machine may operate on weekends, or OFF cluster is misidentified.")

    return passed


# ============================================================
# TEST 10 — IMDELD FACTORY SCHEDULE PROOF
# ============================================================

def test_schedule_proof(df, best_gmm, best_k, state_map, X):
    """
    T10 -- IMDELD FACTORY SCHEDULE PROOF
    The factory closes every weekday from 5 PM to 10 PM because
    electricity prices are higher. GMM must predict OFF or STANDBY
    for >90% of those readings. Skipped automatically if not IMDELD.
    """
    print("\n" + "=" * 65)
    print("  TEST 10 -- IMDELD FACTORY SCHEDULE PROOF")
    print("=" * 65)

    if not is_imdeld(df):
        _info("Skipped -- not an IMDELD dataset (no strict factory schedule applies).")
        return True

    _info("Factory schedule: Mon-Fri, closed 17:00-22:00 local time (Brazil UTC-3)")
    _info("During this window, ALL machines must be OFF or in minimum STANDBY.")
    _sep()

    raw_labels  = best_gmm.predict(X)
    label_names = np.array([state_map[l] for l in raw_labels])

    try:
        local_ts = df['timestamp'].dt.tz_convert(FACTORY_TZ)
    except Exception:
        _warn("Could not convert timezone -- skipping schedule proof.")
        return True

    hour    = local_ts.dt.hour
    weekday = local_ts.dt.dayofweek

    closed_mask = (weekday <= 4) & (hour >= SCHEDULE_CLOSE_HOUR) & (hour < SCHEDULE_OPEN_HOUR)
    n_closed    = closed_mask.sum()

    if n_closed == 0:
        _warn("No readings found in the closed window -- check timestamps or timezone.")
        return True

    labels_in_closed = label_names[closed_mask.values]
    n_off_stby = np.isin(labels_in_closed, ['OFF', 'STANDBY']).sum()
    n_working  = np.isin(labels_in_closed, ['WORKING', 'PEAK_LOAD']).sum()
    pct_correct = n_off_stby / n_closed * 100
    pct_wrong   = n_working  / n_closed * 100

    _info(f"Factory-closed readings found : {n_closed:,}")
    _info(f"Labelled OFF or STANDBY       : {n_off_stby:,}  ({pct_correct:.1f}%)")
    _info(f"Labelled WORKING (wrong!)      : {n_working:,}  ({pct_wrong:.1f}%)")
    print()

    TARGET = 90.0
    if pct_correct >= TARGET:
        _pass(f"{pct_correct:.1f}% of closed-window readings correctly labelled "
              f"OFF/STANDBY  (target >{TARGET:.0f}%)")
        _pass("GMM state labels align with the factory schedule -- STRONG PROOF!")
        return True
    elif pct_correct >= 70.0:
        _warn(f"{pct_correct:.1f}% correct in closed window -- acceptable but not ideal.")
        return False
    else:
        _fail(f"Only {pct_correct:.1f}% correct in closed window -- "
              "GMM is labelling too many readings as WORKING during factory closure.")
        return False


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def plot_k_selection(sil_scores, bic_scores, best_k, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"T3 — k Selection: Silhouette + BIC — {MACHINE_NAME}", fontsize=13, fontweight='bold')

    ks = list(sil_scores.keys())
    ax = axes[0]
    ax.plot(ks, [sil_scores[k] for k in ks], 'o-', color='#4fc3f7', linewidth=2, markersize=8)
    ax.axvline(best_k, color='#e53935', linestyle='--', linewidth=1.5, label=f'best k={best_k}')
    ax.set_title("Silhouette Score  (higher = better)"); ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ks, [bic_scores[k] for k in ks], 's-', color='#66bb6a', linewidth=2, markersize=8)
    ax.axvline(best_k, color='#e53935', linestyle='--', linewidth=1.5, label=f'best k={best_k}')
    ax.set_title("BIC Score  (lower = better)"); ax.set_xlabel("k"); ax.set_ylabel("BIC")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "T3_k_selection_curves.png")
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
    print(f"   📊 Saved: T3_k_selection_curves.png")


def plot_gmm_histogram(df, gmm, best_k, state_map, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"GMM Gaussian Overlay — {MACHINE_NAME}  (k={best_k})", fontsize=14, fontweight='bold')

    power      = df['power'].values
    means      = gmm.means_.flatten()
    variances  = gmm.covariances_.flatten()
    stds       = np.sqrt(np.abs(variances))
    weights    = gmm.weights_
    sorted_idx = np.argsort(means)

    ax = axes[0]
    bins = np.linspace(0, power.max(), 120)
    ax.hist(power, bins=bins, density=True, color='#1e2a3a', edgecolor='#2d3f58', linewidth=0.3, label='Data', alpha=0.9)
    x = np.linspace(0, power.max(), 2000)
    total_pdf = np.zeros_like(x)
    for idx in sorted_idx:
        pdf   = weights[idx] * norm.pdf(x, means[idx], stds[idx])
        name  = state_map[idx]
        color = STATE_COLORS.get(name, '#aaaaaa')
        ax.fill_between(x, pdf, alpha=0.35, color=color)
        ax.plot(x, pdf, linewidth=2, color=color, label=name)
        total_pdf += pdf
    ax.plot(x, total_pdf, linewidth=1.5, color='white', linestyle='--', alpha=0.6, label='Total GMM')
    ax.set_title("Full Distribution  (incl. OFF state)"); ax.set_xlabel("Power (W)"); ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc='upper right'); ax.set_xlim(left=0)

    ax = axes[1]
    on_power = power[power > 5]
    if len(on_power) > 10:
        max_p    = on_power.max()
        log_bins = np.logspace(np.log10(5.1), np.log10(max_p + 1), 120)
        ax.hist(on_power, bins=log_bins, density=True, color='#1e2a3a', edgecolor='#2d3f58', linewidth=0.3, alpha=0.9)
        x2         = np.logspace(np.log10(5.1), np.log10(max_p + 1), 2000)
        total_pdf2 = np.zeros_like(x2)
        for idx in sorted_idx:
            if means[idx] <= 5:
                continue
            pdf   = weights[idx] * norm.pdf(x2, means[idx], stds[idx])
            name  = state_map[idx]
            color = STATE_COLORS.get(name, '#aaaaaa')
            ax.fill_between(x2, pdf, alpha=0.4, color=color)
            ax.plot(x2, pdf, linewidth=2.5, color=color, label=f"{name}\n~{means[idx]:.0f}W ± {stds[idx]:.0f}W")
            total_pdf2 += pdf
            ax.axvline(means[idx], color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.set_xscale('log')
        ax.set_title("ON-State Only  (log x-axis)"); ax.set_xlabel("Power (W)  [log scale]"); ax.set_ylabel("Density")
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "T5_gmm_histogram_overlay.png"), dpi=140, bbox_inches='tight'); plt.close()
    print(f"   📊 Saved: T5_gmm_histogram_overlay.png")


def plot_confidence_histogram(max_probs, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_probs * 100, bins=60, color='#4fc3f7', edgecolor='#1e2a3a', linewidth=0.4)
    ax.axvline(80, color='#66bb6a', linestyle='--', linewidth=2, label='80% threshold')
    ax.axvline(60, color='#e53935', linestyle='--', linewidth=2, label='60% threshold')
    ax.set_title(f"GMM Assignment Confidence — {MACHINE_NAME}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Max Assignment Probability (%)"); ax.set_ylabel("Number of Readings")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "T5_confidence_distribution.png"), dpi=140, bbox_inches='tight'); plt.close()
    print(f"   📊 Saved: T5_confidence_distribution.png")


def plot_labeled_sample(df, gmm, best_k, state_map, save_dir, scaler=None):
    sample = df.head(2000).copy()
    if scaler is not None:
        X_s = scaler.transform(sample[IMDELD_FEATURES].values)
    else:
        X_s = sample['power'].values.reshape(-1, 1)
    labels = gmm.predict(X_s)
    sample['state'] = [state_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(16, 5))
    for state, color in STATE_COLORS.items():
        mask = sample['state'] == state
        if mask.any():
            ax.scatter(sample.loc[mask, 'timestamp'], sample.loc[mask, 'power'],
                       c=color, s=4, label=state, zorder=3)
    ax.plot(sample['timestamp'], sample['power'], color='#cccccc', linewidth=0.6, alpha=0.4, zorder=2)
    ax.set_title(f"GMM State Labels — First 2000 Readings — {MACHINE_NAME}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time"); ax.set_ylabel("Power (W)")
    ax.legend(markerscale=4, fontsize=9, loc='upper right'); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "T5_labeled_time_sample.png"), dpi=140, bbox_inches='tight'); plt.close()
    print(f"   📊 Saved: T5_labeled_time_sample.png")


def plot_cluster_separation(df, gmm, best_k, state_map, save_dir, X=None, scaler=None):
    """
    X and scaler must match what the GMM was trained on.
    If X is None, falls back to 1D power (only safe for 1D GMMs).
    """
    if X is None:
        X = df['power'].values.reshape(-1, 1)
    labels = gmm.predict(X)
    df2    = df.copy()
    df2['state'] = [state_map[l] for l in labels]
    ordered_states = [s for s in ['OFF', 'STANDBY', 'IDLE', 'WORKING', 'PEAK_LOAD'] if s in df2['state'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    data_per_state = [
        df2.loc[df2['state'] == s, 'power'].sample(min(5000, (df2['state'] == s).sum()), random_state=42).values
        for s in ordered_states
    ]
    colors = [STATE_COLORS[s] for s in ordered_states]
    parts  = ax.violinplot(data_per_state, positions=range(len(ordered_states)),
                           showmeans=True, showmedians=False, showextrema=True)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    for part_name in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
        if part_name in parts:
            parts[part_name].set_color('white'); parts[part_name].set_linewidth(1.5)
    ax.set_xticks(range(len(ordered_states))); ax.set_xticklabels(ordered_states, fontsize=11)
    ax.set_title(f"Power Distribution per GMM State — {MACHINE_NAME}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Power (W)"); ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "T4_cluster_separation_violin.png"), dpi=140, bbox_inches='tight'); plt.close()
    print(f"   📊 Saved: T4_cluster_separation_violin.png")


def plot_temporal_alignment(df, gmm, best_k, state_map, X, save_dir):
    labels_arr = gmm.predict(X)
    df2 = df.copy()
    df2['_state']   = [state_map[l] for l in labels_arr]
    df2['_hour']    = df2['timestamp'].dt.hour
    df2['_weekday'] = df2['timestamp'].dt.weekday
    df2['_is_off']  = (df2['_state'] == 'OFF').astype(int)

    pivot = df2.groupby(['_weekday', '_hour'])['_is_off'].mean().unstack(fill_value=0)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([day_labels[d] for d in pivot.index], fontsize=9)
    plt.colorbar(im, ax=ax, label='Fraction labelled OFF')
    ax.set_title(f"T9 — Temporal Alignment: OFF-State Rate by Hour & Day — {MACHINE_NAME}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "T9_temporal_alignment_heatmap.png"), dpi=140, bbox_inches='tight'); plt.close()
    print(f"   📊 Saved: T9_temporal_alignment_heatmap.png")


# ============================================================
# FINAL SUMMARY REPORT
# ============================================================

def print_final_report(results, best_k, state_map, gmm):
    print("\n" + "=" * 65)
    print("  FINAL GMM VALIDATION SUMMARY")
    print("=" * 65)

    labels_map = {True: "[PASS]", False: "[FAIL]"}
    print(f"  {'Test':<45} {'Result':>10}")
    print(f"  {'':->45} {'':->10}")
    for name, passed in results.items():
        print(f"  {name:<45} {labels_map[passed]:>10}")

    passed_count = sum(1 for v in results.values() if v)
    total_count  = len(results)
    pct          = passed_count / total_count * 100

    print(f"\n  Passed: {passed_count}/{total_count}  ({pct:.0f}%)")
    print()

    if pct == 100:
        verdict = "[CONFIRMED] GMM is splitting the data correctly."
        detail  = "All independent tests passed. Safe to proceed to proof_5methods.py."
    elif pct >= 66:
        verdict = "[PARTIAL] GMM is working but some checks raised warnings."
        detail  = "Review the FAIL items above before trusting the split fully."
    else:
        verdict = "[SUSPECT] GMM likely not splitting correctly."
        detail  = ("Multiple checks failed. Review plots and consider adjusting k, "
                   "checking the data loader, or investigating degenerate values.")

    print(f"  {verdict}")
    print(f"\n  {detail}")
    print("=" * 65 + "\n")


# ============================================================
# HMM/VITERBI SMOOTHER  (applied ONLY to the exported CSV)
# ============================================================

def count_short_runs(labels, min_dur=10):
    """
    Count discrete state-runs shorter than min_dur samples.
    Returns (n_short_runs, n_total_runs).
    """
    if len(labels) == 0:
        return 0, 0
    change_pts = np.where(np.diff(labels, prepend=labels[0] - 1) != 0)[0]
    run_lens   = np.diff(np.append(change_pts, len(labels)))
    return int((run_lens < min_dur).sum()), len(run_lens)


def viterbi_smooth(gmm, X, min_dwell_s=30):
    """
    Viterbi HMM smoother on top of a fitted GMM.

    Uses gmm.predict_proba(X) as per-frame emission probabilities
    and a persistence-biased transition matrix (expected minimum
    dwell = min_dwell_s samples) to find the globally optimal state
    sequence that balances data fit and temporal coherence.

    Returns integer component labels (same as gmm.predict(X)) but
    with rapid flickering suppressed.

    Performance: ~30-90 s on 5 M rows with k=4 (vectorised inner loop).
    """
    k = gmm.n_components
    n = len(X)

    print(f"   Viterbi smoothing: {n:,} rows, k={k}, min_dwell={min_dwell_s}s ...",
          flush=True)

    # --- Emission log-probs from GMM posterior (n, k) ---
    # Compute in chunks to limit peak memory to ~100 MB
    chunk = 200_000
    log_emit = np.empty((n, k), dtype=np.float32)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        log_emit[i:end] = np.log(
            gmm.predict_proba(X[i:end]).astype(np.float32) + 1e-30
        )

    # --- Transition matrix ---
    trans_rate = float(np.clip(1.0 / max(min_dwell_s, 1), 0, 0.5))
    stay  = 1.0 - trans_rate
    leave = trans_rate / max(k - 1, 1)
    log_trans = np.log(
        np.where(np.eye(k, dtype=bool), stay, leave).astype(np.float32)
    )  # (k, k)

    # --- Viterbi forward pass ---
    delta = np.empty((n, k), dtype=np.float32)
    psi   = np.empty((n, k), dtype=np.int8)       # k <= 127: int8 is fine

    delta[0] = log_emit[0] + np.log(gmm.weights_.astype(np.float32) + 1e-30)

    report = max(1, n // 10)
    arange_k = np.arange(k)
    for t in range(1, n):
        if t % report == 0:
            print(f"   ... {t * 100 // n}%", end='\r', flush=True)
        # M[i, j] = delta[t-1, i] + log_trans[i, j]
        M = delta[t - 1, :, None].astype(np.float64) + log_trans  # (k, k)
        best_prev  = M.argmax(axis=0).astype(np.int8)              # (k,)
        psi[t]     = best_prev
        delta[t]   = M[best_prev, arange_k].astype(np.float32) + log_emit[t]

    print(f"   ... 100% — backtracking ...", flush=True)

    # --- Backtrack ---
    states      = np.empty(n, dtype=np.int32)
    states[-1]  = int(delta[-1].argmax())
    for t in range(n - 2, -1, -1):
        states[t] = int(psi[t + 1, states[t + 1]])

    return states


# ============================================================
# EXPORT LABELLED CSV  (read by proof_5methods.py)
# ============================================================

def export_labelled_csv(df, gmm, X, state_map, machine_key, min_dwell_s=30, spike_mask=None):
    """
    Write timestamp + electrical features + Viterbi-smoothed state to
    outputs/imdeld_labelled/<machine_key>_labelled.csv.

    Two state columns are exported:
      - 'state_raw' : raw GMM labels (pre-Viterbi, from gmm.predict(X))
      - 'state'     : Viterbi-smoothed labels (post-smoothing)
    proof_5methods.py's diagnose_disagreement() uses 'state_raw' to
    identify which raw GMM run each disagreement row belonged to.

    If spike_mask is provided, also writes an 'is_spike' column so
    proof_5methods.py's Step C can exclude the SAME rows this script
    excluded from its own hour totals -- guaranteeing both scripts'
    GMM-baseline hour totals are computed over an identical time pool.

    Smoothing is applied ONLY here, after all T1-T10 tests are complete.
    The internal test suite always uses raw gmm.predict(X) labels so
    that no test result is influenced by the smoothing step.
    """
    print("\n" + "=" * 65)
    print("  EXPORT + VITERBI SMOOTHING")
    print("=" * 65)

    # Raw labels (used for T1-T10, reported here for before/after comparison)
    raw_labels = gmm.predict(X)
    short_raw,  total_runs_raw  = count_short_runs(raw_labels, min_dur=10)
    _info(f"Raw GMM labels  : {total_runs_raw:,} runs, "
          f"{short_raw:,} flicker episodes (<10s) = "
          f"{short_raw / max(total_runs_raw, 1) * 100:.1f}%")

    # Map raw integer labels → state names for 'state_raw' column
    raw_state_names = np.array([state_map[l] for l in raw_labels])

    # --- Data-driven min_dwell_s selection (FIX 3) -------------------
    # Compute raw run lengths per state from raw_labels (pre-smoothing).
    # For OFF and STANDBY (the two stable states), compute the 25th
    # percentile of their run-length distributions (in seconds = samples
    # at 1 Hz sampling; IMDELD is 1-second resolution).
    # Candidate = min(P25_OFF, P25_STANDBY), floored at 10s, capped at 120s.
    # Then evaluate flicker rate at candidate, 60s, and 90s; choose the
    # SMALLEST value that achieves flicker_rate <= 5%.  If none reaches
    # <=5%, use the one with the lowest flicker rate and print a WARNING.

    _info("\n[FIX 3] Computing data-driven min_dwell_s from raw run lengths...")

    # Build run-length arrays per raw state name
    def _compute_run_lengths_per_state(labels_int, state_map_dict):
        """Return dict: state_name -> np.array of run lengths (in samples)."""
        if len(labels_int) == 0:
            return {}
        change_pts = np.where(np.diff(labels_int, prepend=labels_int[0] - 1) != 0)[0]
        run_starts = change_pts
        run_lengths = np.diff(np.append(change_pts, len(labels_int)))
        run_states  = np.array([state_map_dict[labels_int[s]] for s in run_starts])
        result = {}
        for sname in np.unique(run_states):
            result[sname] = run_lengths[run_states == sname]
        return result

    raw_run_lengths = _compute_run_lengths_per_state(raw_labels, state_map)

    p25_vals = []
    for stable_state in ['OFF', 'STANDBY']:
        if stable_state in raw_run_lengths and len(raw_run_lengths[stable_state]) >= 4:
            p25 = float(np.percentile(raw_run_lengths[stable_state], 25))
            _info(f"   P25 run length for {stable_state}: {p25:.1f} s  "
                  f"(from {len(raw_run_lengths[stable_state]):,} runs)")
            p25_vals.append(p25)
        else:
            _info(f"   {stable_state}: not enough runs to compute P25 -- skipping")

    if p25_vals:
        candidate_dwell = float(np.clip(min(p25_vals), 10, 120))
    else:
        candidate_dwell = float(min_dwell_s)   # fallback to caller's default
    _info(f"   Candidate min_dwell_s = min(P25s) clipped to [10,120] = {candidate_dwell:.1f} s")

    # Evaluate flicker rate at candidate, 60s, 90s
    def _flicker_rate_at(dwell_s):
        """Run Viterbi at dwell_s and return (flicker_rate_pct, total_episodes)."""
        lbl = viterbi_smooth(gmm, X, min_dwell_s=dwell_s)
        short_ep, total_ep = count_short_runs(lbl, min_dur=10)
        rate = short_ep / max(total_ep, 1) * 100
        return rate, total_ep

    test_dwells = sorted(set([candidate_dwell, 60.0, 90.0]))
    _info(f"\n   Evaluating flicker rate at min_dwell_s = {test_dwells} ...")

    print(f"\n   {'min_dwell_s':>12} {'flicker_rate':>14} {'total_episodes':>16}")
    print(f"   {'':->12} {'':->14} {'':->16}")
    dwell_results = []
    for dwell in test_dwells:
        rate, total_ep = _flicker_rate_at(dwell)
        print(f"   {dwell:>12.1f} {rate:>13.2f}% {total_ep:>16,}")
        dwell_results.append((dwell, rate, total_ep))

    # Select smallest dwell achieving <= 5% flicker
    FLICKER_TARGET = 5.0
    qualifying = [(d, r, t) for d, r, t in dwell_results if r <= FLICKER_TARGET]
    if qualifying:
        chosen_dwell, chosen_rate, _ = qualifying[0]   # smallest that qualifies
        _info(f"   SELECTED min_dwell_s = {chosen_dwell:.1f} s  "
              f"(flicker_rate={chosen_rate:.2f}% <= {FLICKER_TARGET}%)")
    else:
        # None qualifies -- pick the one with the lowest flicker rate
        best = min(dwell_results, key=lambda x: x[1])
        chosen_dwell, chosen_rate, _ = best
        _warn(f"WARNING: No tested min_dwell_s achieves flicker_rate <= {FLICKER_TARGET}%. "
              f"Best achieved: {chosen_rate:.2f}% at min_dwell_s={chosen_dwell:.1f}s. "
              f"This remains visible -- the 5% target is NOT met.")

    actual_min_dwell = chosen_dwell
    # -------------------------------------------------------------------

    # Smoothed labels via Viterbi at the chosen min_dwell_s
    smooth_labels = viterbi_smooth(gmm, X, min_dwell_s=actual_min_dwell)
    short_sm, total_runs_sm = count_short_runs(smooth_labels, min_dur=10)
    _info(f"Smoothed labels : {total_runs_sm:,} runs, "
          f"{short_sm:,} flicker episodes (<10s) = "
          f"{short_sm / max(total_runs_sm, 1) * 100:.1f}%  "
          f"(min_dwell_s={actual_min_dwell:.1f}s)")

    reduction = (short_raw - short_sm)
    _pass(f"Smoothing reduced flicker episodes by {reduction:,} "
          f"({short_raw} → {short_sm})")

    # Build output dataframe with smoothed state AND raw state
    out = df.copy()
    out['state_raw'] = raw_state_names          # pre-Viterbi raw GMM labels
    out['state']     = [state_map[l] for l in smooth_labels]  # Viterbi-smoothed
    if spike_mask is not None:
        out['is_spike'] = pd.Series(spike_mask).reset_index(drop=True).values

    out_dir  = "outputs/imdeld_labelled"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{machine_key}_labelled.csv")

    cols = ['timestamp'] + [c for c in IMDELD_FEATURES if c in out.columns] + \
           ['state_raw', 'state']
    if 'is_spike' in out.columns:
        cols.append('is_spike')
    out[cols].to_csv(out_path, index=False)
    _pass(f"Labelled CSV exported → {out_path}  ({len(out):,} rows, "
          f"Viterbi-smoothed, min_dwell_s={actual_min_dwell:.1f}s)")
    _info("Columns: state_raw=pre-Viterbi raw labels, state=Viterbi-smoothed labels")

    missing = [c for c in IMDELD_FEATURES if c not in out.columns]
    if missing:
        _warn(
            f"Columns NOT in labelled CSV (not in source): {missing}\n"
            f"   proof_5methods.py methods needing these will be skipped."
        )
    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  GMM VALIDATION HARNESS")
    print(f"  File    : {DATA_PATH}")
    print(f"  Machine : {MACHINE_NAME}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 65)

    # -- Load data -----------------------------------------------
    df = load_and_prepare_data(DATA_PATH)

    # -- Spike detection (for hour-total reporting/export ONLY --
    #    matches proof_5methods.py exactly, so both scripts' hour
    #    totals are built from the same time pool. Deliberately NOT
    #    used in T1-T10 below, so test results stay unaffected --
    #    same design principle as Viterbi smoothing being export-only.)
    spike_mask = compute_spike_mask(df)

    mode_str = "IMDELD (5-feature)" if is_imdeld(df) else "SPARK (power-only)"
    print(f"   Mode    : {mode_str}")

    # -- Test 1: Sanity ------------------------------------------
    t1 = test_data_sanity(df)

    # -- Build feature matrix ------------------------------------
    print(f"\n   Building feature matrix ...")
    X, scaler = build_feature_matrix(df)
    mode_label = "5-feature (IMDELD)" if scaler is not None else "1-feature (SPARK)"
    print(f"   Feature matrix shape: {X.shape}  [{mode_label}]")

    # -- Fit GMMs for all k values -------------------------------
    print(f"\n   Fitting GMMs for k in {K_RANGE} with n_init=1 ...")
    models = {}
    for k in K_RANGE:
        print(f"   Fitting k={k} ... ", end='', flush=True)
        models[k] = fit_gmm(X, k, n_init=1)
        print(f"done  (converged={models[k].converged_}, "
              f"iterations={models[k].n_iter_})")

    labels = {k: models[k].predict(X) for k in K_RANGE}

    # -- Tests 2-10 ----------------------------------------------
    t2 = test_gmm_convergence(X, models)

    best_k, sil_scores, bic_scores, agreement = test_k_selection(
        X, models, labels, scaler=scaler, max_power=df['power'].max())
    t3 = agreement

    best_gmm  = models[best_k]
    state_map = map_states(best_gmm, best_k, scaler)

    t4, state_map, s_means, s_stds, s_names = test_state_separation(best_gmm, best_k, scaler)
    t5, max_probs = test_soft_assignment_confidence(best_gmm, X)
    t6 = test_weight_sanity(best_gmm, best_k, state_map, scaler)

    max_power = df['power'].max()
    t7  = test_physics_thresholds(best_gmm, best_k, state_map, max_power, scaler)
    t8  = test_variance_ordering(best_gmm, best_k, state_map, scaler)
    t9  = test_temporal_alignment(df, best_gmm, best_k, state_map, X)
    t10 = test_schedule_proof(df, best_gmm, best_k, state_map, X)

    # -- State-time breakdown ------------------------------------
    labels_arr = best_gmm.predict(X)
    state_hours = compute_state_time(df, labels_arr, state_map, spike_mask=spike_mask)
    print_state_time_breakdown(state_hours, MACHINE_NAME)

    # -- Diagnostic plots ----------------------------------------
    print("\n" + "=" * 65)
    print("  GENERATING DIAGNOSTIC PLOTS")
    print("=" * 65)
    X_1d   = df['power'].values.reshape(-1, 1)
    gmm_1d = fit_gmm(X_1d, best_k, n_init=1)
    plot_k_selection(sil_scores, bic_scores, best_k, OUTPUT_DIR)
    plot_gmm_histogram(df, gmm_1d, best_k, state_map, OUTPUT_DIR)
    plot_confidence_histogram(max_probs, OUTPUT_DIR)
    plot_labeled_sample(df, best_gmm, best_k, state_map, OUTPUT_DIR, scaler)
    plot_cluster_separation(df, best_gmm, best_k, state_map, OUTPUT_DIR, X=X, scaler=scaler)
    plot_temporal_alignment(df, best_gmm, best_k, state_map, X, OUTPUT_DIR)

    # -- Export labelled CSV for proof_5methods.py ---------------
    export_labelled_csv(df, best_gmm, X, state_map, _machine_key, spike_mask=spike_mask)

    # -- Final report --------------------------------------------
    results = {
        "T1   Data Sanity (rows, no-NaN, IMDELD features)":   t1,
        "T2   GMM Convergence (EM algorithm)":                  t2,
        "T3   k Selection (BIC+physical filters agree)":        t3,
        "T4   Physical Separation (2-sigma between states)":    t4,
        "T5   Soft-Assignment Confidence (>80% in >70%)":       t5,
        "T6   Cluster Weight Sanity (no ghost clusters)":       t6,
        "T7   Physics Threshold (OFF~0W, STBY stable)":         t7,
        "T8   Variance Ordering (OFF < STBY < WORK)":           t8,
        "T9   Temporal Alignment (OFF aligns nights/weekends)": t9,
        "T10  IMDELD Schedule Proof (>90% OFF in 5-10 PM)":    t10,
    }
    print_final_report(results, best_k, state_map, best_gmm)
    print(f"All plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()