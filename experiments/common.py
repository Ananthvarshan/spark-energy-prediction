"""
============================================================
SHARED EXPERIMENT INFRASTRUCTURE  --  experiments/common.py
============================================================

PURPOSE
-------
Everything Phase II's three experiments need in common:

  * loading the GMM-HMM-labelled CSV and attaching the Task 3
    engineered feature set
  * carving the record into contiguous evaluation blocks that
    never span an acquisition gap
  * the clustering feature matrix (log1p + standardise), built
    identically to validate_gmm.build_feature_matrix so the
    baselines in Task 4 are compared on the SAME representation
    the proposed method uses
  * mapping arbitrary cluster ids to physical state names
  * the ground-truth-free evaluation metrics

WHY GROUND-TRUTH-FREE METRICS
-----------------------------
IMDELD ships no per-timestamp state annotation (Martins et al.,
2018 describe the machines as three-state systems but deliver no
label column).  Accuracy, precision and recall are therefore
undefined for Task 4.  Every metric here instead scores a
partition against evidence EXTERNAL to the model that produced
it: induction-motor physics, dwell-time plausibility, and the
factory's published shift schedule.

============================================================
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (               # noqa: E402
    add_engineered_features,
    add_segment_ids,
    compute_sample_interval_s,
    all_feature_names,
)


# ── Machine registry (mirrors validate_gmm.MACHINES) ──────────────────────────

MACHINES: dict[str, dict] = {
    "pelletizer-I": {
        "labelled": "outputs/imdeld_labelled/pelletizer-I_labelled.csv",
        "name": "Pelletizer I (IMDELD)",
        "factory_tz": "America/Sao_Paulo",
        "close_hour": 17,
        "open_hour": 22,
    },
    "pelletizer-II": {
        "labelled": "outputs/imdeld_labelled/pelletizer-II_labelled.csv",
        "name": "Pelletizer II (IMDELD)",
        "factory_tz": "America/Sao_Paulo",
        "close_hour": 17,
        "open_hour": 22,
    },
}
DEFAULT_MACHINE = "pelletizer-I"

PHASE2_DIR = "outputs/phase2"

# Canonical state ordering, ascending in active power.
STATE_ORDER = ["OFF", "STANDBY", "WORKING", "PEAK_LOAD"]

# States that represent the machine doing real mechanical work.  Used by the
# physics checks, which contrast "loaded" against "unloaded but energised".
PRODUCTIVE_STATES = ("WORKING", "PEAK_LOAD")

# Clustering representation -- identical to validate_gmm.build_feature_matrix.
CLUSTER_FEATURES = [
    "active_power", "reactive_power", "apparent_power",
    "current", "voltage", "power_factor",
]
CLUSTER_LOG_COLS = ["active_power", "reactive_power", "apparent_power", "current"]

# Dwell shorter than this is physically implausible for a heavy industrial
# machine and is counted as flicker (five-method proof, Method 4).
FLICKER_TAU_S = 10.0

# Physics acceptance bands (five-method proof, Methods 1 and 2).
PF_SEPARATION_MIN = 0.40          # PF_productive - PF_standby
CURRENT_RATIO_BAND = (0.25, 0.50)  # I_standby / I_productive
OFF_POWER_MAX_W = 50.0


# ── Small helpers ─────────────────────────────────────────────────────────────

def info(msg: str) -> None:
    print(f"   [INFO] {msg}", flush=True)


def section(title: str) -> None:
    print("\n" + "=" * 68, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 68, flush=True)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    info(f"wrote {path}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labelled(
    machine_key: str = DEFAULT_MACHINE,
    drop_spikes: bool = True,
    add_features: bool = True,
    roll_window_s: float = 300.0,
    nrows: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the GMM-HMM-labelled CSV produced by validate_gmm.py and attach the
    Task 3 engineered features.

    Spike rows (rolling-MAD flagged, ~19.5% on pelletizer-I) are dropped by
    default, matching lstm_pipeline.py.  They are measurement artefacts and
    carry no state information; keeping them would let every model be scored
    partly on its ability to fit sensor noise.

    Returns (df, meta).
    """
    if machine_key not in MACHINES:
        raise KeyError(f"Unknown machine '{machine_key}'. "
                       f"Choose from {list(MACHINES)}")
    cfg = MACHINES[machine_key]
    path = cfg["labelled"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labelled CSV not found: {path}\n"
            f"Run:  python validate_gmm.py {machine_key}"
        )

    if verbose:
        info(f"loading {path}")
    df = pd.read_csv(path, nrows=nrows)
    n_raw = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if drop_spikes and "is_spike" in df.columns:
        n_spike = int(df["is_spike"].sum())
        df = df[~df["is_spike"].astype(bool)].reset_index(drop=True)
        if verbose:
            info(f"dropped {n_spike:,} spike rows "
                 f"({n_spike / max(n_raw, 1) * 100:.2f}%) -> {len(df):,} clean rows")

    sample_interval_s = compute_sample_interval_s(df)

    if add_features:
        df = add_engineered_features(
            df,
            factory_tz=cfg["factory_tz"],
            schedule_close_hour=cfg["close_hour"],
            schedule_open_hour=cfg["open_hour"],
            roll_window_s=roll_window_s,
            sample_interval_s=sample_interval_s,
            verbose=verbose,
        )
    else:
        df = add_segment_ids(df, sample_interval_s=sample_interval_s)

    meta = {
        "machine_key": machine_key,
        "machine_name": cfg["name"],
        "labelled_path": path,
        "n_rows_raw": n_raw,
        "n_rows_clean": len(df),
        "sample_interval_s": sample_interval_s,
        "n_segments": int(df["segment_id"].nunique()),
        "factory_tz": cfg["factory_tz"],
        "close_hour": cfg["close_hour"],
        "open_hour": cfg["open_hour"],
        "reference_states": sorted(df["state"].unique().tolist())
        if "state" in df.columns else [],
    }
    if verbose:
        info(f"span {df['timestamp'].min()} -> {df['timestamp'].max()}")
        if "state" in df.columns:
            vc = df["state"].value_counts()
            for s, c in vc.items():
                info(f"  reference label {s:<10} {c:>10,} ({c/len(df)*100:5.2f}%)")
    return df, meta


# ── Contiguous evaluation blocks ──────────────────────────────────────────────

def make_blocks(
    df: pd.DataFrame,
    block_seconds: float = 7200.0,
    n_blocks: int = 48,
    sample_interval_s: float = 1.0,
    segment_col: str = "segment_id",
    random_state: int = 42,
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    Carve the record into `n_blocks` contiguous blocks of `block_seconds`
    each, spread evenly across the whole span and never crossing a gap.

    Returns a list of integer positional-index arrays into `df`.

    Rationale
    ---------
    Two of the Task 4 metrics -- flicker rate and dwell-time distribution --
    are only defined on a contiguous time series, so the evaluation set
    cannot be a random row sample.  Spreading blocks evenly rather than
    taking one long slice keeps seasonal and shift-pattern coverage: the
    pelletizer record spans 155 days and its duty cycle is not stationary.
    """
    block_rows = max(2, int(round(block_seconds / sample_interval_s)))
    rng = np.random.default_rng(random_state)

    # Candidate start positions, evenly spaced within each contiguous segment.
    seg_bounds = []
    seg_ids = df[segment_col].values
    starts = np.flatnonzero(np.r_[True, seg_ids[1:] != seg_ids[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        if e - s >= block_rows:
            seg_bounds.append((int(s), int(e)))

    if not seg_bounds:
        raise ValueError(
            f"No contiguous segment is long enough for a {block_seconds:.0f}s "
            f"block ({block_rows} rows)."
        )

    # Allocate blocks across segments in proportion to segment length.
    lengths = np.array([e - s for s, e in seg_bounds], dtype=float)
    share = lengths / lengths.sum()
    alloc = np.maximum(1, np.round(share * n_blocks).astype(int))

    blocks: list[np.ndarray] = []
    for (s, e), k in zip(seg_bounds, alloc):
        capacity = (e - s) // block_rows
        k = int(min(k, capacity))
        if k <= 0:
            continue
        # Evenly spaced, then jittered within the free space so repeated runs
        # with different seeds do not always land on the same clock times.
        edges = np.linspace(s, e - block_rows, k).astype(int)
        slack = max(0, ((e - s) - k * block_rows) // max(k, 1))
        for pos in edges:
            jitter = int(rng.integers(0, slack + 1)) if slack > 0 else 0
            p = min(max(pos + jitter, s), e - block_rows)
            blocks.append(np.arange(p, p + block_rows))

    blocks = blocks[:n_blocks]
    if verbose:
        total = sum(len(b) for b in blocks)
        info(f"{len(blocks)} contiguous blocks x {block_rows} rows "
             f"({block_seconds/3600:.1f}h each) = {total:,} rows "
             f"across {len(seg_bounds)} segment(s)")
    return blocks


def split_blocks(
    blocks: list[np.ndarray],
    fit_fraction: float = 0.5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Alternate blocks into a FIT set and an EVAL set.

    Alternating rather than splitting chronologically matters here: the
    models are unsupervised, so the concern is not label leakage but
    representativeness.  A chronological split would fit the emission
    distributions on the first half of the record and evaluate on the
    second, confounding method differences with seasonal drift in the
    plant's duty cycle.  Alternating gives both sets the same coverage.
    """
    fit_every = max(1, int(round(1.0 / max(fit_fraction, 1e-9))))
    fit = [b for i, b in enumerate(blocks) if i % fit_every == 0]
    ev = [b for i, b in enumerate(blocks) if i % fit_every != 0]
    if not ev:                      # degenerate; fall back to a halving
        fit, ev = blocks[0::2], blocks[1::2]
    return fit, ev


# ── Clustering feature matrix ─────────────────────────────────────────────────

def build_cluster_matrix(
    df: pd.DataFrame,
    idx: np.ndarray | None = None,
    scaler=None,
    feature_cols: list[str] | None = None,
    log_cols: list[str] | None = None,
):
    """
    Build the (log1p, standardised) matrix every Task 4 method is fitted on.

    Reproduces validate_gmm.build_feature_matrix exactly: right-skewed
    power and current channels are log1p-compressed (Box & Cox 1964;
    standard NILM preprocessing per Zeifman & Roth 2011), voltage and
    power factor are left alone, and the result is standardised.

    Pass `scaler=None` to fit a new scaler; pass an existing one to apply
    it (used to transform the EVAL blocks with the FIT blocks' statistics).
    """
    from sklearn.preprocessing import StandardScaler

    feature_cols = feature_cols or CLUSTER_FEATURES
    log_cols = CLUSTER_LOG_COLS if log_cols is None else log_cols

    sub = df.iloc[idx] if idx is not None else df
    X = sub[feature_cols].to_numpy(dtype=np.float64, copy=True)
    for col in log_cols:
        if col in feature_cols:
            j = feature_cols.index(col)
            X[:, j] = np.log1p(np.abs(X[:, j]))

    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


# ── Cluster -> state mapping ──────────────────────────────────────────────────

def map_clusters_to_states(
    labels: np.ndarray,
    power: np.ndarray,
    state_names: list[str] | None = None,
) -> dict[int, str]:
    """
    Map arbitrary cluster ids to physical state names by mean active
    power: the lowest-power group is OFF, the highest is the top load
    state.

    This is the physics-anchored ranking used by
    src/state_mapping.map_clusters_to_states, restated here so that any
    method can be mapped by the identical rule.  Noise (-1, emitted by
    DBSCAN) is left unmapped and reported separately rather than being
    forced into a state.

    WHEN A METHOD RETURNS MORE CLUSTERS THAN THERE ARE STATES
    ---------------------------------------------------------
    DBSCAN chooses its own cluster count and returned 28 here.  Naively
    handing the four canonical names to the four lowest-power clusters
    and inventing names for the other 24 would be a mapping artefact
    rather than a finding: all four named states would then sit inside
    the machine's low-power range, the "productive" group would contain
    almost no load, and the power-factor separation check would fail for
    reasons that have nothing to do with the clustering.

    Instead the cluster MEANS are themselves grouped into
    len(state_names) power bands by 1-D k-means, weighted by cluster
    size, and every cluster inherits its band's state name.  This gives
    an over-segmenting method its best legitimate shot at the physics
    checks: sub-clusters of one physical state are re-merged, and only a
    genuine failure to separate load from no-load can still fail C3/C4.
    """
    valid = np.unique(labels[labels >= 0])
    if len(valid) == 0:
        return {}

    means = {int(c): float(np.mean(power[labels == c])) for c in valid}
    sizes = {int(c): int((labels == c).sum()) for c in valid}
    ranked = sorted(means, key=means.get)

    if state_names is None:
        state_names = STATE_ORDER[:min(len(ranked), len(STATE_ORDER))]

    n_states = len(state_names)
    if len(ranked) <= n_states:
        return {c: state_names[i] for i, c in enumerate(ranked)}

    # More clusters than states: merge them into n_states power bands.
    from sklearn.cluster import KMeans

    mu = np.array([[means[c]] for c in ranked], dtype=float)
    w = np.array([sizes[c] for c in ranked], dtype=float)
    km = KMeans(n_clusters=n_states, n_init=10, random_state=0).fit(mu,
                                                                   sample_weight=w)
    band_of = km.labels_
    # Rank the bands by their own centre so band ids follow power order.
    order = np.argsort(km.cluster_centers_.ravel())
    band_rank = {int(b): r for r, b in enumerate(order)}
    return {c: state_names[band_rank[int(band_of[i])]]
            for i, c in enumerate(ranked)}


def labels_to_states(labels: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    """Vector-map cluster ids to state-name strings; noise becomes 'NOISE'."""
    out = np.full(len(labels), "NOISE", dtype=object)
    for c, name in mapping.items():
        out[labels == c] = name
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def run_lengths(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (run_values, run_lengths) for a 1D sequence."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return np.array([]), np.array([], dtype=int)
    change = np.r_[True, seq[1:] != seq[:-1]]
    starts = np.flatnonzero(change)
    lengths = np.diff(np.r_[starts, len(seq)])
    return seq[starts], lengths


def flicker_rate(
    state_seq_blocks: list[np.ndarray],
    sample_interval_s: float = 1.0,
    tau_s: float = FLICKER_TAU_S,
) -> dict:
    """
    Fraction of state runs shorter than `tau_s` seconds.

    Method 4 of the five-method proof.  A heavy industrial machine cannot
    physically change operating state several times per second; a high
    flicker rate therefore means the partition is tracking measurement
    noise rather than machine behaviour.  Computed per contiguous block
    and pooled, so no run is ever counted across a block boundary.

    THREE VIEWS ARE REPORTED, and they answer different questions.

    `flicker_rate_pct` pools every state transition.  On a machine whose
    productive range is split into two states (WORKING / PEAK_LOAD), this
    number is dominated by churn along that internal boundary, which is a
    consequence of the chosen model order rather than a defect in the
    temporal model.

    `flicker_rate_standby_binary_pct` collapses the sequence to
    STANDBY vs not-STANDBY first.  This is the DECISION-RELEVANT figure:
    the downstream shutdown rule only ever asks "is the machine idle, and
    for how long" -- it is indifferent to whether a productive second was
    WORKING or PEAK_LOAD.  A partition can therefore be perfectly usable
    for the decision layer while scoring poorly on the pooled measure.

    `per_state` gives the dwell distribution for each state separately,
    which localises where any flicker actually lives.
    """
    if not state_seq_blocks:
        return {"flicker_rate_pct": float("nan"), "n_runs": 0,
                "median_dwell_s": float("nan"), "mean_dwell_s": float("nan")}

    tau_rows = tau_s / sample_interval_s

    def _pool(blocks):
        vals, lens = [], []
        for seq in blocks:
            v, ln = run_lengths(seq)
            vals.append(v)
            lens.append(ln)
        return np.concatenate(vals), np.concatenate(lens)

    values, lens = _pool(state_seq_blocks)
    dwell_s = lens * sample_interval_s

    # Decision-relevant view: binarise to STANDBY vs everything else.
    binary_blocks = [np.where(s == "STANDBY", "STANDBY", "OTHER")
                     for s in state_seq_blocks]
    b_values, b_lens = _pool(binary_blocks)
    b_standby = b_lens[b_values == "STANDBY"]

    out = {
        "flicker_rate_pct": float((lens < tau_rows).mean() * 100.0),
        "n_runs": int(len(lens)),
        "median_dwell_s": float(np.median(dwell_s)),
        "mean_dwell_s": float(np.mean(dwell_s)),
        "p90_dwell_s": float(np.percentile(dwell_s, 90)),
        "flicker_rate_standby_binary_pct": float((b_lens < tau_rows).mean() * 100.0),
        "n_runs_standby_binary": int(len(b_lens)),
        "median_dwell_standby_binary_s": float(np.median(b_lens) * sample_interval_s),
        "standby_run_flicker_pct": (float((b_standby < tau_rows).mean() * 100.0)
                                    if len(b_standby) else float("nan")),
        "median_standby_run_s": (float(np.median(b_standby) * sample_interval_s)
                                 if len(b_standby) else float("nan")),
        "n_standby_runs": int(len(b_standby)),
    }

    per_state = {}
    for s in np.unique(values):
        ln = lens[values == s]
        per_state[str(s)] = {
            "n_runs": int(len(ln)),
            "median_dwell_s": float(np.median(ln) * sample_interval_s),
            "mean_dwell_s": float(np.mean(ln) * sample_interval_s),
            "flicker_pct": float((ln < tau_rows).mean() * 100.0),
        }
    out["per_state_dwell"] = per_state
    return out


def physics_compliance(
    df: pd.DataFrame,
    idx: np.ndarray,
    states: np.ndarray,
) -> dict:
    """
    Score a partition against five independent physical checks.

    C1  power ordering       mean P strictly increasing OFF < STANDBY < ...
    C2  OFF is truly off     mean P of the OFF state < OFF_POWER_MAX_W
    C3  PF separation        PF(productive) - PF(STANDBY) >= 0.40
    C4  no-load current      I(STANDBY) / I(productive) in [0.25, 0.50]
    C5  variance ordering    sigma_P(OFF) < sigma_P(STANDBY) < sigma_P(prod.)

    C3 and C4 are the two checks that do not depend on power magnitude at
    all, and are therefore the ones that actually test whether STANDBY has
    been separated from a lightly-loaded WORKING state rather than merely
    from a lower-power one.  C3's threshold comes from induction-motor
    theory (an unloaded motor's current is predominantly magnetising);
    C4's band is the standard no-load/full-load current ratio for
    industrial induction motors.

    Returns the individual check values plus `score` = fraction passed.
    """
    sub = df.iloc[idx]
    P = sub["active_power"].to_numpy(float)
    I = sub["current"].to_numpy(float) if "current" in sub else None
    PF = sub["power_factor"].to_numpy(float) if "power_factor" in sub else None

    present = [s for s in STATE_ORDER if (states == s).any()]
    prod_mask = np.isin(states, PRODUCTIVE_STATES)
    standby_mask = states == "STANDBY"
    off_mask = states == "OFF"

    def _mean(arr, mask):
        return float(np.mean(arr[mask])) if arr is not None and mask.any() else float("nan")

    per_state = {
        s: {
            "n": int((states == s).sum()),
            "frac": float((states == s).mean()),
            "mean_active_power_w": _mean(P, states == s),
            "std_active_power_w": (float(np.std(P[states == s]))
                                   if (states == s).any() else float("nan")),
            "mean_current_a": _mean(I, states == s),
            "mean_power_factor": _mean(PF, states == s),
        }
        for s in present
    }

    checks: dict[str, bool | None] = {}
    values: dict[str, float] = {}

    # C1 -- power ordering
    ordered_means = [per_state[s]["mean_active_power_w"] for s in present]
    checks["C1_power_ordering"] = bool(
        len(ordered_means) >= 2
        and all(a < b for a, b in zip(ordered_means, ordered_means[1:]))
    )

    # C2 -- OFF is truly off
    if off_mask.any():
        values["off_mean_power_w"] = _mean(P, off_mask)
        checks["C2_off_near_zero"] = bool(values["off_mean_power_w"] < OFF_POWER_MAX_W)
    else:
        checks["C2_off_near_zero"] = None

    # C3 -- power-factor separation
    if PF is not None and standby_mask.any() and prod_mask.any():
        pf_gap = _mean(PF, prod_mask) - _mean(PF, standby_mask)
        values["pf_standby"] = _mean(PF, standby_mask)
        values["pf_productive"] = _mean(PF, prod_mask)
        values["pf_separation"] = pf_gap
        checks["C3_pf_separation"] = bool(pf_gap >= PF_SEPARATION_MIN)
    else:
        checks["C3_pf_separation"] = None

    # C4 -- no-load current ratio
    if I is not None and standby_mask.any() and prod_mask.any():
        i_prod = _mean(I, prod_mask)
        ratio = _mean(I, standby_mask) / i_prod if i_prod > 0 else float("nan")
        values["current_ratio"] = ratio
        lo, hi = CURRENT_RATIO_BAND
        checks["C4_current_ratio"] = bool(lo <= ratio <= hi)
    else:
        checks["C4_current_ratio"] = None

    # C5 -- variance ordering
    if off_mask.any() and standby_mask.any() and prod_mask.any():
        s_off = float(np.std(P[off_mask]))
        s_sby = float(np.std(P[standby_mask]))
        s_prd = float(np.std(P[prod_mask]))
        values.update({"std_off": s_off, "std_standby": s_sby,
                       "std_productive": s_prd})
        checks["C5_variance_ordering"] = bool(s_off < s_sby < s_prd)
    else:
        checks["C5_variance_ordering"] = None

    applicable = [v for v in checks.values() if v is not None]
    score = float(np.mean(applicable)) if applicable else float("nan")

    return {
        "score": score,
        "n_checks_applicable": len(applicable),
        "n_checks_passed": int(sum(applicable)),
        "checks": checks,
        "values": values,
        "per_state": per_state,
    }


def schedule_accuracy(
    df: pd.DataFrame,
    idx: np.ndarray,
    states: np.ndarray,
    non_productive: tuple[str, ...] = ("OFF", "STANDBY"),
) -> dict:
    """
    Fraction of readings inside the factory's published closed window that
    are labelled non-productive (OFF or STANDBY).

    Test T10 of validate_gmm.py, restated as a comparable score.  This is
    the only metric in Task 4 that uses information genuinely external to
    the electrical data: the plant halts every weekday between 17:00 and
    22:00 local time to avoid the peak tariff, and does not run at
    weekends.  A partition that assigns productive states during the
    closed window is contradicting a known operational fact.

    Also reports the converse -- the fraction of open-window readings
    labelled productive -- because a degenerate partition that calls
    everything OFF would otherwise score 100%.
    """
    sub = df.iloc[idx]
    if "is_factory_open" not in sub.columns:
        return {"closed_window_correct_pct": float("nan"),
                "open_window_productive_pct": float("nan"),
                "balanced_schedule_score": float("nan")}

    is_open = sub["is_factory_open"].to_numpy(float) > 0.5
    is_np = np.isin(states, non_productive)

    closed_correct = (float(is_np[~is_open].mean() * 100.0)
                      if (~is_open).any() else float("nan"))
    open_productive = (float((~is_np[is_open]).mean() * 100.0)
                       if is_open.any() else float("nan"))

    both = [v for v in (closed_correct, open_productive) if not np.isnan(v)]
    balanced = float(np.mean(both)) if both else float("nan")

    return {
        "closed_window_correct_pct": closed_correct,
        "open_window_productive_pct": open_productive,
        "balanced_schedule_score": balanced,
        "n_closed": int((~is_open).sum()),
        "n_open": int(is_open.sum()),
    }


def agreement_with_reference(
    states: np.ndarray,
    reference: np.ndarray,
) -> dict:
    """
    Adjusted Rand Index and adjusted mutual information against the
    pipeline's own exported labels.

    NOT a correctness measure -- the reference is itself a model output,
    not ground truth.  It is reported only to show HOW DIFFERENT each
    baseline's partition is from the proposed method's, which is what
    makes the physics scores interpretable: a baseline that scores worse
    while producing a near-identical partition would indicate a metric
    problem rather than a method difference.
    """
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    mask = (states != "NOISE")
    if mask.sum() < 2:
        return {"ari_vs_reference": float("nan"), "ami_vs_reference": float("nan")}
    return {
        "ari_vs_reference": float(adjusted_rand_score(reference[mask], states[mask])),
        "ami_vs_reference": float(adjusted_mutual_info_score(reference[mask], states[mask])),
    }


def standby_hours(
    states_blocks: list[np.ndarray],
    sample_interval_s: float,
) -> float:
    """Total hours labelled STANDBY across the evaluated blocks."""
    n = sum(int((s == "STANDBY").sum()) for s in states_blocks)
    return n * sample_interval_s / 3600.0
