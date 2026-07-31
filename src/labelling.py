"""
============================================================
UNIFORM MULTI-MACHINE LABELLER  --  src/labelling.py
============================================================

PURPOSE
-------
Phase IV.  Everything up to Phase III was produced for a single
machine, pelletizer-I, by `validate_gmm.py`.  A multi-machine
comparison needs the labelling procedure to be a FUNCTION that can
be applied identically to every record, so that a difference in the
results is a difference between machines and not between runs of a
script whose configuration drifted.

This module is that function.  It reproduces `validate_gmm.py`'s
preprocessing exactly and then applies Phase II's own conclusion
about how the state sequence should be decoded:

    load  ->  spike mask (rolling MAD)          [validate_gmm]
          ->  OFF-state denoise                 [validate_gmm]
          ->  log1p + standardise 6 channels    [validate_gmm]
          ->  GMM emissions (full covariance)   [Task 4]
          ->  GaussianHMM, transitions only     [Task 4 correction 2]
          ->  Viterbi decode, per segment       [spec 1.4]
          ->  minimum-dwell constraint          [Task 4b/4c]

THREE CHOICES THAT DIFFER FROM validate_gmm.py, AND WHY
--------------------------------------------------------
1.  EMISSIONS ARE WARM-STARTED FROM A GMM AND HELD FIXED
    (`params="t"`).  Task 4 correction 2 found that fitting HMM
    emissions from scratch by Baum-Welch converged to a materially
    worse local optimum on pelletizer-I -- its STANDBY component
    absorbed part of the productive range instead of finding the
    no-load band.  Across eight machines that failure mode would be
    hit at random and would be indistinguishable from a genuine
    machine difference.

2.  A MINIMUM-DWELL CONSTRAINT IS APPLIED AT DECODE TIME.  Task 4c
    showed the transition prior cannot control flicker (the maximum
    switching penalty is bounded at ~9 nats while the emission
    log-likelihood gap exceeds it at 56% of timesteps), and that an
    explicit minimum dwell takes flicker to zero at no cost to any
    physics score.  It is set to `FLICKER_TAU_S` (10 s), i.e. to the
    flicker threshold itself, so it removes exactly what the metric
    defines as implausible and nothing more.

3.  TRANSITIONS ARE FITTED ON SAMPLED CONTIGUOUS BLOCKS, NOT THE
    WHOLE RECORD.  Only k(k-1) free parameters are being estimated
    and each machine supplies ~10^5 transitions per hour of record,
    so the estimate is saturated long before the record is exhausted;
    fitting Baum-Welch over 5.5M rows x 8 machines would cost hours
    to change the self-transition probabilities in the fourth decimal
    place.  Decoding still runs over EVERY row.

k IS FIXED AT 4 FOR EVERY MACHINE
---------------------------------
Not because 4 is right for all of them -- `bic_scan` records what
each machine's own BIC prefers, and Task 8 reports it -- but because
a cross-machine comparison of STANDBY hours is meaningless if the
state count varies per machine.  The k-selection question belongs to
the Phase IX ablation; here it is a diagnostic, not a control.

============================================================
"""

from __future__ import annotations

import os
import sys
import time
import heapq

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import compute_sample_interval_s, GAP_MULTIPLE   # noqa: E402


# ── Constants shared with the rest of the project ─────────────────────────────

# Canonical state ordering, ascending in active power (experiments.common).
STATE_ORDER = ["OFF", "STANDBY", "WORKING", "PEAK_LOAD"]

# Electrical channels the clustering uses, in validate_gmm's order.
IMDELD_FEATURES = ["active_power", "reactive_power", "apparent_power",
                   "current", "voltage", "power_factor"]
LOG_COLS = ["active_power", "reactive_power", "apparent_power", "current"]

# Spike detection -- must match proof_5methods.py and validate_gmm.py exactly.
MAD_WINDOW = 11
MAD_THRESHOLD = 5.0

# OFF-state denoising (validate_gmm.denoise_off_state).
OFF_DENOISE_THRESHOLD_W = 5.0
OFF_DENOISE_WINDOW = 5

# Model configuration.
K_STATES = 4
STICKY_TARGET_DWELL_S = 30.0     # Fox et al. (2011) prior, as in validate_gmm
HMM_N_ITER = 100
MIN_DWELL_S = 10.0               # = FLICKER_TAU_S (experiments.common)
N_EMISSION_FIT_ROWS = 200_000    # rows drawn to fit the GMM emissions
N_TRANSITION_BLOCKS = 64         # contiguous blocks used to fit transitions
TRANSITION_BLOCK_S = 7200.0      # 2 h each
RANDOM_STATE = 42

PHASE4_LABELLED_DIR = "outputs/phase4/labelled"


def _info(msg: str) -> None:
    print(f"   [INFO] {msg}", flush=True)


# ── Raw loading ───────────────────────────────────────────────────────────────

def parse_timestamps(s: pd.Series) -> pd.Series:
    """
    Parse IMDELD's "YYYY-MM-DD HH:MM:SS±HH" stamps to UTC.

    `pd.to_datetime` cannot use its C fast path on a column with mixed UTC
    offsets (IMDELD spans a daylight-saving change, so the record contains
    both -02 and -03), and falls back to a per-element Python parse: 115 s
    per million rows, i.e. over ten minutes for one machine and more than an
    hour for the eight of them.

    Splitting the naive part from the offset lets the fast path run and is
    exact, not approximate: the offset is a whole number of hours in this
    dataset and is subtracted arithmetically.  Verified equal to
    `pd.to_datetime(..., utc=True)` element-for-element in
    `tests/test_labelling.py`; any deviation from the expected layout falls
    back to the slow, general parser rather than guessing.
    """
    try:
        naive = pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
        off = s.str.slice(19, 22).astype(np.int8)
        if naive.isna().any() or off.isna().any():
            raise ValueError("unparsed rows")
        return (naive - pd.to_timedelta(off, unit="h")).dt.tz_localize("UTC")
    except Exception:                                   # any layout surprise
        return pd.to_datetime(s, utc=True)


def load_raw(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load one raw appliance CSV into the frame the pipeline expects.

    Reproduces `validate_gmm.load_and_prepare_data` for the IMDELD file
    layout: canonical column names, UTC timestamps, numeric coercion,
    clipping of active/apparent power, current and voltage at zero (reactive
    power is deliberately NOT clipped -- a negative reading is a valid
    capacitive-load signal), the `power` alias, and power factor as
    |P| / S with S = 0 mapped to 0.

    Electrical columns are stored as float32.  At 5.5M rows x 8 machines the
    float64 originals do not fit in this machine's free memory, and the
    sensors deliver ~6 significant digits, so float32 is lossless with
    respect to the measurement.  Every model fit below casts back to float64.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    t0 = time.time()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "wsdatetime" in df.columns:
        df = df.rename(columns={"wsdatetime": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError(f"no timestamp column in {path}: {list(df.columns)}")

    df["timestamp"] = parse_timestamps(df["timestamp"].astype(str))

    for col in [c for c in IMDELD_FEATURES if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    for col in ["active_power", "apparent_power", "current", "voltage"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # -- Missing current on a de-energised meter ------------------------------
    # The milling-machine meters leave `current` blank on 6.4% of rows, and on
    # every one of them apparent power is exactly 0 VA with voltage present.
    # S = V*I with V > 0 admits only I = 0 A, so the blank is a reporting
    # convention for zero and is filled as such -- an unavoidable decision,
    # since the alternative (dropping the rows) would delete a fourteenth of
    # the record precisely where the machine is OFF and would bias every
    # state-time total.  Blanks NOT explained by S = 0 are dropped and counted.
    n_before = len(df)
    n_zero_current = 0
    if "current" in df.columns and "apparent_power" in df.columns:
        fillable = df["current"].isna() & (df["apparent_power"] == 0)
        n_zero_current = int(fillable.sum())
        if n_zero_current:
            df.loc[fillable, "current"] = np.float32(0.0)

    elec = [c for c in IMDELD_FEATURES if c in df.columns and c != "power_factor"]
    df = df.dropna(subset=["timestamp"] + elec)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "apparent_power" in df.columns:
        df["power_factor"] = (df["active_power"].abs()
                              / df["apparent_power"].replace(0, np.nan)
                              ).fillna(0).astype(np.float32)
    df["power"] = df["active_power"]

    if verbose:
        _info(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s "
              f"({n_zero_current:,} blank currents filled as 0 A at S=0 VA; "
              f"{n_before - len(df):,} rows dropped for other NaN); "
              f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
    df.attrs["n_zero_current_filled"] = n_zero_current
    df.attrs["n_rows_dropped_nan"] = int(n_before - len(df))
    return df


def add_segments(df: pd.DataFrame, sample_interval_s: float | None = None
                 ) -> tuple[pd.DataFrame, float]:
    """Attach `segment_id` (increments at every acquisition gap) in place."""
    dt = sample_interval_s or compute_sample_interval_s(df)
    gap = df["timestamp"].diff().dt.total_seconds() > GAP_MULTIPLE * dt
    df["segment_id"] = gap.fillna(False).cumsum().astype(np.int32)
    return df, float(dt)


# ── Preprocessing, identical to validate_gmm.py ───────────────────────────────

def compute_spike_mask(df: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    """
    Rolling-MAD spike flags -- same algorithm, window and threshold as
    `validate_gmm.compute_spike_mask` and `proof_5methods.clean_raw_electrical_data`,
    so the hour totals of all three remain computed over one identical time pool.

    Rows are flagged, never deleted: Task 4 correction 1 showed that dropping
    them destroys the row-count-to-seconds correspondence every dwell time and
    episode duration depends on.
    """
    cols = [c for c in IMDELD_FEATURES if c in df.columns]
    mask = np.zeros(len(df), dtype=bool)
    for col in cols:
        # float64 for the rolling median and the threshold comparison: at
        # float32 a handful of rows per million sit on the wrong side of
        # `dev > 5*MAD` by one ulp, and the flags would then not match the
        # ones proof_5methods.py computes on the same data.
        s = df[col].astype(np.float64)
        med = s.rolling(MAD_WINDOW, center=True, min_periods=1).median()
        dev = (s - med).abs()
        mad = dev.rolling(MAD_WINDOW, center=True, min_periods=1).median()
        mask |= (dev > MAD_THRESHOLD * mad.replace(0, np.nan)).fillna(False).to_numpy()
    if verbose:
        _info(f"spike mask: {mask.sum():,} rows flagged "
              f"({mask.mean()*100:.2f}%) over {len(cols)} channels")
    return mask


def denoise_off_state(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Rolling-median filter applied ONLY to rows already below
    `OFF_DENOISE_THRESHOLD_W`, reproducing `validate_gmm.denoise_off_state`
    (Hart 1992; Zeifman & Roth 2011).  It targets OFF-cluster sensor noise
    without touching ON-state dynamics.  Derived columns are recomputed
    afterwards so power factor stays consistent with the smoothed channels.
    """
    off = df["active_power"] < OFF_DENOISE_THRESHOLD_W
    n_off = int(off.sum())
    for col in [c for c in IMDELD_FEATURES if c in df.columns and c != "power_factor"]:
        smooth = df[col].rolling(OFF_DENOISE_WINDOW, center=True, min_periods=1).median()
        df[col] = df[col].where(~off, smooth).astype(np.float32)
    if "apparent_power" in df.columns:
        df["power_factor"] = (df["active_power"].abs()
                              / df["apparent_power"].replace(0, np.nan)
                              ).fillna(0).astype(np.float32)
    df["power"] = df["active_power"]
    if verbose:
        _info(f"OFF denoise: {n_off:,} rows (<{OFF_DENOISE_THRESHOLD_W:.0f} W) "
              f"median-smoothed (window={OFF_DENOISE_WINDOW})")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, object, list[str]]:
    """
    log1p the right-skewed channels, standardise everything, return float32.

    Identical to `validate_gmm.build_feature_matrix` (Box & Cox 1964; Zeifman
    & Roth 2011) and to `experiments.common.build_cluster_matrix`, so Phase IV
    labels live in the same representation Phase II compared its baselines in.

    Records with only an active-power channel (the SPARK datasets of Task 10)
    fall back to that single channel, and the caller is told which channels
    were available -- Task 10's finding is precisely that two of the five
    physics checks cannot be evaluated without current and power factor.
    """
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in IMDELD_FEATURES if c in df.columns]
    if "active_power" not in cols:
        raise ValueError("active_power is required")
    X = df[cols].to_numpy(dtype=np.float64, copy=True)
    for col in LOG_COLS:
        if col in cols:
            j = cols.index(col)
            X[:, j] = np.log1p(np.abs(X[:, j]))
    scaler = StandardScaler().fit(X)
    return scaler.transform(X).astype(np.float32), scaler, cols


# ── Minimum-dwell constraint ──────────────────────────────────────────────────

def run_lengths(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(run_values, run_lengths) for a 1-D sequence."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return np.array([]), np.array([], dtype=int)
    change = np.r_[True, seq[1:] != seq[:-1]]
    starts = np.flatnonzero(change)
    return seq[starts], np.diff(np.r_[starts, len(seq)])


def enforce_min_dwell(labels: np.ndarray, min_rows: int) -> np.ndarray:
    """
    Absorb every run shorter than `min_rows` into a neighbouring run, shortest
    run first, ties going to the leftmost, each absorbed run taking the label
    of its longer neighbour.

    SAME OPERATOR AS `experiments.task4_flicker_diagnosis.enforce_min_dwell`,
    reimplemented over a linked list with a priority queue because Phase IV
    applies it to whole records rather than to 2-hour blocks.  The Phase II
    version recomputes every run length after each merge, which is O(R^2) in
    the number of runs; a flickery 5.5M-row record has ~10^6 runs and that
    version does not terminate in practical time.  This one is O(R log R) and
    is verified to produce bit-identical output on random sequences in
    `tests/test_labelling.py` -- the Phase II result depends on this operator's
    exact semantics, so it is reproduced rather than approximated.
    """
    labels = np.asarray(labels)
    if len(labels) == 0 or min_rows <= 1:
        return labels.copy()

    vals, lens = run_lengths(labels)
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    n = len(vals)
    if n <= 1:
        return labels.copy()

    val = list(vals)
    length = list(lens.astype(int))
    start = list(starts.astype(int))
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    alive = [True] * n

    heap = [(length[i], start[i], i) for i in range(n) if length[i] < min_rows]
    heapq.heapify(heap)

    while heap:
        ln, st, i = heapq.heappop(heap)
        # Stale entry: the run has since been merged away or grown.
        if not alive[i] or length[i] != ln or start[i] != st or ln >= min_rows:
            continue
        p, q = prev[i], nxt[i]
        if p < 0 and q < 0:
            break                                   # a single run: nothing to do
        left_len = length[p] if p >= 0 else -1
        right_len = length[q] if q >= 0 else -1
        take = (val[p] if (left_len >= right_len and p >= 0)
                else (val[q] if right_len > 0 else val[p]))
        val[i] = take

        # Coalesce with either neighbour that now carries the same label.
        for j, side in ((p, "left"), (q, "right")):
            if j is None or j < 0 or not alive[j] or val[j] != take:
                continue
            length[i] += length[j]
            if side == "left":
                start[i] = start[j]
                prev[i] = prev[j]
                if prev[i] >= 0:
                    nxt[prev[i]] = i
            else:
                nxt[i] = nxt[j]
                if nxt[i] >= 0:
                    prev[nxt[i]] = i
            alive[j] = False
        if length[i] < min_rows:
            heapq.heappush(heap, (length[i], start[i], i))

    out = np.empty_like(labels)
    for i in range(n):
        if alive[i]:
            out[start[i]:start[i] + length[i]] = val[i]
    return out


# ── Model fitting and decoding ────────────────────────────────────────────────

def segment_bounds(seg: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) row ranges of each contiguous segment."""
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(seg)]
    return list(zip(starts.tolist(), ends.tolist()))


def sample_blocks(bounds, block_rows: int, n_blocks: int, rng) -> list[tuple[int, int]]:
    """
    Contiguous blocks spread across the record in proportion to segment
    length, never crossing a gap -- the same construction as
    `experiments.common.make_blocks`, restated here because Phase IV works on
    raw arrays rather than on a loaded labelled frame.
    """
    usable = [(s, e) for s, e in bounds if e - s >= block_rows]
    if not usable:
        usable = [max(bounds, key=lambda b: b[1] - b[0])]
        block_rows = max(2, usable[0][1] - usable[0][0])
    lengths = np.array([e - s for s, e in usable], dtype=float)
    alloc = np.maximum(1, np.round(lengths / lengths.sum() * n_blocks).astype(int))
    out = []
    for (s, e), k in zip(usable, alloc):
        k = int(min(k, max(1, (e - s) // block_rows)))
        for pos in np.linspace(s, e - block_rows, k).astype(int):
            out.append((int(pos), int(pos) + block_rows))
    rng.shuffle(out)
    return out[:n_blocks]


def bic_scan(X: np.ndarray, ks=(2, 3, 4, 5), seed: int = RANDOM_STATE) -> dict:
    """
    Per-machine BIC over candidate state counts, as a DIAGNOSTIC only.

    Phase IV fixes k = 4 everywhere so that STANDBY hours are comparable
    across machines.  Reporting what each machine's own BIC would have chosen
    is what makes that choice auditable: where BIC prefers a smaller k, the
    machine's productive range is being split, and Task 8 says so.
    """
    from sklearn.mixture import GaussianMixture

    out = {}
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="full",
                            n_init=2, random_state=seed).fit(X)
        out[int(k)] = {"bic": float(g.bic(X)), "aic": float(g.aic(X)),
                       "converged": bool(g.converged_)}
    best = min(out, key=lambda k: out[k]["bic"])
    return {"per_k": out, "bic_selected_k": int(best)}


def fit_state_model(
    X: np.ndarray,
    bounds: list[tuple[int, int]],
    sample_interval_s: float,
    k: int = K_STATES,
    seed: int = RANDOM_STATE,
    n_emission_rows: int = N_EMISSION_FIT_ROWS,
    n_blocks: int = N_TRANSITION_BLOCKS,
    block_seconds: float = TRANSITION_BLOCK_S,
    verbose: bool = True,
):
    """
    Fit the emission model on a random row sample and the transition matrix on
    contiguous blocks, returning a GaussianHMM ready to decode.

    The two fits use different samples because they estimate different things.
    Emissions are a property of the marginal distribution of the readings, so
    a uniform random sample of rows is the right sample and a contiguous slice
    would be biased toward whatever the machine happened to be doing.
    Transitions are a property of the sequence and can only be estimated from
    contiguous runs.
    """
    from sklearn.mixture import GaussianMixture
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(seed)
    n = len(X)
    sel = (rng.choice(n, size=n_emission_rows, replace=False)
           if n > n_emission_rows else np.arange(n))
    g = GaussianMixture(n_components=k, covariance_type="full",
                        n_init=5, random_state=seed).fit(X[sel].astype(np.float64))

    block_rows = max(2, int(round(block_seconds / sample_interval_s)))
    blocks = sample_blocks(bounds, block_rows, n_blocks, rng)
    Xt = np.vstack([X[s:e] for s, e in blocks]).astype(np.float64)
    lengths = [e - s for s, e in blocks]

    target_dwell_samples = max(STICKY_TARGET_DWELL_S / sample_interval_s, 1.0)
    m = GaussianHMM(
        n_components=k, covariance_type="full", n_iter=HMM_N_ITER,
        random_state=seed,
        init_params="",            # nothing auto-initialised
        params="t",                # ONLY the transition matrix is learned
        transmat_prior=np.ones((k, k)) + np.eye(k) * target_dwell_samples,
    )
    m.startprob_ = g.weights_.copy()
    m.means_ = g.means_.copy()
    m.covars_ = g.covariances_.copy()
    m.transmat_ = np.full((k, k), 1.0 / k)
    m.fit(Xt, lengths)

    if verbose:
        _info(f"emissions fitted on {len(sel):,} rows; transitions on "
              f"{len(blocks)} blocks / {len(Xt):,} rows "
              f"(converged={m.monitor_.converged}, iters={m.monitor_.iter}); "
              f"mean self-transition {np.mean(np.diag(m.transmat_)):.6f}")
    return m, g


def decode(
    model,
    X: np.ndarray,
    bounds: list[tuple[int, int]],
    min_rows: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Viterbi-decode every segment separately, then impose the minimum dwell.

    Decoding per segment is not an optimisation: a Viterbi path across an
    acquisition gap would assert a transition probability between two readings
    that are up to thirty days apart.  Returns (viterbi_labels,
    min_dwell_labels) so the constraint's effect is measurable rather than
    assumed.
    """
    raw = np.empty(len(X), dtype=np.int8)
    smoothed = np.empty(len(X), dtype=np.int8)
    for s, e in bounds:
        seg = X[s:e].astype(np.float64)
        lab = model.predict(seg)
        raw[s:e] = lab
        smoothed[s:e] = enforce_min_dwell(lab, min_rows) if min_rows > 1 else lab
    if verbose:
        changed = float((raw != smoothed).mean() * 100.0)
        _info(f"decoded {len(bounds)} segment(s); the min-dwell constraint "
              f"relabelled {changed:.2f}% of rows")
    return raw, smoothed


def map_labels_to_states(labels: np.ndarray, power: np.ndarray,
                         k: int = K_STATES) -> dict[int, str]:
    """
    Rank clusters by mean active power and hand out the canonical state names.

    The same physics-anchored rule as `src.state_mapping.map_clusters_to_states`
    and `experiments.common.map_clusters_to_states`: nothing about the cluster
    indices is meaningful, only their power ordering is.
    """
    present = np.unique(labels)
    means = {int(c): float(np.mean(power[labels == c])) for c in present}
    ranked = sorted(means, key=means.get)
    names = STATE_ORDER[:len(ranked)] if len(ranked) <= len(STATE_ORDER) else (
        STATE_ORDER + [f"STATE_{i}" for i in range(len(STATE_ORDER), len(ranked))])
    return {c: names[i] for i, c in enumerate(ranked)}


# ── End-to-end ────────────────────────────────────────────────────────────────

def label_record(
    df: pd.DataFrame,
    k: int = K_STATES,
    min_dwell_s: float = MIN_DWELL_S,
    seed: int = RANDOM_STATE,
    denoise: bool = True,
    with_bic_scan: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label an already-loaded record end to end and return (df, model_info).

    `df` is modified to carry `is_spike`, `segment_id`, `state_raw` (Viterbi)
    and `state` (Viterbi + minimum dwell).  `state_raw` is retained under the
    name `validate_gmm.py` gave its pre-smoothing column so that anything
    written against the Phase I-III exports keeps working, but note the
    difference: there `state_raw` was the framewise MAP assignment, here it is
    the unconstrained Viterbi path.  The pair (`state_raw`, `state`) is what
    isolates the minimum-dwell constraint, which is the comparison Phase IV
    needs; the framewise-vs-Viterbi comparison was Phase II's job and is
    already reported there.
    """
    info: dict = {}
    df, dt = add_segments(df)
    info["sample_interval_s"] = dt

    spike = compute_spike_mask(df, verbose=verbose)
    df["is_spike"] = spike
    info["spike_fraction"] = float(spike.mean())

    if denoise:
        df = denoise_off_state(df, verbose=verbose)

    X, scaler, cols = build_feature_matrix(df)
    info["feature_channels"] = cols
    info["n_channels"] = len(cols)

    bounds = segment_bounds(df["segment_id"].to_numpy())
    info["n_segments"] = len(bounds)

    if with_bic_scan:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(X), size=min(len(X), 100_000), replace=False)
        info["bic_scan"] = bic_scan(X[sub].astype(np.float64), seed=seed)

    model, gmm = fit_state_model(X, bounds, dt, k=k, seed=seed, verbose=verbose)
    min_rows = max(1, int(round(min_dwell_s / dt)))
    raw, smooth = decode(model, X, bounds, min_rows, verbose=verbose)

    power = df["active_power"].to_numpy(np.float64)
    mapping = map_labels_to_states(smooth, power, k=k)
    df["state_raw"] = pd.Categorical([mapping[int(c)] for c in raw],
                                     categories=STATE_ORDER[:k])
    df["state"] = pd.Categorical([mapping[int(c)] for c in smooth],
                                 categories=STATE_ORDER[:k])

    info.update({
        "k": k,
        "min_dwell_s": min_dwell_s,
        "min_dwell_rows": min_rows,
        "cluster_to_state": {str(c): s for c, s in mapping.items()},
        "transmat": model.transmat_.tolist(),
        "startprob": model.startprob_.tolist(),
        "mean_self_transition": float(np.mean(np.diag(model.transmat_))),
        "hmm_converged": bool(model.monitor_.converged),
        "hmm_iterations": int(model.monitor_.iter),
        "gmm_converged": bool(gmm.converged_),
        "rows_relabelled_by_min_dwell_pct": float((raw != smooth).mean() * 100.0),
        "emission_means_scaled": gmm.means_.tolist(),
        "seed": seed,
    })
    return df, info


PARQUET_COLUMNS = ["timestamp", "active_power", "reactive_power",
                   "apparent_power", "current", "voltage", "power_factor",
                   "segment_id", "is_spike", "state_raw", "state"]


def save_labelled(df: pd.DataFrame, path: str, verbose: bool = True) -> str:
    """
    Persist labels as parquet rather than as the 480 MB CSV the Phase I-III
    export produced.  Eight machines at that size would be 3.8 GB of text;
    parquet with float32 columns and a dictionary-encoded state column is
    about a twentieth of it and loads an order of magnitude faster.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [c for c in PARQUET_COLUMNS if c in df.columns]
    df[cols].to_parquet(path, index=False, compression="snappy")
    if verbose:
        _info(f"wrote {path} ({os.path.getsize(path)/1e6:.0f} MB, {len(df):,} rows)")
    return path


def label_machine(
    machine_key: str,
    raw_path: str,
    out_dir: str = PHASE4_LABELLED_DIR,
    k: int = K_STATES,
    min_dwell_s: float = MIN_DWELL_S,
    seed: int = RANDOM_STATE,
    nrows: int | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load, label and (optionally) persist one machine.  Returns (df, info)."""
    t0 = time.time()
    df = load_raw(raw_path, verbose=verbose)
    if nrows:
        df = df.iloc[:nrows].reset_index(drop=True)
    attrs = dict(df.attrs)
    df, info = label_record(df, k=k, min_dwell_s=min_dwell_s, seed=seed,
                            verbose=verbose)
    info.update({"machine_key": machine_key, "raw_path": raw_path,
                 "n_rows": int(len(df)), **attrs})
    if save:
        info["labelled_path"] = save_labelled(
            df, os.path.join(out_dir, f"{machine_key}_labelled.parquet"),
            verbose=verbose)
    info["label_seconds"] = round(time.time() - t0, 1)
    if verbose:
        _info(f"{machine_key} labelled in {info['label_seconds']/60:.1f} min")
    return df, info
