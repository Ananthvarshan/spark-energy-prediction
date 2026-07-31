"""
============================================================
DECISION & OPTIMISATION LAYER  --  src/decision.py
============================================================

PURPOSE
-------
Phase III.  Replace the fixed break-even heuristic in
`src/lstm_inference.should_recommend_shutdown` with a stated
optimisation problem, and make the break-even duration a DERIVED
quantity of that problem rather than an input constant.

THE PROBLEM
-----------
The machine alternates between productive operation and idle
periods.  During an idle period it may be left energised (drawing
`standby_power_w` and producing nothing) or de-energised.
De-energising saves standby energy but incurs a restart: extra
energy, extra ramp time before production resumes, and mechanical
wear.

    minimise   C_total = C_standby * h_standby
                       + C_restart * N_restarts
                       + C_delay   * h_delay

    subject to  h_off      >= h_min_off        (thermal cooling)
                N_restarts <= N_max per day    (contactor / mechanical wear)
                x(t)       in {0, 1}           (energised / de-energised)

`EconomicParams` holds the coefficients; `episode_savings` is the
objective restated per idle episode; `break_even_s` is its root.

REDUCTION TO EPISODES
---------------------
x(t) is defined at every second, but the objective only ever
changes value at an idle-period boundary, and within one idle
episode the standby cost is linear and non-decreasing in the time
spent energised.  So if an episode is going to be shut down at
all, it is optimal to shut it down at its onset, and the whole
timeline collapses to one scalar per episode:

    tau_i = elapsed seconds before de-energising  (inf = never)

This is what makes an exact optimum computable over a 5.5-million
row record: `optimal_offline` solves the reduced problem exactly
rather than approximating the original.

WHY THIS IS A SKI-RENTAL PROBLEM
--------------------------------
Online, the episode duration is unknown when the decision must be
made.  Pay a small cost repeatedly (standby power) or a large
one-off cost (restart) that ends the payments -- this is exactly
the ski-rental / spin-block problem, and the classical result
applies: de-energising once accumulated standby cost equals the
restart cost is 2-competitive, i.e. never worse than twice the
cost of the offline optimum, whatever the duration distribution
(Karlin et al. 1994).  That is the `elapsed_time` policy below,
and it is the correct baseline for the forecast-driven policy to
beat -- beating "never shut down" proves nothing.

WHY THE FORECAST ONLY NEEDS A CONDITIONAL MEAN
----------------------------------------------
The saving from de-energising with R seconds of idle remaining is

    c_standby * R - restart_cost

which is AFFINE in R.  Its expectation therefore depends on the
predictive distribution of R only through E[R], so a mean
regression is sufficient for the decision rule and a full
predictive distribution buys nothing here.  (It would matter if
the minimum-off constraint were priced rather than reported, or
if the tariff were non-linear.)

STATUS-QUO BASELINE
-------------------
Three reference points are reported, not one:

  observed        what the plant actually did, read off the labels
  never_shutdown  the machine left energised through every idle
                  episode -- a hypothetical upper bound on cost
  oracle          the exact offline optimum -- a lower bound

`never_shutdown` is NOT the honest baseline: the operators already
de-energise this machine for long gaps (575 of 663 idle hours).
Savings quoted against it would be savings the plant has already
banked.  The deployment-relevant number is the saving against
`observed`.
============================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd


# Machine states that represent real mechanical work.  Everything else is an
# opportunity to de-energise.
PRODUCTIVE_STATES = ("WORKING", "PEAK_LOAD")

# Fraction of the machine's normal productive power that counts as "production
# has resumed", used to time the restart ramp.
PRODUCTION_RESUMED_FRAC = 0.90

SECONDS_PER_HOUR = 3600.0


# ── Economic parameters ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class EconomicParams:
    """
    Coefficients of the objective, plus the operational constraints.

    Every cost is expressed in one currency unit per the stated basis, so
    `restart_cost` and `standby_cost_per_s` are directly comparable and their
    ratio is a duration -- which is the break-even point.

    standby_power_w
        Mean active power drawn while idle but energised.  Data-derived: use
        the STANDBY cluster's own mean, never a spec-sheet figure.
    tariff_per_kwh
        Electricity price.  0.12 USD/kWh is the Brazilian industrial average
        for the period the IMDELD record covers.
    restart_energy_kwh
        Extra electrical energy attributable to a cold start, over and above
        resuming from standby.  `measure_restart_signature` estimates it from
        the record; it is NOT a free parameter.
    restart_labour_cost
        Any non-energy per-restart cost (operator attention, consumables).
        Its presence is what makes the break-even point tariff-dependent --
        see `break_even_s`.
    restart_delay_s
        Extra time to reach production from cold versus from standby.  Also
        measured, not assumed.
    delay_cost_per_hour
        Value of lost production per hour of restart delay.  This is the one
        coefficient the electrical record cannot supply; it is swept in the
        Task 7 sensitivity analysis rather than asserted.
    min_off_s
        Minimum time the machine must remain de-energised once shut down
        (motor thermal cooling).  A shutdown whose idle period turns out
        shorter than this is a constraint violation, reported, never hidden.
    max_restarts_per_day
        Cap on de-energisation cycles per calendar day (mechanical wear).
    """

    standby_power_w: float
    tariff_per_kwh: float = 0.12
    restart_energy_kwh: float = 0.0
    restart_labour_cost: float = 0.0
    restart_delay_s: float = 0.0
    delay_cost_per_hour: float = 0.0
    min_off_s: float = 0.0
    max_restarts_per_day: int = 24
    currency: str = "USD"

    # -- derived coefficients --------------------------------------------------

    @property
    def standby_cost_per_s(self) -> float:
        """Cost of leaving the machine energised for one idle second."""
        return self.tariff_per_kwh * (self.standby_power_w / 1000.0) / SECONDS_PER_HOUR

    @property
    def restart_cost(self) -> float:
        """
        Total cost of one de-energise/re-energise cycle: energy, labour and
        the value of the production delay.
        """
        return (
            self.tariff_per_kwh * self.restart_energy_kwh
            + self.restart_labour_cost
            + self.delay_cost_per_hour * (self.restart_delay_s / SECONDS_PER_HOUR)
        )

    @property
    def break_even_s(self) -> float:
        """
        Idle duration at which de-energising exactly pays for itself --
        the root of `episode_savings(D, tau=0) = 0`:

            D* = restart_cost / standby_cost_per_s

        Note what this expression does NOT say.  If the restart cost were
        purely electrical (labour = 0, delay cost = 0) the tariff cancels:

            D* = E_restart / P_standby

        and the break-even point is INDEPENDENT of electricity price.  A
        sensitivity heat-map with tariff on one axis and break-even time as
        the response is therefore degenerate unless a non-energy cost term is
        present.  It is the labour and delay terms that make price matter,
        and that is why they are carried explicitly here.
        """
        c = self.standby_cost_per_s
        if c <= 0:
            return math.inf
        return self.restart_cost / c

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "standby_cost_per_s": self.standby_cost_per_s,
            "restart_cost": self.restart_cost,
            "break_even_s": self.break_even_s,
            "break_even_min": self.break_even_s / 60.0,
            "break_even_h": self.break_even_s / SECONDS_PER_HOUR,
        })
        return d

    def with_(self, **kw) -> "EconomicParams":
        """Return a copy with fields replaced (for sensitivity sweeps)."""
        return replace(self, **kw)


def params_for_break_even(
    base: EconomicParams,
    target_break_even_s: float,
) -> EconomicParams:
    """
    Return a copy of `base` whose break-even point equals `target_break_even_s`,
    achieved by solving for the production-delay valuation.

    WHY SCENARIOS ARE DEFINED THIS WAY.  Every coefficient of the objective is
    measurable from the electrical record except the value of lost production,
    which no ammeter can supply.  Asserting a figure for it would make that
    single invented number the hidden driver of every result -- and because the
    break-even point is linear in it, the choice would silently decide whether
    the answer is "always shut down" or "never shut down".

    So the scenarios are indexed by the quantity that actually matters to the
    decision -- where break-even falls relative to the observed idle-duration
    distribution -- and the delay valuation each one implies is REPORTED as an
    output.  A plant can then compare the implied figure with its own
    production economics instead of inheriting ours.

    Solves  c_standby * D* = tariff * E_restart + labour + c_delay * (T_d/3600)
    for c_delay.  Raises if the measured delay is zero, in which case the
    break-even point cannot be steered through the delay term.
    """
    if base.restart_delay_s <= 0:
        raise ValueError("cannot solve for a delay valuation when the measured "
                         "restart delay is zero; vary restart_labour_cost instead")
    fixed = base.tariff_per_kwh * base.restart_energy_kwh + base.restart_labour_cost
    needed = base.standby_cost_per_s * target_break_even_s
    c_delay = (needed - fixed) / (base.restart_delay_s / SECONDS_PER_HOUR)
    return base.with_(delay_cost_per_hour=float(c_delay))


# ── Idle-episode extraction ───────────────────────────────────────────────────

def extract_idle_episodes(
    df: pd.DataFrame,
    sample_interval_s: float = 1.0,
    state_col: str = "state",
    segment_col: str = "segment_id",
    productive_states: tuple[str, ...] = PRODUCTIVE_STATES,
) -> pd.DataFrame:
    """
    Collapse the labelled time series into one row per idle episode.

    An idle episode is a maximal run of non-productive readings bounded by
    productive operation.  Its `duration_s` is the resource the decision layer
    is bidding for.

    SPIKE ROWS MUST NOT HAVE BEEN DROPPED.  This repeats Task 4's protocol
    correction: removing ~19.5% of a 1 Hz series makes the retained rows
    non-uniformly spaced, so a run of N rows no longer spans N seconds and
    every episode duration is silently compressed.  Load with
    `drop_spikes=False`.

    TRUNCATED EPISODES ARE FLAGGED, NOT DROPPED HERE.  An episode touching a
    segment boundary (the record has 31 acquisition gaps, the largest 30 days)
    has an unknown true duration -- the machine may have run for a week in the
    gap.  Counting such an episode as an enormous shutdown opportunity would
    manufacture savings out of missing data.  `is_truncated` marks them so the
    experiment can exclude them and report how much was excluded.

    Returns one row per episode with timing, composition (how much of it the
    plant actually spent energised), and the onset context a forecaster needs.
    """
    for col in (state_col, segment_col):
        if col not in df.columns:
            raise KeyError(f"extract_idle_episodes needs column '{col}'")

    states = df[state_col].to_numpy()
    seg = df[segment_col].to_numpy()
    idle = ~np.isin(states, productive_states)

    # Run boundaries: a new run starts wherever idleness or the segment changes.
    change = np.r_[True, (idle[1:] != idle[:-1]) | (seg[1:] != seg[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(df)]              # exclusive

    ts = df["timestamp"].to_numpy()
    power = (df["active_power"].to_numpy(float)
             if "active_power" in df.columns else np.zeros(len(df)))

    # Segment extent, to detect episodes that abut an acquisition gap.
    seg_first = pd.Series(np.arange(len(df))).groupby(seg).transform("min").to_numpy()
    seg_last = pd.Series(np.arange(len(df))).groupby(seg).transform("max").to_numpy()

    rows = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        if not idle[a]:
            continue
        n = b - a
        sub_states = states[a:b]
        # Context: the productive run immediately before this episode.
        prev_a = starts[k - 1] if k > 0 else None
        prev_prod_s = float((a - prev_a) * sample_interval_s) if prev_a is not None and not idle[prev_a] else np.nan
        prev_prod_power = float(np.mean(power[prev_a:a])) if prev_a is not None and not idle[prev_a] and a > prev_a else np.nan

        rows.append({
            "episode_id": len(rows),
            "i0": int(a),
            "i1": int(b - 1),
            "n_rows": int(n),
            "duration_s": float(n * sample_interval_s),
            "start_ts": ts[a],
            "end_ts": ts[b - 1],
            "segment_id": int(seg[a]),
            # Composition of what the plant actually did during the episode.
            "standby_s": float((sub_states == "STANDBY").sum() * sample_interval_s),
            "off_s": float((sub_states == "OFF").sum() * sample_interval_s),
            "observed_shutdown": bool((sub_states == "OFF").any()),
            # An episode that starts or ends at a segment edge has unknown extent.
            "is_truncated": bool(a == seg_first[a] or (b - 1) == seg_last[b - 1]),
            "prev_productive_s": prev_prod_s,
            "prev_productive_power_w": prev_prod_power,
        })

    ep = pd.DataFrame(rows)
    if ep.empty:
        return ep

    ep["start_ts"] = pd.to_datetime(ep["start_ts"], utc=True)
    ep["end_ts"] = pd.to_datetime(ep["end_ts"], utc=True)
    return ep


def attach_local_calendar(
    ep: pd.DataFrame,
    factory_tz: str = "America/Sao_Paulo",
    close_hour: int = 17,
    open_hour: int = 22,
) -> pd.DataFrame:
    """
    Add local-time onset context and the calendar day used by the
    restarts-per-day constraint.

    The day key is the FACTORY-LOCAL date, not the UTC date: the wear
    constraint is a plant operating rule and a UTC day boundary would split
    a Brazilian night shift in two.
    """
    ep = ep.copy()
    local = ep["start_ts"].dt.tz_convert(factory_tz)
    ep["local_day"] = local.dt.date.astype(str)
    ep["hour"] = local.dt.hour + local.dt.minute / 60.0
    ep["dow"] = local.dt.dayofweek
    ep["is_weekend"] = (ep["dow"] >= 5).astype(int)
    # Factory closed window: weekday close_hour..open_hour, plus all weekend.
    h = local.dt.hour
    closed = ((h >= close_hour) & (h < open_hour)) | (ep["dow"] >= 5)
    ep["is_factory_open"] = (~closed).astype(int)
    return ep


# ── Data-derived restart signature ────────────────────────────────────────────

def measure_restart_signature(
    df: pd.DataFrame,
    sample_interval_s: float = 1.0,
    state_col: str = "state",
    segment_col: str = "segment_id",
    productive_states: tuple[str, ...] = PRODUCTIVE_STATES,
    min_source_run_s: float = 60.0,
    max_ramp_s: float = 7200.0,
    resumed_frac: float = PRODUCTION_RESUMED_FRAC,
    common_window_s: float = 1800.0,
    common_window_min_productive_frac: float = 0.5,
    n_boot: int = 2000,
    random_state: int = 0,
) -> dict:
    """
    Estimate the restart delay and restart energy from the record instead of
    assuming them.

    Every resumption of production is classified by the state the machine
    resumed FROM -- cold (OFF) or warm (STANDBY) -- and the cost of having
    de-energised is the DIFFERENCE between the two.  Charging the whole
    cold-start ramp to the shutdown decision would bill it for a ramp the
    machine would have paid anyway; only the increment is caused by the
    decision.

    DELAY: milestone timing.

        restart_delay_s = median(t to reach `resumed_frac` of production | cold)
                        - median(same | warm)

    ENERGY: two estimators, because the obvious one is not valid on its own.

      (a) `energy_milestone_kwh` integrates each ramp up to the production
          milestone and differences the medians.  This compares energy over
          UNEQUAL durations -- the cold ramp is longer -- so it is reported
          as descriptive only and must not be used as a cost coefficient.

      (b) `restart_energy_kwh` integrates both cases over a COMMON window of
          `common_window_s` measured from the moment production resumes, and
          differences those.  Equal windows make the two energies comparable.
          The window is required to be at least
          `common_window_min_productive_frac` productive so that a resumption
          which immediately collapses back to idle is not counted; the
          threshold cannot be 100% because the cold ramp itself passes through
          the low-power range and is partly labelled non-productive.

    Both estimators come out NEGATIVE on the pelletizer record, and the reason
    matters for how the result is reported.  A cold start reaches production
    more slowly, so within any fixed window it has produced less and therefore
    drawn less.  The energy difference and the delay are two views of the same
    physical fact, and adding a negative energy term to a positive delay term
    would double-count it.  The defensible conclusion is that this machine has
    no separately identifiable electrical restart penalty -- its restart cost
    is a production-delay cost -- so the caller should floor the energy
    coefficient at zero and let the delay term carry it.  `restart_energy_kwh`
    is still returned with a bootstrap interval so the claim can be checked.
    """
    rng = np.random.default_rng(random_state)

    states = df[state_col].to_numpy()
    seg = df[segment_col].to_numpy()
    P = df["active_power"].to_numpy(float)
    prod_mask = np.isin(states, productive_states)

    if not prod_mask.any():
        return {"n_cold": 0, "n_warm": 0}

    # Reference production level: the machine's own top-load median.
    top_state = max(productive_states,
                    key=lambda s: np.mean(P[states == s]) if (states == s).any() else -np.inf)
    prod_level = float(np.median(P[states == top_state]))
    target = resumed_frac * prod_level

    # Run boundaries over the raw state labels.
    change = np.r_[True, (states[1:] != states[:-1]) | (seg[1:] != seg[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(df)]

    min_rows = int(round(min_source_run_s / sample_interval_s))
    max_rows = int(round(max_ramp_s / sample_interval_s))

    ramps = {"OFF": [], "STANDBY": []}
    for k in range(len(starts) - 1):
        a, b = starts[k], ends[k]
        src = states[a]
        if src not in ramps or (b - a) < min_rows:
            continue
        a2, b2 = starts[k + 1], ends[k + 1]
        if seg[a2] != seg[a] or not prod_mask[a2]:
            continue                       # not a resumption of production
        win = P[a2:min(a2 + max_rows, len(P))]
        hit = np.flatnonzero(win >= target)
        if len(hit) == 0:
            continue                       # never reached production in the window
        t_rows = int(hit[0]) + 1
        t_s = t_rows * sample_interval_s
        e_kwh = float(win[:t_rows].sum() * sample_interval_s / SECONDS_PER_HOUR / 1000.0)
        ramps[src].append((t_s, e_kwh))

    cold = np.array(ramps["OFF"], dtype=float).reshape(-1, 2)
    warm = np.array(ramps["STANDBY"], dtype=float).reshape(-1, 2)

    # -- estimator (b): energy over a common post-resumption window ------------
    # Idle runs are re-derived here (rather than reusing the state runs above)
    # because a cold ramp passes through the low-power range and is partly
    # labelled non-productive, so an idle run may contain both OFF and STANDBY.
    idle = ~prod_mask
    ichg = np.r_[True, (idle[1:] != idle[:-1]) | (seg[1:] != seg[:-1])]
    ist = np.flatnonzero(ichg)
    ien = np.r_[ist[1:], len(idle)]
    w_rows = int(round(common_window_s / sample_interval_s))

    cw = {"cold": [], "warm": []}
    for a, b in zip(ist, ien):
        if not idle[a] or b >= len(idle) or seg[b] != seg[a]:
            continue
        if b + w_rows > len(idle) or seg[b + w_rows - 1] != seg[a]:
            continue
        win = slice(b, b + w_rows)
        if prod_mask[win].mean() < common_window_min_productive_frac:
            continue
        e = float(P[win].sum() * sample_interval_s / SECONDS_PER_HOUR / 1000.0)
        cw["cold" if (states[a:b] == "OFF").any() else "warm"].append(e)
    cw_cold = np.asarray(cw["cold"], dtype=float)
    cw_warm = np.asarray(cw["warm"], dtype=float)

    def _boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        """Median difference a - b with a percentile bootstrap interval."""
        if len(a) == 0 or len(b) == 0:
            return float("nan"), float("nan"), float("nan")
        point = float(np.median(a) - np.median(b))
        draws = np.empty(n_boot)
        for i in range(n_boot):
            draws[i] = (np.median(rng.choice(a, len(a), replace=True))
                        - np.median(rng.choice(b, len(b), replace=True)))
        return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    d_t, d_t_lo, d_t_hi = _boot_diff(cold[:, 0], warm[:, 0]) if len(cold) and len(warm) else (np.nan,) * 3
    m_e, m_e_lo, m_e_hi = _boot_diff(cold[:, 1], warm[:, 1]) if len(cold) and len(warm) else (np.nan,) * 3
    d_e, d_e_lo, d_e_hi = _boot_diff(cw_cold, cw_warm)

    def _describe(arr, name):
        if len(arr) == 0:
            return {}
        return {
            f"{name}_n": int(len(arr)),
            f"{name}_ramp_median_s": float(np.median(arr[:, 0])),
            f"{name}_ramp_mean_s": float(np.mean(arr[:, 0])),
            f"{name}_energy_median_kwh": float(np.median(arr[:, 1])),
            f"{name}_energy_mean_kwh": float(np.mean(arr[:, 1])),
        }

    out = {
        "production_level_w": prod_level,
        "resumed_threshold_w": target,
        "n_cold": int(len(cold)),
        "n_warm": int(len(warm)),
        "restart_delay_s": d_t,
        "restart_delay_ci95_s": [d_t_lo, d_t_hi],
        # Estimator (b) -- the one that may legitimately be used as a cost.
        "restart_energy_kwh": d_e,
        "restart_energy_ci95_kwh": [d_e_lo, d_e_hi],
        "restart_energy_is_significant": bool(
            not math.isnan(d_e_lo) and (d_e_lo > 0 or d_e_hi < 0)
        ),
        "common_window_s": common_window_s,
        "n_cold_common_window": int(len(cw_cold)),
        "n_warm_common_window": int(len(cw_warm)),
        "cold_common_window_median_kwh": (float(np.median(cw_cold))
                                          if len(cw_cold) else float("nan")),
        "warm_common_window_median_kwh": (float(np.median(cw_warm))
                                          if len(cw_warm) else float("nan")),
        # Estimator (a) -- descriptive only, unequal windows.
        "energy_milestone_kwh": m_e,
        "energy_milestone_ci95_kwh": [m_e_lo, m_e_hi],
    }
    out.update(_describe(cold, "cold"))
    out.update(_describe(warm, "warm"))
    return out


# ── Objective, restated per episode ───────────────────────────────────────────

def episode_savings(
    duration_s: np.ndarray | float,
    tau_s: np.ndarray | float,
    params: EconomicParams,
) -> np.ndarray:
    """
    Saving from de-energising an idle episode of length `duration_s` after
    `tau_s` seconds, relative to leaving it energised throughout.

        saving = c_standby * (D - tau)  -  restart_cost      if tau < D
                 0                                           otherwise

    Monotone decreasing in tau, which is the formal reason `optimal_offline`
    only ever considers tau = 0.
    """
    D = np.asarray(duration_s, dtype=float)
    tau = np.asarray(tau_s, dtype=float)
    acts = np.isfinite(tau) & (tau < D)
    off_s = np.where(acts, D - np.minimum(tau, D), 0.0)
    return np.where(acts, params.standby_cost_per_s * off_s - params.restart_cost, 0.0)


# ── Policies ──────────────────────────────────────────────────────────────────
#
# Every policy returns tau_s per episode: seconds of idling to tolerate before
# de-energising, with inf meaning "leave it energised".  Separating the policy
# (which may only use information available at the time) from the evaluation
# (which uses the realised durations) is what keeps the comparison honest.

def policy_never(ep: pd.DataFrame, params: EconomicParams) -> np.ndarray:
    """Leave the machine energised through every idle episode."""
    return np.full(len(ep), np.inf)


def policy_observed(ep: pd.DataFrame, params: EconomicParams) -> np.ndarray:
    """
    The plant's actual behaviour, read off the labels.

    The machine was de-energised after `standby_s` seconds of idling in every
    episode where an OFF reading appears; elsewhere it was left energised.
    This is the status quo the framework has to beat to be worth deploying.
    """
    return np.where(ep["observed_shutdown"].to_numpy(bool),
                    ep["standby_s"].to_numpy(float), np.inf)


def policy_immediate(ep: pd.DataFrame, params: EconomicParams) -> np.ndarray:
    """De-energise the instant the machine goes idle. Ignores restart cost."""
    return np.zeros(len(ep))


def policy_elapsed_time(ep: pd.DataFrame, params: EconomicParams) -> np.ndarray:
    """
    Ski-rental policy: idle until accumulated standby cost equals the restart
    cost, then de-energise.

    Uses NO forecast and no future information -- only a clock.  It is
    2-competitive against the offline optimum for any duration distribution
    (Karlin et al. 1994), which makes it the reference a forecast-driven
    policy must beat before the forecast can be said to add value.
    """
    return np.full(len(ep), params.break_even_s)


def policy_static_threshold(
    ep: pd.DataFrame,
    params: EconomicParams,
    predicted_duration_s: np.ndarray,
) -> np.ndarray:
    """
    The heuristic this phase replaces: forecast the idle duration once at
    onset, de-energise immediately if the forecast exceeds break-even.

    Equivalent to `should_recommend_shutdown` in src/lstm_inference.py, but
    with the break-even value derived from `EconomicParams` instead of being
    supplied as a constant.  It is included to quantify what the optimisation
    layer actually buys over it.
    """
    pred = np.asarray(predicted_duration_s, float)
    return np.where(pred > params.break_even_s, 0.0, np.inf)


def policy_forecast_optimising(
    ep: pd.DataFrame,
    params: EconomicParams,
    predict_remaining_s,
    epoch_s: float = 60.0,
    max_epochs: int = 240,
    predict_feasible=None,
    confidence: float = 0.0,
) -> np.ndarray:
    """
    The proposed policy: re-decide every `epoch_s` seconds while the machine
    stays idle, using a forecast of the REMAINING idle time.

    At elapsed time tau the rule de-energises when

        c_standby * E[R | features, tau] > restart_cost         (economics)
        P(R >= min_off_s | features, tau) >= confidence         (feasibility)

    WHY TWO CONDITIONS AND NOT ONE.  The saving is affine in R, so an
    UNCONSTRAINED risk-neutral rule needs only E[R] -- that is the argument
    made in the module docstring, and it is why a mean regression suffices for
    the economics.  The minimum-off constraint breaks the affinity: a shutdown
    whose idle period turns out shorter than the motor's cooling time is
    infeasible, not merely unprofitable, and no expectation over R can express
    that.  Feasibility is therefore imposed as a CHANCE CONSTRAINT at a stated
    confidence level rather than folded into the objective with an invented
    penalty price.

    This matters empirically, not just formally.  The idle-duration
    distribution is heavy-tailed enough that at episode onset the conditional
    mean sits far above the conditional median, so a mean-driven rule with no
    feasibility test fires on the many short episodes whose mean is inflated
    by the tail, and most of those shutdowns violate the cooling constraint.

    `confidence = 0` recovers the pure risk-neutral rule, which is how the
    contribution of the chance constraint is isolated.

    Two further properties are worth stating in the paper.

    It is a sequential rule, not a one-shot classification.  A forecast made
    at onset is made when the least information is available; every additional
    idle minute is evidence about how long the episode will run, and the rule
    exploits it.  This is why it can beat `policy_static_threshold` using the
    same forecaster.

    It degrades gracefully to ski-rental.  If the forecaster is uninformative
    and returns a constant, the rule fires at a fixed elapsed time and inherits
    the 2-competitive guarantee.

    `predict_remaining_s(ep_rows, elapsed_s) -> array` and
    `predict_feasible(ep_rows, elapsed_s) -> array of probabilities` must use
    only information available at that elapsed time.
    """
    n = len(ep)
    tau = np.full(n, np.inf)
    duration = ep["duration_s"].to_numpy(float)
    undecided = np.ones(n, dtype=bool)

    for step in range(max_epochs):
        elapsed = step * epoch_s
        # Only episodes still idle at this elapsed time can still be decided.
        alive = undecided & (duration > elapsed)
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        rows = ep.iloc[idx]
        r_hat = np.asarray(predict_remaining_s(rows, elapsed), float)
        fire = params.standby_cost_per_s * r_hat - params.restart_cost > 0

        if params.min_off_s > 0:
            if predict_feasible is not None:
                p_ok = np.asarray(predict_feasible(rows, elapsed), float)
                fire &= p_ok >= confidence
            else:
                # No feasibility model available: fall back to the plug-in test.
                fire &= r_hat >= params.min_off_s

        tau[idx[fire]] = elapsed
        undecided[idx[fire]] = False
    return tau


def optimal_offline(ep: pd.DataFrame, params: EconomicParams) -> np.ndarray:
    """
    Exact offline optimum with perfect foresight -- the oracle bound.

    Because `episode_savings` is decreasing in tau, any episode that is shut
    down at all is shut down at tau = 0, so the problem reduces to SELECTING a
    subset of episodes subject to:

      * saving_i > 0                       (never take a loss-making shutdown)
      * D_i >= min_off_s                   (thermal cooling)
      * at most N_max selections per local day  (mechanical wear)

    Savings are additive across episodes and the only coupling is the per-day
    cardinality cap, so the exact optimum is: within each day, take the
    N_max largest positive savings.  No dynamic programme is needed -- greedy
    selection under a cardinality constraint on a modular objective is
    provably optimal -- and this is the exact solution of the stated problem,
    not a heuristic for it.

    NOTE the asymmetry this creates with the online policies: the oracle
    allocates its daily restart budget to the best episodes of the day, while
    an online policy must commit without knowing what the rest of the day
    holds.  Any gap between them under a binding cap is a cost of causality,
    not a deficiency of the forecaster, and should be reported as such.
    """
    D = ep["duration_s"].to_numpy(float)
    sav = episode_savings(D, np.zeros(len(ep)), params)
    eligible = (sav > 0) & (D >= params.min_off_s)

    tau = np.full(len(ep), np.inf)
    cap = params.max_restarts_per_day
    if cap is None or cap >= len(ep):
        tau[eligible] = 0.0
        return tau

    days = ep["local_day"].to_numpy() if "local_day" in ep.columns else np.zeros(len(ep))
    order = np.argsort(-sav)                       # best first
    used: dict = {}
    for i in order:
        if not eligible[i]:
            continue
        d = days[i]
        if used.get(d, 0) >= cap:
            continue
        used[d] = used.get(d, 0) + 1
        tau[i] = 0.0
    return tau


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(
    ep: pd.DataFrame,
    tau_s: np.ndarray,
    params: EconomicParams,
    enforce_daily_cap: bool = True,
) -> dict:
    """
    Score a policy's decisions against the REALISED episode durations.

    The daily restart cap is applied here, chronologically, to every policy
    alike: a shutdown requested after the day's budget is spent is refused and
    the machine stays energised.  Applying it at evaluation time rather than
    inside each policy guarantees no policy can quietly exceed it, and models
    what a plant interlock would actually do.

    Minimum-off violations are counted, not priced.  A violation means the
    policy de-energised and production resumed before the motor had cooled;
    the appropriate response is to report the rate so a reviewer can judge it,
    not to fold it into the objective with an invented penalty.
    """
    tau = np.asarray(tau_s, float).copy()
    D = ep["duration_s"].to_numpy(float)
    days = ep["local_day"].to_numpy() if "local_day" in ep.columns else np.zeros(len(ep))

    acts = np.isfinite(tau) & (tau < D)

    if enforce_daily_cap and params.max_restarts_per_day is not None:
        # Chronological order, so the budget goes to whoever asks first.
        chrono = np.argsort(ep["start_ts"].to_numpy())
        used: dict = {}
        for i in chrono:
            if not acts[i]:
                continue
            d = days[i]
            if used.get(d, 0) >= params.max_restarts_per_day:
                acts[i] = False
                tau[i] = np.inf
            else:
                used[d] = used.get(d, 0) + 1

    off_s = np.where(acts, D - np.minimum(tau, D), 0.0)
    standby_s = D - off_s
    n_restarts = int(acts.sum())

    energy_standby_kwh = float(standby_s.sum() * params.standby_power_w / 1000.0
                               / SECONDS_PER_HOUR)
    energy_restart_kwh = n_restarts * params.restart_energy_kwh
    delay_h = n_restarts * params.restart_delay_s / SECONDS_PER_HOUR

    cost_standby = params.tariff_per_kwh * energy_standby_kwh
    cost_restart = n_restarts * (params.tariff_per_kwh * params.restart_energy_kwh
                                 + params.restart_labour_cost)
    cost_delay = params.delay_cost_per_hour * delay_h

    violations = int((acts & (off_s < params.min_off_s)).sum())
    loss_making = int((acts & (episode_savings(D, tau, params) < 0)).sum())

    return {
        "n_episodes": int(len(ep)),
        "n_shutdowns": n_restarts,
        "shutdown_rate_pct": float(n_restarts / max(len(ep), 1) * 100.0),
        "idle_h": float(D.sum() / SECONDS_PER_HOUR),
        "standby_h": float(standby_s.sum() / SECONDS_PER_HOUR),
        "off_h": float(off_s.sum() / SECONDS_PER_HOUR),
        "energy_standby_kwh": energy_standby_kwh,
        "energy_restart_kwh": energy_restart_kwh,
        "energy_net_kwh": energy_standby_kwh + energy_restart_kwh,
        "cost_standby": cost_standby,
        "cost_restart": cost_restart,
        "cost_delay": cost_delay,
        "cost_total": cost_standby + cost_restart + cost_delay,
        "delay_h": delay_h,
        "min_off_violations": violations,
        "min_off_violation_pct": float(violations / max(n_restarts, 1) * 100.0),
        "loss_making_shutdowns": loss_making,
        "mean_off_duration_h": (float(off_s[acts].mean() / SECONDS_PER_HOUR)
                                if n_restarts else 0.0),
    }


def compare_policies(
    ep: pd.DataFrame,
    policies: dict[str, np.ndarray],
    params: EconomicParams,
    baseline: str = "observed",
    oracle: str = "oracle",
) -> pd.DataFrame:
    """
    Evaluate every policy and express each one against the status quo and
    against the oracle.

    `savings_vs_baseline` is the deployment number: currency saved per unit of
    record versus what the plant did.  `pct_of_oracle` is the research number:
    how much of the theoretically attainable saving the policy captured.  A
    policy can look excellent on the first and poor on the second, and both
    facts belong in the paper.
    """
    rows = {name: evaluate_policy(ep, tau, params) for name, tau in policies.items()}
    base_cost = rows[baseline]["cost_total"] if baseline in rows else np.nan
    orac_cost = rows[oracle]["cost_total"] if oracle in rows else np.nan
    head_room = base_cost - orac_cost

    for name, r in rows.items():
        r["savings_vs_baseline"] = base_cost - r["cost_total"]
        r["savings_vs_baseline_pct"] = (
            (base_cost - r["cost_total"]) / base_cost * 100.0 if base_cost else np.nan
        )
        r["pct_of_oracle"] = (
            (base_cost - r["cost_total"]) / head_room * 100.0
            if head_room and abs(head_room) > 1e-12 else np.nan
        )
        r["energy_saved_vs_baseline_kwh"] = (
            rows[baseline]["energy_net_kwh"] - r["energy_net_kwh"]
            if baseline in rows else np.nan
        )

    out = pd.DataFrame(rows).T
    out.index.name = "policy"
    return out


def annualise(value: float, covered_seconds: float) -> float:
    """
    Scale a per-record figure to a year using the seconds actually COVERED by
    the record, not its calendar span.

    The IMDELD pelletizer record spans 155 days but contains 31 acquisition
    gaps totalling ~83 days, so only ~63 days of data exist.  Annualising on
    the span would understate every rate by a factor of 2.4.
    """
    if covered_seconds <= 0:
        return float("nan")
    return value * (365.25 * 86400.0) / covered_seconds
