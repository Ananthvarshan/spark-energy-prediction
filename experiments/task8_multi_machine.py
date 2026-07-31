"""
============================================================
TASK 8 -- MULTI-MACHINE ANALYSIS
experiments/task8_multi_machine.py
============================================================

WHAT THIS ANSWERS
-----------------
Phases I-III were computed on one machine.  Three questions follow
from that, and this script answers them for all eight IMDELD
machines under one identical procedure:

  Q1  Does the state model hold up away from pelletizer-I?  Per
      machine: physics compliance, factory-schedule agreement,
      flicker and dwell, and the Otsu-derived power-factor and
      current thresholds of the five-method proof.

  Q2  Which machines are HARD, and why?  Difficulty is measured,
      not asserted: no-load/load separability, the BIC-preferred
      state count, spike rate, and how much of the record the
      minimum-dwell constraint has to repair.

  Q3  Phase III found only 86.8 h of energised idle on
      pelletizer-I and concluded that standby recovery there
      cannot fund a retrofit, leaving open whether any machine in
      the facility looks better.  The decision layer is therefore
      re-run per machine and the head-room reported for each.

WHY EVERY MACHINE IS RE-LABELLED
--------------------------------
The Phase I-III labels exist only for pelletizer-I and were
produced by a script, not a function.  Comparing that export with
seven fresh ones would confound machine differences with procedure
differences, so all eight -- pelletizer-I included -- are labelled
by `src/labelling.py` (`--source phase4`), and the agreement
between the two pelletizer-I labellings is reported as a
continuity check rather than assumed.

k IS FIXED AT 4 EVERYWHERE
--------------------------
STANDBY hours are only comparable across machines if the state
count is.  Each machine's own BIC-preferred k is reported beside
the result so the cost of that decision is visible.

USAGE
-----
    python -m experiments.task8_multi_machine
    python -m experiments.task8_multi_machine --machines pelletizer-I milling-I
    python -m experiments.task8_multi_machine --skip label
============================================================
"""

from __future__ import annotations

import os
import sys
import gc
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                    # noqa: E402
    MACHINES, IMDELD_MACHINES, PHASE4_DIR, FLICKER_TAU_S,
    STATE_ORDER, PRODUCTIVE_STATES,
    load_labelled, resolve_labelled,
    flicker_rate, physics_compliance, schedule_accuracy,
    degenerate_reason, MIN_COLD_STARTS,
    ensure_dir, save_json, info, section,
)
from src.labelling import label_machine, PHASE4_LABELLED_DIR   # noqa: E402
from src.decision import SECONDS_PER_HOUR           # noqa: E402

TASK8_DIR = f"{PHASE4_DIR}/task8"

# Rows used for the distributional checks (physics, schedule, Otsu).  A
# uniform stride over the whole record, not a random subset: means and
# thresholds converge long before 2M rows, and a stride keeps the sample
# spread over every shift and season instead of concentrating it wherever the
# random draw happened to land.  Hour totals and flicker are ALWAYS computed
# on every row -- they are the numbers the paper quotes.
CHECK_SAMPLE_ROWS = 2_000_000

# Active power above which a reading counts as "the machine is energised",
# matching proof_5methods.py's ON-state definition for the Otsu thresholds.
ON_STATE_MIN_W = 5.0

# Cold starts required before the measured restart signature is treated as
# identified (experiments.common.MIN_COLD_STARTS).  The coefficient is a
# difference of two medians, so a handful of cold starts gives an estimate
# whose bootstrap interval spans the plausible range; pelletizer-I supplies
# 126, but a machine the plant rarely de-energises can supply three.  The
# scenarios are still reported for such a machine -- with this flag set, and
# read as conditional on an unidentified coefficient.
MIN_COLD_STARTS_FOR_SIGNATURE = MIN_COLD_STARTS


def _import_proof():
    """Import proof_5methods with a clean argv -- it reads sys.argv on import."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import proof_5methods as pf
        return pf
    finally:
        sys.argv = saved


# ── Step 1: labelling ─────────────────────────────────────────────────────────

def label_all(machines: list[str], force: bool = False, seed: int = 42) -> dict:
    """Label every machine with the one Phase IV procedure."""
    section("TASK 8 -- STEP 1: uniform labelling")
    out = {}
    for key in machines:
        path = MACHINES[key]["labelled_phase4"]
        if os.path.exists(path) and not force:
            info(f"{key}: {path} exists, skipping (use --relabel to force)")
            continue
        info(f"labelling {key} from {MACHINES[key]['raw']}")
        _, meta = label_machine(key, MACHINES[key]["raw"],
                                out_dir=PHASE4_LABELLED_DIR, seed=seed)
        out[key] = meta
        save_json(meta, f"{TASK8_DIR}/{key}/label_info.json")
    return out


# ── Step 2: characterisation ──────────────────────────────────────────────────

def _add_calendar(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Factory-open flag, computed here rather than through
    `src.features.add_engineered_features`.

    The schedule metric needs one boolean column; the full engineered set
    costs eight rolling windows over millions of rows and none of them is used
    by any Task 8 metric.  The rule is the same one
    `src.decision.attach_local_calendar` applies: closed on weekdays between
    the plant's close and open hours, and closed all weekend.
    """
    local = df["timestamp"].dt.tz_convert(cfg["factory_tz"])
    h, dow = local.dt.hour, local.dt.dayofweek
    closed = ((h >= cfg["close_hour"]) & (h < cfg["open_hour"])) | (dow >= 5)
    df["is_factory_open"] = (~closed).astype(np.int8)
    return df


def _otsu_methods(df: pd.DataFrame, states: np.ndarray, otsu, dt: float) -> dict:
    """
    Methods 1 and 2 of the five-method proof, re-derived per machine.

    Both split the ENERGISED rows in two by an Otsu (1979) threshold -- on
    power factor for Method 1, on current for Method 2 -- and call the low
    side STANDBY.  Neither uses the state model, so the agreement between them
    and the GMM-HMM labels is an independent check on the labelling, and the
    two thresholds are the per-machine numbers the proof document quotes.

    Agreement is computed on ON rows only.  Including OFF rows would inflate
    every figure with the trivially easy part of the problem: the whole
    difficulty is separating no-load from light load, not separating either
    from a de-energised meter.
    """
    on = (df["active_power"].to_numpy(np.float64) >= ON_STATE_MIN_W)
    if "is_spike" in df.columns:
        on &= ~df["is_spike"].to_numpy(bool)
    n_on = int(on.sum())
    if n_on < 1000:
        return {"n_on_rows": n_on, "insufficient_on_rows": True}

    res: dict = {"n_on_rows": n_on,
                 "on_hours": n_on * dt / SECONDS_PER_HOUR}
    gmm_standby = (states[on] == "STANDBY")

    for name, col in (("m1_power_factor", "power_factor"), ("m2_current", "current")):
        if col not in df.columns:
            res[name] = {"available": False}
            continue
        v = df[col].to_numpy(np.float64)[on]
        v = np.where(np.isfinite(v), v, 0.0)
        thr = float(otsu(v))
        pred_standby = v < thr
        agree = float((pred_standby == gmm_standby).mean() * 100.0)
        res[name] = {
            "available": True,
            "otsu_threshold": thr,
            "standby_hours": float(pred_standby.sum() * dt / SECONDS_PER_HOUR),
            "gmm_standby_hours_on_rows": float(gmm_standby.sum() * dt / SECONDS_PER_HOUR),
            "row_agreement_pct": agree,
            "hour_divergence_pct": float(
                abs(pred_standby.sum() - gmm_standby.sum())
                / max(gmm_standby.sum(), 1) * 100.0),
        }
    return res


def _separability(df: pd.DataFrame, states: np.ndarray) -> dict:
    """
    How hard is this machine's no-load / load boundary?

    Two statistics, both on log active power because the states span orders of
    magnitude:

    `standby_working_d`  the two-sample effect size (Cohen's d) between
                         STANDBY and the productive states.  Large d means the
                         clusters are far apart relative to their own spread.
    `bhattacharyya`      the Bhattacharyya distance between Gaussian fits of
                         the same two groups, which unlike d penalises a
                         difference in variance as well as in mean.

    Neither uses labels from outside the model, so they measure the SHAPE of
    the problem the machine presents, not the quality of the answer -- which
    is what makes them usable as a difficulty axis against which the physics
    scores can be read.
    """
    P = np.log1p(df["active_power"].to_numpy(np.float64))
    sby = P[states == "STANDBY"]
    prod = P[np.isin(states, PRODUCTIVE_STATES)]
    if len(sby) < 100 or len(prod) < 100:
        return {"standby_working_d": float("nan"), "bhattacharyya": float("nan")}
    m1, m2 = float(sby.mean()), float(prod.mean())
    v1, v2 = float(sby.var()) + 1e-12, float(prod.var()) + 1e-12
    pooled = np.sqrt((v1 + v2) / 2.0)
    bc = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2)) + 0.25 * (m2 - m1) ** 2 / (v1 + v2)
    return {"standby_working_d": float(abs(m2 - m1) / pooled),
            "bhattacharyya": float(bc)}


def bic_diagnostic(machine: str, ks=(2, 3, 4, 5, 6, 7, 8), n_rows: int = 100_000,
                   source: str = "phase4", seed: int = 42) -> dict:
    """
    Does BIC identify a state count on these records?

    Phase IV fixes k = 4 for comparability and the labelling metadata already
    records the BIC-preferred k over 2..5.  On every machine that came back as
    5 -- the largest value tested -- which is not an answer, it is a boundary
    hit.  This widens the scan to 8 so the shape of the curve can be reported:
    if BIC is still decreasing at k = 8 on every machine, then BIC is not
    selecting a number of physical states here at all, and the physics vetoes
    that `validate_gmm.py` applies on top of it are not a belt-and-braces
    addition but the thing actually doing the selecting.

    That is a claim about the criterion, not about the data: at n > 10^6 the
    BIC penalty is negligible against the likelihood gain from splitting a
    non-Gaussian component, so a mixture model will keep buying components to
    approximate a distribution that is not a mixture of four Gaussians.
    """
    from experiments.common import build_cluster_matrix
    from sklearn.mixture import GaussianMixture

    df, _ = load_labelled(machine, drop_spikes=False, add_features=False,
                          source=source)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_rows, len(df)), replace=False)
    idx.sort()
    X, _ = build_cluster_matrix(df, idx)
    del df
    gc.collect()

    out = {}
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="full",
                            n_init=2, random_state=seed).fit(X)
        out[int(k)] = {"bic": float(g.bic(X)), "aic": float(g.aic(X))}
    best = min(out, key=lambda k: out[k]["bic"])
    res = {"machine": machine, "n_rows": int(len(X)), "ks": list(ks),
           "per_k": out, "bic_selected_k": int(best),
           "bic_still_falling_at_max_k": bool(best == max(ks)),
           "bic_drop_4_to_max_pct": float(
               (out[4]["bic"] - out[max(ks)]["bic"]) / abs(out[4]["bic"]) * 100.0)}
    info(f"{machine}: BIC over k={list(ks)} selects k={best}"
         f"{' (boundary of the scan)' if res['bic_still_falling_at_max_k'] else ''}")
    save_json(res, f"{TASK8_DIR}/{machine}/bic_scan.json")
    return res


def characterise(machine: str, source: str = "phase4", seed: int = 42) -> dict:
    """Every Task 8 read-out for one machine."""
    cfg = MACHINES[machine]
    section(f"TASK 8 -- STEP 2: characterising {machine}")
    t0 = time.time()

    df, meta = load_labelled(machine, drop_spikes=False, add_features=False,
                             source=source)
    dt = float(meta["sample_interval_s"])
    n = len(df)
    states = df["state"].to_numpy()
    covered_s = n * dt
    span_s = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

    # -- record ---------------------------------------------------------------
    rec = {
        "machine": machine,
        "machine_name": cfg["name"],
        "label_source": meta["label_source"],
        "n_rows": n,
        "sample_interval_s": dt,
        "coverage_days": covered_s / 86400.0,
        "span_days": span_s / 86400.0,
        "coverage_fraction": covered_s / max(span_s, 1.0),
        "n_segments": int(df["segment_id"].nunique()),
        "spike_pct": float(df["is_spike"].mean() * 100.0) if "is_spike" in df else np.nan,
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }

    # -- state hours (every row; spikes reported separately) ------------------
    counts = pd.Series(states).value_counts()
    hours = {s: float(counts.get(s, 0) * dt / SECONDS_PER_HOUR) for s in STATE_ORDER}
    total_h = sum(hours.values())
    rec["state_hours"] = hours
    rec["state_share_pct"] = {s: (h / total_h * 100.0 if total_h else np.nan)
                              for s, h in hours.items()}
    rec["standby_hours"] = hours["STANDBY"]
    rec["standby_hours_per_covered_day"] = hours["STANDBY"] / (covered_s / 86400.0)

    sby_mask = states == "STANDBY"
    rec["standby_power_w"] = (float(df.loc[sby_mask, "active_power"].mean())
                              if sby_mask.any() else float("nan"))
    rec["standby_energy_kwh"] = (rec["standby_power_w"] / 1000.0 * hours["STANDBY"]
                                 if np.isfinite(rec["standby_power_w"]) else float("nan"))

    # -- flicker and dwell, per contiguous segment, on every row --------------
    seg = df["segment_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], n]
    seg_blocks = [states[s:e] for s, e in zip(starts, ends) if e - s > 1]
    rec["flicker"] = flicker_rate(seg_blocks, dt, FLICKER_TAU_S)
    rec["flicker"].pop("per_state_dwell", None)

    if "state_raw" in df.columns:
        raw_blocks = [df["state_raw"].to_numpy()[s:e]
                      for s, e in zip(starts, ends) if e - s > 1]
        raw = flicker_rate(raw_blocks, dt, FLICKER_TAU_S)
        rec["flicker_before_min_dwell"] = {
            "flicker_rate_pct": raw["flicker_rate_pct"],
            "flicker_rate_standby_binary_pct": raw["flicker_rate_standby_binary_pct"],
            "median_dwell_s": raw["median_dwell_s"],
        }
        del raw_blocks
    del seg_blocks

    # -- distributional checks on a strided sample ----------------------------
    stride = max(1, n // CHECK_SAMPLE_ROWS)
    idx = np.arange(0, n, stride)
    cols = [c for c in ("timestamp", "active_power", "current", "power_factor",
                        "is_spike") if c in df.columns]
    sub = df.iloc[idx][cols].reset_index(drop=True)
    sub_states = states[idx]
    sub = _add_calendar(sub, cfg)
    rec["check_sample"] = {"n_rows": int(len(sub)), "stride": int(stride)}

    phys = physics_compliance(sub, np.arange(len(sub)), sub_states)
    rec["physics"] = {"score": phys["score"],
                      "n_passed": phys["n_checks_passed"],
                      "n_applicable": phys["n_checks_applicable"],
                      "checks": phys["checks"],
                      "values": phys["values"]}
    rec["per_state"] = phys["per_state"]
    rec["schedule"] = schedule_accuracy(sub, np.arange(len(sub)), sub_states)

    pf = _import_proof()
    rec["otsu_methods"] = _otsu_methods(sub, sub_states, pf.otsu_threshold, dt * stride)
    rec["separability"] = _separability(sub, sub_states)
    del sub

    # -- continuity with the Phase I-III export, where one exists -------------
    ref_path = cfg.get("labelled")
    if source == "phase4" and ref_path and os.path.exists(ref_path):
        rec["reference_agreement"] = _compare_with_reference(machine, df, dt)

    rec["elapsed_s"] = round(time.time() - t0, 1)
    del df, states
    save_json(rec, f"{TASK8_DIR}/{machine}/characterisation.json")

    info(f"{machine}: physics {rec['physics']['score']:.2f} "
         f"({rec['physics']['n_passed']}/{rec['physics']['n_applicable']}), "
         f"schedule {rec['schedule']['balanced_schedule_score']:.1f}%, "
         f"flicker {rec['flicker']['flicker_rate_pct']:.2f}%, "
         f"STANDBY {rec['standby_hours']:.1f} h @ {rec['standby_power_w']:.0f} W")
    return rec


def _compare_with_reference(machine: str, df4: pd.DataFrame, dt: float) -> dict:
    """
    Agreement between the Phase IV labelling and the Phase I-III export.

    This is the continuity check that makes the rest of Phase IV readable
    against Phases II and III: if the uniform labeller reproduces the export
    on the machine both cover, then a Phase IV number that differs from a
    Phase III number is a consequence of the analysis, not of the labels.
    Where they diverge, the divergence is reported here rather than left for a
    reader to discover.
    """
    ref_path, _ = resolve_labelled(machine, "reference")
    ref = pd.read_csv(ref_path, usecols=["timestamp", "state"])
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True)

    p4 = pd.DataFrame({"timestamp": df4["timestamp"].to_numpy(),
                       "p4": df4["state"].to_numpy()})

    # THE RECORD CONTAINS REPEATED TIMESTAMPS.  A plain merge on the timestamp
    # is a cartesian product within each repeated second and returns MORE rows
    # than either input (5.91M against 5.47M here), which inflates every hour
    # total it is used to compute.  The rows are in acquisition order in both
    # files, so the occurrence index within a timestamp identifies a reading
    # uniquely and the join on (timestamp, occurrence) is one-to-one.
    p4["occ"] = p4.groupby("timestamp").cumcount()
    ref["occ"] = ref.groupby("timestamp").cumcount()
    merged = pd.merge(p4, ref.rename(columns={"state": "ref"}),
                      on=["timestamp", "occ"], how="inner")
    del ref, p4
    if merged.empty:
        return {"n_common_rows": 0}

    a, b = merged["p4"].to_numpy(), merged["ref"].to_numpy()
    from sklearn.metrics import adjusted_rand_score
    sample = np.arange(0, len(a), max(1, len(a) // 500_000))
    out = {
        "reference_path": ref_path,
        "n_common_rows": int(len(a)),
        "row_agreement_pct": float((a == b).mean() * 100.0),
        "standby_binary_agreement_pct": float(
            ((a == "STANDBY") == (b == "STANDBY")).mean() * 100.0),
        "idle_binary_agreement_pct": float(
            (~np.isin(a, PRODUCTIVE_STATES) == ~np.isin(b, PRODUCTIVE_STATES)).mean()
            * 100.0),
        "ari": float(adjusted_rand_score(b[sample], a[sample])),
        "standby_hours_phase4": float((a == "STANDBY").sum() * dt / SECONDS_PER_HOUR),
        "standby_hours_reference": float((b == "STANDBY").sum() * dt / SECONDS_PER_HOUR),
    }
    info(f"continuity vs Phase I-III export: rows {out['row_agreement_pct']:.1f}%, "
         f"STANDBY-binary {out['standby_binary_agreement_pct']:.1f}%, "
         f"ARI {out['ari']:.3f}, "
         f"STANDBY {out['standby_hours_phase4']:.1f} h vs "
         f"{out['standby_hours_reference']:.1f} h")
    return out


# ── Step 3: the decision layer, per machine ───────────────────────────────────

def decision_for_machine(machine: str, source: str = "phase4",
                         seed: int = 0, tag: str = "", split_ts=None) -> dict:
    """
    Run Phase III's decision layer unchanged on one machine.

    Nothing is re-tuned per machine: the same tariff, the same minimum-off
    time, the same restarts-per-day cap, the same three break-even scenarios,
    and the same policy set.  Every coefficient that CAN be measured
    (standby power, restart delay, restart energy) is re-measured from that
    machine's own record, which is the point -- the framework is supposed to
    calibrate itself to a machine it has not seen.
    """
    from experiments.task6_optimization import (
        prepare, build_policies, BREAK_EVEN_SCENARIOS, HEADLINE_SCENARIO,
        TARIFF_USD_PER_KWH, MIN_OFF_S, MAX_RESTARTS_PER_DAY, DECISION_EPOCH_S,
    )
    from src.decision import compare_policies, params_for_break_even, annualise

    section(f"TASK 8 -- STEP 3: decision layer on {machine}")
    prep = prepare(machine, tariff=TARIFF_USD_PER_KWH, min_off_s=MIN_OFF_S,
                   max_restarts_per_day=MAX_RESTARTS_PER_DAY,
                   epoch_s=DECISION_EPOCH_S, seed=seed, source=source,
                   split_ts=split_ts)

    ep, ep_te = prep["episodes"], prep["ep_test"]
    base = prep["base_params"]
    sig = prep["restart_signature"]
    test_span_s = prep["test_span_s"]

    out: dict = {
        "machine": machine,
        "label_source": source,
        "standby_power_w": prep["standby_power_w"],
        "coverage_days": prep["covered_s"] / 86400.0,
        "restart_delay_s": sig.get("restart_delay_s"),
        "restart_energy_kwh": sig.get("restart_energy_kwh"),
        "restart_energy_significant": sig.get("restart_energy_is_significant"),
        "n_cold": sig.get("n_cold"), "n_warm": sig.get("n_warm"),
        "restart_signature_identified": bool(
            (sig.get("n_cold") or 0) >= MIN_COLD_STARTS_FOR_SIGNATURE),
        "restart_delay_ci95_s": sig.get("restart_delay_ci95_s"),
        "n_episodes_usable": int(len(ep)),
        "idle_hours": float(ep["duration_s"].sum() / SECONDS_PER_HOUR),
        "standby_hours_observed": float(ep["standby_s"].sum() / SECONDS_PER_HOUR),
        "off_hours_observed": float(ep["off_s"].sum() / SECONDS_PER_HOUR),
        "energised_idle_fraction": float(
            ep["standby_s"].sum() / max(ep["duration_s"].sum(), 1)),
        "max_episode_h": float(ep["duration_s"].max() / SECONDS_PER_HOUR),
        "median_episode_s": float(ep["duration_s"].median()),
        "n_test_episodes": int(len(ep_te)),
        "test_span_days": test_span_s / 86400.0,
        "forecaster": {},
        "scenarios": {},
    }

    # Forecast quality on this machine's own held-out epochs.
    yte = prep["yte"]
    pred = prep["predictors"]["mean"]
    from experiments.task6_optimization import _feature_matrix       # noqa: F401
    p = np.clip(prep["forecaster_models"]["mean"].predict(prep["Xte"]), 0, None)
    out["forecaster"] = {
        "n_train_epochs": int(len(prep["ytr"])),
        "n_test_epochs": int(len(yte)),
        "mae_s": float(np.mean(np.abs(p - yte))),
        "spearman": float(pd.Series(p).corr(pd.Series(yte), method="spearman")),
        "mean_ratio": float(p.mean() / max(yte.mean(), 1e-9)),
    }

    degenerate = degenerate_reason(prep["standby_power_w"], sig)
    out["degenerate_reason"] = degenerate
    if degenerate is None:
        for name, be in BREAK_EVEN_SCENARIOS.items():
            params = params_for_break_even(base, be)
            policies = build_policies(ep_te, params, prep["predictors"],
                                      prep["feasible"], prep["epoch_s"])
            tab = compare_policies(ep_te, policies, params)
            row = {
                "break_even_min": be / 60.0,
                "implied_delay_cost_per_h": params.delay_cost_per_hour,
                "restart_cost_usd": params.restart_cost,
                "head_room_usd_per_year": float(annualise(
                    tab.loc["observed", "cost_total"] - tab.loc["oracle", "cost_total"],
                    test_span_s)),
            }
            for pol in ("never_shutdown", "ski_rental", "static_break_even",
                        "forecast_opt", "oracle"):
                if pol in tab.index:
                    row[f"{pol}_usd_per_year"] = float(annualise(
                        tab.loc[pol, "savings_vs_baseline"], test_span_s))
                    row[f"{pol}_pct_of_oracle"] = float(tab.loc[pol, "pct_of_oracle"])
            row["forecast_opt_min_off_violations"] = int(
                tab.loc["forecast_opt", "min_off_violations"])
            row["observed_min_off_violations"] = int(
                tab.loc["observed", "min_off_violations"])
            row["observed_loss_making"] = int(
                tab.loc["observed", "loss_making_shutdowns"])
            out["scenarios"][name] = row
            info(f"  {name}: proposed {row.get('forecast_opt_usd_per_year', float('nan')):+.0f} USD/yr "
                 f"({row.get('forecast_opt_pct_of_oracle', float('nan')):.0f}% of oracle), "
                 f"head-room {row['head_room_usd_per_year']:.0f} USD/yr")
        out["headline_scenario"] = HEADLINE_SCENARIO
    else:
        # The decision problem is ill-posed on this machine.  Its scenarios are
        # left empty rather than filled with figures produced by dividing by a
        # standby power of ~0 W or by a restart delay that was never observed.
        out["scenarios_unavailable"] = degenerate
        info(f"  no decision problem here: {degenerate}")

    save_json(out, f"{TASK8_DIR}/{machine}/decision{tag}.json")
    return out


def label_source_sensitivity(machine: str = "pelletizer-I", seed: int = 0) -> dict:
    """
    The same decision layer on the same machine under BOTH labellings.

    Phase III measured its head-room and its policy value on the Phase I-III
    export.  Phase IV re-labels with the minimum-dwell constraint that Phase II
    recommended, which removes the sub-10-second state runs -- and idle
    EPISODES are built from exactly those runs.  Any difference between Phase
    III's numbers and Phase IV's therefore has two possible causes, the change
    of labels and the change of machine, and on pelletizer-I the second is
    absent.  Running both here separates them.

    BOTH RUNS ARE SPLIT AT THE SAME INSTANT.  The default 70/30 split is on
    episode count, and the two labellings do not produce the same number of
    episodes, so that split would test them on different calendar windows and
    the comparison would confound the labelling with the weather of one month.
    The instant is taken from the record itself -- the timestamp 70% of the way
    through its covered rows -- and is therefore a property of the data rather
    than of either labelling.

    Requires the Phase I-III CSV export, so it is skipped where that is absent.
    """
    section(f"TASK 8 -- label-source sensitivity on {machine}")

    ts = pd.read_parquet(MACHINES[machine]["labelled_phase4"], columns=["timestamp"])
    split_ts = pd.Timestamp(ts["timestamp"].iloc[int(len(ts) * 0.70)])
    del ts
    info(f"common chronological split at {split_ts}")

    ref = decision_for_machine(machine, source="reference", seed=seed,
                               tag="_reference", split_ts=split_ts)
    p4 = decision_for_machine(machine, source="phase4", seed=seed,
                              tag="_aligned_split", split_ts=split_ts)

    def _head(d):
        return d.get("scenarios", {}).get("S2_moderate", {})

    cmp = {
        "machine": machine,
        "split_ts": str(split_ts),
        "reference": {
            "standby_power_w": ref["standby_power_w"],
            "n_episodes_usable": ref["n_episodes_usable"],
            "median_episode_s": ref["median_episode_s"],
            "idle_hours": ref["idle_hours"],
            "standby_hours_observed": ref["standby_hours_observed"],
            "restart_delay_s": ref["restart_delay_s"],
            "forecast_mae_s": ref["forecaster"].get("mae_s"),
            **{k: _head(ref).get(k) for k in
               ("head_room_usd_per_year", "forecast_opt_usd_per_year",
                "forecast_opt_pct_of_oracle", "ski_rental_usd_per_year")},
        },
        "phase4": {
            "standby_power_w": p4.get("standby_power_w"),
            "n_episodes_usable": p4.get("n_episodes_usable"),
            "median_episode_s": p4.get("median_episode_s"),
            "idle_hours": p4.get("idle_hours"),
            "standby_hours_observed": p4.get("standby_hours_observed"),
            "restart_delay_s": p4.get("restart_delay_s"),
            "forecast_mae_s": p4.get("forecaster", {}).get("mae_s"),
            **{k: _head(p4).get(k) for k in
               ("head_room_usd_per_year", "forecast_opt_usd_per_year",
                "forecast_opt_pct_of_oracle", "ski_rental_usd_per_year")},
        },
    }
    save_json(cmp, f"{TASK8_DIR}/{machine}/label_source_sensitivity.json")
    print(pd.DataFrame(cmp["reference"], index=["reference"]).T.join(
        pd.DataFrame(cmp["phase4"], index=["phase4"]).T).to_string())
    return cmp


# ── Tables ────────────────────────────────────────────────────────────────────

def _labelling_diagnostics(machine: str) -> dict:
    """
    Diagnostics that live in the labelling metadata rather than in the labels:
    the BIC-preferred k over the labeller's own narrow scan, how much of the
    record the minimum-dwell constraint relabelled, and the learned mean
    self-transition.  Attached to the characterisation whether it was just
    computed or loaded from cache, so a partial re-run produces the same table
    as a full one.
    """
    import json
    p = f"{TASK8_DIR}/{machine}/label_info.json"
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        lm = json.load(fh)
    return {
        "bic_selected_k": lm.get("bic_scan", {}).get("bic_selected_k"),
        "min_dwell_relabelled_pct": lm.get("rows_relabelled_by_min_dwell_pct"),
        "mean_self_transition": lm.get("mean_self_transition"),
        "label_seconds": lm.get("label_seconds"),
        "hmm_iterations": lm.get("hmm_iterations"),
    }


def build_tables(chars: dict, decisions: dict) -> dict[str, pd.DataFrame]:
    """Assemble the three Task 8 tables the paper needs."""
    rows_state, rows_diff, rows_dec = [], [], []
    for m, c in chars.items():
        phys, sched = c["physics"], c["schedule"]
        otsu = c.get("otsu_methods", {})
        m1 = otsu.get("m1_power_factor", {})
        m2 = otsu.get("m2_current", {})
        rows_state.append({
            "machine": m,
            "name": c["machine_name"],
            "coverage_days": c["coverage_days"],
            "span_days": c["span_days"],
            "n_segments": c["n_segments"],
            "spike_pct": c["spike_pct"],
            "standby_h": c["standby_hours"],
            "standby_h_per_day": c["standby_hours_per_covered_day"],
            "standby_power_w": c["standby_power_w"],
            "off_pct": c["state_share_pct"]["OFF"],
            "standby_pct": c["state_share_pct"]["STANDBY"],
            "productive_pct": (c["state_share_pct"]["WORKING"]
                               + c["state_share_pct"]["PEAK_LOAD"]),
            "physics_score": phys["score"],
            "physics_passed": f"{phys['n_passed']}/{phys['n_applicable']}",
            "pf_separation": phys["values"].get("pf_separation", np.nan),
            "current_ratio": phys["values"].get("current_ratio", np.nan),
            "schedule_score": sched["balanced_schedule_score"],
            "closed_window_pct": sched["closed_window_correct_pct"],
            "flicker_pct": c["flicker"]["flicker_rate_pct"],
            "flicker_standby_pct": c["flicker"]["flicker_rate_standby_binary_pct"],
            "flicker_before_min_dwell_pct": c.get("flicker_before_min_dwell", {}).get(
                "flicker_rate_pct", np.nan),
            "median_dwell_s": c["flicker"]["median_dwell_s"],
            "otsu_pf_threshold": m1.get("otsu_threshold", np.nan),
            "m1_agreement_pct": m1.get("row_agreement_pct", np.nan),
            "otsu_current_threshold": m2.get("otsu_threshold", np.nan),
            "m2_agreement_pct": m2.get("row_agreement_pct", np.nan),
        })
        rows_diff.append({
            "machine": m,
            "standby_working_d": c["separability"]["standby_working_d"],
            "bhattacharyya": c["separability"]["bhattacharyya"],
            "bic_selected_k": c.get("bic_selected_k", np.nan),
            "bic_selected_k_wide": c.get("bic_selected_k_wide", np.nan),
            "bic_still_falling": c.get("bic_still_falling", np.nan),
            "spike_pct": c["spike_pct"],
            "min_dwell_relabelled_pct": c.get("min_dwell_relabelled_pct", np.nan),
            "flicker_before_min_dwell_pct": c.get("flicker_before_min_dwell", {}).get(
                "flicker_rate_pct", np.nan),
            "physics_score": phys["score"],
            "m1_agreement_pct": m1.get("row_agreement_pct", np.nan),
            "n_segments": c["n_segments"],
            "coverage_fraction": c["coverage_fraction"],
        })

    for m, d in decisions.items():
        head = d.get("scenarios", {}).get("S2_moderate", {})
        rows_dec.append({
            "machine": m,
            "standby_power_w": d["standby_power_w"],
            "idle_h": d["idle_hours"],
            "standby_h_energised": d["standby_hours_observed"],
            "off_h": d["off_hours_observed"],
            "energised_idle_pct": d["energised_idle_fraction"] * 100.0,
            "restart_delay_s": d["restart_delay_s"],
            "n_cold_starts": d.get("n_cold", np.nan),
            "signature_identified": d.get("restart_signature_identified", False),
            "decision_problem": ("degenerate" if d.get("degenerate_reason")
                                 else ("ok" if d.get("scenarios") else "error")),
            "restart_energy_kwh": d["restart_energy_kwh"],
            "forecast_mae_s": d["forecaster"].get("mae_s", np.nan),
            "forecast_spearman": d["forecaster"].get("spearman", np.nan),
            "head_room_usd_yr": head.get("head_room_usd_per_year", np.nan),
            "proposed_usd_yr": head.get("forecast_opt_usd_per_year", np.nan),
            "proposed_pct_oracle": head.get("forecast_opt_pct_of_oracle", np.nan),
            "ski_rental_usd_yr": head.get("ski_rental_usd_per_year", np.nan),
            "observed_violations": head.get("observed_min_off_violations", np.nan),
            "proposed_violations": head.get("forecast_opt_min_off_violations", np.nan),
        })

    return {
        "state_identification": pd.DataFrame(rows_state).set_index("machine"),
        "difficulty": pd.DataFrame(rows_diff).set_index("machine"),
        "decision": pd.DataFrame(rows_dec).set_index("machine") if rows_dec
        else pd.DataFrame(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machines: list[str] | None = None,
    steps: tuple[str, ...] = ("label", "characterise", "bic", "decision",
                              "labelsource"),
    source: str = "phase4",
    relabel: bool = False,
    seed: int = 42,
) -> dict:
    machines = machines or IMDELD_MACHINES
    ensure_dir(TASK8_DIR)
    section(f"PHASE IV / TASK 8 -- {len(machines)} machines: {machines}")

    label_meta = label_all(machines, force=relabel, seed=seed) if "label" in steps else {}

    chars: dict = {}
    if "characterise" not in steps:
        # Re-use the cached characterisation so that re-running only the
        # decision step still rebuilds every table.  Characterisation is the
        # expensive part (it loads eight multi-million-row records), and a
        # partial re-run should not silently drop two of the three tables.
        import json
        for m in machines:
            p = f"{TASK8_DIR}/{m}/characterisation.json"
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    chars[m] = json.load(fh)
                chars[m].update(_labelling_diagnostics(m))
        if chars:
            info(f"loaded cached characterisation for {len(chars)} machine(s)")
    else:
        for m in machines:
            c = characterise(m, source=source, seed=seed)
            c.update(_labelling_diagnostics(m))
            chars[m] = c

    bics: dict = {}
    if "bic" in steps:
        for m in machines:
            bics[m] = bic_diagnostic(m, source=source, seed=seed)
    else:
        import json
        for m in machines:
            p = f"{TASK8_DIR}/{m}/bic_scan.json"
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    bics[m] = json.load(fh)
    for m, b in bics.items():
        if m in chars:
            chars[m]["bic_selected_k_wide"] = b["bic_selected_k"]
            chars[m]["bic_still_falling"] = b["bic_still_falling_at_max_k"]

    decisions: dict = {}
    if "decision" in steps:
        for m in machines:
            try:
                decisions[m] = decision_for_machine(m, source=source, seed=0)
            except Exception as exc:                # a machine that cannot be
                info(f"{m}: decision layer FAILED -- {type(exc).__name__}: {exc}")
                decisions[m] = {"machine": m,
                                "error": f"{type(exc).__name__}: {exc}",
                                "standby_power_w": np.nan, "idle_hours": np.nan,
                                "standby_hours_observed": np.nan,
                                "off_hours_observed": np.nan,
                                "energised_idle_fraction": np.nan,
                                "restart_delay_s": np.nan,
                                "restart_energy_kwh": np.nan,
                                "forecaster": {}, "scenarios": {}}

    sens = {}
    if "labelsource" in steps:
        for m in machines:
            ref = MACHINES[m].get("labelled")
            if ref and os.path.exists(ref):
                sens[m] = label_source_sensitivity(m)

    tables = build_tables(chars, decisions) if chars else {}
    for name, tab in tables.items():
        if len(tab):
            tab.to_csv(f"{TASK8_DIR}/table_{name}.csv")
            section(f"TASK 8 -- {name.replace('_', ' ').upper()}")
            print(tab.round(3).to_string())

    save_json({"machines": machines, "source": source, "seed": seed,
               "characterisation": chars, "decision": decisions,
               "bic_scan": bics, "label_source_sensitivity": sens,
               "label_meta": label_meta},
              f"{TASK8_DIR}/task8_results.json")
    section("TASK 8 COMPLETE")
    return {"characterisation": chars, "decision": decisions, "tables": tables}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase IV Task 8 -- multi-machine analysis")
    ap.add_argument("--machines", nargs="*", default=None,
                    help=f"subset of {IMDELD_MACHINES}")
    ap.add_argument("--only", nargs="*", default=None,
                    help="steps: label characterise bic decision labelsource")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--source", default="phase4", choices=["phase4", "reference", "auto"])
    ap.add_argument("--relabel", action="store_true", help="re-label even if parquet exists")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    all_steps = ("label", "characterise", "bic", "decision", "labelsource")
    steps = tuple(s for s in (a.only or all_steps) if s not in a.skip)
    main(machines=a.machines, steps=steps, source=a.source,
         relabel=a.relabel, seed=a.seed)
