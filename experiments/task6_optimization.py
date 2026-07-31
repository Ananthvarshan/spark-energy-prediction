"""
============================================================
TASK 6 -- OPTIMISATION-BASED DECISION LAYER
experiments/task6_optimization.py
============================================================

WHAT THIS ANSWERS
-----------------
Phase III replaces a fixed threshold rule ("if predicted standby
> break-even then shut down") with the optimisation problem stated
in `src/decision.py`.  This script instantiates that problem on the
GMM-HMM-labelled record and answers four questions:

  Q1  What do the restart coefficients equal for this machine?
      Measured from the record with bootstrap intervals, not taken
      from a spec sheet.

  Q2  Where does the derived break-even point fall relative to the
      observed idle-duration distribution?  This, not the algorithm,
      is what decides whether a decision layer can do anything.

  Q3  How much of the attainable saving does each policy capture,
      out-of-sample, against the plant's own status quo?

  Q4  Does the forecast earn its place?  A forecast-driven policy
      must beat the forecast-free ski-rental rule -- beating
      "never shut down" proves nothing.

WHY SCENARIOS RATHER THAN ONE COST SETTING
------------------------------------------
Every coefficient is measurable except the value of lost
production.  Since break-even is linear in that coefficient,
asserting a value for it would silently decide the answer.  The
scenarios are therefore indexed by the break-even point itself,
spanning the observed idle-duration distribution, and the delay
valuation each implies is reported as an output.  Task 7 sweeps the
whole plane.

PROTOCOL
--------
  * Spike rows are KEPT (Task 4 correction 1): dropping 19.5% of a
    1 Hz series destroys the row-count-to-seconds correspondence
    every episode duration depends on.
  * Episodes touching an acquisition gap are excluded -- their true
    duration is unknown and counting them would manufacture savings
    out of missing data.
  * Chronological split.  The forecaster is fitted on the earlier
    70% of episodes and every policy scored on the later 30%.
    Unlike Task 4's alternating blocks this must be chronological:
    a decision policy is deployed forward in time, and a shuffled
    split would let it learn the future duty cycle.
  * The daily restart cap is enforced at evaluation time, equally
    for every policy.

    python -m experiments.task6_optimization --machine pelletizer-I
============================================================
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                       # noqa: E402
    DEFAULT_MACHINE, MACHINES,
    load_labelled, ensure_dir, save_json, info, section,
)
from src.decision import (                             # noqa: E402
    EconomicParams, extract_idle_episodes, attach_local_calendar,
    measure_restart_signature, compare_policies, annualise,
    params_for_break_even,
    policy_never, policy_observed, policy_immediate, policy_elapsed_time,
    policy_static_threshold, policy_forecast_optimising, optimal_offline,
    SECONDS_PER_HOUR,
)

PHASE3_DIR = "outputs/phase3"

# ── Measured / stated constants ───────────────────────────────────────────────

TARIFF_USD_PER_KWH = 0.12      # Brazilian industrial average over the record
RESTART_LABOUR_COST = 0.0      # no operator attendance for a contactor cycle
MIN_OFF_S = 600.0              # motor thermal cooling; swept in Task 7
MAX_RESTARTS_PER_DAY = 4       # mechanical wear ceiling; swept in Task 7

# Scenarios, indexed by where break-even falls in the idle-duration
# distribution.  Chosen before looking at any policy result: 10 min sits below
# the bulk of the useful episodes, 1 h near the top of them, 4 h above all but
# a handful.  The implied production-delay valuation is an OUTPUT.
BREAK_EVEN_SCENARIOS = {
    "S1_low":      10 * 60.0,
    "S2_moderate": 60 * 60.0,
    "S3_high":     4 * 3600.0,
}
HEADLINE_SCENARIO = "S2_moderate"

DECISION_EPOCH_S = 60.0        # how often the online policy re-decides
MAX_DECISION_HOURS = 24.0

POLICY_ORDER = ["never_shutdown", "observed", "immediate",
                "static_break_even", "ski_rental", "forecast_opt_median",
                "forecast_opt_nochance", "forecast_opt", "oracle"]

# Confidence required by the chance constraint before de-energising: the
# policy must be at least this sure the machine will stay idle long enough to
# satisfy the motor cooling time.  A stated operating point, not a tuned one;
# swept in Task 7.
CHANCE_CONFIDENCE = 0.90


# ── Remaining-duration forecaster ─────────────────────────────────────────────

FEATURE_NAMES = [
    "elapsed_log", "onset_hour", "onset_dow", "is_weekend",
    "cur_hour_sin", "cur_hour_cos", "cur_is_open", "cur_is_weekend",
    "prev_productive_log", "prev_productive_power_w",
]


def _feature_matrix(
    ep: pd.DataFrame,
    elapsed_s: float | np.ndarray,
    close_hour: int = 17,
    open_hour: int = 22,
) -> np.ndarray:
    """
    Features available to the decision rule at a given elapsed idle time.

    Deliberately contains only what a controller genuinely holds at that
    instant: how long the machine has been idle, where the CURRENT wall-clock
    time sits in the factory calendar, and the load history of the production
    run that just ended.  No feature is drawn from the future of the episode.

    The current-time features matter more than the onset-time ones.  An
    episode that began at 16:50 looks unremarkable at onset and decisive forty
    minutes later, once the clock has crossed the plant's 17:00 shutdown: the
    remaining idle time jumps from minutes to hours.  Encoding only the onset
    hour would hide exactly the transition the policy needs to see.
    """
    n = len(ep)
    elapsed = np.broadcast_to(np.asarray(elapsed_s, float), (n,)).astype(float)
    onset_hour = ep["hour"].to_numpy(float)
    onset_dow = ep["dow"].to_numpy(float)

    hours_elapsed = elapsed / SECONDS_PER_HOUR
    cur_hour = np.mod(onset_hour + hours_elapsed, 24.0)
    cur_dow = np.mod(onset_dow + np.floor((onset_hour + hours_elapsed) / 24.0), 7.0)
    cur_weekend = (cur_dow >= 5).astype(float)
    closed = ((cur_hour >= close_hour) & (cur_hour < open_hour)) | (cur_weekend > 0)

    prev = np.nan_to_num(ep["prev_productive_s"].to_numpy(float), nan=0.0)
    prev_pow = np.nan_to_num(ep["prev_productive_power_w"].to_numpy(float), nan=0.0)

    return np.column_stack([
        np.log1p(elapsed),
        onset_hour,
        onset_dow,
        ep["is_weekend"].to_numpy(float),
        np.sin(2 * np.pi * cur_hour / 24.0),
        np.cos(2 * np.pi * cur_hour / 24.0),
        (~closed).astype(float),
        cur_weekend,
        np.log1p(prev),
        prev_pow,
    ])


def build_epoch_table(
    ep: pd.DataFrame,
    epoch_s: float = DECISION_EPOCH_S,
    max_hours: float = MAX_DECISION_HOURS,
    close_hour: int = 17,
    open_hour: int = 22,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand episodes into (features, remaining-seconds) training pairs, one per
    decision epoch the online policy would actually reach.

    Training on epochs rather than on episode onsets is the point.  The policy
    is asked "how much longer will this run?" at every elapsed time, so the
    training distribution must contain those same conditional questions.  A
    model fitted only at onset would be applied far outside its support the
    moment an episode survived its first minute.
    """
    X_parts, y_parts = [], []
    max_steps = int(max_hours * SECONDS_PER_HOUR / epoch_s)
    D = ep["duration_s"].to_numpy(float)
    for step in range(max_steps + 1):
        elapsed = step * epoch_s
        alive = D > elapsed
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        X_parts.append(_feature_matrix(ep.iloc[idx], elapsed, close_hour, open_hour))
        y_parts.append(D[idx] - elapsed)
    return np.vstack(X_parts), np.concatenate(y_parts)


def fit_forecaster(X: np.ndarray, y: np.ndarray, seed: int = 0, mode: str = "mean"):
    """
    Gradient-boosted regression of remaining idle time.

    WHICH FUNCTIONAL OF THE PREDICTIVE DISTRIBUTION TO TARGET IS NOT A DETAIL.
    The saving from de-energising is affine in the remaining duration, so the
    risk-neutral rule compares E[R] with the break-even point: the conditional
    MEAN is the sufficient statistic, and nothing else about the predictive
    distribution matters.

    `mode="log_median"` is the natural-looking choice and is wrong here.
    Idle durations span four orders of magnitude, so fitting log1p(R) with
    squared error is well conditioned -- but exp(.) of a log-scale fit
    estimates the conditional MEDIAN, and this distribution is heavy enough
    that at episode onset the median is 7 s while the mean is over 2,000 s.
    Duan's smearing factor corrects the retransformation with a single global
    constant and cannot repair a bias that varies this strongly with the
    covariates.  A policy driven by it waits far too long before acting.

    `mode="mean"` fits the raw scale with squared error, which targets E[R]
    directly.  It is noisier per-observation, and it over-predicts on the many
    very short episodes, but it is the estimand the decision rule actually
    needs.  Both are fitted so the decision-layer consequence of getting this
    wrong can be reported rather than asserted.
    """
    from xgboost import XGBRegressor

    kw = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=seed, n_jobs=-1, tree_method="hist")
    if mode == "mean":
        model = XGBRegressor(**kw).fit(X, y)
        return model, {"mode": "mean", "smearing": 1.0}
    if mode == "log_median":
        model = XGBRegressor(**kw)
        ylog = np.log1p(y)
        model.fit(X, ylog)
        resid = ylog - model.predict(X)
        return model, {"mode": "log_median", "smearing": float(np.mean(np.exp(resid)))}
    raise ValueError(f"unknown mode '{mode}'")


def fit_feasibility(X: np.ndarray, y: np.ndarray, min_off_s: float, seed: int = 0):
    """
    Classifier for P(remaining idle time >= min_off_s), the chance constraint's
    left-hand side.

    Kept separate from the duration regressor rather than derived from it: a
    point forecast of E[R] carries no information about the probability that R
    clears a threshold once the distribution is this skewed, which is the whole
    reason the constraint needs its own model.
    """
    from xgboost import XGBClassifier

    labels = (y >= min_off_s).astype(int)
    if labels.min() == labels.max():          # degenerate; constraint never binds
        return None
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, tree_method="hist",
        eval_metric="logloss",
    )
    model.fit(X, labels)
    return model


def make_feasibility(model, close_hour: int, open_hour: int):
    if model is None:
        return None

    def predict_feasible(ep_rows: pd.DataFrame, elapsed_s: float) -> np.ndarray:
        X = _feature_matrix(ep_rows, elapsed_s, close_hour, open_hour)
        return model.predict_proba(X)[:, 1]
    return predict_feasible


def make_predictor(model, spec: dict, close_hour: int, open_hour: int):
    """Wrap the fitted model into the callable the policy expects."""
    def predict_remaining_s(ep_rows: pd.DataFrame, elapsed_s: float) -> np.ndarray:
        X = _feature_matrix(ep_rows, elapsed_s, close_hour, open_hour)
        raw = model.predict(X)
        if spec["mode"] == "log_median":
            return np.expm1(raw) * spec["smearing"]
        return np.clip(raw, 0.0, None)
    return predict_remaining_s


# ── Policy set ────────────────────────────────────────────────────────────────

def build_policies(ep_te, params, predictors, feasible, epoch_s,
                   confidence: float = CHANCE_CONFIDENCE) -> dict[str, np.ndarray]:
    """
    Assemble the policy set for one economic scenario.

    Two ablation members sit alongside the proposed policy so that its two
    ingredients are each priced rather than assumed:

      `forecast_opt_median`     same rule, log-scale forecaster -- isolates the
                                cost of targeting the wrong functional.
      `forecast_opt_nochance`   same rule, no chance constraint -- isolates the
                                contribution of the feasibility model.
    """
    steps = int(MAX_DECISION_HOURS * SECONDS_PER_HOUR / epoch_s)
    predict = predictors["mean"]
    return {
        "never_shutdown": policy_never(ep_te, params),
        "observed": policy_observed(ep_te, params),
        "immediate": policy_immediate(ep_te, params),
        "static_break_even": policy_static_threshold(ep_te, params,
                                                     predict(ep_te, 0.0)),
        "ski_rental": policy_elapsed_time(ep_te, params),
        "forecast_opt_median": policy_forecast_optimising(
            ep_te, params, predictors["log_median"], epoch_s=epoch_s,
            max_epochs=steps, predict_feasible=feasible, confidence=confidence),
        "forecast_opt_nochance": policy_forecast_optimising(
            ep_te, params, predict, epoch_s=epoch_s, max_epochs=steps),
        "forecast_opt": policy_forecast_optimising(
            ep_te, params, predict, epoch_s=epoch_s, max_epochs=steps,
            predict_feasible=feasible, confidence=confidence),
        "oracle": optimal_offline(ep_te, params),
    }


# ── Shared preparation ────────────────────────────────────────────────────────

def prepare(
    machine: str = DEFAULT_MACHINE,
    tariff: float = TARIFF_USD_PER_KWH,
    min_off_s: float = MIN_OFF_S,
    max_restarts_per_day: int = MAX_RESTARTS_PER_DAY,
    train_fraction: float = 0.70,
    epoch_s: float = DECISION_EPOCH_S,
    seed: int = 0,
) -> dict:
    """
    Everything Task 6 and Task 7 both need: the labelled record reduced to idle
    episodes, the measured restart signature, the chronological split, and the
    fitted forecaster and feasibility models.

    Shared rather than duplicated so the sensitivity analysis is guaranteed to
    sweep the SAME fitted models and the SAME held-out episodes the headline
    comparison used.  A sweep that silently refitted would confound parameter
    sensitivity with model-fitting variance.
    """
    cfg = MACHINES[machine]

    df, meta = load_labelled(machine, drop_spikes=False, add_features=False)
    dt = float(meta["sample_interval_s"])
    covered_s = len(df) * dt
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days

    sig = measure_restart_signature(df, sample_interval_s=dt, random_state=seed)
    restart_energy = float(max(0.0, sig["restart_energy_kwh"]))
    restart_delay = float(max(0.0, sig["restart_delay_s"]))

    ep_all = extract_idle_episodes(df, sample_interval_s=dt)
    ep_all = attach_local_calendar(ep_all, cfg["factory_tz"],
                                   cfg["close_hour"], cfg["open_hour"])
    ep = ep_all[~ep_all["is_truncated"]].reset_index(drop=True)
    standby_power_w = float(df.loc[df["state"] == "STANDBY", "active_power"].mean())
    del df

    ep = ep.sort_values("start_ts").reset_index(drop=True)
    cut = int(len(ep) * train_fraction)
    ep_tr = ep.iloc[:cut].reset_index(drop=True)
    ep_te = ep.iloc[cut:].reset_index(drop=True)

    Xtr, ytr = build_epoch_table(ep_tr, epoch_s, MAX_DECISION_HOURS,
                                 cfg["close_hour"], cfg["open_hour"])
    Xte, yte = build_epoch_table(ep_te, epoch_s, MAX_DECISION_HOURS,
                                 cfg["close_hour"], cfg["open_hour"])

    predictors, specs, models = {}, {}, {}
    for mode in ("mean", "log_median"):
        mdl, spec = fit_forecaster(Xtr, ytr, seed=seed, mode=mode)
        predictors[mode] = make_predictor(mdl, spec, cfg["close_hour"], cfg["open_hour"])
        specs[mode], models[mode] = spec, mdl

    clf = fit_feasibility(Xtr, ytr, min_off_s, seed=seed)
    feasible = make_feasibility(clf, cfg["close_hour"], cfg["open_hour"])

    base = EconomicParams(
        standby_power_w=standby_power_w,
        tariff_per_kwh=tariff,
        restart_energy_kwh=restart_energy,
        restart_labour_cost=RESTART_LABOUR_COST,
        restart_delay_s=restart_delay,
        delay_cost_per_hour=0.0,
        min_off_s=min_off_s,
        max_restarts_per_day=max_restarts_per_day,
    )

    return {
        "cfg": cfg, "meta": meta, "sample_interval_s": dt,
        "covered_s": covered_s, "span_days": span_days,
        "restart_signature": sig, "standby_power_w": standby_power_w,
        "episodes_all": ep_all, "episodes": ep, "ep_train": ep_tr, "ep_test": ep_te,
        "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
        "predictors": predictors, "forecaster_specs": specs,
        "forecaster_models": models,
        "feasibility_model": clf, "feasible": feasible,
        "base_params": base, "epoch_s": epoch_s,
        "test_span_s": float((ep_te["end_ts"].max()
                              - ep_te["start_ts"].min()).total_seconds()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machine: str = DEFAULT_MACHINE,
    tariff: float = TARIFF_USD_PER_KWH,
    min_off_s: float = MIN_OFF_S,
    max_restarts_per_day: int = MAX_RESTARTS_PER_DAY,
    train_fraction: float = 0.70,
    epoch_s: float = DECISION_EPOCH_S,
    seed: int = 0,
) -> dict:
    cfg = MACHINES[machine]
    out_dir = ensure_dir(f"{PHASE3_DIR}/task6/{machine}")

    section("TASK 6 -- OPTIMISATION-BASED DECISION LAYER")

    # ── Load. Spikes KEPT; row-level engineered features not needed. ──────────
    df, meta = load_labelled(machine, drop_spikes=False, add_features=False)
    dt = float(meta["sample_interval_s"])
    covered_s = len(df) * dt
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days
    info(f"{len(df):,} rows @ {dt:.2f}s = {covered_s/86400:.1f} days of coverage "
         f"over a {span_days}-day span")

    # ── Q1: measure the restart coefficients ──────────────────────────────────
    section("Q1 -- restart signature measured from the record")
    sig = measure_restart_signature(df, sample_interval_s=dt, random_state=seed)
    info(f"cold starts n={sig['n_cold']}, warm resumes n={sig['n_warm']}")
    info(f"time to reach {sig['resumed_threshold_w']/1000:.1f} kW: "
         f"cold {sig['cold_ramp_median_s']:.0f}s vs warm {sig['warm_ramp_median_s']:.0f}s")
    info(f"restart DELAY  = {sig['restart_delay_s']:.0f} s "
         f"(95% CI [{sig['restart_delay_ci95_s'][0]:.0f}, "
         f"{sig['restart_delay_ci95_s'][1]:.0f}])")
    info(f"restart ENERGY over a common {sig['common_window_s']:.0f}s window = "
         f"{sig['restart_energy_kwh']:+.2f} kWh "
         f"(95% CI [{sig['restart_energy_ci95_kwh'][0]:+.2f}, "
         f"{sig['restart_energy_ci95_kwh'][1]:+.2f}], "
         f"n={sig['n_cold_common_window']} cold / {sig['n_warm_common_window']} warm)")
    info(f"   milestone-based estimator (descriptive only): "
         f"{sig['energy_milestone_kwh']:+.2f} kWh")

    # Both energy estimators are negative: a cold start reaches production more
    # slowly, so in any fixed window it has produced less and drawn less.  That
    # is the delay, already priced by its own term -- adding a negative energy
    # term would double-count it.  Floor at zero and let the delay carry it.
    restart_energy = float(max(0.0, sig["restart_energy_kwh"]))
    restart_delay = float(max(0.0, sig["restart_delay_s"]))
    if sig["restart_energy_kwh"] < 0:
        info("no positive electrical restart penalty is identifiable; the energy "
             "coefficient is floored at 0 kWh and the restart cost is carried "
             "entirely by the measured production delay")

    # ── Episodes ──────────────────────────────────────────────────────────────
    section("Q2 -- idle episodes and where break-even falls")
    ep_all = extract_idle_episodes(df, sample_interval_s=dt)
    ep_all = attach_local_calendar(ep_all, cfg["factory_tz"],
                                   cfg["close_hour"], cfg["open_hour"])
    n_trunc = int(ep_all["is_truncated"].sum())
    trunc_h = float(ep_all.loc[ep_all["is_truncated"], "duration_s"].sum() / SECONDS_PER_HOUR)
    ep = ep_all[~ep_all["is_truncated"]].reset_index(drop=True)
    info(f"{len(ep_all):,} idle episodes, "
         f"{ep_all['duration_s'].sum()/SECONDS_PER_HOUR:.1f} h total")
    info(f"excluded {n_trunc} truncated by an acquisition gap ({trunc_h:.1f} h)")
    info(f"{len(ep):,} usable episodes, {ep['duration_s'].sum()/SECONDS_PER_HOUR:.1f} h idle, "
         f"of which the plant left {ep['standby_s'].sum()/SECONDS_PER_HOUR:.1f} h energised")

    standby_power_w = float(df.loc[df["state"] == "STANDBY", "active_power"].mean())
    info(f"STANDBY power (data-derived cluster mean) = {standby_power_w:.0f} W")

    base = EconomicParams(
        standby_power_w=standby_power_w,
        tariff_per_kwh=tariff,
        restart_energy_kwh=restart_energy,
        restart_labour_cost=RESTART_LABOUR_COST,
        restart_delay_s=restart_delay,
        delay_cost_per_hour=0.0,
        min_off_s=min_off_s,
        max_restarts_per_day=max_restarts_per_day,
    )

    # How much idle time each candidate break-even point can reach.
    D = ep["duration_s"].to_numpy(float)
    reach = []
    for name, be in BREAK_EVEN_SCENARIOS.items():
        p = params_for_break_even(base, be)
        over = D > be
        reach.append({
            "scenario": name,
            "break_even_min": be / 60.0,
            "implied_delay_cost_per_h": p.delay_cost_per_hour,
            "restart_cost_usd": p.restart_cost,
            "n_episodes_over": int(over.sum()),
            "idle_h_over": float(D[over].sum() / SECONDS_PER_HOUR),
            "recoverable_kwh": float(
                np.maximum(D[over] - be, 0).sum() / SECONDS_PER_HOUR
                * standby_power_w / 1000.0),
        })
    reach_df = pd.DataFrame(reach)
    print()
    print(reach_df.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))
    info(f"longest usable idle episode = {D.max()/SECONDS_PER_HOUR:.2f} h -- "
         f"any break-even beyond this makes shutdown unconditionally unprofitable")

    # ── Chronological split ───────────────────────────────────────────────────
    ep = ep.sort_values("start_ts").reset_index(drop=True)
    cut = int(len(ep) * train_fraction)
    ep_tr = ep.iloc[:cut].reset_index(drop=True)
    ep_te = ep.iloc[cut:].reset_index(drop=True)
    info(f"train {len(ep_tr):,} episodes up to {ep_tr['start_ts'].max()}; "
         f"test {len(ep_te):,} from {ep_te['start_ts'].min()}")

    # ── Forecaster ────────────────────────────────────────────────────────────
    section("remaining-idle-duration forecaster")
    Xtr, ytr = build_epoch_table(ep_tr, epoch_s, MAX_DECISION_HOURS,
                                 cfg["close_hour"], cfg["open_hour"])
    Xte, yte = build_epoch_table(ep_te, epoch_s, MAX_DECISION_HOURS,
                                 cfg["close_hour"], cfg["open_hour"])
    info(f"{len(ytr):,} training epochs / {len(yte):,} test epochs")

    predictors, fc, models, specs = {}, {}, {}, {}
    const = float(np.mean(ytr))
    for mode in ("mean", "log_median"):
        mdl, spec = fit_forecaster(Xtr, ytr, seed=seed, mode=mode)
        models[mode], specs[mode] = mdl, spec
        predictors[mode] = make_predictor(mdl, spec, cfg["close_hour"], cfg["open_hour"])
        p = (np.expm1(mdl.predict(Xte)) * spec["smearing"] if mode == "log_median"
             else np.clip(mdl.predict(Xte), 0, None))
        fc[mode] = {
            "smearing_factor": spec["smearing"],
            "mae_s": float(np.mean(np.abs(p - yte))),
            "median_ae_s": float(np.median(np.abs(p - yte))),
            "rmse_s": float(np.sqrt(np.mean((p - yte) ** 2))),
            "spearman": float(pd.Series(p).corr(pd.Series(yte), method="spearman")),
            # Calibration of the estimand the decision rule consumes: does the
            # average prediction match the average realised remaining time?
            "mean_prediction_s": float(p.mean()),
            "mean_actual_s": float(yte.mean()),
            "mean_ratio": float(p.mean() / yte.mean()),
            "importance_gain": dict(zip(FEATURE_NAMES,
                                        [float(v) for v in mdl.feature_importances_])),
        }
        fc[mode]["skill_vs_constant_pct"] = (
            1 - fc[mode]["mae_s"] / float(np.mean(np.abs(const - yte)))) * 100.0
        info(f"[{mode:>10}] MAE {fc[mode]['mae_s']:.0f}s  "
             f"Spearman {fc[mode]['spearman']:.3f}  "
             f"mean predicted/actual = {fc[mode]['mean_ratio']:.2f}")

    fc["features"] = FEATURE_NAMES
    fc["n_train_epochs"] = int(len(ytr))
    fc["n_test_epochs"] = int(len(yte))
    fc["mae_constant_predictor_s"] = float(np.mean(np.abs(const - yte)))
    predict = predictors["mean"]          # the estimand the decision rule needs

    # Chance-constraint model: P(remaining >= min_off_s).
    clf = fit_feasibility(Xtr, ytr, min_off_s, seed=seed)
    feasible = make_feasibility(clf, cfg["close_hour"], cfg["open_hour"])
    if clf is not None:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        p_ok = clf.predict_proba(Xte)[:, 1]
        y_ok = (yte >= min_off_s).astype(int)
        fc["feasibility"] = {
            "min_off_s": min_off_s,
            "confidence_level": CHANCE_CONFIDENCE,
            "positive_rate_train": float((ytr >= min_off_s).mean()),
            "positive_rate_test": float(y_ok.mean()),
            "roc_auc": float(roc_auc_score(y_ok, p_ok)),
            "brier": float(brier_score_loss(y_ok, p_ok)),
        }
        info(f"feasibility model P(R>={min_off_s:.0f}s): "
             f"AUC {fc['feasibility']['roc_auc']:.3f}, "
             f"Brier {fc['feasibility']['brier']:.3f}, "
             f"base rate {fc['feasibility']['positive_rate_test']:.3f}")

    top = sorted(fc["mean"]["importance_gain"].items(), key=lambda kv: -kv[1])[:4]
    info("top features (mean model): " + ", ".join(f"{k} {v:.2f}" for k, v in top))

    # ── Q3/Q4: policies under each scenario ───────────────────────────────────
    section("Q3/Q4 -- policy comparison on held-out episodes")
    test_span_s = float((ep_te["end_ts"].max() - ep_te["start_ts"].min()).total_seconds())
    scenario_tables: dict[str, pd.DataFrame] = {}
    scenario_params: dict[str, dict] = {}
    scenario_policies: dict[str, dict] = {}

    for name, be in BREAK_EVEN_SCENARIOS.items():
        params = params_for_break_even(base, be)
        policies = build_policies(ep_te, params, predictors, feasible, epoch_s)
        tab = compare_policies(ep_te, policies, params).reindex(POLICY_ORDER)
        tab["annual_savings_usd"] = [annualise(v, test_span_s)
                                     for v in tab["savings_vs_baseline"]]
        tab["annual_energy_saved_kwh"] = [annualise(v, test_span_s)
                                          for v in tab["energy_saved_vs_baseline_kwh"]]
        tab.insert(0, "scenario", name)
        scenario_tables[name] = tab
        scenario_policies[name] = policies
        scenario_params[name] = params.to_dict()

        print(f"\n--- {name}: break-even {be/60:.0f} min, "
              f"implied delay valuation {params.delay_cost_per_hour:,.0f} USD/h, "
              f"restart cost {params.restart_cost:.3f} USD ---")
        show = ["n_shutdowns", "standby_h", "off_h", "cost_total",
                "savings_vs_baseline", "pct_of_oracle",
                "min_off_violations", "loss_making_shutdowns", "annual_savings_usd"]
        print(tab[show].to_string(float_format=lambda v: f"{v:9.3f}"))

    combined = pd.concat(scenario_tables.values())

    # ── Read-outs ─────────────────────────────────────────────────────────────
    section("read-outs")
    readouts = {}
    for name, tab in scenario_tables.items():
        head_room = tab.loc["observed", "cost_total"] - tab.loc["oracle", "cost_total"]
        prop = tab.loc["forecast_opt", "savings_vs_baseline"]
        ski = tab.loc["ski_rental", "savings_vs_baseline"]
        stat = tab.loc["static_break_even", "savings_vs_baseline"]
        never = tab.loc["never_shutdown", "savings_vs_baseline"]
        med = tab.loc["forecast_opt_median", "savings_vs_baseline"]
        readouts[name] = {
            "forecast_value_vs_median_estimand_usd": float(prop - med),
            "head_room_usd": float(head_room),
            "head_room_usd_per_year": float(annualise(head_room, test_span_s)),
            "forecast_opt_pct_of_oracle": float(tab.loc["forecast_opt", "pct_of_oracle"]),
            "ski_rental_pct_of_oracle": float(tab.loc["ski_rental", "pct_of_oracle"]),
            "static_pct_of_oracle": float(tab.loc["static_break_even", "pct_of_oracle"]),
            "never_pct_of_oracle": float(tab.loc["never_shutdown", "pct_of_oracle"]),
            "forecast_value_vs_ski_usd": float(prop - ski),
            "forecast_value_vs_static_usd": float(prop - stat),
            "forecast_value_vs_ski_usd_per_year": float(annualise(prop - ski, test_span_s)),
            "beats_trivial_policies": bool(prop > never and prop > 0),
        }
        r = readouts[name]
        info(f"{name}: head-room {head_room:7.2f} USD "
             f"({r['head_room_usd_per_year']:8.0f} USD/yr) | "
             f"proposed {r['forecast_opt_pct_of_oracle']:6.1f}% of oracle, "
             f"ski-rental {r['ski_rental_pct_of_oracle']:6.1f}%, "
             f"never {r['never_pct_of_oracle']:6.1f}% | "
             f"forecast worth {r['forecast_value_vs_ski_usd']:+.2f} USD vs ski-rental")

    # ── Persist ───────────────────────────────────────────────────────────────
    combined.to_csv(f"{out_dir}/policy_comparison.csv")
    info(f"wrote {out_dir}/policy_comparison.csv")
    reach_df.to_csv(f"{out_dir}/break_even_reach.csv", index=False)
    ep.to_csv(f"{out_dir}/idle_episodes.csv", index=False)
    info(f"wrote {out_dir}/idle_episodes.csv")

    # Forecaster diagnostics, so the figures need no refit.
    np.savez_compressed(
        f"{out_dir}/forecast_test.npz",
        actual_remaining_s=yte,
        pred_mean_s=np.clip(models["mean"].predict(Xte), 0, None),
        pred_median_s=np.expm1(models["log_median"].predict(Xte))
        * specs["log_median"]["smearing"],
        p_feasible=(clf.predict_proba(Xte)[:, 1] if clf is not None
                    else np.full(len(yte), np.nan)),
        test_episode_duration_s=ep_te["duration_s"].to_numpy(float),
    )
    np.savez_compressed(
        f"{out_dir}/policy_tau.npz",
        duration_s=ep_te["duration_s"].to_numpy(float),
        **{f"{s}__{k}": v for s, pol in scenario_policies.items()
           for k, v in pol.items()})

    head = scenario_tables[HEADLINE_SCENARIO]
    results = {
        "machine": machine,
        "coverage_days": covered_s / 86400.0,
        "span_days": span_days,
        "sample_interval_s": dt,
        "restart_signature": sig,
        "base_params": base.to_dict(),
        "scenarios": {"break_even_s": BREAK_EVEN_SCENARIOS,
                      "params": scenario_params,
                      "headline": HEADLINE_SCENARIO},
        "break_even_reach": reach_df.to_dict(orient="records"),
        "episodes": {
            "n_total": int(len(ep_all)),
            "n_truncated_excluded": n_trunc,
            "truncated_hours_excluded": trunc_h,
            "n_usable": int(len(ep)),
            "idle_hours": float(ep["duration_s"].sum() / SECONDS_PER_HOUR),
            "standby_hours_observed": float(ep["standby_s"].sum() / SECONDS_PER_HOUR),
            "off_hours_observed": float(ep["off_s"].sum() / SECONDS_PER_HOUR),
            "max_duration_h": float(D.max() / SECONDS_PER_HOUR),
            "median_duration_s": float(np.median(D)),
            "n_train": int(len(ep_tr)),
            "n_test": int(len(ep_te)),
            "test_span_days": test_span_s / 86400.0,
            "test_idle_hours": float(ep_te["duration_s"].sum() / SECONDS_PER_HOUR),
        },
        "forecaster": fc,
        "policy_tables": {k: v.to_dict(orient="index") for k, v in scenario_tables.items()},
        "readouts": readouts,
        "headline_table": head.to_dict(orient="index"),
    }
    save_json(results, f"{out_dir}/task6_results.json")
    section("TASK 6 COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task 6 -- optimisation decision layer")
    ap.add_argument("--machine", default=DEFAULT_MACHINE, choices=list(MACHINES))
    ap.add_argument("--tariff", type=float, default=TARIFF_USD_PER_KWH)
    ap.add_argument("--min-off", type=float, default=MIN_OFF_S)
    ap.add_argument("--max-restarts", type=int, default=MAX_RESTARTS_PER_DAY)
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--epoch", type=float, default=DECISION_EPOCH_S)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(machine=a.machine, tariff=a.tariff, min_off_s=a.min_off,
         max_restarts_per_day=a.max_restarts, train_fraction=a.train_fraction,
         epoch_s=a.epoch, seed=a.seed)
