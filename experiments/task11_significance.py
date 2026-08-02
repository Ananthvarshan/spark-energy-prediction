"""
============================================================
TASK 11 -- STATISTICAL SIGNIFICANCE
experiments/task11_significance.py
============================================================

WHAT THIS ANSWERS
-----------------
Phases I-IV report point estimates.  This script asks, for each of
the paper's claims, whether the difference behind it survives an
interval:

  A  FORECASTING.  Every pair of the seven models (six trained plus
     the untrained persistence reference) compared window by window
     on the same 5,449 held-out windows.  Paired Wilcoxon, effect
     size, Hodges-Lehmann shift, Holm and Benjamini-Hochberg
     adjustment, Friedman over seeds with Kendall's W, and a Nemenyi
     critical-difference diagram.

     The headline test is `seq2seq_lstm` against `persistence`.  It
     is stated as a pre-registered comparison rather than picked
     after looking, because Phase IV found the gap to be 0.010 F1
     and the point of Phase V is to say whether that is real.

  B  DECISION LAYER.  Day-block bootstrap of the policy comparison:
     factory days are resampled with replacement, not episodes,
     because the objective carries a per-day restart cap and
     resampling episodes independently would break it.  Reports the
     interval for every policy and for the two differences that
     carry the paper's claim (proposed - status quo, proposed -
     ski-rental), with the economic magnitude beside the p-value.

  C  STATE LAYER.  Moving-block bootstrap (one-week blocks) of
     STANDBY hours per machine, plus the labelling-seed spread from
     Task 12B, so the state-time totals quoted throughout the paper
     carry an interval.

  D  CROSS-MACHINE.  The Phase IV transfer results tested across
     machines, where n is 8 (or 4 for the motor-driven subset) and
     the tests have almost no power -- which is measured by
     simulation rather than conceded in prose.

EVERY TEST IS REPORTED WITH AN EFFECT SIZE AND, WHERE THE QUANTITY
IS ECONOMIC, WITH ITS VALUE IN USD/YR.  A p-value on thousands of
paired windows is a statement about detectability, not importance;
`economically_significant` is a separate column and is allowed to be
False when `p_holm` is below 0.05.

    python -m experiments.task11_significance
    python -m experiments.task11_significance --only forecasting
============================================================
"""

from __future__ import annotations

import os
import sys
import json
import itertools
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                    # noqa: E402
    DEFAULT_MACHINE, MACHINES, IMDELD_MACHINES, PHASE4_DIR,
    load_labelled, ensure_dir, save_json, info, section,
)
from src.stats import (                             # noqa: E402
    wilcoxon_paired, cohens_d_paired, friedman_test, nemenyi_matrix,
    holm, benjamini_hochberg, block_mean_bootstrap,
    describe_runs, friedman_power, wilcoxon_power, wilcoxon_p_floor, stars,
)

PHASE5_DIR = "outputs/phase5"
TASK11_DIR = f"{PHASE5_DIR}/task11"
TASK12_DIR = f"{PHASE5_DIR}/task12"

ALPHA = 0.05
N_BOOT = 2000

# The comparison the paper's forecasting claim rests on, named before the
# tests run so that it cannot be chosen afterwards from whatever came out
# significant.
HEADLINE_PAIR = ("seq2seq_lstm", "persistence")

# Smallest difference in STANDBY-seconds MAE that could change a decision.
# The decision layer consumes a duration; at the measured standby power of
# ~4.2 kW and 0.12 USD/kWh, one second of standby costs 1.4e-4 USD, so a
# per-window MAE difference below a second is worth ~0.5 USD/yr at this
# machine's window count.  Stated here so "economically significant" has a
# fixed meaning rather than being decided per result.
MAE_PRACTICAL_THRESHOLD_S = 1.0
F1_PRACTICAL_THRESHOLD = 0.02
USD_PRACTICAL_THRESHOLD = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# A -- forecasting models
# ══════════════════════════════════════════════════════════════════════════════

def _load_predictions(machine: str) -> dict:
    """Per-window predictions for every model and seed from Task 12A."""
    d = f"{TASK12_DIR}/{machine}/forecasting"
    path = f"{d}/forecasting_seeds.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run `python -m experiments.task12_multiple_runs` first")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)

    preds = {}
    for key in meta["results"]:
        f = f"{d}/preds_{key}.npz"
        if os.path.exists(f):
            preds[key] = np.load(f)
    return {"meta": meta, "preds": preds}


def _per_window_scores(z, standby_id: int, step_s: float):
    """
    Two per-window quantities for every seed of one model:

      `abs_err`   |predicted STANDBY seconds - true STANDBY seconds| per window,
                  the duration error the decision layer would consume;
      `correct`   fraction of horizon steps whose state is predicted correctly,
                  a per-window accuracy that can be paired across models.

    Both are per WINDOW, which is what makes the comparison paired: model A's
    error on window i and model B's error on window i are the same forecasting
    problem attempted twice.
    """
    y_pred = z["y_pred"]                       # (seeds, windows, horizon)
    y_true = z["y_true"]                       # (windows, horizon)
    true_s = z["true_standby_s"]
    pred_s = (y_pred == standby_id).sum(axis=2) * step_s
    abs_err = np.abs(pred_s - true_s[None, :])
    correct = (y_pred == y_true[None, :, :]).mean(axis=2)
    return abs_err, correct


def _f1_standby(y_true: np.ndarray, y_pred: np.ndarray, standby_id: int) -> float:
    tp = np.sum((y_pred == standby_id) & (y_true == standby_id))
    fp = np.sum((y_pred == standby_id) & (y_true != standby_id))
    fn = np.sum((y_pred != standby_id) & (y_true == standby_id))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def forecasting_tests(machine: str = DEFAULT_MACHINE, alpha: float = ALPHA) -> dict:
    """Pairwise paired tests, Friedman over seeds, and the Nemenyi diagram."""
    section("TASK 11A -- forecasting model comparison")
    out_dir = ensure_dir(f"{TASK11_DIR}/{machine}")
    bundle = _load_predictions(machine)
    meta, preds = bundle["meta"], bundle["preds"]
    standby_id = meta["config"]["state_encoder"]["STANDBY"]
    step_s = meta["config"]["step_seconds"]
    keys = [k for k in meta["results"] if k in preds]

    # Per-window scores, averaged over seeds: the paired unit is the window,
    # and averaging over seeds first removes the training noise that is not
    # the object of this comparison (it is reported separately in 12A).
    err, acc, f1, y_true = {}, {}, {}, None
    for k in keys:
        z = preds[k]
        y_true = z["y_true"]
        e, c = _per_window_scores(z, standby_id, step_s)
        err[k] = e.mean(axis=0)
        acc[k] = c.mean(axis=0)
        f1[k] = [_f1_standby(y_true, z["y_pred"][s], standby_id)
                 for s in range(z["y_pred"].shape[0])]
        info(f"{k:<14} MAE {err[k].mean():7.3f}s   "
             f"F1 {np.mean(f1[k]):.4f} over {len(f1[k])} seed(s)")

    # -- pairwise paired Wilcoxon on per-window absolute duration error -------
    rows = []
    for a, b in itertools.combinations(keys, 2):
        w = wilcoxon_paired(err[a], err[b])
        rows.append({
            "model_a": a, "model_b": b,
            "mae_a": float(err[a].mean()), "mae_b": float(err[b].mean()),
            "mae_diff": float(err[a].mean() - err[b].mean()),
            "median_diff": w["median_diff"], "hl_shift": w["hl_shift"],
            "rank_biserial": w["rank_biserial"],
            "cohens_d": cohens_d_paired(err[a], err[b]),
            "p_raw": w["p_value"], "n_windows": w["n"],
        })
    tab = pd.DataFrame(rows)
    tab["p_holm"] = holm(tab["p_raw"].to_numpy())
    tab["p_bh"] = benjamini_hochberg(tab["p_raw"].to_numpy())
    tab["significant_holm"] = tab["p_holm"] < alpha
    # Statistical significance and practical relevance are separate columns.
    tab["economically_significant"] = tab["mae_diff"].abs() >= MAE_PRACTICAL_THRESHOLD_S
    tab["verdict"] = np.where(
        tab["significant_holm"] & tab["economically_significant"], "real and material",
        np.where(tab["significant_holm"], "detectable but immaterial",
                 "not detectable"))
    tab = tab.sort_values("p_raw").reset_index(drop=True)
    tab.to_csv(f"{out_dir}/forecasting_pairwise.csv", index=False)

    section("pairwise comparisons on per-window duration error (MAE, seconds)")
    show = ["model_a", "model_b", "mae_a", "mae_b", "mae_diff", "hl_shift",
            "rank_biserial", "p_raw", "p_holm", "verdict"]
    print(tab[show].round(4).to_string(index=False))

    # -- the pre-registered headline test -------------------------------------
    a, b = HEADLINE_PAIR
    head = {}
    if a in err and b in err:
        w_mae = wilcoxon_paired(err[a], err[b])
        w_acc = wilcoxon_paired(acc[a], acc[b])
        # F1 is a set-level statistic, so its interval comes from a bootstrap
        # over windows rather than from a paired test over per-window values.
        # The interval is computed on ONE run of each model (seed 0), because a
        # set-level statistic cannot be averaged over seeds the way a
        # per-window error can -- averaging categorical predictions is
        # undefined.  The seed-to-seed spread of F1 is reported separately, so
        # the two sources of variability stay distinguishable instead of being
        # blended into one interval that means neither.
        rng = np.random.default_rng(0)
        n = len(y_true)
        draws = np.empty(N_BOOT)
        pa, pb = preds[a]["y_pred"][0], preds[b]["y_pred"][0]
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            draws[i] = (_f1_standby(y_true[idx], pa[idx], standby_id)
                        - _f1_standby(y_true[idx], pb[idx], standby_id))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        head = {
            "pair": f"{a} vs {b}",
            "mae_a": float(err[a].mean()), "mae_b": float(err[b].mean()),
            "mae_diff_s": float(err[a].mean() - err[b].mean()),
            "mae_p_value": w_mae["p_value"],
            "mae_rank_biserial": w_mae["rank_biserial"],
            "mae_hl_shift_s": w_mae["hl_shift"],
            "step_accuracy_p_value": w_acc["p_value"],
            "step_accuracy_diff": float(acc[a].mean() - acc[b].mean()),
            "f1_a": float(np.mean(f1[a])), "f1_b": float(np.mean(f1[b])),
            "f1_diff": float(np.mean(f1[a]) - np.mean(f1[b])),
            "f1_a_seed0": float(f1[a][0]), "f1_b_seed0": float(f1[b][0]),
            "f1_diff_seed0": float(f1[a][0] - f1[b][0]),
            "f1_a_seed_sd": float(np.std(f1[a], ddof=1)) if len(f1[a]) > 1 else 0.0,
            "f1_b_seed_sd": float(np.std(f1[b], ddof=1)) if len(f1[b]) > 1 else 0.0,
            "f1_diff_ci95": [float(lo), float(hi)],
            "f1_ci_basis": "seed 0 of each model, bootstrap over test windows",
            "f1_diff_ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_windows": int(n),
        }
        section(f"pre-registered headline test: {a} vs {b}")
        info(f"STANDBY-seconds MAE {head['mae_a']:.2f}s vs {head['mae_b']:.2f}s "
             f"(difference {head['mae_diff_s']:+.2f}s), Wilcoxon p="
             f"{head['mae_p_value']:.3g} {stars(head['mae_p_value'])}, "
             f"rank-biserial {head['mae_rank_biserial']:+.3f}")
        info(f"STANDBY F1 {head['f1_a']:.4f} vs {head['f1_b']:.4f} "
             f"(difference {head['f1_diff']:+.4f}, 95% CI "
             f"[{lo:+.4f}, {hi:+.4f}] -> "
             f"{'excludes' if head['f1_diff_ci_excludes_zero'] else 'includes'} zero)")

    # -- Friedman over seeds, for the models that have several ----------------
    fried, nem = {}, {}
    multi = [k for k in keys if len(f1[k]) >= 3]
    if len(multi) >= 3:
        n_common = min(len(f1[k]) for k in multi)
        S = np.column_stack([f1[k][:n_common] for k in multi])
        fried = friedman_test(S, lower_is_better=False)
        fried["models"] = multi
        diff, sig, cd = nemenyi_matrix(fried["mean_ranks"], len(multi), n_common,
                                       alpha)
        nem = {"models": multi, "mean_ranks": fried["mean_ranks"],
               "critical_difference": cd, "rank_diff": diff.tolist(),
               "significant": sig.tolist(), "n_blocks": int(n_common)}
        section("Friedman over seeds (blocks = seeds, treatments = models)")
        info(f"models {multi}, {n_common} seeds each")
        info(f"chi2 = {fried['statistic']:.3f}, p = {fried['p_value']:.4g}, "
             f"Kendall's W = {fried['kendalls_w']:.3f}, CD = {cd:.3f}")
    else:
        info("fewer than three models have >=3 seeds; Friedman over seeds skipped")

    # -- how much power the seed-blocked design actually has ------------------
    power = {}
    if multi:
        n_common = min(len(f1[k]) for k in multi)
        power = {
            "n_blocks": int(n_common), "k": len(multi),
            "power_at_1sd": friedman_power(n_common, len(multi), 1.0, n_sim=800),
            "power_at_2sd": friedman_power(n_common, len(multi), 2.0, n_sim=800),
        }
        info(f"Friedman power at this design: {power['power_at_1sd']:.2f} for a "
             f"1-sd effect, {power['power_at_2sd']:.2f} for 2 sd")

    res = {"machine": machine, "models": keys,
           "seed_counts": {k: len(f1[k]) for k in keys},
           "mae_per_model": {k: float(err[k].mean()) for k in keys},
           "f1_per_model": {k: describe_runs(f1[k]) for k in keys},
           "headline": head, "friedman_over_seeds": fried, "nemenyi": nem,
           "power": power,
           "thresholds": {"mae_s": MAE_PRACTICAL_THRESHOLD_S,
                          "f1": F1_PRACTICAL_THRESHOLD}}
    save_json(res, f"{out_dir}/forecasting_tests.json")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# B -- decision layer
# ══════════════════════════════════════════════════════════════════════════════

def decision_bootstrap(machine: str = DEFAULT_MACHINE, source: str = "phase4",
                       scenario: str = "S2_moderate", seed: int = 0,
                       n_boot: int = N_BOOT, alpha: float = ALPHA,
                       prep: dict | None = None) -> dict:
    """
    Day-block bootstrap of the policy comparison.

    THE RESAMPLING UNIT IS THE FACTORY DAY, not the episode.  Two reasons, and
    both would bias the interval if ignored: the daily restart cap couples the
    episodes within a day, so resampling episodes independently evaluates a
    constraint that no longer corresponds to any real day; and idle episodes on
    the same day are not independent draws -- they are the same shift, the same
    production plan and the same operator.

    Every policy's tau is computed ONCE on the real episodes and then re-scored
    on each resample, so the bootstrap measures sampling variability of the
    RESULT, not of the policy's fitting.  Fitting variability is Task 12C.
    """
    from experiments.task6_optimization import (
        prepare, build_policies, BREAK_EVEN_SCENARIOS, DECISION_EPOCH_S,
    )
    from src.decision import (
        evaluate_policy, params_for_break_even, annualise, SECONDS_PER_HOUR,
    )

    section(f"TASK 11B -- decision-layer bootstrap on {machine} [{scenario}]")
    out_dir = ensure_dir(f"{TASK11_DIR}/{machine}")

    # `prep` is accepted so the three scenarios share one preparation: it
    # reloads a 5.5-million-row record and refits the forecaster, and the
    # scenarios differ only in the economic coefficients, not in the data.
    prep = prep if prep is not None else prepare(machine, seed=seed,
                                                 source=source)
    be = BREAK_EVEN_SCENARIOS[scenario]
    params = params_for_break_even(prep["base_params"], be)
    ep = prep["ep_test"].reset_index(drop=True)
    pol = build_policies(ep, params, prep["predictors"], prep["feasible"],
                         DECISION_EPOCH_S)

    days = ep["local_day"].to_numpy()
    span_s = prep["test_span_s"]
    n_days = len(np.unique(days))
    info(f"{len(ep):,} held-out episodes over {n_days} factory days "
         f"({span_s/86400:.1f} days of span)")

    # Annualisation factor is FIXED at the real test span: every resample
    # contains the same number of days as the original, so the amount of
    # calendar time represented does not change and rescaling per draw would
    # only inject noise from the day-length variation.
    scale = (365.25 * 86400.0) / span_s

    uniq_days = np.unique(days)
    day_index = {d: np.flatnonzero(days == d) for d in uniq_days}
    policy_names = list(pol)
    rng = np.random.default_rng(seed)

    def _costs_on(sub: pd.DataFrame, taus: dict) -> dict:
        return {k: evaluate_policy(sub, taus[k], params)["cost_total"]
                for k in policy_names}

    point_costs = _costs_on(ep, pol)

    # EVERY POLICY IS SCORED ON THE SAME RESAMPLE.  Drawing separate resamples
    # per policy would break the pairing and inflate the interval of every
    # contrast, which is exactly the quantity the paper's claim rests on.
    draws = np.empty((n_boot, len(policy_names)))
    for b in range(n_boot):
        picked = rng.choice(uniq_days, size=len(uniq_days), replace=True)
        idx = np.concatenate([day_index[d] for d in picked])
        # A day drawn twice must count as two SEPARATE days: the restart cap is
        # per calendar day, and leaving the label unchanged would let the two
        # copies share one day's budget and silently tighten the constraint.
        keys = np.concatenate([np.full(len(day_index[d]), f"{d}#{j}")
                               for j, d in enumerate(picked)])
        sub = ep.iloc[idx].reset_index(drop=True)
        sub["local_day"] = keys
        taus = {k: pol[k][idx] for k in policy_names}
        c = _costs_on(sub, taus)
        draws[b] = [c[k] for k in policy_names]

    base_j = policy_names.index("observed")
    rows = []
    for j, name in enumerate(policy_names):
        d = (draws[:, base_j] - draws[:, j]) * scale
        point = (point_costs["observed"] - point_costs[name]) * scale
        lo, hi = np.percentile(d, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        p_two = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows.append({
            "policy": name, "savings_usd_yr": float(point),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(d.std(ddof=1)), "p_vs_zero": p_two,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_days": int(len(uniq_days)),
        })
        info(f"{name:<22} {point:+9.2f} USD/yr  "
             f"95% CI [{lo:+8.2f}, {hi:+8.2f}]  "
             f"{'excludes' if rows[-1]['ci_excludes_zero'] else 'includes'} zero")

    tab = pd.DataFrame(rows)
    tab["p_holm"] = holm(tab["p_vs_zero"].to_numpy())
    tab["economically_significant"] = tab["savings_usd_yr"].abs() >= USD_PRACTICAL_THRESHOLD
    tab.to_csv(f"{out_dir}/decision_bootstrap_{scenario}.csv", index=False)

    # -- the differences the paper's claim rests on ---------------------------
    contrasts = {}
    prop_j = policy_names.index("forecast_opt")
    for name, other in (("proposed_vs_status_quo", "observed"),
                        ("proposed_vs_ski_rental", "ski_rental"),
                        ("proposed_vs_static_heuristic", "static_break_even"),
                        ("proposed_vs_oracle", "oracle")):
        if other not in policy_names:
            continue
        j = policy_names.index(other)
        d = (draws[:, j] - draws[:, prop_j]) * scale
        point = (point_costs[other] - point_costs["forecast_opt"]) * scale
        lo, hi = np.percentile(d, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        contrasts[name] = {
            "difference_usd_yr": float(point),
            "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "economically_significant": bool(abs(point) >= USD_PRACTICAL_THRESHOLD),
            "n_days": int(len(uniq_days)),
        }
        c = contrasts[name]
        info(f"{name}: {c['difference_usd_yr']:+.2f} USD/yr, 95% CI "
             f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}], p={c['p_two_sided']:.3f}")

    np.savez_compressed(f"{out_dir}/decision_bootstrap_{scenario}_draws.npz",
                        draws=draws, policies=np.array(policy_names),
                        scale=scale)

    res = {"machine": machine, "scenario": scenario, "source": source,
           "n_episodes": int(len(ep)), "n_days": int(n_days),
           "test_span_days": span_s / 86400.0,
           "break_even_s": be, "policies": tab.to_dict(orient="records"),
           "contrasts": contrasts, "n_boot": n_boot}
    save_json(res, f"{out_dir}/decision_bootstrap_{scenario}.json")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# C -- state layer
# ══════════════════════════════════════════════════════════════════════════════

BLOCK_DAYS = 7.0


def state_bootstrap(machines: list[str] | None = None, source: str = "phase4",
                    block_days: float = BLOCK_DAYS, n_boot: int = 500,
                    alpha: float = ALPHA) -> dict:
    """
    Moving-block bootstrap of the STANDBY share, one machine at a time.

    The block is one week.  The plant's duty cycle is organised by shift and by
    weekday, so a block shorter than a week resamples fragments of a pattern
    that only closes over seven days, and the interval comes out too narrow; a
    block much longer leaves single-digit blocks to resample.  The statistic is
    the STANDBY FRACTION of readings, converted to hours per covered day, so
    machines with different record lengths are comparable.
    """
    section("TASK 11C -- state-layer block bootstrap")
    out_dir = ensure_dir(TASK11_DIR)
    machines = machines or IMDELD_MACHINES
    rows = []
    for m in machines:
        df, meta = load_labelled(m, drop_spikes=False, add_features=False,
                                 source=source)
        dt = float(meta["sample_interval_s"])
        is_sby = (df["state"].to_numpy() == "STANDBY").astype(np.int8)
        del df
        block_len = int(round(block_days * 86400.0 / dt))
        boot = block_mean_bootstrap(is_sby, block_len=block_len,
                                    n_boot=n_boot, random_state=0, alpha=alpha)
        h_per_day = 24.0
        rows.append({
            "machine": m,
            "standby_pct": boot["point"] * 100,
            "standby_h_per_day": boot["point"] * h_per_day,
            "ci_lo_h_per_day": boot["ci_lo"] * h_per_day,
            "ci_hi_h_per_day": boot["ci_hi"] * h_per_day,
            "se_h_per_day": boot["se"] * h_per_day,
            "block_days": block_days,
            "n_blocks": boot["n_blocks"],
            "n_rows": int(len(is_sby)),
        })
        info(f"{m:<16} STANDBY {rows[-1]['standby_h_per_day']:5.2f} h/day  "
             f"95% CI [{rows[-1]['ci_lo_h_per_day']:5.2f}, "
             f"{rows[-1]['ci_hi_h_per_day']:5.2f}]  "
             f"({boot['n_blocks']} one-week blocks)")
        del is_sby

    tab = pd.DataFrame(rows).set_index("machine")
    tab.to_csv(f"{out_dir}/state_block_bootstrap.csv")
    return {"block_days": block_days, "rows": rows}


# ══════════════════════════════════════════════════════════════════════════════
# D -- cross-machine tests
# ══════════════════════════════════════════════════════════════════════════════

MOTOR_MACHINES = ["pelletizer-I", "pelletizer-II", "milling-I", "milling-II"]


def cross_machine_tests(alpha: float = ALPHA) -> dict:
    """
    Test the Phase IV cross-machine claims, and measure how little power the
    tests have.

    Three families:

      1  own-machine model vs persistence, paired over the 8 machines
         (Task 9B).  n = 8, so the smallest attainable two-sided p-value is
         0.0078 and no effect, however large, can be called significant below
         that floor.
      2  the same restricted to the 4 motor-driven machines, where the state
         model is valid.  n = 4 gives a p-value floor of 0.125: the test CANNOT
         reject at 0.05 whatever the data show, which is the sharpest possible
         statement of the limitation.
      3  transferred vs local sequence models, paired over machines.
    """
    section("TASK 11D -- cross-machine tests")
    out_dir = ensure_dir(TASK11_DIR)
    path = f"{PHASE4_DIR}/task9/part_b/transfer_matrix.csv"
    if not os.path.exists(path):
        info("Phase IV Task 9B matrix missing; cross-machine tests skipped")
        return {}

    t = pd.read_csv(path)
    pers = t[t["source"] == "persistence"].set_index("target")
    own = t[(t["source"] != "persistence") & (t["same_machine"])
            & (t["variant"] == "target_scaler")].set_index("target")
    trans = (t[(t["source"] != "persistence") & (~t["same_machine"])
               & (t["variant"] == "target_scaler")]
             .groupby("target")["f1_STANDBY"].median())

    families, results = {}, {}
    for name, machines in (("all_machines", list(own.index)),
                           ("motor_machines", MOTOR_MACHINES)):
        ms = [m for m in machines if m in own.index and m in pers.index]
        a = own.loc[ms, "f1_STANDBY"].to_numpy()
        b = pers.loc[ms, "f1_STANDBY"].to_numpy()
        w = wilcoxon_paired(a, b)
        n = len(ms)
        entry = {
            "machines": ms, "n": n,
            "mean_own": float(a.mean()), "mean_persistence": float(b.mean()),
            "mean_diff": float((a - b).mean()),
            "median_diff": w["median_diff"],
            "p_value": w["p_value"],
            "p_value_floor": wilcoxon_p_floor(n),
            "can_reject_at_alpha": bool(wilcoxon_p_floor(n) <= alpha),
            "rank_biserial": w["rank_biserial"],
            "cohens_d": cohens_d_paired(a, b),
            "power_at_observed_effect": wilcoxon_power(
                n, abs(cohens_d_paired(a, b)), n_sim=2000),
            "power_at_1sd": wilcoxon_power(n, 1.0, n_sim=2000),
        }
        results[f"own_vs_persistence_{name}"] = entry
        info(f"own vs persistence [{name}, n={n}]: mean diff "
             f"{entry['mean_diff']:+.3f} F1, p={entry['p_value']:.4f}, "
             f"floor={entry['p_value_floor']:.4f}, "
             f"power at the observed effect {entry['power_at_observed_effect']:.2f}")
        if not entry["can_reject_at_alpha"]:
            info(f"  NOTE: with n={n} the signed-rank test cannot produce "
                 f"p<{alpha} under any data; a non-significant verdict here is "
                 f"a statement about the design, not about the models")

    ms = [m for m in own.index if m in trans.index]
    a = own.loc[ms, "f1_STANDBY"].to_numpy()
    b = trans.loc[ms].to_numpy()
    w = wilcoxon_paired(a, b)
    results["local_vs_transferred"] = {
        "machines": ms, "n": len(ms),
        "mean_local": float(a.mean()), "mean_transferred": float(b.mean()),
        "mean_diff": float((a - b).mean()),
        "p_value": w["p_value"], "p_value_floor": wilcoxon_p_floor(len(ms)),
        "rank_biserial": w["rank_biserial"],
        "cohens_d": cohens_d_paired(a, b),
    }
    r = results["local_vs_transferred"]
    info(f"local vs transferred [n={r['n']}]: {r['mean_local']:.3f} vs "
         f"{r['mean_transferred']:.3f} F1 (diff {r['mean_diff']:+.3f}), "
         f"p={r['p_value']:.4f}, d={r['cohens_d']:+.2f}")

    # Multiple-comparison control over the whole cross-machine family.
    names = list(results)
    p = np.array([results[k]["p_value"] for k in names])
    for k, ph, pb in zip(names, holm(p), benjamini_hochberg(p)):
        results[k]["p_holm"] = float(ph)
        results[k]["p_bh"] = float(pb)

    # What effect WOULD have been detectable?
    detect = {}
    for n in (4, 8):
        detect[n] = {f"{e}sd": wilcoxon_power(n, e, n_sim=2000)
                     for e in (0.5, 1.0, 1.5, 2.0, 3.0)}
    results["detectable_effects"] = detect
    families["cross_machine"] = results

    save_json(results, f"{out_dir}/cross_machine_tests.json")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = DEFAULT_MACHINE,
         steps: tuple[str, ...] = ("forecasting", "decision", "state", "cross"),
         source: str = "phase4", n_boot: int = N_BOOT) -> dict:
    ensure_dir(TASK11_DIR)
    section(f"PHASE V / TASK 11 -- significance testing ({machine})")
    out = {}
    if "forecasting" in steps:
        try:
            out["forecasting"] = forecasting_tests(machine)
        except FileNotFoundError as exc:
            info(f"forecasting tests skipped: {exc}")
    if "decision" in steps:
        from experiments.task6_optimization import prepare
        prep = prepare(machine, seed=0, source=source)
        for sc in ("S1_low", "S2_moderate", "S3_high"):
            out[f"decision_{sc}"] = decision_bootstrap(
                machine, source=source, scenario=sc, n_boot=n_boot, prep=prep)
    if "state" in steps:
        out["state"] = state_bootstrap(source=source)
    if "cross" in steps:
        out["cross_machine"] = cross_machine_tests()
    section("TASK 11 COMPLETE")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase V Task 11 -- significance")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--only", nargs="*", default=None,
                    help="steps: forecasting decision state cross")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--source", default="phase4",
                    choices=["auto", "reference", "phase4"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    all_steps = ("forecasting", "decision", "state", "cross")
    steps = tuple(s for s in (a.only or all_steps) if s not in a.skip)
    main(a.machine, steps, a.source, a.n_boot)
