"""
============================================================
TASK 7 -- BREAK-EVEN AND ECONOMIC SENSITIVITY
experiments/task7_sensitivity.py
============================================================

WHAT THIS ANSWERS
-----------------
Task 6 reports the decision layer at three break-even points.
Task 7 sweeps the whole economic plane, and answers the questions a
plant engineer and a reviewer each ask first:

  A  How much saving is physically available as a function of the
     break-even point?  This is a property of the machine's idle
     duration distribution alone -- no algorithm can exceed it.

  B  Over what region of (restart energy, delay valuation, tariff)
     does the proposed policy actually beat both trivial policies
     and the forecast-free ski-rental rule?

  C  How sensitive is the result to the three constraint constants
     that were stated rather than measured -- minimum off time,
     restarts per day, and the chance-constraint confidence?

  D  What retrofit cost does this pay back, and over what horizon?

A CORRECTION TO THE PLANNED ANALYSIS
------------------------------------
The improvement plan specifies a heat-map of break-even time over
electricity tariff x restart energy.  That map is DEGENERATE when
the restart cost is purely electrical.  Break-even is

    D* = restart_cost / (tariff * P_standby)
       = (tariff*E_r + labour + c_delay*T_d) / (tariff * P_standby)

so with labour = 0 and c_delay = 0 the tariff cancels exactly:

    D* = E_r / P_standby

and every column of the planned heat-map is identical.  Tariff
enters only through the NON-energy terms.  Sweep C1 below
demonstrates this numerically, and the headline map is drawn over
the axes that do carry information: restart energy against the
production-delay valuation.

    python -m experiments.task7_sensitivity --machine pelletizer-I
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
    DEFAULT_MACHINE, MACHINES, ensure_dir, save_json, info, section,
)
from src.decision import (                             # noqa: E402
    compare_policies, annualise, optimal_offline,
    policy_never, policy_observed, policy_elapsed_time, policy_static_threshold,
    SECONDS_PER_HOUR,
)
from experiments.task6_optimization import (           # noqa: E402
    PHASE3_DIR, prepare, fit_feasibility, make_feasibility,
    MAX_DECISION_HOURS, DECISION_EPOCH_S, CHANCE_CONFIDENCE,
    MIN_OFF_S, MAX_RESTARTS_PER_DAY, TARIFF_USD_PER_KWH,
)

# ── Sweep grids ───────────────────────────────────────────────────────────────

RESTART_ENERGY_KWH = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
TARIFFS = [0.06, 0.12, 0.20, 0.30]
DELAY_COST_PER_HOUR = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]

MIN_OFF_GRID = [0.0, 300.0, 600.0, 1200.0, 1800.0]
MAX_RESTART_GRID = [1, 2, 4, 8, 100]
CONFIDENCE_GRID = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]

# Break-even grid for the attainability curve: log-spaced from one minute to
# two days, which brackets the machine's whole idle-duration range with room
# above the longest observed episode -- the point beyond which de-energising
# can never pay and "never shut down" becomes provably optimal.
BREAK_EVEN_GRID_S = np.unique(np.round(np.logspace(np.log10(60), np.log10(172800), 44)))

# Retrofit economics for the payback analysis.  Stated, and swept.
SYSTEM_COST_USD = [500, 1000, 2000, 5000, 10000]
N_MACHINES = [1, 2, 5, 10]
PAYBACK_TARGET_YEARS = 3.0


# ── Decision traces ───────────────────────────────────────────────────────────

def decision_traces(ep, predict, feasible, epoch_s, max_hours):
    """
    Pre-compute the forecaster's output at every decision epoch of every
    episode, once.

    The online rule fires when `c_standby * E[R] > restart_cost`, and that
    inequality depends on the economic parameters ONLY through the ratio
    `restart_cost / c_standby`, which is the break-even duration.  So the
    forecast trajectory does not depend on the economics at all: a single
    (episodes x epochs) matrix of predictions can be thresholded at any
    break-even value to recover exactly the policy Task 6 would have run.

    This makes the sweep exact rather than approximate, and turns a
    several-hundred-model-evaluation sweep into one pass.
    """
    n_steps = int(max_hours * SECONDS_PER_HOUR / epoch_s) + 1
    D = ep["duration_s"].to_numpy(float)
    R = np.full((len(ep), n_steps), np.nan)
    Pok = np.full((len(ep), n_steps), np.nan)
    for s in range(n_steps):
        elapsed = s * epoch_s
        alive = D > elapsed
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        rows = ep.iloc[idx]
        R[idx, s] = predict(rows, elapsed)
        if feasible is not None:
            Pok[idx, s] = feasible(rows, elapsed)
    return R, Pok


def tau_from_traces(R, Pok, break_even_s, epoch_s,
                    confidence=None, min_off_s=0.0):
    """
    Recover the proposed policy's decision times from the pre-computed traces.

    `confidence=None` disables the chance constraint and falls back to the
    plug-in feasibility test, matching `policy_forecast_optimising`'s
    behaviour when no feasibility model is supplied.
    """
    ok = np.isfinite(R) & (R > break_even_s)
    if min_off_s > 0:
        if confidence is not None and np.isfinite(Pok).any():
            ok &= np.nan_to_num(Pok, nan=-1.0) >= confidence
        else:
            ok &= R >= min_off_s
    fires = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    return np.where(fires, first * epoch_s, np.inf)


def _row(tab, policy, params, test_span_s):
    """
    One policy's results, flattened for the sweep tables.

    Energy saved is carried alongside currency saved because the two can point
    in opposite directions.  Where the restart cost is large the cheapest
    policy is to STOP de-energising, which saves money by avoiding restarts
    while spending MORE standby energy.  A currency figure quoted on its own
    would read as an energy-efficiency result when it is the reverse.
    """
    r = tab.loc[policy]
    return {
        f"{policy}_savings": float(r["savings_vs_baseline"]),
        f"{policy}_annual": float(annualise(r["savings_vs_baseline"], test_span_s)),
        f"{policy}_energy_saved_kwh": float(r["energy_saved_vs_baseline_kwh"]),
        f"{policy}_shutdowns": int(r["n_shutdowns"]),
        f"{policy}_pct_oracle": float(r["pct_of_oracle"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machine: str = DEFAULT_MACHINE,
    seed: int = 0,
    epoch_s: float = DECISION_EPOCH_S,
) -> dict:
    out_dir = ensure_dir(f"{PHASE3_DIR}/task7/{machine}")
    section("TASK 7 -- BREAK-EVEN AND ECONOMIC SENSITIVITY")

    prep = prepare(machine, seed=seed, epoch_s=epoch_s)
    ep_te = prep["ep_test"]
    base = prep["base_params"]
    test_span_s = prep["test_span_s"]
    P_sb = prep["standby_power_w"]
    info(f"{len(ep_te):,} held-out episodes over {test_span_s/86400:.0f} days; "
         f"STANDBY power {P_sb:.0f} W; measured restart delay "
         f"{base.restart_delay_s:.0f} s")

    R, Pok = decision_traces(ep_te, prep["predictors"]["mean"], prep["feasible"],
                             epoch_s, MAX_DECISION_HOURS)
    info(f"decision traces: {R.shape[0]} episodes x {R.shape[1]} epochs")

    fixed_policies = {
        "never_shutdown": policy_never(ep_te, base),
        "observed": policy_observed(ep_te, base),
    }
    onset_pred = prep["predictors"]["mean"](ep_te, 0.0)

    def evaluate(params, tau_prop):
        pol = dict(fixed_policies)
        pol["ski_rental"] = policy_elapsed_time(ep_te, params)
        pol["static_break_even"] = policy_static_threshold(ep_te, params, onset_pred)
        pol["forecast_opt"] = tau_prop
        pol["oracle"] = optimal_offline(ep_te, params)
        return compare_policies(ep_te, pol, params)

    # ── A. Attainability curve ────────────────────────────────────────────────
    section("A -- what is attainable as a function of break-even")
    D = ep_te["duration_s"].to_numpy(float)
    rows_a = []
    for be in BREAK_EVEN_GRID_S:
        # Solve for the delay valuation that puts break-even here, so the whole
        # curve is realisable by some combination of the real coefficients.
        c_delay = (base.standby_cost_per_s * be) / (base.restart_delay_s / SECONDS_PER_HOUR)
        p = base.with_(delay_cost_per_hour=float(c_delay))
        tau = tau_from_traces(R, Pok, be, epoch_s, CHANCE_CONFIDENCE, p.min_off_s)
        tab = evaluate(p, tau)
        rows_a.append({
            "break_even_s": float(be),
            "break_even_h": float(be / SECONDS_PER_HOUR),
            "implied_delay_cost_per_h": float(c_delay),
            "restart_cost_usd": float(p.restart_cost),
            "n_episodes_over": int((D > be).sum()),
            "idle_h_over": float(D[D > be].sum() / SECONDS_PER_HOUR),
            "attainable_kwh": float(D[D > be].sum() / SECONDS_PER_HOUR * P_sb / 1000.0),
            **_row(tab, "oracle", p, test_span_s),
            **_row(tab, "forecast_opt", p, test_span_s),
            **_row(tab, "ski_rental", p, test_span_s),
            **_row(tab, "static_break_even", p, test_span_s),
            **_row(tab, "never_shutdown", p, test_span_s),
        })
    curve = pd.DataFrame(rows_a)
    curve.to_csv(f"{out_dir}/attainability_curve.csv", index=False)
    info(f"wrote {out_dir}/attainability_curve.csv ({len(curve)} points)")

    # Crossovers -- the numbers the discussion section needs.
    cross = {}
    beats_ski = curve[curve["forecast_opt_savings"] > curve["ski_rental_savings"]]
    cross["forecast_beats_ski_rental_from_h"] = (
        float(beats_ski["break_even_h"].min()) if len(beats_ski) else None)
    beats_obs = curve[curve["forecast_opt_savings"] > 0]
    cross["forecast_beats_status_quo_from_h"] = (
        float(beats_obs["break_even_h"].min()) if len(beats_obs) else None)
    never_best = curve[curve["never_shutdown_savings"] >= curve["oracle_savings"] - 1e-9]
    cross["never_shutdown_optimal_from_h"] = (
        float(never_best["break_even_h"].min()) if len(never_best) else None)
    cross["max_episode_h"] = float(D.max() / SECONDS_PER_HOUR)
    for k, v in cross.items():
        info(f"{k}: {v}")

    # ── B. The economic plane ─────────────────────────────────────────────────
    section("B -- restart energy x delay valuation x tariff")
    rows_b = []
    for tariff in TARIFFS:
        for e_r in RESTART_ENERGY_KWH:
            for c_d in DELAY_COST_PER_HOUR:
                p = base.with_(tariff_per_kwh=tariff, restart_energy_kwh=e_r,
                               delay_cost_per_hour=c_d)
                be = p.break_even_s
                tau = tau_from_traces(R, Pok, be, epoch_s,
                                      CHANCE_CONFIDENCE, p.min_off_s)
                tab = evaluate(p, tau)
                rows_b.append({
                    "tariff_per_kwh": tariff,
                    "restart_energy_kwh": e_r,
                    "delay_cost_per_hour": c_d,
                    "restart_cost_usd": float(p.restart_cost),
                    "break_even_h": float(be / SECONDS_PER_HOUR),
                    **_row(tab, "oracle", p, test_span_s),
                    **_row(tab, "forecast_opt", p, test_span_s),
                    **_row(tab, "ski_rental", p, test_span_s),
                    **_row(tab, "never_shutdown", p, test_span_s),
                })
    grid = pd.DataFrame(rows_b)
    grid.to_csv(f"{out_dir}/economic_grid.csv", index=False)
    info(f"wrote {out_dir}/economic_grid.csv ({len(grid)} cells)")

    profitable = grid[grid["forecast_opt_annual"] > 0]
    best = grid.loc[grid["forecast_opt_annual"].idxmax()]
    info(f"proposed policy is profitable in {len(profitable)}/{len(grid)} cells "
         f"({len(profitable)/len(grid)*100:.0f}%)")
    info(f"largest currency saving {best['forecast_opt_annual']:.0f} USD/yr at "
         f"tariff {best['tariff_per_kwh']:.2f}, E_r {best['restart_energy_kwh']:.0f} kWh, "
         f"delay {best['delay_cost_per_hour']:.0f} USD/h "
         f"(break-even {best['break_even_h']:.1f} h)")

    # That corner is where restarts are so expensive the best move is to stop
    # de-energising, so the currency saving comes from avoided restarts while
    # standby energy RISES.  Report the energy-positive region separately.
    energy_pos = grid[grid["forecast_opt_energy_saved_kwh"] > 0]
    info(f"{len(energy_pos)}/{len(grid)} cells also save ENERGY; "
         f"at the largest-currency-saving cell energy changes by "
         f"{best['forecast_opt_energy_saved_kwh']:+.0f} kWh over the test window")
    if len(energy_pos):
        be_ = energy_pos.loc[energy_pos["forecast_opt_annual"].idxmax()]
        info(f"best energy-positive cell: {be_['forecast_opt_annual']:.0f} USD/yr and "
             f"{be_['forecast_opt_energy_saved_kwh']:+.0f} kWh at break-even "
             f"{be_['break_even_h']:.2f} h")

    # ── C1. Tariff-invariance of break-even ───────────────────────────────────
    section("C1 -- is break-even tariff-dependent?")
    rows_c1 = []
    for tariff in [0.04, 0.08, 0.12, 0.20, 0.30, 0.50]:
        for labour in [0.0, 0.25, 1.0]:
            p = base.with_(tariff_per_kwh=tariff, restart_energy_kwh=10.0,
                           restart_labour_cost=labour, delay_cost_per_hour=0.0)
            rows_c1.append({"tariff_per_kwh": tariff,
                            "restart_labour_cost": labour,
                            "restart_energy_kwh": 10.0,
                            "break_even_h": p.break_even_s / SECONDS_PER_HOUR})
    c1 = pd.DataFrame(rows_c1)
    c1.to_csv(f"{out_dir}/tariff_invariance.csv", index=False)
    flat = c1[c1["restart_labour_cost"] == 0.0]["break_even_h"]
    info(f"with a purely electrical restart cost, break-even is "
         f"{flat.min():.3f}-{flat.max():.3f} h across a 12x tariff range "
         f"(exactly invariant: E_r/P_standby = "
         f"{10.0/(P_sb/1000.0):.3f} h)")
    withlab = c1[c1["restart_labour_cost"] == 1.0]
    info(f"with a 1.00 USD non-energy restart cost it spans "
         f"{withlab['break_even_h'].min():.2f}-{withlab['break_even_h'].max():.2f} h "
         f"-- tariff dependence enters only through non-energy terms")

    # ── C2. Constraint constants ──────────────────────────────────────────────
    section("C2 -- sensitivity to the stated constraint constants")
    headline_be = 3600.0
    c_delay_head = (base.standby_cost_per_s * headline_be) / (base.restart_delay_s / SECONDS_PER_HOUR)
    head = base.with_(delay_cost_per_hour=float(c_delay_head))

    rows_c2 = []
    # confidence -- free, the traces already hold P(feasible)
    for conf in CONFIDENCE_GRID:
        tau = tau_from_traces(R, Pok, headline_be, epoch_s, conf, head.min_off_s)
        tab = evaluate(head, tau)
        r = tab.loc["forecast_opt"]
        rows_c2.append({"knob": "chance_confidence", "value": conf,
                        "savings": float(r["savings_vs_baseline"]),
                        "annual": float(annualise(r["savings_vs_baseline"], test_span_s)),
                        "n_shutdowns": int(r["n_shutdowns"]),
                        "min_off_violations": int(r["min_off_violations"]),
                        "pct_of_oracle": float(r["pct_of_oracle"])})
    # restarts per day -- evaluation-time constraint only
    for cap in MAX_RESTART_GRID:
        p = head.with_(max_restarts_per_day=cap)
        tau = tau_from_traces(R, Pok, headline_be, epoch_s, CHANCE_CONFIDENCE, p.min_off_s)
        tab = evaluate(p, tau)
        r = tab.loc["forecast_opt"]
        rows_c2.append({"knob": "max_restarts_per_day", "value": cap,
                        "savings": float(r["savings_vs_baseline"]),
                        "annual": float(annualise(r["savings_vs_baseline"], test_span_s)),
                        "n_shutdowns": int(r["n_shutdowns"]),
                        "min_off_violations": int(r["min_off_violations"]),
                        "pct_of_oracle": float(r["pct_of_oracle"])})
    # minimum off time -- the feasibility model must be refitted per value
    for m_off in MIN_OFF_GRID:
        p = head.with_(min_off_s=m_off)
        if m_off > 0:
            clf = fit_feasibility(prep["Xtr"], prep["ytr"], m_off, seed=seed)
            feas = make_feasibility(clf, prep["cfg"]["close_hour"],
                                    prep["cfg"]["open_hour"])
            _, Pok_m = decision_traces(ep_te, prep["predictors"]["mean"], feas,
                                       epoch_s, MAX_DECISION_HOURS)
            tau = tau_from_traces(R, Pok_m, headline_be, epoch_s,
                                  CHANCE_CONFIDENCE, m_off)
        else:
            tau = tau_from_traces(R, Pok, headline_be, epoch_s, None, 0.0)
        tab = evaluate(p, tau)
        r = tab.loc["forecast_opt"]
        rows_c2.append({"knob": "min_off_s", "value": m_off,
                        "savings": float(r["savings_vs_baseline"]),
                        "annual": float(annualise(r["savings_vs_baseline"], test_span_s)),
                        "n_shutdowns": int(r["n_shutdowns"]),
                        "min_off_violations": int(r["min_off_violations"]),
                        "pct_of_oracle": float(r["pct_of_oracle"])})
    c2 = pd.DataFrame(rows_c2)
    c2.to_csv(f"{out_dir}/constraint_sensitivity.csv", index=False)
    print()
    print(c2.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    # ── D. Payback ────────────────────────────────────────────────────────────
    section("D -- retrofit payback")
    # Anchored on the CENTRAL case -- the measured restart coefficients at the
    # headline break-even -- not on the best cell of the plane.  Quoting the
    # maximum would advertise the corner where the policy's advice is "stop
    # de-energising", which is a restart-cost saving rather than an energy one
    # and does not generalise to a machine with different economics.
    central_annual = float(annualise(
        evaluate(head, tau_from_traces(R, Pok, headline_be, epoch_s,
                                       CHANCE_CONFIDENCE, head.min_off_s))
        .loc["forecast_opt", "savings_vs_baseline"], test_span_s))
    best_annual_any = float(grid["forecast_opt_annual"].max())

    rows_d = []
    for label, annual in (("central", central_annual), ("best_case", best_annual_any)):
        for cost in SYSTEM_COST_USD:
            for n in N_MACHINES:
                total = annual * n
                rows_d.append({
                    "case": label,
                    "annual_savings_per_machine_usd": annual,
                    "system_cost_usd": cost,
                    "n_machines": n,
                    "annual_savings_usd": total,
                    "payback_years": (cost / total if total > 0 else np.inf),
                })
    pay = pd.DataFrame(rows_d)
    pay.to_csv(f"{out_dir}/payback.csv", index=False)

    max_cost_3yr = central_annual * PAYBACK_TARGET_YEARS
    info(f"central case (measured coefficients, break-even "
         f"{headline_be/SECONDS_PER_HOUR:.0f} h): {central_annual:.0f} USD/yr per machine")
    info(f"best case anywhere on the plane: {best_annual_any:.0f} USD/yr per machine")
    info(f"a {PAYBACK_TARGET_YEARS:.0f}-year payback in the central case requires a "
         f"retrofit cost below {max_cost_3yr:.0f} USD per machine")
    for cost in (2000, 5000):
        yrs = cost / central_annual if central_annual > 0 else np.inf
        info(f"  a {cost} USD/machine retrofit pays back in {yrs:,.0f} years "
             f"at the central case -- standby recovery alone does not fund it")
    viable = pay[(pay["case"] == "central")
                 & (pay["payback_years"] <= PAYBACK_TARGET_YEARS)]
    info(f"{len(viable)}/{len(pay[pay['case']=='central'])} central-case "
         f"(cost, fleet) combinations reach a "
         f"{PAYBACK_TARGET_YEARS:.0f}-year payback")

    # ── Persist ───────────────────────────────────────────────────────────────
    results = {
        "machine": machine,
        "standby_power_w": P_sb,
        "restart_delay_s": base.restart_delay_s,
        "test_span_days": test_span_s / 86400.0,
        "n_test_episodes": int(len(ep_te)),
        "grids": {
            "restart_energy_kwh": RESTART_ENERGY_KWH,
            "tariffs": TARIFFS,
            "delay_cost_per_hour": DELAY_COST_PER_HOUR,
            "min_off_s": MIN_OFF_GRID,
            "max_restarts_per_day": MAX_RESTART_GRID,
            "chance_confidence": CONFIDENCE_GRID,
            "system_cost_usd": SYSTEM_COST_USD,
            "n_machines": N_MACHINES,
        },
        "crossovers": cross,
        "tariff_invariance": {
            "analytic_break_even_h_at_10kwh": 10.0 / (P_sb / 1000.0),
            "numeric_range_no_labour_h": [float(flat.min()), float(flat.max())],
            "numeric_range_with_labour_h": [float(withlab["break_even_h"].min()),
                                            float(withlab["break_even_h"].max())],
        },
        "economic_plane": {
            "n_cells": int(len(grid)),
            "n_profitable": int(len(profitable)),
            "n_energy_positive": int(len(energy_pos)),
            "best_annual_usd": best_annual_any,
            "best_cell": {k: float(best[k]) for k in
                          ("tariff_per_kwh", "restart_energy_kwh",
                           "delay_cost_per_hour", "break_even_h",
                           "forecast_opt_annual", "forecast_opt_energy_saved_kwh")},
        },
        "payback": {
            "headline_break_even_h": headline_be / SECONDS_PER_HOUR,
            "central_annual_usd_per_machine": central_annual,
            "best_annual_usd_per_machine": best_annual_any,
            "max_retrofit_cost_for_3yr_payback_usd": max_cost_3yr,
            "payback_years_at_5000_usd": (5000.0 / central_annual
                                          if central_annual > 0 else None),
            "n_viable_central_combinations": int(len(viable)),
        },
    }
    save_json(results, f"{out_dir}/task7_results.json")
    section("TASK 7 COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task 7 -- economic sensitivity")
    ap.add_argument("--machine", default=DEFAULT_MACHINE, choices=list(MACHINES))
    ap.add_argument("--epoch", type=float, default=DECISION_EPOCH_S)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(machine=a.machine, seed=a.seed, epoch_s=a.epoch)
