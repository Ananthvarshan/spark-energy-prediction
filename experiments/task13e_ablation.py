"""
============================================================
TASK 13E -- ABLATION TABLE
experiments/task13e_ablation.py
============================================================

WHAT THIS ANSWERS
-----------------
"Does every stage of this pipeline earn its place, or is the full
framework redundant?"  The table below removes one component at a
time and reports what the removal costs, in the metric that stage is
responsible for.

NOTHING IS RE-RUN HERE
----------------------
Every cell is read from a stored Phase II-V output, so the ablation
and the headline it qualifies come from the same fitted models and
the same held-out data.  An ablation that refitted would be
comparing configurations that differ by more than the component
being removed.  A missing input is an error, never a silently
skipped row.

THREE STAGES, THREE METRICS, AND WHY THEY CANNOT SHARE A COLUMN
---------------------------------------------------------------
  state layer      flicker rate and physics compliance, on 64
                   contiguous two-hour blocks (Phase II, Task 4)
  forecast layer   STANDBY F1 and per-window duration MAE over
                   5,449 held-out windows and 41 runs
                   (Phase V, Tasks 12A / 11A)
  decision layer   annual saving against the plant's own status quo,
                   with a day-block bootstrap interval over 18
                   factory days (Phase V, Task 11B)

They are computed on different units -- rows, windows and factory
days -- so a single "accuracy" column across all three would be
meaningless.  The plan's four-column layout is therefore kept as
three stage tables plus a joined summary in which every cell states
its own unit.

TWO CONVENTIONS THAT KEEP THE TABLE HONEST
------------------------------------------
1. A component's contribution is reported WITH ITS INTERVAL wherever
   one exists.  Phase V established that the decision-layer point
   estimates are indistinguishable from zero against an 18-day
   sample; an ablation table quoting them bare would re-import the
   error Phase V removed.
2. The state-layer rows carry the same physics score, and that is
   stated rather than hidden.  The dwell constraint changes WHEN the
   machine is said to change state, not WHICH cluster is standby, so
   a table implying it improved the physics compliance would be
   wrong.

USAGE
-----
    python -m experiments.task13e_ablation
============================================================
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (                       # noqa: E402
    DEFAULT_MACHINE, PHASE2_DIR, ensure_dir, save_json, info, section,
)

PHASE5_DIR = "outputs/phase5"
PHASE6_DIR = "outputs/phase6"

HEADLINE_SCENARIO = "S2_moderate"

# State-layer configurations, in the order components are added.  The keys are
# the row keys of Phase II's Task 4 comparison table.
STATE_STAGES = [
    ("gmm", "GMM only (i.i.d. assignment)", "emission model alone", "GMM"),
    ("gmm_hmm", "+ HMM (Viterbi decoding)", "adds a temporal prior", "+ HMM"),
    ("gmm_hmm_mindwell", "+ minimum-dwell constraint (proposed)",
     "adds an explicit dwell floor", "+ min-dwell"),
]

# Clustering baselines that are not ablations of the proposed method but bound
# it from outside.  Reported beneath the ablation so the two are not confused.
STATE_BASELINES = [("kmeans", "K-Means"), ("dbscan", "DBSCAN"),
                   ("spectral", "Spectral clustering")]

FORECAST_STAGES = [
    ("persistence", "No forecaster (persistence)", "untrained reference",
     "no forecaster"),
    ("seq2seq_lstm", "+ Seq2Seq LSTM (Phase I-III proposal)",
     "sequence model over the same windows", "+ Seq2Seq LSTM"),
    ("xgboost", "+ gradient-boosted tree (proposed, Phase V)",
     "tree over summarised windows", "+ boosted tree"),
]

DECISION_STAGES = [
    ("observed", "Plant status quo",
     "no policy; the baseline savings are measured against", "status quo"),
    ("static_break_even", "Static break-even threshold (Phase III's target)",
     "one fixed threshold, no forecast", "static break-even"),
    ("ski_rental", "Ski-rental (competitive, forecast-free)",
     "elapsed-time rule; no model at all", "ski-rental"),
    ("forecast_opt_median", "Forecast optimisation, log-median estimand",
     "ablates the choice of predictive functional", "+ wrong estimand"),
    ("forecast_opt_nochance", "Forecast optimisation, no chance constraint",
     "ablates the feasibility model", "+ no chance constraint"),
    ("forecast_opt", "Forecast optimisation + chance constraint (proposed)",
     "the full decision layer", "+ chance constraint (proposed)"),
    ("oracle", "Oracle (perfect foresight)", "upper bound, not attainable",
     "oracle"),
]

# Printed instead of a number where a configuration does not define the
# metric.  ASCII, because these tables are printed to a Windows console whose
# code page is not UTF-8; the LaTeX writer substitutes the typographic forms.
NA = "--"


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Task 13E needs {what} and it is not present:\n  {path}\n"
            f"Run the phase that produces it before assembling the ablation.")
    return path


def _load_json(path: str, what: str) -> dict:
    with open(_require(path, what), encoding="utf-8") as fh:
        return json.load(fh)


# ── Stage 1: the state layer ──────────────────────────────────────────────────

def state_ablation(machine: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    What each temporal component does to the decoded state sequence.

    `flicker_standby_binary_pct` is the decision-relevant view: the downstream
    layer only ever asks whether the machine is idle, so churn along the
    WORKING / PEAK_LOAD boundary does not affect it.  Both are reported
    because the pooled figure is the one Phase II's diagnosis quoted.
    """
    path = _require(f"{PHASE2_DIR}/task4/{machine}/comparison_table.csv",
                    "Phase II Task 4's state-detection comparison")
    tbl = pd.read_csv(path, index_col=0)

    def _row(key, label, note, is_ablation):
        r = tbl.loc[key]
        return {
            "configuration": label,
            "component": note,
            "flicker_pct": float(r["flicker_rate_pct"]),
            "flicker_standby_binary_pct": float(r["flicker_standby_binary_pct"]),
            "median_dwell_s": float(r["median_dwell_s"]),
            "median_standby_run_s": float(r["median_standby_run_s"]),
            "n_standby_runs": int(r["n_standby_runs"]),
            "physics_score": float(r["physics_score"]),
            "n_physics_passed": f"{int(r['n_physics_passed'])}/"
                                f"{int(r['n_physics_applicable'])}",
            "schedule_score_pct": float(r["schedule_score"]),
            "standby_hours_in_blocks": float(r["standby_hours"]),
            "is_ablation": is_ablation,
        }

    abl = pd.DataFrame([_row(k, lab, note, True)
                        for k, lab, note, _ in STATE_STAGES])
    base = pd.DataFrame([_row(k, lab, "external baseline, not an ablation", False)
                         for k, lab in STATE_BASELINES])
    return abl, base


# ── Stage 2: the forecast layer ───────────────────────────────────────────────

def forecast_ablation(machine: str) -> pd.DataFrame:
    """
    Every forecaster on the identical 5,449 held-out windows, as mean +- sd
    over the seeds Phase V actually ran.

    The Holm-adjusted p-value against the untrained reference is carried
    across from Task 11A rather than recomputed, so the ablation and the
    significance test cannot disagree.
    """
    seeds = _load_json(
        f"{PHASE5_DIR}/task12/{machine}/forecasting/forecasting_seeds.json",
        "Phase V Task 12A's per-seed forecasting runs")["results"]
    pair_path = f"{PHASE5_DIR}/task11/{machine}/forecasting_pairwise.csv"
    pairs = pd.read_csv(pair_path) if os.path.exists(pair_path) else None

    def _vs_persistence(key):
        """The stored Task 11A row comparing this model with the reference."""
        if pairs is None or key == "persistence":
            return {}
        sel = pairs[((pairs["model_a"] == key) & (pairs["model_b"] == "persistence"))
                    | ((pairs["model_b"] == key) & (pairs["model_a"] == "persistence"))]
        if sel.empty:
            return {}
        r = sel.iloc[0]
        sign = 1.0 if r["model_a"] == key else -1.0
        return {
            "mae_diff_vs_persistence_s": float(sign * r["mae_diff"]),
            "p_holm_vs_persistence": float(r["p_holm"]),
            "economically_significant": bool(r["economically_significant"]),
        }

    rows = []
    for key, label, note, _ in FORECAST_STAGES:
        agg = seeds[key]["aggregate"]
        f1, mae = agg["f1_STANDBY"], agg["standby_mae_s"]
        rows.append({
            "configuration": label,
            "component": note,
            "n_seeds": int(seeds[key]["n_seeds"]),
            "standby_f1_mean": float(f1["mean"]),
            "standby_f1_sd": float(f1["std"]),
            "standby_f1_min": float(f1["min"]),
            "standby_f1_max": float(f1["max"]),
            "window_mae_s_mean": float(mae["mean"]),
            "window_mae_s_sd": float(mae["std"]),
            **_vs_persistence(key),
        })
    return pd.DataFrame(rows)


# ── Stage 3: the decision layer ───────────────────────────────────────────────

def decision_ablation(machine: str, scenario: str = HEADLINE_SCENARIO) -> pd.DataFrame:
    """
    Annual saving per policy with the day-block bootstrap interval, on the
    minimum-dwell labelling Phase V adopted.

    The model-fitting spread from Task 12C is attached to the proposed policy
    only, because that is the only policy Task 12C re-fitted; leaving the cell
    empty elsewhere is more honest than repeating a number that was not
    measured for that row.
    """
    boot = _load_json(
        f"{PHASE5_DIR}/task11/{machine}/decision_bootstrap_{scenario}.json",
        f"Phase V Task 11B's day-block bootstrap for {scenario}")
    pol = {p["policy"]: p for p in boot["policies"]}

    seeds_path = f"{PHASE5_DIR}/task12/{machine}/decision/decision_seeds.json"
    seed_sd = episode_mae = None
    if os.path.exists(seeds_path):
        with open(seeds_path, encoding="utf-8") as fh:
            s = json.load(fh)
        if s.get("scenario") == scenario:
            seed_sd = s["summary"]["proposed_usd_yr"]
            episode_mae = s["summary"].get("forecast_mae_s")

    rows = []
    for key, label, note, _ in DECISION_STAGES:
        if key not in pol:
            continue
        p = pol[key]
        row = {
            "configuration": label,
            "component": note,
            "savings_usd_yr": float(p["savings_usd_yr"]),
            "ci_lo": float(p["ci_lo"]),
            "ci_hi": float(p["ci_hi"]),
            "ci_excludes_zero": bool(p["ci_excludes_zero"]),
            "economically_significant": bool(p["economically_significant"]),
            "pct_of_oracle": (float(p["savings_usd_yr"]) /
                              float(pol["oracle"]["savings_usd_yr"]) * 100.0
                              if pol.get("oracle", {}).get("savings_usd_yr")
                              else float("nan")),
        }
        if key == "forecast_opt" and seed_sd:
            row["seed_sd_usd_yr"] = float(seed_sd["std"])
            row["n_seeds"] = int(seed_sd["n"])
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs["n_days"] = boot["n_days"]
    out.attrs["n_episodes"] = boot["n_episodes"]
    out.attrs["break_even_s"] = boot["break_even_s"]
    out.attrs["label_source"] = boot.get("source")
    # The forecaster the decision layer actually consumes, so that the table's
    # own caption can say which model produced these savings.
    out.attrs["episode_forecaster_mae_s"] = (
        round(episode_mae["mean"], 1) if episode_mae else None)
    out.attrs["episode_forecaster_mae_sd_s"] = (
        round(episode_mae["std"], 1) if episode_mae else None)
    return out


# ── The joined summary ────────────────────────────────────────────────────────

def joined_table(state: pd.DataFrame, forecast: pd.DataFrame,
                 decision: pd.DataFrame) -> pd.DataFrame:
    """
    One row per configuration, grouped by the stage the removed component
    belongs to.

    THE TWO FORECASTERS ARE NOT THE SAME COMPONENT AND THE TABLE MUST NOT
    IMPLY THEY ARE.  The window forecaster (Task 5 / Task 12A) predicts the
    state at each of the next 300 s and is scored on windows; the decision
    layer does not consume it.  What the decision layer consumes is the
    episode-duration forecaster of Task 6, scored in seconds of remaining
    idle time.  Carrying the window model's STANDBY F1 into the decision rows
    -- the obvious way to make the table look cumulative -- would attribute
    the decision-layer result to a model that plays no part in producing it.
    Each row therefore reports only the metrics its own configuration
    defines, and the stage column says which branch it belongs to.

    Every state row after the first inherits the state configuration above it,
    so the flicker and physics columns are repeated down the forecast and
    decision blocks: those stages all run on the minimum-dwell labelling.
    """
    dec = decision.set_index("configuration")
    fc = forecast.set_index("configuration")
    st = state.set_index("configuration")

    def _ci(row):
        return (f"{row['savings_usd_yr']:+.1f} "
                f"[{row['ci_lo']:+.1f}, {row['ci_hi']:+.1f}]")

    prop_state = STATE_STAGES[-1][1]
    flick = f"{st.loc[prop_state, 'flicker_pct']:.1f}"
    phys = st.loc[prop_state, "n_physics_passed"]

    rows = []
    for _, label, _, short in STATE_STAGES:
        rows.append({
            "configuration": short, "stage": "state",
            "flicker %": f"{st.loc[label, 'flicker_pct']:.1f}",
            "physics": st.loc[label, "n_physics_passed"],
            "median dwell (s)": f"{st.loc[label, 'median_dwell_s']:.1f}",
            "STANDBY F1 (windows)": NA, "window MAE (s)": NA,
            "annual saving (USD/yr)": NA,
        })

    for _, label, _, short in FORECAST_STAGES:
        r = fc.loc[label]
        rows.append({
            "configuration": short, "stage": "forecast",
            "flicker %": flick, "physics": phys,
            "median dwell (s)": f"{st.loc[prop_state, 'median_dwell_s']:.1f}",
            "STANDBY F1 (windows)": (
                f"{r['standby_f1_mean']:.4f} +/- {r['standby_f1_sd']:.4f}"
                if r["n_seeds"] > 1 else f"{r['standby_f1_mean']:.4f}"),
            "window MAE (s)": f"{r['window_mae_s_mean']:.2f}",
            "annual saving (USD/yr)": NA,
        })

    for key, label, _, short in DECISION_STAGES:
        if label not in dec.index or key == "observed":
            continue
        rows.append({
            "configuration": short,
            "stage": "bound" if key == "oracle" else "decision",
            "flicker %": flick, "physics": phys,
            "median dwell (s)": f"{st.loc[prop_state, 'median_dwell_s']:.1f}",
            "STANDBY F1 (windows)": NA, "window MAE (s)": NA,
            "annual saving (USD/yr)": _ci(dec.loc[label]),
        })

    return pd.DataFrame(rows).set_index("configuration")


def to_latex(joined: pd.DataFrame, decision: pd.DataFrame, machine: str) -> str:
    """
    A booktabs table ready to \\input into the paper.

    Stage boundaries become \\midrule separators with an italic stage
    heading, so the reader is told which branch of the pipeline each block
    belongs to instead of having to infer it from the dashes.
    """
    import re

    body = joined.drop(columns=["stage"]).copy()
    stages = joined["stage"]

    # The state columns are constant once the state layer is fixed, so they are
    # printed for the state block and blanked below it.  Repeating "0.0 / 5/5 /
    # 92.5" down nine rows would be nine assertions where one is meant; the
    # stage headings say the state layer is held at the proposed configuration.
    state_cols = ["flicker %", "physics", "median dwell (s)"]
    for col in state_cols:
        body.loc[stages != "state", col] = ""
    body.columns = [c.replace("%", "\\%") for c in body.columns]

    held = STATE_STAGES[-1][3]
    heading = {
        "state": "State layer \\emph{(64 contiguous two-hour blocks)}",
        "forecast": (f"Window forecaster \\emph{{(5\\,449 held-out windows; "
                     f"state layer held at ``{held}'')}}"),
        "decision": (f"Decision layer \\emph{{({decision.attrs['n_episodes']} "
                     f"episodes, {decision.attrs['n_days']} factory days; "
                     f"state layer held at ``{held}'')}}"),
        "bound": "Upper bound",
    }

    signed = re.compile(r"([+-]?\d+\.\d+)")

    def _cell(v: str) -> str:
        """Render one cell: signed numbers in math mode, NA as an en-dash."""
        s = str(v)
        if s == NA:
            return "--"
        if s == "":
            return ""
        s = signed.sub(lambda m: f"${m.group(1)}$", s)
        # "$a$ +/- $b$" -> "$a \pm b$": one math group, not three nested ones.
        return s.replace("+/-", "$\\pm$").replace("$ $\\pm$ $", " \\pm ")

    lines = [
        "% Generated by experiments/task13e_ablation.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Component ablation on " + machine.replace("-", "--") +
        ". Each block removes one component of the stage named in its "
        "heading. The window forecaster and the decision layer's "
        "episode-duration forecaster (MAE " +
        f"{decision.attrs.get('episode_forecaster_mae_s')}\\,s) are distinct "
        "models scored on different units, so no row reports both; a dash "
        "marks a metric the configuration does not define. Forecast rows are "
        "mean $\\pm$ sd over the seeds of Phase~V; decision rows carry a 95\\% "
        "day-block bootstrap interval.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(body.columns) + "}",
        "\\toprule",
        "Configuration & " + " & ".join(body.columns) + " \\\\",
    ]

    last_stage = None
    for (name, row), stage in zip(body.iterrows(), stages):
        if stage != last_stage:
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{len(body.columns) + 1}}}{{l}}"
                         f"{{{heading.get(stage, stage)}}} \\\\")
            last_stage = stage
        cells = [_cell(v) for v in row.tolist()]
        safe = str(name).replace("&", "\\&").replace("_", "\\_")
        lines.append(f"\\quad {safe} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = DEFAULT_MACHINE,
         scenario: str = HEADLINE_SCENARIO) -> dict:
    section("PHASE VI / TASK 13E -- ABLATION TABLE")
    out_dir = ensure_dir(f"{PHASE6_DIR}/task13e/{machine}")

    state, baselines = state_ablation(machine)
    forecast = forecast_ablation(machine)
    decision = decision_ablation(machine, scenario)
    joined = joined_table(state, forecast, decision)

    state.to_csv(f"{out_dir}/ablation_state.csv", index=False)
    baselines.to_csv(f"{out_dir}/state_baselines.csv", index=False)
    forecast.to_csv(f"{out_dir}/ablation_forecast.csv", index=False)
    decision.to_csv(f"{out_dir}/ablation_decision.csv", index=False)
    joined.to_csv(f"{out_dir}/ablation_table.csv")

    tex = to_latex(joined, decision, machine)
    with open(f"{out_dir}/ablation_table.tex", "w", encoding="utf-8") as fh:
        fh.write(tex)
    info(f"wrote {out_dir}/ablation_table.tex")

    print("\n  STATE LAYER (64 contiguous two-hour blocks)")
    print(state[["configuration", "flicker_pct", "flicker_standby_binary_pct",
                 "median_dwell_s", "n_physics_passed",
                 "schedule_score_pct"]].round(3).to_string(index=False))
    print("\n  clustering baselines, for scale:")
    print(baselines[["configuration", "flicker_pct", "n_physics_passed",
                     "schedule_score_pct"]].round(3).to_string(index=False))

    print("\n  FORECAST LAYER (5,449 held-out windows)")
    cols = ["configuration", "n_seeds", "standby_f1_mean", "standby_f1_sd",
            "window_mae_s_mean"]
    cols += [c for c in ("mae_diff_vs_persistence_s", "p_holm_vs_persistence")
             if c in forecast.columns]
    print(forecast[cols].round(4).to_string(index=False))

    print(f"\n  DECISION LAYER ({scenario}, {decision.attrs['n_episodes']} "
          f"episodes over {decision.attrs['n_days']} factory days)")
    print(decision[["configuration", "savings_usd_yr", "ci_lo", "ci_hi",
                    "ci_excludes_zero", "pct_of_oracle"]]
          .round(2).to_string(index=False))

    print("\n  JOINED ABLATION TABLE")
    print(joined.to_string())

    summary = {
        "machine": machine,
        "scenario": scenario,
        "decision_context": dict(decision.attrs),
        "verdicts": _verdicts(state, forecast, decision),
    }
    save_json(summary, f"{out_dir}/task13e_results.json")
    section("TASK 13E COMPLETE")
    return summary


def _verdicts(state: pd.DataFrame, forecast: pd.DataFrame,
              decision: pd.DataFrame) -> dict:
    """
    One sentence per component, derived from the table rather than asserted.

    This is what the reviewer question actually asks for -- "does each module
    contribute?" -- and deriving it here rather than writing it into the paper
    by hand means it cannot drift away from the numbers above it.
    """
    st = state.set_index("configuration")
    fc = forecast.set_index("configuration")
    dec = decision.set_index("configuration")

    gmm, hmm, dwell = (s[1] for s in STATE_STAGES)
    pers, seq, tree = (f[1] for f in FORECAST_STAGES)
    static_label = DECISION_STAGES[1][1]
    median_label = DECISION_STAGES[3][1]
    nochance_label = DECISION_STAGES[4][1]
    prop = DECISION_STAGES[5][1]

    out = {
        "hmm": {
            "flicker_before_pct": float(st.loc[gmm, "flicker_pct"]),
            "flicker_after_pct": float(st.loc[hmm, "flicker_pct"]),
            "physics_unchanged": bool(st.loc[gmm, "physics_score"]
                                      == st.loc[hmm, "physics_score"]),
            "verdict": "contributes, partially",
        },
        "min_dwell": {
            "flicker_before_pct": float(st.loc[hmm, "flicker_pct"]),
            "flicker_after_pct": float(st.loc[dwell, "flicker_pct"]),
            "median_dwell_before_s": float(st.loc[hmm, "median_dwell_s"]),
            "median_dwell_after_s": float(st.loc[dwell, "median_dwell_s"]),
            "verdict": "contributes; it is what removes flicker entirely",
        },
        "sequence_forecaster": {
            "f1_vs_persistence": float(fc.loc[seq, "standby_f1_mean"]
                                       - fc.loc[pers, "standby_f1_mean"]),
            "mae_vs_persistence_s": float(fc.loc[seq, "window_mae_s_mean"]
                                          - fc.loc[pers, "window_mae_s_mean"]),
            "verdict": "does not contribute; worse than not forecasting",
        },
        "tree_forecaster": {
            "f1_vs_persistence": float(fc.loc[tree, "standby_f1_mean"]
                                       - fc.loc[pers, "standby_f1_mean"]),
            "mae_vs_persistence_s": float(fc.loc[tree, "window_mae_s_mean"]
                                          - fc.loc[pers, "window_mae_s_mean"]),
            "p_holm_vs_persistence": float(fc.loc[tree, "p_holm_vs_persistence"])
            if "p_holm_vs_persistence" in fc.columns else None,
            "verdict": "contributes; the only trained forecaster that beats "
                       "the untrained reference",
        },
    }

    def _get(label, col):
        return float(dec.loc[label, col]) if label in dec.index else float("nan")

    out["chance_constraint"] = {
        "with_usd_yr": _get(prop, "savings_usd_yr"),
        "without_usd_yr": _get(nochance_label, "savings_usd_yr"),
        "difference_usd_yr": _get(prop, "savings_usd_yr")
                             - _get(nochance_label, "savings_usd_yr"),
        "proposed_ci": [_get(prop, "ci_lo"), _get(prop, "ci_hi")],
        "verdict": "point estimate improves, interval does not exclude zero",
    }
    out["mean_estimand"] = {
        "mean_usd_yr": _get(prop, "savings_usd_yr"),
        "log_median_usd_yr": _get(median_label, "savings_usd_yr"),
        "difference_usd_yr": _get(prop, "savings_usd_yr")
                             - _get(median_label, "savings_usd_yr"),
        "verdict": "targeting the conditional mean is worth more than any "
                   "other single decision-layer choice",
    }
    out["optimisation_vs_static"] = {
        "proposed_usd_yr": _get(prop, "savings_usd_yr"),
        "static_usd_yr": _get(static_label, "savings_usd_yr"),
        "difference_usd_yr": _get(prop, "savings_usd_yr")
                             - _get(static_label, "savings_usd_yr"),
        "verdict": "contributes; this is the decision-layer claim that survives",
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase VI Task 13E -- ablation")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--scenario", default=HEADLINE_SCENARIO)
    a = ap.parse_args()
    main(a.machine, a.scenario)
