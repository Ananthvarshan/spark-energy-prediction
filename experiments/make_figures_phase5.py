"""
============================================================
PHASE V RESULT FIGURES  --  experiments/make_figures_phase5.py
============================================================

PURPOSE
-------
Render the Phase V figures from the JSON/CSV the tests wrote.

    python -m experiments.make_figures_phase5

DESIGN NOTES
------------
Same palette and rules as Phases II-IV.  Three additions specific to
statistical figures:

  * An interval is never drawn without the zero line it has to be
    read against, and intervals that cross zero are drawn in the
    neutral grey rather than in the signed colour -- the whole point
    of the figure is which ones do.
  * The critical-difference diagram follows Demsar (2006) exactly:
    mean ranks on a single axis, best on the left, and horizontal
    bars joining models whose rank difference is smaller than the
    critical difference.  A bar means "these are not distinguishable
    by this test at this n", which is a statement about the design as
    much as about the models.
  * Power curves carry the p-value FLOOR as an annotation, because at
    n = 4 the floor is above alpha and no amount of effect can produce
    a significant result -- a fact no confidence interval on the
    figure would otherwise reveal.
============================================================
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ensure_dir, info, section        # noqa: E402
from src.stats import wilcoxon_power, wilcoxon_p_floor, friedman_power  # noqa: E402

PHASE5_DIR = "outputs/phase5"
FIG_DIR = f"{PHASE5_DIR}/figures"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
NEUTRAL = "#9a9894"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

LABEL = {
    "persistence": "Persistence (none)",
    "seq2seq_lstm": "Seq2Seq LSTM",
    "vanilla_lstm": "Vanilla LSTM",
    "gru": "GRU",
    "tcn": "TCN",
    "transformer": "Transformer",
    "xgboost": "XGBoost",
    "never_shutdown": "Never shut down",
    "observed": "Observed (status quo)",
    "immediate": "Immediate",
    "static_break_even": "Static break-even",
    "ski_rental": "Ski-rental",
    "forecast_opt_median": "ablation: median estimand",
    "forecast_opt_nochance": "ablation: no chance constraint",
    "forecast_opt": "Forecast optimisation",
    "oracle": "Oracle",
}
SHORT = {"pelletizer-I": "Pel-I", "pelletizer-II": "Pel-II",
         "milling-I": "Mill-I", "milling-II": "Mill-II",
         "exhaust-fan-I": "Fan-I", "exhaust-fan-II": "Fan-II",
         "dpc-I": "DPC-I", "dpc-II": "DPC-II"}


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)
    info(f"wrote {path}")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Critical-difference diagram ───────────────────────────────────────────────

def figure_cd_diagram(res: dict, out_dir: str) -> None:
    """
    Demsar (2006) critical-difference diagram over the seed blocks.

    Reading it: models sit on a rank axis with the best (lowest mean rank) on
    the left; any two joined by a bar are NOT significantly different at the
    stated alpha, given the number of blocks.  With few blocks the critical
    difference is large and almost everything is joined, which is the
    figure honestly showing that the design cannot separate the models rather
    than the models being equal.
    """
    nem = res.get("nemenyi")
    fried = res.get("friedman_over_seeds")
    if not nem or not fried:
        return
    models = nem["models"]
    ranks = np.asarray(nem["mean_ranks"], float)
    cd = nem["critical_difference"]
    order = np.argsort(ranks)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    lo, hi = 0.5, len(models) + 0.5
    ax.set_xlim(lo, hi)                       # best (lowest rank) on the left
    # Headroom above the axis for the clique bars and the CD ruler, which are
    # drawn at y0 + 0.30 and would otherwise land outside the axes and collide
    # with the title.
    ax.set_ylim(0, 1.25)
    ax.axis("off")
    y0 = 0.78
    ax.plot([lo, hi], [y0, y0], color=INK, linewidth=1.2)
    for r in range(1, len(models) + 1):
        ax.plot([r, r], [y0, y0 + 0.03], color=INK, linewidth=1.0)
        ax.text(r, y0 + 0.06, str(r), ha="center", fontsize=8, color=INK_MUTED)

    # Model labels, alternating sides so they do not collide.
    for i, idx in enumerate(order):
        r = ranks[idx]
        left = i < len(order) / 2
        y = y0 - 0.12 - 0.11 * (i if left else len(order) - 1 - i)
        x_end = lo + 0.15 if left else hi - 0.15
        ax.plot([r, r], [y0, y], color=INK_MUTED, linewidth=0.9)
        ax.plot([r, x_end], [y, y], color=INK_MUTED, linewidth=0.9)
        # Text runs outward from the anchor, away from the plot, so the
        # connector never passes through the label.
        ax.text(x_end, y, f" {LABEL.get(models[idx], models[idx])} ({r:.2f}) ",
                ha="right" if left else "left", va="center", fontsize=8)

    # Cliques: maximal groups whose extreme ranks differ by less than CD.
    sorted_ranks = ranks[order]
    bars, i = [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
            j += 1
        if j > i:
            bars.append((sorted_ranks[i], sorted_ranks[j]))
        i += 1
    drawn = []
    for k, (a, b) in enumerate(bars):
        if any(a >= x and b <= y for x, y in drawn):
            continue
        drawn.append((a, b))
        ax.plot([a - 0.03, b + 0.03], [y0 + 0.10 + 0.05 * len(drawn)] * 2,
                color=CATEGORICAL[0], linewidth=3.2, solid_capstyle="butt")

    ax.plot([lo, lo + cd], [y0 + 0.30, y0 + 0.30], color=INK, linewidth=2.0)
    ax.text(lo + cd / 2, y0 + 0.33, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_title(
        f"Critical-difference diagram: STANDBY F1 ranked over "
        f"{nem['n_blocks']} seeds  (Friedman p = {fried['p_value']:.3g}, "
        f"Kendall's W = {fried['kendalls_w']:.2f})", pad=14)
    _save(fig, f"{out_dir}/fig_task11_cd_diagram.png")


# ── Forecasting: significance against effect size ─────────────────────────────

def figure_forecast_pairwise(tab: pd.DataFrame, res: dict, out_dir: str) -> None:
    """
    Panel A: every model's per-window duration error, with the seed spread.
    Panel B: statistical significance against economic magnitude, which is the
    figure that keeps a p-value of 1e-40 on a 0.2-second difference from being
    read as an important result.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    keys = list(res["mae_per_model"])
    order = sorted(keys, key=lambda k: res["mae_per_model"][k])
    x = np.arange(len(order))
    vals = [res["mae_per_model"][k] for k in order]
    colors = [NEUTRAL if k == "persistence" else CATEGORICAL[0] for k in order]
    b = ax.bar(x, vals, color=colors, zorder=3)
    for xi, k in enumerate(order):
        d = res["f1_per_model"][k]
        ax.text(xi, vals[xi] + 0.15, f"{vals[xi]:.2f}", ha="center", fontsize=7)
        if d.get("n", 0) > 1:
            ax.text(xi, 0.4, f"F1 {d['mean']:.3f}\n±{d['std']:.3f}",
                    ha="center", fontsize=6.5, color="white")
        else:
            ax.text(xi, 0.4, f"F1 {d.get('mean', float('nan')):.3f}",
                    ha="center", fontsize=6.5, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(k, k) for k in order], rotation=30, ha="right")
    ax.set_ylabel("per-window STANDBY-seconds MAE")
    ax.set_title("A  Duration error by model (seed mean)")
    _grid(ax)

    ax = axes[1]
    sig = tab["p_holm"] < 0.05
    econ = tab["economically_significant"]
    ax.axhline(0.05, color=INK, linestyle="--", linewidth=1.0)
    ax.axvline(res["thresholds"]["mae_s"], color=INK, linestyle=":", linewidth=1.0)
    ax.scatter(tab.loc[sig & econ, "mae_diff"].abs(),
               tab.loc[sig & econ, "p_holm"].clip(lower=1e-300),
               s=42, color=CATEGORICAL[1], zorder=3, label="real and material")
    ax.scatter(tab.loc[sig & ~econ, "mae_diff"].abs(),
               tab.loc[sig & ~econ, "p_holm"].clip(lower=1e-300),
               s=42, color=CATEGORICAL[0], zorder=3,
               label="detectable but immaterial")
    ax.scatter(tab.loc[~sig, "mae_diff"].abs(),
               tab.loc[~sig, "p_holm"].clip(lower=1e-300),
               s=42, color=NEUTRAL, zorder=3, label="not detectable")
    for _, r in tab.iterrows():
        ax.annotate(f"{LABEL.get(r['model_a'], r['model_a'])[:6]}/"
                    f"{LABEL.get(r['model_b'], r['model_b'])[:6]}",
                    (abs(r["mae_diff"]), max(r["p_holm"], 1e-300)),
                    textcoords="offset points", xytext=(4, 3), fontsize=5.5,
                    color=INK_MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("|difference in MAE| (seconds per window)")
    ax.set_ylabel("Holm-adjusted p-value (log scale)")
    ax.set_title("B  Statistical significance vs economic magnitude")
    ax.text(res["thresholds"]["mae_s"], ax.get_ylim()[1],
            " practical threshold", fontsize=6.5, va="top", color=INK_MUTED)
    ax.legend(fontsize=7, loc="lower right")
    _grid(ax, "both")

    fig.suptitle("Task 11A -- forecasting model comparison",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task11_forecast_pairwise.png")


# ── Decision layer: forest plot ───────────────────────────────────────────────

def figure_decision_ci(scenarios: dict, out_dir: str) -> None:
    """
    Forest plot of the annual saving of every policy, by scenario.

    An interval that crosses zero is drawn grey: on this machine most of them
    do, and that is the result.  The oracle's interval is the head-room, so a
    policy whose interval overlaps the oracle's is not distinguishable from
    optimal on this sample either -- both readings matter and both are visible.
    """
    if not scenarios:
        return
    names = list(scenarios)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.6),
                             sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, sc in zip(axes, names):
        rows = pd.DataFrame(scenarios[sc]["policies"])
        rows = rows[rows["policy"] != "observed"]
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows()):
            crosses = not r["ci_excludes_zero"]
            c = NEUTRAL if crosses else (CATEGORICAL[2] if r["savings_usd_yr"] > 0
                                         else CATEGORICAL[1])
            ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, linewidth=2.4,
                    solid_capstyle="round", zorder=3)
            ax.plot([r["savings_usd_yr"]], [yi], "o", color=c, markersize=5,
                    zorder=4)
        ax.axvline(0, color=INK, linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL.get(p, p) for p in rows["policy"]], fontsize=8)
        ax.set_xlabel("annual saving vs status quo (USD/yr)")
        ax.set_title(f"{sc}  (break-even "
                     f"{scenarios[sc]['break_even_s']/60:.0f} min)")
        _grid(ax, "x")
    fig.suptitle(
        f"Task 11B -- day-block bootstrap of the decision layer "
        f"({scenarios[names[0]]['n_days']} factory days, "
        f"{scenarios[names[0]]['n_episodes']} held-out episodes)",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task11_decision_ci.png")


# ── State layer ───────────────────────────────────────────────────────────────

def figure_state_ci(state: pd.DataFrame, label_seeds: dict, out_dir: str) -> None:
    """
    STANDBY hours per covered day with a one-week block-bootstrap interval, and
    beside it the spread produced by re-fitting the labeller at ten seeds.

    The two intervals answer different questions -- how much the machine's duty
    cycle varies week to week, and how much the answer depends on the model's
    random initialisation -- and the paper needs both, because a tight seed
    spread inside a wide sampling interval means the model is stable and the
    plant is not.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    m = list(state.index)
    x = np.arange(len(m))
    ax.bar(x, state["standby_h_per_day"], color=CATEGORICAL[0], zorder=3)
    ax.errorbar(x, state["standby_h_per_day"],
                yerr=[state["standby_h_per_day"] - state["ci_lo_h_per_day"],
                      state["ci_hi_h_per_day"] - state["standby_h_per_day"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
    for xi, (v, lo, hi) in enumerate(zip(state["standby_h_per_day"],
                                         state["ci_lo_h_per_day"],
                                         state["ci_hi_h_per_day"])):
        ax.text(xi, hi + 0.12, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(k, k) for k in m], rotation=30, ha="right")
    ax.set_ylabel("STANDBY hours per covered day")
    ax.set_title("A  Sampling interval (one-week moving blocks)")
    _grid(ax)

    ax = axes[1]
    if label_seeds:
        runs = pd.DataFrame(label_seeds["runs"])
        ax.plot(runs["seed"], runs["standby_hours"], "o-", color=CATEGORICAL[1],
                markersize=4)
        s = label_seeds["summary"]["standby_hours"]
        ax.axhline(s["mean"], color=INK, linewidth=1.0, linestyle="--")
        ax.fill_between([runs["seed"].min(), runs["seed"].max()],
                        s["mean"] - s["std"], s["mean"] + s["std"],
                        color=CATEGORICAL[1], alpha=0.15)
        ax.set_xlabel("labelling seed")
        ax.set_ylabel("STANDBY hours (whole record)")
        ax.set_title(f"B  Labelling-seed spread\n"
                     f"{s['mean']:.1f} ± {s['std']:.1f} h over {s['n']} seeds")
        _grid(ax)
    else:
        ax.axis("off")

    fig.suptitle("Task 11C -- intervals on the state-time totals",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task11_state_ci.png")


# ── Power ─────────────────────────────────────────────────────────────────────

def figure_power(cross: dict, out_dir: str) -> None:
    """
    What the cross-machine design could have detected.

    The horizontal line is alpha; the annotation is the p-value floor, which at
    n = 4 sits ABOVE alpha -- so the n = 4 curve is pinned at zero power by
    arithmetic, not by the data.  This is the figure to point at when a
    reviewer asks whether the non-significant Friedman means the models are
    equivalent.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    effects = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    ax = axes[0]
    for i, n in enumerate((4, 8, 16, 30)):
        p = [wilcoxon_power(n, e, n_sim=1200) for e in effects]
        ax.plot(effects, p, "o-", markersize=3.5,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=f"n = {n} pairs")
    ax.axhline(0.80, color=INK, linestyle="--", linewidth=1.0)
    ax.text(effects[-1], 0.81, "80% power", ha="right", fontsize=7)
    ax.set_xlabel("true effect (paired difference, in standard deviations)")
    ax.set_ylabel("power at alpha = 0.05")
    ax.set_title("A  Paired Wilcoxon: power against the number of machines")
    ax.text(0.28, 0.06,
            f"at n = 4 the smallest attainable p is "
            f"{wilcoxon_p_floor(4):.3f} > 0.05:\nno effect can be significant",
            fontsize=7, color=CATEGORICAL[1])
    ax.legend(fontsize=7)
    _grid(ax, "both")

    ax = axes[1]
    for i, n in enumerate((4, 5, 8, 15)):
        p = [friedman_power(n, 4, e, n_sim=600) for e in effects]
        ax.plot(effects, p, "o-", markersize=3.5,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=f"{n} blocks")
    ax.axhline(0.80, color=INK, linestyle="--", linewidth=1.0)
    ax.set_xlabel("true effect (in standard deviations)")
    ax.set_ylabel("power at alpha = 0.05")
    ax.set_title("B  Friedman with 4 treatments: power against blocks")
    ax.legend(fontsize=7)
    _grid(ax, "both")

    fig.suptitle("Task 11D -- what these designs could have detected",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task11_power.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = "pelletizer-I") -> None:
    section("PHASE V FIGURES")
    out_dir = ensure_dir(FIG_DIR)
    t11 = f"{PHASE5_DIR}/task11"
    t12 = f"{PHASE5_DIR}/task12/{machine}"

    res = _load_json(f"{t11}/{machine}/forecasting_tests.json")
    if res:
        figure_cd_diagram(res, out_dir)
        pair_csv = f"{t11}/{machine}/forecasting_pairwise.csv"
        if os.path.exists(pair_csv):
            figure_forecast_pairwise(pd.read_csv(pair_csv), res, out_dir)

    scenarios = {}
    for sc in ("S1_low", "S2_moderate", "S3_high"):
        d = _load_json(f"{t11}/{machine}/decision_bootstrap_{sc}.json")
        if d:
            scenarios[sc] = d
    figure_decision_ci(scenarios, out_dir)

    state_csv = f"{t11}/state_block_bootstrap.csv"
    if os.path.exists(state_csv):
        figure_state_ci(pd.read_csv(state_csv, index_col=0),
                        _load_json(f"{t12}/labelling/labelling_seeds.json"),
                        out_dir)

    cross = _load_json(f"{t11}/cross_machine_tests.json")
    figure_power(cross or {}, out_dir)

    section("PHASE V FIGURES COMPLETE")


if __name__ == "__main__":
    main()
