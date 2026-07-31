"""
============================================================
PHASE II RESULT FIGURES  --  experiments/make_figures.py
============================================================

PURPOSE
-------
Render the figures that present Phase II's three results tables.
Reads only the CSV/JSON written by the experiment scripts, so it
can be re-run without recomputing anything.

    python -m experiments.make_figures

DESIGN NOTES
------------
Palettes are taken from a validated categorical set and a
validated single-hue ordinal ramp; both were checked for
colour-vision-deficiency separation and lightness banding rather
than chosen by eye.  Consequences for the code below:

  * Categorical hues are assigned in FIXED slot order and never
    cycled.  A method keeps its colour across every figure, so
    'the orange one' means the same thing in Fig. 2 and Fig. 3.
  * The four machine states are an ORDERED quantity (increasing
    load), so they use a one-hue light-to-dark ramp rather than
    four unrelated hues -- an ordered variable encoded with
    categorical colour is a misencoding.
  * Every bar carries a direct value label.  Three of the
    categorical slots sit below 3:1 contrast on a white surface,
    which is legal only when the value is also readable as text.
  * No chart has two y-scales.  Where two measures of different
    units must be compared, they become separate panels.
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
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.ticker import MaxNLocator            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import PHASE2_DIR, DEFAULT_MACHINE, ensure_dir, info, section  # noqa: E402


# ── Palette (validated; see module docstring) ─────────────────────────────────

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STATE_RAMP = {                       # ordinal, one hue, light -> dark
    "OFF": "#86b6ef",
    "STANDBY": "#3987e5",
    "WORKING": "#256abf",
    "PEAK_LOAD": "#104281",
}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
PROPOSED_EDGE = "#0b0b0b"            # emphasis ring on the proposed method

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _label_bars(ax, bars, values, fmt="{:.2f}", horizontal=False, pad=0.01,
                errors=None):
    """
    Direct value labels -- required by the palette's relief rule.

    `errors` lifts each label clear of its error bar; without it the text
    lands on top of the whisker and both become unreadable.
    """
    span = (ax.get_xlim()[1] - ax.get_xlim()[0]) if horizontal else \
           (ax.get_ylim()[1] - ax.get_ylim()[0])
    errs = errors if errors is not None else [0.0] * len(bars)
    for b, v, e in zip(bars, values, errs):
        if not np.isfinite(v):
            continue
        e = e if np.isfinite(e) else 0.0
        if horizontal:
            ax.text(b.get_width() + e + span * pad,
                    b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center", ha="left",
                    fontsize=7.5, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + span * pad,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=7.5, color=INK)


def _save(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)
    info(f"wrote {path}")


# ── Task 3 ────────────────────────────────────────────────────────────────────

def figure_task3(machine: str, out_dir: str, top_n: int = 12):
    """
    Feature importance under three independent measures.

    Grouped horizontal bars: one row per feature, one bar per measure.
    Horizontal because feature names are long and would otherwise be
    rotated; grouped rather than stacked because the three measures are
    alternative estimates of the same quantity, not components of it --
    stacking would imply they sum to something meaningful.
    """
    src = f"{PHASE2_DIR}/task3/{machine}"
    path = f"{src}/importance_standby_vs_productive.csv"
    if not os.path.exists(path):
        info(f"skip Task 3 figure -- {path} not found")
        return

    df = pd.read_csv(path, index_col=0).head(top_n).iloc[::-1]
    measures = [("mutual_information", "Mutual information"),
                ("xgb_gain", "XGBoost gain"),
                ("shap_mean_abs", "SHAP (mean |value|)")]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6),
                             gridspec_kw={"width_ratios": [2.15, 1]})

    # -- panel A: per-feature -------------------------------------------
    ax = axes[0]
    y = np.arange(len(df))
    h = 0.26
    for i, (col, label) in enumerate(measures):
        off = (i - 1) * h
        bars = ax.barh(y + off, df[col].to_numpy(), height=h * 0.9,
                       color=CATEGORICAL[i], label=label, zorder=3)
        if col == "shap_mean_abs":
            _label_bars(ax, bars, df[col].to_numpy(), "{:.3f}", horizontal=True)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("Normalised importance (each measure sums to 1)")
    ax.set_title("A   Feature importance: STANDBY vs productive", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, max(df[[m[0] for m in measures]].to_numpy().max() * 1.22, 0.01))
    _grid(ax, axis="x")

    # highlight the two physics features the five-method proof relies on
    for tick in ax.get_yticklabels():
        if tick.get_text() in ("power_factor", "q_p_ratio"):
            tick.set_fontweight("bold")
            tick.set_color(INK)

    # -- panel B: per-group ---------------------------------------------
    gpath = f"{src}/importance_by_group_standby_vs_productive.csv"
    ax = axes[1]
    if os.path.exists(gpath):
        g = pd.read_csv(gpath, index_col=0).iloc[::-1]
        y = np.arange(len(g))
        for i, (col, label) in enumerate(measures):
            off = (i - 1) * h
            ax.barh(y + off, g[col].to_numpy(), height=h * 0.9,
                    color=CATEGORICAL[i], label=label, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([s.replace("_", " ") for s in g.index], fontsize=8)
        ax.set_xlabel("Summed importance")
        ax.set_title("B   By feature group", loc="left")
        ax.set_xlim(0, g[[m[0] for m in measures]].to_numpy().max() * 1.18)
        _grid(ax, axis="x")

    fig.suptitle(
        "Which features explain the inferred STANDBY / productive split "
        f"({machine})",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.02)
    _save(fig, f"{out_dir}/fig_task3_feature_importance.png")


def figure_task3_states(machine: str, out_dir: str):
    """
    Per-state distribution of the two physics features, as a grouped bar
    with error bars (mean +/- 1 sd).

    This is the figure that makes the physics argument visible: if the
    STANDBY bar for power factor does not sit far below the productive
    ones, Methods 1 and 2 of the five-method proof have no basis.
    """
    path = f"{PHASE2_DIR}/task3/{machine}/feature_stats_by_state.csv"
    if not os.path.exists(path):
        info(f"skip Task 3 state figure -- {path} not found")
        return

    df = pd.read_csv(path, header=[0, 1], index_col=0)
    states = [s for s in STATE_RAMP if s in df.index]
    feats = [("power_factor", "Power factor  P/S", "{:.3f}"),
             ("q_p_ratio", "Q/P ratio  log(1+|Q|/P)", "{:.2f}"),
             ("active_power", "Active power (W)", "{:.0f}")]

    fig, axes = plt.subplots(1, len(feats), figsize=(10.5, 3.4))
    for ax, (col, title, fmt) in zip(axes, feats):
        if (col, "mean") not in df.columns:
            continue
        m = [df.loc[s, (col, "mean")] for s in states]
        sd = [df.loc[s, (col, "std")] for s in states]
        x = np.arange(len(states))
        bars = ax.bar(x, m, yerr=sd, capsize=3, width=0.62,
                      color=[STATE_RAMP[s] for s in states],
                      error_kw=dict(ecolor=INK_MUTED, lw=1), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
        ax.margins(y=0.28)
        _label_bars(ax, bars, m, fmt, errors=sd)
        _grid(ax)
        ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.suptitle(
        "Physics features by inferred state (mean ± 1 sd) — "
        "STANDBY is separated by power factor, not by power alone",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.05)
    _save(fig, f"{out_dir}/fig_task3_state_physics.png")


# ── Task 4 ────────────────────────────────────────────────────────────────────

def figure_task4(machine: str, out_dir: str):
    """
    State-detection method comparison across three metrics.

    Three panels rather than one chart with three y-scales: flicker rate
    is a percentage where LOW is good, physics score is a 0-1 fraction
    where HIGH is good, and schedule agreement is a percentage where high
    is good.  Overlaying them on shared axes would be the dual-axis
    anti-pattern and would also hide the sign flip on flicker.
    """
    path = f"{PHASE2_DIR}/task4/{machine}/comparison_table.csv"
    if not os.path.exists(path):
        info(f"skip Task 4 figure -- {path} not found")
        return

    df = pd.read_csv(path, index_col=0)
    df = df[df.index != "_reference_export"]
    if "flicker_rate_pct" not in df.columns:
        info("skip Task 4 figure -- table has no metrics (all methods failed?)")
        return
    df = df.dropna(subset=["flicker_rate_pct"])

    order = [k for k in ["kmeans", "dbscan", "spectral", "gmm", "gmm_hmm",
                         "gmm_hmm_mindwell"] if k in df.index]
    df = df.loc[order]
    # Short tick labels: the full method names collide at six categories.
    # "(proposed)" is carried by the outline ring and the caption instead.
    SHORT = {
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "spectral": "Spectral",
        "gmm": "GMM",
        "gmm_hmm": "GMM-HMM",
        "gmm_hmm_mindwell": "+min-dwell",
    }
    colors = [CATEGORICAL[i] for i in range(len(df))]
    labels = [SHORT.get(k, str(k)) for k in df.index]
    x = np.arange(len(df))

    panels = [
        ("flicker_rate_pct", "A   Flicker, all states", "% of runs < 10 s",
         "{:.1f}", "lower is better"),
        ("flicker_standby_binary_pct", "B   Flicker, STANDBY vs rest",
         "% of runs < 10 s", "{:.1f}", "lower is better — decision-relevant"),
        ("physics_score", "C   Physics compliance", "fraction of checks passed",
         "{:.2f}", "higher is better"),
        ("standby_hours", "D   STANDBY hours found", "hours in 64 h evaluated",
         "{:.2f}", "reference export = 4.18 h"),
    ]

    # 2x2 rather than 1x4: six category labels need roughly twice the panel
    # width to sit horizontally without colliding, and rotating them instead
    # forces the reader to tilt their head at every panel.
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
    axes = axes.ravel()
    for ax, (col, title, ylab, fmt, sense) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].to_numpy(float)
        bars = ax.bar(x, vals, width=0.64, color=colors, zorder=3)
        # emphasis ring on the full proposed method
        for b, key in zip(bars, df.index):
            if key == "gmm_hmm_mindwell":
                b.set_edgecolor(PROPOSED_EDGE)
                b.set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8, rotation=0, ha="center")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(f"{title}\n{sense}", loc="left", fontsize=8.5)
        ax.margins(y=0.20)
        _label_bars(ax, bars, vals, fmt)
        _grid(ax)

    fig.suptitle(
        "Task 4 — unsupervised state detection compared on identical data, "
        f"features and metrics ({machine})",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.03)
    fig.text(0.02, 0.985,
             "Outlined bar = full proposed method.  GMM = i.i.d. per-sample "
             "assignment; GMM-HMM = same emissions, Viterbi decode.  Panel D: "
             "a method finding far more or far less STANDBY than the rest is "
             "mis-partitioning, whatever its other scores.",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.subplots_adjust(wspace=0.28, hspace=0.42)
    _save(fig, f"{out_dir}/fig_task4_state_detection.png")


# ── Task 5 ────────────────────────────────────────────────────────────────────

def figure_task5(machine: str, out_dir: str):
    """
    Forecasting model comparison: sequence quality and duration quality.

    Two panels because the two metrics answer different questions and one
    does not imply the other -- a model can place STANDBY steps correctly
    per-step (good F1) yet mis-total the interval the decision layer
    consumes (bad MAE).  Error bars are +/- 1 sd across seeds where more
    than one seed was run.
    """
    path = f"{PHASE2_DIR}/task5/{machine}/comparison_table.csv"
    if not os.path.exists(path):
        info(f"skip Task 5 figure -- {path} not found")
        return

    df = pd.read_csv(path, index_col=0)
    if "f1_STANDBY" not in df.columns:
        info("skip Task 5 figure -- table incomplete")
        return
    df = df.dropna(subset=["f1_STANDBY"])

    order = [k for k in ["xgboost", "vanilla_lstm", "gru", "tcn",
                         "transformer", "seq2seq_lstm"] if k in df.index]
    df = df.loc[order]
    SHORT = {
        "xgboost": "XGBoost",
        "vanilla_lstm": "Vanilla\nLSTM",
        "gru": "GRU",
        "tcn": "TCN",
        "transformer": "Trans-\nformer",
        "seq2seq_lstm": "Seq2Seq\nLSTM",
    }
    colors = [CATEGORICAL[i] for i in range(len(df))]
    labels = [SHORT.get(k, str(k)) for k in df.index]
    x = np.arange(len(df))

    panels = [
        ("f1_STANDBY", "A   STANDBY F1 (per-step)", "F1", "{:.3f}",
         "higher is better"),
        ("standby_mae_s", "B   STANDBY duration MAE", "seconds", "{:.1f}",
         "lower is better"),
        ("standby_rmse_s", "C   STANDBY duration RMSE", "seconds", "{:.1f}",
         "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (col, title, ylab, fmt, sense) in zip(axes, panels):
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(float)
        err = (df[f"{col}_std"].to_numpy(float)
               if f"{col}_std" in df.columns else None)
        bars = ax.bar(x, vals, yerr=err, capsize=3, width=0.64, color=colors,
                      error_kw=dict(ecolor=INK_MUTED, lw=1), zorder=3)
        for b, k in zip(bars, df.index):
            if k == "seq2seq_lstm":
                b.set_edgecolor(PROPOSED_EDGE)
                b.set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=0, ha="center")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(f"{title}\n{sense}", loc="left", fontsize=9)
        ax.margins(y=0.22)
        _label_bars(ax, bars, vals, fmt, errors=err)
        _grid(ax)

    fig.suptitle(
        "Task 5 — STANDBY-duration forecasting baselines under one protocol "
        f"({machine})",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.09)
    fig.text(0.02, 0.985,
             "Outlined bar = model proposed by the project. Identical windows, "
             "features, chronological split and class weights for every model; "
             "single seed.",
             fontsize=8, color=INK_MUTED, ha="left")
    _save(fig, f"{out_dir}/fig_task5_forecasting.png")


def figure_task4_transitions(machine: str, out_dir: str):
    """
    The HMM transition matrix as a heatmap.

    A sequential single-hue ramp on log10 probability: the matrix is
    dominated by four self-transition entries near 1 while every
    off-diagonal entry is 1e-3 or smaller, so a linear ramp renders as a
    black diagonal on an otherwise blank grid and shows nothing.  Every
    cell is also printed as text, which is the honest way to present a
    4x4 matrix regardless of the colour scale.
    """
    path = f"{PHASE2_DIR}/task4/{machine}/task4_results.json"
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    hmm = res.get("results", {}).get("gmm_hmm", {})
    T = hmm.get("transmat")
    if not T:
        info("skip transition figure -- no HMM transition matrix in results")
        return

    T = np.asarray(T, dtype=float)
    mapping = hmm.get("cluster_to_state", {})
    names = [mapping.get(str(i), f"cluster {i}") for i in range(len(T))]

    order = [i for s in STATE_RAMP for i, n in enumerate(names) if n == s]
    if len(order) == len(T):
        T = T[np.ix_(order, order)]
        names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(np.log10(np.clip(T, 1e-8, 1.0)), cmap="Blues",
                   vmin=-8, vmax=0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("state at t+1")
    ax.set_ylabel("state at t")
    ax.set_title("Learned HMM transition matrix", loc="left")

    for i in range(len(names)):
        for j in range(len(names)):
            v = T[i, j]
            # 4 dp, not 3: a self-transition of 0.99994 rounds to "1.000" at
            # 3 dp, which reads as an absorbing state that can never be left.
            txt = f"{v:.4f}" if v >= 1e-4 else f"{v:.1e}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if np.log10(max(v, 1e-8)) > -1.2 else INK)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log$_{10}$ transition probability", fontsize=8)
    ax.grid(False)
    _save(fig, f"{out_dir}/fig_task4_transition_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = DEFAULT_MACHINE):
    section("PHASE II -- FIGURES")
    out_dir = ensure_dir(f"{PHASE2_DIR}/figures/{machine}")
    figure_task3(machine, out_dir)
    figure_task3_states(machine, out_dir)
    figure_task4(machine, out_dir)
    figure_task4_transitions(machine, out_dir)
    figure_task5(machine, out_dir)
    section("FIGURES COMPLETE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase II result figures")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    a = ap.parse_args()
    main(a.machine)
