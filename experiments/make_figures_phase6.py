"""
============================================================
PHASE VI FIGURES  --  experiments/make_figures_phase6.py
============================================================

PURPOSE
-------
Two jobs in one module, because they share a stylesheet:

  * the explainability figures Phase VI produces (Tasks 13A-13C
    and the 13E ablation)
  * publication-quality re-renders of the five Phase V result
    figures (Task 13D)

    python -m experiments.make_figures_phase6

WHAT "PUBLICATION QUALITY" CHANGES, AND WHY EACH CHANGE
-------------------------------------------------------
Phase V's figures were built to be read on screen while the tests
were being written.  A journal figure has different constraints,
and the differences here are deliberate rather than cosmetic:

  * FIGURE WIDTHS ARE COLUMN WIDTHS.  Every figure is sized to a
    single (3.5 in) or double (7.16 in) IEEE column, so it is placed
    at 100% scale and the 8 pt type in it stays 8 pt on the page.  A
    figure drawn at 12 in and scaled down is how a caption ends up
    larger than its axis labels.
  * NO SUPTITLES.  A rendered title duplicates the caption and
    cannot be edited in proof.  Panel letters stay, because the
    caption refers to them.
  * PDF AND PNG.  Vector for the manuscript, raster for the notes
    and for anything that has to be pasted into a slide.
  * COLOURS SURVIVE GREYSCALE AND CVD.  The categorical set is
    checked for monotone luminance so a printed figure is still
    ordered, and no result is encoded by hue alone -- an interval
    that crosses zero is grey AND unfilled, so the distinction does
    not vanish on a monochrome printer.

============================================================
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap          # noqa: E402
from matplotlib.patches import FancyArrowPatch                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ensure_dir, info, section        # noqa: E402
from src.stats import wilcoxon_power, wilcoxon_p_floor          # noqa: E402

PHASE5_DIR = "outputs/phase5"
PHASE6_DIR = "outputs/phase6"
FIG_DIR = f"{PHASE6_DIR}/figures"

# IEEE two-column geometry, in inches.
COL, DOUBLE = 3.5, 7.16

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e3e2df", "#9a9894"
POS, NEG = "#1baf7a", "#eb6834"

# Diverging map for the SHAP beeswarm: low feature value to high.  Blue-to-red
# is conventional in the SHAP literature, so a reader who knows the library
# reads this without consulting the legend.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shapbr", ["#2a78d6", "#a0a0a8", "#eb6834"])
# Sequential map for the transition heatmap: white at zero so the forbidden
# entries read as absent rather than as "the lowest of four shades".
TRANS_CMAP = LinearSegmentedColormap.from_list(
    "trans", ["#ffffff", "#cfe0f5", "#7fb0e6", "#2a78d6", "#123f77"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,       # embed real glyphs, not paths
})

LABEL = {
    "persistence": "Persistence (untrained)",
    "seq2seq_lstm": "Seq2Seq LSTM", "vanilla_lstm": "Vanilla LSTM",
    "gru": "GRU", "tcn": "TCN", "transformer": "Transformer",
    "xgboost": "XGBoost",
    "never_shutdown": "Never shut down", "observed": "Status quo",
    "immediate": "Immediate", "static_break_even": "Static break-even",
    "ski_rental": "Ski-rental",
    "forecast_opt_median": "Forecast opt. (median estimand)",
    "forecast_opt_nochance": "Forecast opt. (no chance constr.)",
    "forecast_opt": "Forecast opt. (proposed)", "oracle": "Oracle",
}
SHORT = {"pelletizer-I": "Pel-I", "pelletizer-II": "Pel-II",
         "milling-I": "Mill-I", "milling-II": "Mill-II",
         "exhaust-fan-I": "Fan-I", "exhaust-fan-II": "Fan-II",
         "dpc-I": "DPC-I", "dpc-II": "DPC-II"}
# The four machines whose physics battery passes 5/5 (Phase IV, Task 8).
MOTOR_MACHINES = {"pelletizer-I", "pelletizer-II", "milling-I", "milling-II"}

PRETTY_FEATURE = {
    "elapsed_log": "log(1 + elapsed idle)",
    "onset_hour": "hour at episode onset",
    "onset_dow": "day of week at onset",
    "is_weekend": "onset is a weekend",
    "cur_hour_sin": "sin(current hour)",
    "cur_hour_cos": "cos(current hour)",
    "cur_is_open": "plant currently open",
    "cur_is_weekend": "currently a weekend",
    "prev_productive_log": "log(1 + previous run length)",
    "prev_productive_power_w": "previous run mean power",
}


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, stem: str) -> None:
    """Write both a vector and a raster copy of one figure."""
    ensure_dir(FIG_DIR)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_DIR}/{stem}.{ext}")
    plt.close(fig)
    info(f"wrote {FIG_DIR}/{stem}.{{pdf,png}}")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _beeswarm(ax, shap_values, X, names, order, max_points=3000, seed=0):
    """
    A SHAP beeswarm: one row per feature, one dot per observation, positioned
    by its Shapley value and coloured by the feature's own value.

    Drawn here rather than with `shap.summary_plot` because that function
    creates and styles its own figure, which cannot then be placed in a
    multi-panel layout or made to match the rest of the paper's figures.
    The vertical scatter is a density-proportional jitter, so the width of a
    row reads as the number of observations at that attribution.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    sel = rng.choice(n, size=min(max_points, n), replace=False)

    for row, j in enumerate(order):
        v = shap_values[sel, j]
        f = X[sel, j].astype(float)
        lo, hi = np.percentile(f, [1, 99])
        c = np.clip((f - lo) / max(hi - lo, 1e-12), 0, 1)

        # Density-proportional jitter: bin the attributions, spread each bin
        # over a height proportional to how full it is.
        counts, edges = np.histogram(v, bins=40)
        idx = np.clip(np.digitize(v, edges) - 1, 0, len(counts) - 1)
        dens = counts[idx] / max(counts.max(), 1)
        y = row + rng.uniform(-1, 1, size=len(v)) * 0.32 * dens

        ax.scatter(v, y, c=c, cmap=SHAP_CMAP, s=2.0, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=3)

    ax.axvline(0, color=INK, linewidth=0.7, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY_FEATURE.get(names[j], names[j]) for j in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.invert_yaxis()
    _grid(ax, "x")


# ══════════════════════════════════════════════════════════════════════════════
# Task 13A -- forecaster attribution
# ══════════════════════════════════════════════════════════════════════════════

def figure_decision_shap(machine: str, source: str = "phase4") -> None:
    """
    Panel A: the beeswarm.  Panel B: SHAP against the split-gain score it
    replaces, which is the figure that shows WHY the 94.5% calendar claim had
    to be restated -- gain and Shapley rank these features differently, and
    the two features that move most are the two the claim rested on.
    """
    base = f"{PHASE6_DIR}/task13a/{machine}"
    tbl_path = f"{base}/shap_decision_forecaster_{source}.csv"
    raw_path = f"{base}/shap_raw_decision_{source}.npz"
    if not (os.path.exists(tbl_path) and os.path.exists(raw_path)):
        return
    tbl = pd.read_csv(tbl_path, index_col=0)
    raw = np.load(raw_path, allow_pickle=True)
    names = [str(s) for s in raw["feature_names"]]
    shap_values, X = raw["shap_values"], raw["X"]

    order = [names.index(f) for f in tbl.index]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    _beeswarm(ax, shap_values, X, names, order)
    # Symmetric log: attributions run from -3e4 to +1e5 seconds but the bulk
    # sits inside +-1e3, so a linear axis would compress every feature except
    # the two calendar ones into a single vertical line at zero.
    ax.set_xscale("symlog", linthresh=1000, linscale=0.6)
    ax.set_xticks([-1e5, -1e4, -1e3, 0, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["$-10^5$", "$-10^4$", "$-10^3$", "0",
                        "$10^3$", "$10^4$", "$10^5$"])
    ax.set_xlabel("SHAP value (seconds of remaining idle time, symlog)")
    ax.set_title("A  Per-prediction attribution", loc="left")
    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP)
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.035, ticks=[0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("feature value", fontsize=7)
    cb.outline.set_linewidth(0.5)

    ax = axes[1]
    lim = max(tbl["gain_share"].max(), tbl["shap_share"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color=NEUTRAL, linewidth=0.7,
            linestyle="--", zorder=2)          # gain == SHAP would sit here
    colours = {"factory calendar": CATEGORICAL[0],
               "machine history": CATEGORICAL[1],
               "elapsed idle time": CATEGORICAL[2]}
    for grp, sub in tbl.groupby("group"):
        ax.scatter(sub["gain_share"], sub["shap_share"], s=26,
                   color=colours.get(grp, NEUTRAL), label=grp, zorder=3)
    for f, r in tbl.iterrows():
        if max(r["gain_share"], r["shap_share"]) > 0.05:
            ax.annotate(f, (r["gain_share"], r["shap_share"]), fontsize=6,
                        textcoords="offset points", xytext=(4, 3),
                        color=INK_MUTED)
    ax.set_xlabel("share of split gain")
    ax.set_ylabel("share of mean |SHAP|")
    ax.set_title("B  Shapley against split gain", loc="left")
    ax.legend(loc="upper left")
    _grid(ax, "both")

    fig.tight_layout()
    _save(fig, f"fig_task13a_decision_shap_{source}")


def figure_window_shap(machine: str) -> None:
    """
    The window forecaster, aggregated two ways.

    Panel A ranks the measured channels; panel B asks whether the tree uses a
    window's LEVEL, its TREND or its VARIABILITY, which is the question the
    rolling-statistic feature group was added to answer.  Panel B is the one
    that explains Phase V's persistence result: if almost half the
    attribution sits on the final observed value, the model is a refinement of
    "nothing changes" and cannot beat it by much.
    """
    base = f"{PHASE6_DIR}/task13a/{machine}"
    chan_path = f"{base}/shap_window_by_channel.csv"
    summ_path = f"{base}/shap_window_by_summary.csv"
    if not (os.path.exists(chan_path) and os.path.exists(summ_path)):
        return
    chan = pd.read_csv(chan_path, index_col=0).head(12)
    summ = pd.read_csv(summ_path, index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.8),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    y = np.arange(len(chan))[::-1]
    ax.barh(y, chan["shap_share"], color=CATEGORICAL[0], height=0.62,
            label="mean |SHAP|", zorder=3)
    ax.barh(y - 0.0, chan["gain_share"], color="none", height=0.62,
            edgecolor=CATEGORICAL[1], linewidth=0.9, label="split gain",
            zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(chan.index)
    ax.set_xlabel("share of total attribution")
    ax.set_title("A  By measured channel", loc="left")
    ax.legend(loc="lower right")
    _grid(ax, "x")

    ax = axes[1]
    x = np.arange(len(summ))
    ax.bar(x, summ["shap_share"], color=CATEGORICAL[0], width=0.6, zorder=3)
    for xi, v in enumerate(summ["shap_share"]):
        ax.text(xi, v + 0.01, f"{v*100:.0f}%", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in summ.index], fontsize=6.5)
    ax.set_ylabel("share of mean |SHAP|")
    ax.set_ylim(0, max(summ["shap_share"]) * 1.25)
    ax.set_title("B  By window summary", loc="left")
    _grid(ax)

    fig.tight_layout()
    _save(fig, "fig_task13a_window_shap")


# ══════════════════════════════════════════════════════════════════════════════
# Task 13B -- state classifier attribution
# ══════════════════════════════════════════════════════════════════════════════

def figure_state_shap(machine: str, source: str = "phase4") -> None:
    """
    The STANDBY-vs-productive split, which is the one the physics argument is
    about.

    Panel A is the beeswarm on the energised rows; panel B puts the three
    independent importance measures beside each other, because agreement
    across measures with different failure modes is what makes the ranking
    worth quoting -- mutual information ignores redundancy, gain is biased
    toward high-cardinality splits, and SHAP inherits the surrogate's biases.
    """
    base = f"{PHASE6_DIR}/task13b/{machine}/{source}"
    tbl_path = f"{base}/importance_standby_vs_productive.csv"
    raw_path = f"{base}/shap_raw_standby_vs_productive.npz"
    if not os.path.exists(tbl_path):
        return
    tbl = pd.read_csv(tbl_path, index_col=0).sort_values("shap_mean_abs",
                                                         ascending=False)
    top = tbl.head(10)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    if os.path.exists(raw_path):
        raw = np.load(raw_path, allow_pickle=True)
        names = [str(s) for s in raw["feature_names"]]
        sv, X = raw["shap_values"], raw["X"]
        if sv.ndim == 3:                    # (n, f, classes) -> the STANDBY column
            classes = [str(c) for c in raw["class_names"]]
            k = classes.index("STANDBY") if "STANDBY" in classes else -1
            sv = sv[:, :, k]
        order = [names.index(f) for f in top.index if f in names]
        _beeswarm(ax, sv, X, names, order)
        ax.set_xlabel("SHAP value (log-odds of STANDBY)")
    ax.set_title("A  STANDBY vs. productive, energised rows", loc="left")

    ax = axes[1]
    measures = [("mutual_information", "mutual information"),
                ("xgb_gain", "split gain"), ("shap_mean_abs", "mean |SHAP|")]
    y = np.arange(len(top))[::-1]
    h = 0.26
    for i, (col, lab) in enumerate(measures):
        share = top[col] / max(tbl[col].sum(), 1e-12)
        ax.barh(y + (1 - i) * h, share, height=h, color=CATEGORICAL[i],
                label=lab, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=6.5)
    ax.set_xlabel("share of that measure's total")
    ax.set_title("B  Three measures, one ranking", loc="left")
    ax.legend(loc="lower right")
    _grid(ax, "x")

    fig.tight_layout()
    _save(fig, f"fig_task13b_state_shap_{source}")


# ══════════════════════════════════════════════════════════════════════════════
# Task 13C -- transition structure
# ══════════════════════════════════════════════════════════════════════════════

def figure_transition_matrix(machine: str) -> None:
    """
    The learned matrix, the realised one, and what the dwell constraint does
    to it.

    Cells are annotated with the probability and, on the realised panels, the
    event count: 1.1e-4 on a five-million-row record is 550 events, and a
    reader cannot tell whether a small number is a rounding artefact or a real
    population without it.  The colour scale is logarithmic because the
    diagonal is ~0.99 and the off-diagonal ~1e-4; on a linear scale every
    off-diagonal cell would be the same white.
    """
    res = _load_json(f"{PHASE6_DIR}/task13c/task13c_results.json")
    if not res or machine not in res:
        return
    r = res[machine]
    panels = [("learned", "A  HMM transition matrix (learned)"),
              ("realised_pre_min_dwell", "B  Realised, before minimum dwell"),
              ("realised_final", "C  Realised, after minimum dwell")]
    panels = [(k, t) for k, t in panels if k in r]

    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 2.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, panels):
        m = pd.DataFrame(r[key]["matrix"]).T
        states = r[key]["states"]
        m = m.loc[states, states]
        M = m.to_numpy()

        shown = np.where(M > 0, np.log10(np.maximum(M, 1e-9)), np.nan)
        ax.imshow(shown, cmap=TRANS_CMAP, vmin=-9, vmax=0, aspect="equal")

        counts = {k2: v for k2, v in (r[key].get("forbidden") or {}).items()}
        for i, a in enumerate(states):
            for j, b in enumerate(states):
                v = M[i, j]
                # Four decimals above the linear threshold: at three, every
                # self-transition rounds to 1.000 and the panel loses the one
                # quantity it exists to show.
                txt = "0" if v == 0 else (f"{v:.4f}" if v >= 0.001
                                          else f"{v:.0e}".replace("e-0", "e-"))
                cnt = counts.get(f"{a}->{b}", {}).get("count")
                if cnt is not None and v > 0:
                    txt += f"\n({cnt})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                        color="white" if v > 0.05 else INK)

        ax.set_xticks(range(len(states)))
        ax.set_yticks(range(len(states)))
        ax.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=6)
        ax.set_yticklabels(states, fontsize=6)
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("from state")
        ax.set_xlabel("to state")

    fig.tight_layout()
    _save(fig, "fig_task13c_transition_matrix")


def figure_transition_graph(machine: str) -> None:
    """
    The same object as a state graph, which is what a reader looks at first.

    Only transitions above a visibility floor are drawn, and the floor is
    stated on the figure: an edge that is not there means "below 1e-4 of the
    outgoing mass", not "impossible".  The two transitions the state
    definition rules out are drawn as absent arcs with their observed counts,
    so the claim is legible as a claim.
    """
    res = _load_json(f"{PHASE6_DIR}/task13c/task13c_results.json")
    if not res or machine not in res:
        return
    r = res[machine]
    key = "realised_final" if "realised_final" in r else "learned"
    m = pd.DataFrame(r[key]["matrix"]).T
    states = r[key]["states"]
    m = m.loc[states, states]
    dwell = r[key]["expected_dwell_s"]

    pos = {s: (np.cos(a), np.sin(a)) for s, a in
           zip(states, np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi,
                                   len(states), endpoint=False))}

    fig, ax = plt.subplots(figsize=(COL, COL * 0.95))
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.55, 1.6)
    ax.axis("off")

    floor = 1e-4
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            p = float(m.loc[a, b])
            if i == j or p < floor:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), connectionstyle="arc3,rad=0.22",
                arrowstyle="-|>", mutation_scale=8,
                linewidth=0.5 + 2.2 * min(1.0, (np.log10(p) + 4) / 4),
                color=CATEGORICAL[0], alpha=0.85,
                shrinkA=17, shrinkB=17, zorder=2))
            mx, my = (x0 + x1) / 2 + 0.16 * (y1 - y0), \
                     (y0 + y1) / 2 - 0.16 * (x1 - x0)
            ax.text(mx, my, f"{p:.1e}".replace("e-0", "e-"), fontsize=5.5,
                    ha="center", va="center", color=INK_MUTED)

    for s in states:
        x, y = pos[s]
        ax.scatter([x], [y], s=1500, color="white", edgecolor=INK,
                   linewidth=0.9, zorder=3)
        ax.text(x, y + 0.045, s.replace("_", "\n"), ha="center", va="center",
                fontsize=6.2, fontweight="bold", zorder=4)
        ax.text(x, y - 0.10, f"{m.loc[s, s]:.4f}", ha="center", va="center",
                fontsize=5.6, color=INK_MUTED, zorder=4)
        d = dwell.get(s, float("nan"))
        ax.text(x, y - 0.185, f"{d/60:.0f} min" if d > 120 else f"{d:.0f} s",
                ha="center", va="center", fontsize=5.6, color=CATEGORICAL[1],
                zorder=4)

    missing = [f"{k.replace('->', ' → ')}: {v.get('count', 0)} events"
               for k, v in (r[key].get("forbidden") or {}).items()
               if v["probability"] < floor]
    ax.text(0, -1.48, f"edges below {floor:.0e} of outgoing mass are not drawn"
            + ("\n" + "; ".join(missing) if missing else ""),
            ha="center", fontsize=5.8, color=INK_MUTED)

    fig.tight_layout()
    _save(fig, "fig_task13c_transition_graph")


# ══════════════════════════════════════════════════════════════════════════════
# Task 13E -- ablation
# ══════════════════════════════════════════════════════════════════════════════

def figure_ablation(machine: str) -> None:
    """
    Three panels because the three stages are measured in three units, and a
    grouped bar chart that pretended otherwise would be the single most
    misleading figure in the paper.

    The decision panel is drawn as intervals rather than bars: Phase V's
    finding there is that the point estimates are indistinguishable from
    zero, and a bar chart cannot say that.
    """
    base = f"{PHASE6_DIR}/task13e/{machine}"
    if not os.path.exists(f"{base}/ablation_state.csv"):
        return
    st = pd.read_csv(f"{base}/ablation_state.csv")
    fc = pd.read_csv(f"{base}/ablation_forecast.csv")
    dec = pd.read_csv(f"{base}/ablation_decision.csv")

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})

    short_state = ["GMM", "+ HMM", "+ min-dwell"]
    ax = axes[0]
    x = np.arange(len(st))
    ax.bar(x - 0.21, st["flicker_pct"], width=0.34, color=CATEGORICAL[0],
           label="all states", zorder=3)
    ax.bar(x + 0.21, st["flicker_standby_binary_pct"], width=0.34,
           color=CATEGORICAL[3], label="STANDBY vs rest", zorder=3)
    for xi, (a, b) in enumerate(zip(st["flicker_pct"],
                                    st["flicker_standby_binary_pct"])):
        # Staggered heights: the two bars are nearly equal at "+ HMM", where
        # labels drawn at the same height overlap.
        ax.text(xi - 0.21, a + 4.2, f"{a:.1f}", ha="center", fontsize=6)
        ax.text(xi + 0.21, b + 1.2, f"{b:.1f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(short_state, fontsize=6.5, rotation=18, ha="right")
    ax.set_ylabel("flicker rate (% of runs < 10 s)")
    ax.set_ylim(0, 78)
    ax.set_title("A  State layer", loc="left")
    ax.legend(loc="upper right")
    _grid(ax)

    ax = axes[1]
    short_fc = ["none", "Seq2Seq\nLSTM", "boosted\ntree"]
    x = np.arange(len(fc))
    ax.bar(x, fc["window_mae_s_mean"], width=0.55, color=CATEGORICAL[0],
           zorder=3)
    ax.errorbar(x, fc["window_mae_s_mean"], yerr=fc["window_mae_s_sd"],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=4)
    ax.axhline(float(fc.loc[0, "window_mae_s_mean"]), color=NEG,
               linestyle="--", linewidth=0.8, zorder=5)
    ax.text(len(fc) - 0.45, float(fc.loc[0, "window_mae_s_mean"]) + 0.4,
            "untrained\nreference", fontsize=6, color=NEG, ha="right")
    for xi, (v, s) in enumerate(zip(fc["window_mae_s_mean"],
                                    fc["standby_f1_sd"])):
        ax.text(xi, 1.0, f"F1 {fc.loc[xi, 'standby_f1_mean']:.3f}",
                ha="center", fontsize=6, color="white", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_fc, fontsize=6.5)
    ax.set_ylabel("per-window MAE (s)")
    ax.set_title("B  Window forecaster", loc="left")
    _grid(ax)

    ax = axes[2]
    dec = dec[dec["configuration"] != "Plant status quo"]
    y = np.arange(len(dec))[::-1]
    for yi, (_, r) in zip(y, dec.iterrows()):
        crosses = not r["ci_excludes_zero"]
        c = NEUTRAL if crosses else (POS if r["savings_usd_yr"] > 0 else NEG)
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, linewidth=1.8,
                solid_capstyle="round", zorder=3)
        ax.plot([r["savings_usd_yr"]], [yi], "o", color=c, markersize=3.6,
                markerfacecolor="white" if crosses else c,
                markeredgewidth=0.9, zorder=4)
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(["static break-even", "ski-rental", "wrong estimand",
                        "no chance constr.", "proposed", "oracle"][:len(dec)],
                       fontsize=6.5)
    ax.set_xlabel("annual saving (USD/yr)")
    ax.set_title("C  Decision layer", loc="left")
    _grid(ax, "x")

    fig.tight_layout()
    _save(fig, "fig_task13e_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# Task 13D -- publication re-renders of the Phase V figures
# ══════════════════════════════════════════════════════════════════════════════

def figure_cd_diagram(machine: str) -> None:
    """Demsar (2006) critical-difference diagram, at column width."""
    res = _load_json(f"{PHASE5_DIR}/task11/{machine}/forecasting_tests.json")
    if not res or not res.get("nemenyi"):
        return
    nem, fried = res["nemenyi"], res["friedman_over_seeds"]
    models = nem["models"]
    ranks = np.asarray(nem["mean_ranks"], float)
    cd = nem["critical_difference"]
    order = np.argsort(ranks)

    fig, ax = plt.subplots(figsize=(DOUBLE, 1.9))
    lo, hi = 0.5, len(models) + 0.5
    ax.set_xlim(lo, hi)
    ax.axis("off")
    y0 = 0.74
    # Bottom edge follows the lowest label rather than a constant, so the
    # figure does not carry a band of white space whose height depends on how
    # many models happen to be in the comparison.
    n_left = int(np.ceil(len(models) / 2))
    ax.set_ylim(y0 - 0.11 - 0.10 * (n_left - 1) - 0.05, 1.14)
    ax.plot([lo, hi], [y0, y0], color=INK, linewidth=0.9)
    for r in range(1, len(models) + 1):
        ax.plot([r, r], [y0, y0 + 0.03], color=INK, linewidth=0.7)
        ax.text(r, y0 + 0.055, str(r), ha="center", fontsize=6.5,
                color=INK_MUTED)

    for i, idx in enumerate(order):
        r = ranks[idx]
        left = i < len(order) / 2
        y = y0 - 0.11 - 0.10 * (i if left else len(order) - 1 - i)
        x_end = lo + 0.1 if left else hi - 0.1
        ax.plot([r, r], [y0, y], color=INK_MUTED, linewidth=0.7)
        ax.plot([r, x_end], [y, y], color=INK_MUTED, linewidth=0.7)
        ax.text(x_end, y, f" {LABEL.get(models[idx], models[idx])} ({r:.2f}) ",
                ha="right" if left else "left", va="center", fontsize=6.5)

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
    for a, b in bars:
        if any(a >= x and b <= y for x, y in drawn):
            continue
        drawn.append((a, b))
        ax.plot([a - 0.03, b + 0.03], [y0 + 0.09 + 0.045 * len(drawn)] * 2,
                color=CATEGORICAL[0], linewidth=2.6, solid_capstyle="butt")

    ax.plot([lo, lo + cd], [y0 + 0.29, y0 + 0.29], color=INK, linewidth=1.4)
    ax.text(lo + cd / 2, y0 + 0.315, f"CD = {cd:.2f}", ha="center", fontsize=6.5)
    ax.text(hi, y0 + 0.315,
            f"Friedman p = {fried['p_value']:.3g}, "
            f"W = {fried['kendalls_w']:.2f}, {nem['n_blocks']} seed blocks",
            ha="right", fontsize=6, color=INK_MUTED)
    fig.tight_layout()
    _save(fig, "fig_p6_cd_diagram")


def figure_decision_forest(machine: str) -> None:
    """
    Forest plot of every policy's annual saving, across the three economic
    scenarios.

    Intervals crossing zero are grey with a hollow marker; the doubling of the
    encoding is what keeps the figure readable in greyscale, where hue alone
    would be lost.
    """
    scenarios = {}
    for sc in ("S1_low", "S2_moderate", "S3_high"):
        d = _load_json(f"{PHASE5_DIR}/task11/{machine}/"
                       f"decision_bootstrap_{sc}.json")
        if d:
            scenarios[sc] = d
    if not scenarios:
        return

    names = list(scenarios)
    fig, axes = plt.subplots(1, len(names), figsize=(DOUBLE, 2.6), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, sc in zip(axes, names):
        rows = pd.DataFrame(scenarios[sc]["policies"])
        rows = rows[rows["policy"] != "observed"]
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows()):
            crosses = not r["ci_excludes_zero"]
            c = NEUTRAL if crosses else (POS if r["savings_usd_yr"] > 0 else NEG)
            ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, linewidth=1.6,
                    solid_capstyle="round", zorder=3)
            ax.plot([r["savings_usd_yr"]], [yi], "o", color=c, markersize=3.4,
                    markerfacecolor="white" if crosses else c,
                    markeredgewidth=0.9, zorder=4)
        ax.axvline(0, color=INK, linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL.get(p, p) for p in rows["policy"]],
                           fontsize=6.5)
        ax.set_xlabel("annual saving (USD/yr)")
        ax.set_title(f"{sc.split('_')[0]}: "
                     f"{scenarios[sc]['break_even_s']/60:.0f} min break-even",
                     loc="left")
        _grid(ax, "x")
    fig.tight_layout()
    # Sample size on the figure, below the axes, so it cannot be separated
    # from the intervals it explains -- the width of every bar above is this
    # number, not the estimator.
    fig.text(0.01, -0.02,
             f"{scenarios[names[0]]['n_episodes']} held-out episodes over "
             f"{scenarios[names[0]]['n_days']} factory days; hollow markers "
             f"mark intervals that include zero",
             fontsize=6, color=INK_MUTED)
    _save(fig, "fig_p6_decision_forest")


def figure_power_curve() -> None:
    """
    What the cross-machine design could have detected.

    The n = 4 curve is pinned at zero not by the data but by arithmetic: the
    smallest attainable two-sided p at four pairs is 0.125, above alpha. The
    annotation carries that number, because it is the answer to "does
    non-significant mean equivalent?" and no confidence band on the figure
    would otherwise reveal it.
    """
    cross = _load_json(f"{PHASE5_DIR}/task11/cross_machine_tests.json") or {}
    effects = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for i, n in enumerate((4, 8, 16, 30)):
        p = [wilcoxon_power(n, e, n_sim=2000) for e in effects]
        ax.plot(effects, p, "o-", markersize=2.6, linewidth=1.1,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=f"n = {n}")
    ax.axhline(0.80, color=INK, linestyle="--", linewidth=0.8)
    ax.text(effects[-1], 0.82, "80% power", ha="right", fontsize=6.5)

    # The effect actually observed on the four motor-driven machines, marked
    # so the reader can read its power off the n = 4 curve directly.
    obs = cross.get("own_vs_persistence_motor_machines") or {}
    d_obs = abs(obs.get("cohens_d") or 0.0)
    if d_obs > 0:
        ax.axvline(d_obs, color=NEUTRAL, linewidth=0.8, linestyle=":")
        ax.text(d_obs + 0.04, 0.62,
                f"observed effect on the\n{obs['n']} motor machines\n"
                f"(d = {d_obs:.2f})",
                fontsize=6, color=INK_MUTED, va="center")
    ax.text(0.28, 0.05,
            f"at n = 4 the smallest attainable p is {wilcoxon_p_floor(4):.3f} "
            f"> 0.05:\nno effect of any size can reach significance",
            fontsize=6, color=NEG)
    ax.set_xlabel("true paired effect (standard deviations)")
    ax.set_ylabel("power at $\\alpha$ = 0.05")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right")
    _grid(ax, "both")
    fig.tight_layout()
    _save(fig, "fig_p6_power_curve")


def figure_standby_hours() -> None:
    """
    STANDBY hours per covered day for all eight machines, with the one-week
    moving-block bootstrap interval.

    The two milling machines are hatched: twelve covered days give two weekly
    blocks, so their intervals are shown to make that visible and are not
    usable.  Marking them on the figure rather than only in the caption is
    what stops the largest bar in the motor group being read as the largest
    opportunity.
    """
    path = f"{PHASE5_DIR}/task11/state_block_bootstrap.csv"
    if not os.path.exists(path):
        return
    st = pd.read_csv(path, index_col=0)

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 2.4))
    x = np.arange(len(st))
    colours = [CATEGORICAL[0] if m in MOTOR_MACHINES else NEUTRAL
               for m in st.index]
    thin = st["n_blocks"] < 4
    bars = ax.bar(x, st["standby_h_per_day"], color=colours, width=0.66,
                  zorder=3)
    for b, is_thin in zip(bars, thin):
        if is_thin:
            b.set_hatch("///")
            b.set_edgecolor(INK)
            b.set_linewidth(0.6)
    ax.errorbar(x, st["standby_h_per_day"],
                yerr=[st["standby_h_per_day"] - st["ci_lo_h_per_day"],
                      st["ci_hi_h_per_day"] - st["standby_h_per_day"]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5, zorder=4)
    for xi, (v, hi, nb) in enumerate(zip(st["standby_h_per_day"],
                                         st["ci_hi_h_per_day"],
                                         st["n_blocks"])):
        ax.text(xi, hi + 0.18, f"{v:.1f}", ha="center", fontsize=6)
        ax.text(xi, 0.12, f"{int(nb)}", ha="center", fontsize=5.5,
                color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(k, k) for k in st.index], rotation=30,
                       ha="right", fontsize=6.5)
    ax.set_ylabel("STANDBY hours per covered day")
    ax.text(0.02, 0.96,
            "blue: physics battery passes 5/5   grey: fails\n"
            "hatched: fewer than four weekly blocks -- interval not usable\n"
            "number in bar: weekly blocks available",
            transform=ax.transAxes, va="top", fontsize=5.8, color=INK_MUTED)
    ax.set_ylim(0, st["ci_hi_h_per_day"].max() * 1.35)
    _grid(ax)
    fig.tight_layout()
    _save(fig, "fig_p6_standby_hours")


def figure_seed_stability(machine: str) -> None:
    """
    STANDBY F1 of every model at every seed.

    The figure exists to make one comparison unmissable: the swing between a
    good and a bad seed of the same neural architecture is larger than the
    whole spread between architectures, so a single-seed table of these models
    is not a ranking.  Seed 0 is ringed, because Phase II reported seed 0 and
    it is the best of five for four of the five neural models.
    """
    path = (f"{PHASE5_DIR}/task12/{machine}/forecasting/forecasting_seeds.json")
    d = _load_json(path)
    if not d:
        return
    res = d["results"]
    keys = sorted(res, key=lambda k: -res[k]["aggregate"]["f1_STANDBY"]["mean"])

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 2.4))
    all_vals = [r["f1_STANDBY"] for k in keys for r in res[k]["runs"]
                if "f1_STANDBY" in r]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad * 2.4, hi + pad)
    sd_y = lo - pad * 1.1          # one row for every sd label, clear of the data

    for i, k in enumerate(keys):
        runs = [r for r in res[k]["runs"] if "f1_STANDBY" in r]
        vals = [r["f1_STANDBY"] for r in runs]
        seeds = [r["seed"] for r in runs]
        colour = NEUTRAL if k == "persistence" else CATEGORICAL[i % len(CATEGORICAL)]
        ax.scatter([i] * len(vals), vals, s=14, color=colour, zorder=3,
                   alpha=0.85)
        for s, v in zip(seeds, vals):
            if s == 0:
                ax.scatter([i], [v], s=52, facecolor="none", edgecolor=INK,
                           linewidth=0.8, zorder=4)
        if len(vals) > 1:
            ax.plot([i - 0.28, i + 0.28], [np.mean(vals)] * 2, color=INK,
                    linewidth=1.1, zorder=5)
            ax.plot([i, i], [min(vals), max(vals)], color=colour,
                    linewidth=0.7, alpha=0.6, zorder=2)
            # All on one row beneath the data, not beside each column: an
            # offset label collides with the next model as soon as two spreads
            # overlap, and a per-column baseline puts them at seven heights.
            ax.text(i, sd_y, f"sd {np.std(vals, ddof=1):.3f}",
                    fontsize=5.8, ha="center", va="center", color=INK_MUTED)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABEL.get(k, k).replace(" (untrained)", "\n(untrained)")
                        for k in keys], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("STANDBY $F_1$")
    ax.text(0.98, 0.97, "ringed: seed 0, the seed Phase II reported",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            color=INK_MUTED)
    _grid(ax)
    fig.tight_layout()
    _save(fig, "fig_p6_seed_stability")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = "pelletizer-I") -> None:
    section("PHASE VI FIGURES")
    ensure_dir(FIG_DIR)

    figure_decision_shap(machine, "phase4")
    figure_decision_shap(machine, "auto")
    figure_window_shap(machine)
    figure_state_shap(machine, "phase4")
    figure_transition_matrix(machine)
    figure_transition_graph(machine)
    figure_ablation(machine)

    figure_cd_diagram(machine)
    figure_decision_forest(machine)
    figure_power_curve()
    figure_standby_hours()
    figure_seed_stability(machine)

    section("PHASE VI FIGURES COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase VI figures")
    ap.add_argument("--machine", default="pelletizer-I")
    a = ap.parse_args()
    main(a.machine)
