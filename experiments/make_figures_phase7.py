"""
============================================================
PHASE VII FIGURES  --  experiments/make_figures_phase7.py
============================================================

PURPOSE
-------
Five results in this paper were carried by a table and nothing
else.  Each is a comparison across a grid -- six state-inference
configurations, a swept constraint, a 64-cell economic plane, two
8x8 transfer matrices, two datasets against five physics checks --
and a grid of numbers is the one thing a reader cannot hold in
their head.  This module renders those five, from the same stored
outputs the tables are typeset from, at the conventions
`make_figures_phase6.py` established:

    python -m experiments.make_figures_phase7

  fig_p7_state_detection   Table 2  ->  state_layer.tex   4.2
  fig_p7_min_dwell         Table 3  ->  state_layer.tex   4.3
  fig_p7_economic_plane    (prose)  ->  decision_layer.tex 6.6
  fig_p7_transfer          Tables 8,10 -> crossmachine.tex 9.1-9.2
  fig_p7_cross_dataset     Table 12 ->  crossmachine.tex  9.3

WHY THESE FIVE AND NOT A RE-RENDER OF THE PHASE II-IV PNGs
----------------------------------------------------------
The screen figures those phases wrote are 12-16 in wide with a
rendered suptitle, which in a two-column manuscript means a
caption in larger type than the axis labels it describes.  They
are also laid out to answer the question the phase was asking at
the time, which is not the question the paper section asks.  Each
figure here is drawn to carry the claim the surrounding prose
actually makes:

  * state detection is drawn as flicker against dwell with the
    decoding path as a connected trajectory, because the claim is
    that the emission model barely moves the result and the
    decoding rule moves all of it -- three points in a line say
    that and six bars do not;
  * the transfer figure puts the duration forecaster and the
    sequence model side by side on a shared layout, because the
    section's claim is the *contrast* between them;
  * the cross-dataset figure is drawn as a check-by-check status
    grid rather than as a bar chart of scores, because the finding
    is which checks were unavailable, not what the surviving ones
    scored.

CONVENTIONS (inherited, deliberately, from Phase VI)
----------------------------------------------------
  * 3.5 in single column / 7.16 in double column, placed at 100%;
  * no suptitle -- that is the caption's job and the caption can
    be edited in proof;
  * PDF for the manuscript and PNG for the notes;
  * `pdf.fonttype = 42`, so glyphs embed as text rather than as
    outlines and the PDF stays searchable and re-typeset-able;
  * nothing is encoded by hue alone.  Physics failures carry a
    hatch, unavailable checks carry a distinct glyph, and every
    diverging map gets an explicit zero contour, so a monochrome
    print keeps every distinction the colour makes.

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ensure_dir, info, section        # noqa: E402

MACHINE = "pelletizer-I"
PHASE2_DIR = f"outputs/phase2/task4/{MACHINE}"
PHASE3_DIR = f"outputs/phase3/task7/{MACHINE}"
PHASE4_T9 = "outputs/phase4/task9"
PHASE4_T10 = "outputs/phase4/task10"
FIG_DIR = "outputs/phase7/figures"

# IEEE two-column geometry, in inches.  Identical to Phase VI: a paper
# whose figures are sized to two different column widths looks like two
# papers.
COL, DOUBLE = 3.5, 7.16

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e3e2df", "#9a9894"
POS, NEG = "#1baf7a", "#eb6834"

SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#ffffff", "#cfe0f5", "#7fb0e6", "#2a78d6", "#123f77"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8c3000", "#eb6834", "#f6d5c4", "#ffffff",
            "#c9ecdd", "#1baf7a", "#0a5c40"])

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

SHORT = {"pelletizer-I": "Pel-I", "pelletizer-II": "Pel-II",
         "milling-I": "Mill-I", "milling-II": "Mill-II",
         "exhaust-fan-I": "Fan-I", "exhaust-fan-II": "Fan-II",
         "dpc-I": "DPC-I", "dpc-II": "DPC-II"}
ORDER8 = ["pelletizer-I", "pelletizer-II", "milling-I", "milling-II",
          "exhaust-fan-I", "exhaust-fan-II", "dpc-I", "dpc-II"]

CHECK_LABEL = {
    "C1_power_ordering": "C1  power ordering",
    "C2_off_near_zero": "C2  OFF near zero",
    "C3_pf_separation": "C3  PF separation",
    "C4_current_ratio": "C4  no-load current ratio",
    "C5_variance_ordering": "C5  variance ordering",
}
CHECK_ORDER = list(CHECK_LABEL)


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


def _need(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(
            f"missing input: {path}\n"
            "Run the phase that produces it before this module; these "
            "figures are rendered from stored outputs and never refit.")
    return path


# ─────────────────────────────────────────────────────────────────────
# Figure 1: state-inference comparison
# ─────────────────────────────────────────────────────────────────────

def figure_state_detection():
    """
    The claim is `the emission model barely matters; the decoding rule
    does'.  Drawn as flicker against median dwell, the three decoding
    configurations -- i.i.d. assignment, Viterbi, Viterbi under a dwell
    floor -- form a connected path over an order of magnitude in both
    coordinates, while the three alternative emission models sit in one
    cluster with the i.i.d. GMM.  That is the whole argument of the
    subsection, and it is one glance.

    Panel B exists because panel A cannot show the spectral failure:
    spectral clustering scores the HIGHEST schedule agreement in the
    table while assigning 3.6x everyone else's STANDBY hours, which is
    the concrete reason the paper argues for a battery of checks rather
    than for a favourite metric.
    """
    df = pd.read_csv(_need(f"{PHASE2_DIR}/comparison_table.csv"))
    df = df[df["key"] != "_reference_export"].set_index("key")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})

    # ── A: flicker vs dwell, with the decoding path drawn as a path ──
    path = ["gmm", "gmm_hmm", "gmm_hmm_mindwell"]
    others = ["kmeans", "dbscan", "spectral"]

    def xy(k):
        return (df.loc[k, "flicker_standby_binary_pct"],
                df.loc[k, "median_dwell_s"])

    for a, b in zip(path[:-1], path[1:]):
        x0, y0 = xy(a)
        x1, y1 = xy(b)
        axA.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
            linewidth=1.1, color=CATEGORICAL[0], zorder=4,
            shrinkA=7, shrinkB=7))

    for k in others:
        x, y = xy(k)
        failed = df.loc[k, "n_physics_passed"] < df.loc[k, "n_physics_applicable"]
        axA.scatter(x, y, s=46, marker="s", zorder=5,
                    facecolor="none" if failed else NEUTRAL,
                    edgecolor=NEG if failed else INK_MUTED,
                    hatch="////" if failed else None, linewidths=1.0)
    for k in path:
        x, y = xy(k)
        axA.scatter(x, y, s=52, marker="o", zorder=6,
                    facecolor=CATEGORICAL[0], edgecolor="white", linewidths=0.8)

    # k-means, DBSCAN and the i.i.d. GMM land within 5 points of flicker
    # and 2 s of dwell of each other -- which IS the result -- so their
    # labels are placed by hand and fanned out rather than auto-offset.
    ann = {
        "kmeans": ("$k$-means", -9, -3, "right"),
        "dbscan": ("DBSCAN", 9, 5, "left"),
        "gmm": ("GMM, i.i.d.", 10, -7, "left"),
        "spectral": ("Spectral\n(physics 4/5)", 0, 9, "center"),
        "gmm_hmm": ("GMM-HMM", 10, 2, "left"),
        "gmm_hmm_mindwell": ("+ min-dwell (proposed)", 9, 3, "left"),
    }
    for k, (txt, dx, dy, ha) in ann.items():
        x, y = xy(k)
        axA.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy),
                     fontsize=6.5, color=INK, ha=ha)

    axA.set_yscale("log")
    axA.set_xlabel("STANDBY-vs-rest flicker (%)")
    axA.set_ylabel("median dwell (s), log scale")
    axA.set_xlim(-8, 118)
    axA.set_ylim(1.3, 900)
    axA.axvline(0, color=INK, linewidth=0.7, linestyle=":", zorder=1)
    axA.set_title("A   Decoding moves the result; emissions do not")
    _grid(axA, "both")

    axA.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=CATEGORICAL[0], markeredgecolor="white",
               label="decoding rule (shared GMM emissions)"),
        Line2D([], [], marker="s", linestyle="none", markersize=6,
               markerfacecolor=NEUTRAL, markeredgecolor=INK_MUTED,
               label="alternative emission model"),
        Line2D([], [], marker="s", linestyle="none", markersize=6,
               markerfacecolor="none", markeredgecolor=NEG,
               label="fails a physics check"),
    ], loc="upper right", fontsize=6.2, handletextpad=0.4,
        borderaxespad=0.3, labelspacing=0.35)

    # ── B: STANDBY hours, the metric that exposes spectral ──
    keys = ["kmeans", "dbscan", "spectral", "gmm", "gmm_hmm",
            "gmm_hmm_mindwell"]
    names = ["$k$-means", "DBSCAN", "Spectral", "GMM i.i.d.",
             "GMM-HMM", "+ min-dwell"]
    vals = [df.loc[k, "standby_hours"] for k in keys]
    fails = [df.loc[k, "n_physics_passed"] < df.loc[k, "n_physics_applicable"]
             for k in keys]
    ypos = np.arange(len(keys))[::-1]

    axB.barh(ypos, vals, height=0.66, zorder=3,
             color=[("none" if f else CATEGORICAL[0]) for f in fails],
             edgecolor=[(NEG if f else "none") for f in fails],
             hatch=[("////" if f else None) for f in fails], linewidth=1.0)
    for y, v in zip(ypos, vals):
        axB.text(v + 0.35, y, f"{v:.2f}", va="center", fontsize=6.5,
                 color=INK, zorder=4)
    axB.set_yticks(ypos)
    axB.set_yticklabels(names, fontsize=7)
    axB.set_xlabel("inferred STANDBY (h) on the evaluation blocks")
    axB.set_xlim(0, 18.2)
    axB.set_title("B   Why no single metric can be quoted alone")
    _grid(axB, "x")

    # Spectral scores the best schedule agreement in the table.  Saying so
    # on the figure is the point of the panel.
    axB.annotate("best schedule score (89.0%)\nof any method here",
                 xy=(vals[2], ypos[2]), xytext=(9.4, ypos[2] - 1.25),
                 fontsize=6.2, color=NEG, ha="left",
                 arrowprops=dict(arrowstyle="->", color=NEG, linewidth=0.8))

    fig.subplots_adjust(wspace=0.28)
    _save(fig, "fig_p7_state_detection")


# ─────────────────────────────────────────────────────────────────────
# Figure 2: minimum-dwell sweep
# ─────────────────────────────────────────────────────────────────────

def figure_min_dwell():
    """
    The sweep's claim is a null result on three of four columns:
    flicker collapses and nothing else moves.  A figure has to make the
    flatness as visible as the collapse, so the STANDBY total is drawn
    on an axis wide enough to show that it *could* have moved -- a
    panel auto-scaled to a 1% change would show a dramatic-looking
    slope and say the opposite of what the data says.
    """
    df = pd.read_csv(_need(f"{PHASE2_DIR}/min_dwell_decode.csv"))
    df = df.sort_values("min_dwell_s").reset_index(drop=True)
    x = np.arange(len(df))
    xt = [f"{int(v)}" for v in df["min_dwell_s"]]

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(COL, 3.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.42})

    # ── A: flicker, both definitions ──
    axA.plot(x, df["flicker_rate_pct"], marker="o", markersize=4.5,
             linewidth=1.4, color=CATEGORICAL[0], zorder=4,
             label="all states")
    axA.plot(x, df["flicker_standby_binary_pct"], marker="^", markersize=4.5,
             linewidth=1.4, color=CATEGORICAL[1], linestyle="--", zorder=4,
             label="STANDBY vs rest")
    axA.set_ylabel("flicker (%)")
    axA.set_ylim(-4, 72)
    axA.set_title("A   Flicker collapses at the first non-zero floor")
    axA.legend(loc="upper right", fontsize=6.5, handletextpad=0.5)
    _grid(axA)
    axA.annotate("0.00% from 10 s onward", xy=(1, 0), xytext=(1.55, 13),
                 fontsize=6.2, color=INK_MUTED,
                 arrowprops=dict(arrowstyle="->", color=INK_MUTED,
                                 linewidth=0.7))

    # ── B: what did NOT move, on axes wide enough to show it ──
    axB.plot(x, df["median_dwell_s"], marker="o", markersize=4.5,
             linewidth=1.4, color=CATEGORICAL[2], zorder=4)
    axB.set_ylabel("median dwell (s)", color=CATEGORICAL[2])
    axB.tick_params(axis="y", colors=CATEGORICAL[2])
    axB.set_ylim(0, 560)
    _grid(axB)

    ax2 = axB.twinx()
    ax2.plot(x, df["standby_hours"], marker="s", markersize=4.5,
             linewidth=1.4, color=CATEGORICAL[3], linestyle="-.", zorder=4)
    ax2.set_ylabel("inferred STANDBY (h)", color=CATEGORICAL[3])
    ax2.tick_params(axis="y", colors=CATEGORICAL[3])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(CATEGORICAL[3])
    # Zero-anchored, so the 1.2% drift over the whole sweep reads as the
    # flat line it is rather than as a trend.
    ax2.set_ylim(0, 7.5)
    ax2.grid(False)

    axB.set_title("B   Dwell rises; the STANDBY total does not move")
    axB.set_xlabel("minimum-dwell floor (s)")
    axB.set_xticks(x)
    axB.set_xticklabels(xt)

    _save(fig, "fig_p7_min_dwell")


# ─────────────────────────────────────────────────────────────────────
# Figure 3: the economic plane
# ─────────────────────────────────────────────────────────────────────

def figure_economic_plane(tariff: float = 0.12):
    """
    Three panels over the two coefficients that carry information --
    restart energy and the valuation of production delay.  Tariff is
    fixed here because the paper shows separately that break-even is
    tariff-invariant absent a non-energy restart cost.

    Panels B and C are the reason this is a figure and not a table:
    they disagree in sign over part of the plane.  Where restarts are
    expensive the profitable advice is to stop de-energising, so money
    is saved while energy is spent.  Both panels therefore carry an
    explicit zero contour -- the one contour an operator needs -- which
    also keeps the sign boundary visible in monochrome, where a
    diverging ramp reads as a single light band through the middle.
    """
    g = pd.read_csv(_need(f"{PHASE3_DIR}/economic_grid.csv"))
    g = g[np.isclose(g["tariff_per_kwh"], tariff)]
    if g.empty:
        raise SystemExit(f"no grid rows at tariff {tariff}")

    # constrained_layout, not subplots_adjust: a colorbar made with
    # `ax=` gets its own axes, and adjusting the subplot grid afterwards
    # moves the panels out from under their own colorbars.
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6),
                             constrained_layout=True)

    piv_be = g.pivot(index="restart_energy_kwh", columns="delay_cost_per_hour",
                     values="break_even_h")
    piv_sav = g.pivot(index="restart_energy_kwh", columns="delay_cost_per_hour",
                      values="forecast_opt_annual")
    piv_kwh = g.pivot(index="restart_energy_kwh", columns="delay_cost_per_hour",
                      values="forecast_opt_energy_saved_kwh")

    # Units live in the panel titles, not on the colorbars: a short
    # vertical colorbar label sits immediately left of the next panel and
    # reads as that panel's y-axis label.
    def _heat(ax, piv, title, cmap, norm=None, zero=False):
        im = ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto",
                       origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=6)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r:g}" for r in piv.index], fontsize=6)
        ax.set_xlabel("delay valuation (USD/h)", fontsize=7)
        ax.set_title(title, fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
        if zero:
            # Drawn on the cell lattice so the contour lands on the cell
            # boundaries the heat-map actually shows.
            ax.contour(np.arange(piv.shape[1]), np.arange(piv.shape[0]),
                       piv.values, levels=[0.0], colors=[INK],
                       linewidths=1.0, linestyles="--")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.6)
        return im

    _heat(axes[0], piv_be, "A   Break-even $D^{\\star}$ (h)", SEQ)
    axes[0].set_ylabel("restart energy (kWh)", fontsize=7)

    lim = float(np.nanmax(np.abs(piv_sav.values))) or 1.0
    _heat(axes[1], piv_sav, "B   Annual saving (USD/yr)",
          DIV, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), zero=True)

    lim2 = float(np.nanmax(np.abs(piv_kwh.values))) or 1.0
    _heat(axes[2], piv_kwh, "C   Energy saved (kWh, test window)", DIV,
          norm=TwoSlopeNorm(vmin=-lim2, vcenter=0, vmax=lim2), zero=True)

    _save(fig, "fig_p7_economic_plane")


# ─────────────────────────────────────────────────────────────────────
# Figure 4: the two transfer matrices, side by side
# ─────────────────────────────────────────────────────────────────────

def figure_transfer():
    """
    The section's claim is a contrast, so the two matrices are drawn on
    one figure with the same axis convention: rows are the machine the
    model was fitted on, columns the machine it was evaluated on, and
    the leading diagonal -- the locally fitted model -- is outlined.

    Panel A is the duration forecaster the decision layer consumes,
    in the raw variant (no target information at all, the honest
    zero-shot condition).  Panel B is the sequence state model under
    its source scaler.  A is close to uniform; B is a bright diagonal
    on a dark field.  What transfers is not machine-specific and what
    is machine-specific does not transfer, which is easier to see than
    to say.
    """
    a = pd.read_csv(_need(f"{PHASE4_T9}/part_a/transfer_matrix.csv"))
    b = pd.read_csv(_need(f"{PHASE4_T9}/part_b/transfer_matrix.csv"))

    a = a[a["variant"] == "raw"]
    b = b[(b["variant"] == "source_scaler") & (b["source"] != "persistence")]

    ma = a.pivot(index="source", columns="target", values="mae_s")
    ma = ma.reindex(index=ORDER8, columns=ORDER8) / 1000.0
    mb = b.pivot(index="source", columns="target", values="f1_STANDBY")
    mb = mb.reindex(index=ORDER8, columns=ORDER8)

    # See figure_economic_plane on why this is not subplots_adjust.
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.4),
                             constrained_layout=True)
    labels = [SHORT[m] for m in ORDER8]

    def _mat(ax, M, title, cbar_label, cmap, fmt, textthresh):
        im = ax.imshow(M.values, cmap=cmap, aspect="equal", origin="upper",
                       interpolation="nearest")
        ax.set_xticks(range(8))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(8))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("evaluated on", fontsize=7)
        ax.set_ylabel("fitted on", fontsize=7)
        ax.set_title(title, fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)

        vmin, vmax = np.nanmin(M.values), np.nanmax(M.values)
        for i in range(8):
            for j in range(8):
                v = M.values[i, j]
                if not np.isfinite(v):
                    continue
                rel = (v - vmin) / max(vmax - vmin, 1e-12)
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=5.4,
                        color="white" if rel > textthresh else INK)
        # The locally fitted model is the reference every off-diagonal
        # cell is read against, so it is outlined rather than left to be
        # located by eye.
        for i in range(8):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=1.3, zorder=5))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.6)

    _mat(axes[0], ma, "A   Duration forecaster (decision layer)",
         "MAE (ks) -- lower is better", SEQ, "{:.1f}", 0.62)
    _mat(axes[1], mb, "B   Sequence state model",
         "STANDBY $F_1$ -- higher is better", SEQ, "{:.2f}", 0.62)

    _save(fig, "fig_p7_transfer")


# ─────────────────────────────────────────────────────────────────────
# Figure 5: cross-dataset, check by check
# ─────────────────────────────────────────────────────────────────────

def figure_cross_dataset():
    """
    The finding is not that the control scored badly -- it nearly did
    not -- but that the two checks which would have caught it were
    unavailable on a single-channel record.  A bar chart of scores
    hides exactly that, so panel A is a status grid over the five
    checks with `unavailable' as its own glyph, against pelletizer I
    where all five are evaluable.

    Panel B carries the one signal that did separate the control, and
    it is a diurnal one rather than a physical one: a photovoltaic
    inverter is productive 0.0% of the time at night, which no driven
    load is.  That the battery had to fall back on this is the section's
    point.
    """
    tbl = pd.read_csv(_need(f"{PHASE4_T10}/table_cross_dataset.csv"))
    cnc = _load_json(_need(f"{PHASE4_T10}/spark-cnc/characterisation.json"))
    pv = _load_json(_need(f"{PHASE4_T10}/spark-solar/characterisation.json"))

    # pelletizer-I is the four-channel reference: all five checks
    # evaluable and passing under the proposed configuration.
    ref = pd.read_csv(_need(f"{PHASE2_DIR}/comparison_table.csv"))
    ref = ref.set_index("key").loc["gmm_hmm_mindwell"]
    assert int(ref["n_physics_applicable"]) == 5, "expected 5 applicable checks"
    ref_checks = {c: True for c in CHECK_ORDER}
    assert int(ref["n_physics_passed"]) == 5, "expected 5 passing checks"

    cols = [
        ("Pelletizer I\n(4-channel)", ref_checks),
        ("CNC centre\n(1-channel)", cnc["physics"]["checks"]),
        ("PV inverter\n(control)", pv["physics"]["checks"]),
    ]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})

    # ── A: status grid ──
    for jx, (_, checks) in enumerate(cols):
        for iy, ck in enumerate(CHECK_ORDER):
            v = checks.get(ck)
            y = len(CHECK_ORDER) - 1 - iy
            if v is None:
                face, edge, hatch, mark, mcol = ("#f4f3f1", NEUTRAL,
                                                 "xxx", "n/a", INK_MUTED)
            elif v:
                face, edge, hatch, mark, mcol = (POS, POS, None,
                                                 "pass", "white")
            else:
                face, edge, hatch, mark, mcol = ("none", NEG, "////",
                                                 "FAIL", NEG)
            axA.add_patch(Rectangle((jx - 0.44, y - 0.4), 0.88, 0.8,
                                    facecolor=face, edgecolor=edge,
                                    hatch=hatch, linewidth=1.0, zorder=3))
            axA.text(jx, y, mark, ha="center", va="center", fontsize=6.4,
                     color=mcol, zorder=4,
                     fontweight="bold" if mark == "FAIL" else "normal")

    # Right margin left clear for the note on C3/C4 below.
    axA.set_xlim(-0.6, len(cols) + 0.75)
    axA.set_ylim(-0.6, len(CHECK_ORDER) - 0.4)
    axA.set_xticks(range(len(cols)))
    axA.set_xticklabels([c[0] for c in cols], fontsize=6.5)
    axA.set_yticks(range(len(CHECK_ORDER)))
    axA.set_yticklabels([CHECK_LABEL[c] for c in CHECK_ORDER][::-1],
                        fontsize=6.5)
    axA.set_title("A   The two discriminating checks are the two lost")
    for s in ("top", "right", "left", "bottom"):
        axA.spines[s].set_visible(False)
    axA.tick_params(length=0)

    # C3 and C4 are the only checks independent of power magnitude, i.e.
    # the only two that can separate energised idle from a lightly loaded
    # productive state.  Marking them is the figure's entire argument.
    # Rows are drawn top-down, so C3 sits at y=2 and C4 at y=1; the box
    # spans the two cell heights between 0.6 and 2.4.
    axA.add_patch(Rectangle((-0.52, 0.55), len(cols) - 0.36, 1.9,
                            fill=False, edgecolor=NEG, linewidth=1.0,
                            linestyle=":", zorder=6))
    axA.text(len(cols) - 0.38, 1.5, "the only two\nindependent of\npower magnitude",
             fontsize=6, color=NEG, va="center", ha="left")

    # ── B: the diurnal signal that did separate the control ──
    names = ["CNC centre", "PV inverter\n(control)"]
    night = tbl.set_index("dataset").loc[
        ["spark-cnc", "spark-solar"], "productive_night_pct"].values
    midday = tbl.set_index("dataset").loc[
        ["spark-cnc", "spark-solar"], "productive_midday_pct"].values

    xpos = np.arange(2)
    w = 0.36
    axB.bar(xpos - w / 2, night, width=w, color=CATEGORICAL[0], zorder=3,
            label="productive at night")
    axB.bar(xpos + w / 2, midday, width=w, color=CATEGORICAL[3], zorder=3,
            label="productive at midday")
    for x, v in zip(xpos - w / 2, night):
        axB.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=6.2, color=INK)
    for x, v in zip(xpos + w / 2, midday):
        axB.text(x, v + 1.4, f"{v:.1f}", ha="center", fontsize=6.2, color=INK)
    axB.set_xticks(xpos)
    axB.set_xticklabels(names, fontsize=6.5)
    axB.set_ylabel("% of the window (state share)", fontsize=7)
    axB.set_ylim(0, 66)
    axB.set_title("B   What the battery fell back on")
    axB.legend(loc="upper left", fontsize=6.2, handletextpad=0.5)
    _grid(axB)

    fig.subplots_adjust(wspace=0.3)
    _save(fig, "fig_p7_cross_dataset")


# ─────────────────────────────────────────────────────────────────────

FIGURES = {
    "state_detection": figure_state_detection,
    "min_dwell": figure_min_dwell,
    "economic_plane": figure_economic_plane,
    "transfer": figure_transfer,
    "cross_dataset": figure_cross_dataset,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(FIGURES),
                    help="render one figure instead of all five")
    args = ap.parse_args()

    section("PHASE VII FIGURES")
    ensure_dir(FIG_DIR)
    todo = {args.only: FIGURES[args.only]} if args.only else FIGURES
    for name, fn in todo.items():
        info(f"rendering {name}")
        fn()
    info(f"{len(todo)} figure(s) written to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
