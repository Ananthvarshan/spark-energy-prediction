"""
============================================================
PHASE IV RESULT FIGURES  --  experiments/make_figures_phase4.py
============================================================

PURPOSE
-------
Render the figures for Tasks 8, 9 and 10 from the CSV/JSON the
experiments wrote, so this re-runs in seconds and recomputes
nothing.

    python -m experiments.make_figures_phase4

DESIGN NOTES
------------
Same palette and rules as `make_figures.py` and
`make_figures_phase3.py`, so a quantity keeps its colour across the
whole paper.  Three consequences specific to Phase IV:

  * Machines are CATEGORIES, not an ordered scale, so they are never
    encoded in a colour ramp -- they are positions on an axis, and
    colour is reserved for the quantity being compared.
  * The transfer matrices are the one place a heat map is the right
    form: a source x target grid is genuinely two-dimensional.  Error
    is an unsigned magnitude and gets the single-hue ramp; currency
    saved is signed and gets the diverging ramp with a neutral zero,
    because the sign is the entire result.
  * Every heat-map cell carries its number.  At eight machines the
    grid is small enough that the colour is an aid to reading, not
    the encoding, which keeps the figure legible in greyscale print.
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import (                       # noqa: E402
    PHASE4_DIR, ensure_dir, info, section,
)

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
NEUTRAL = "#9a9894"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eaf1fb", "#104281"])
SEQ_R = LinearSegmentedColormap.from_list("seq_blue_r", ["#104281", "#eaf1fb"])
DIV = LinearSegmentedColormap.from_list("div_or_bl", ["#c0512a", "#f2f1ee", "#1c5ba8"])

FIG_DIR = f"{PHASE4_DIR}/figures"

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

SHORT = {
    "pelletizer-I": "Pel-I", "pelletizer-II": "Pel-II",
    "milling-I": "Mill-I", "milling-II": "Mill-II",
    "exhaust-fan-I": "Fan-I", "exhaust-fan-II": "Fan-II",
    "dpc-I": "DPC-I", "dpc-II": "DPC-II",
    "spark-cnc": "CNC", "spark-solar": "PV",
    "persistence": "persist.",
}


def _short(keys):
    return [SHORT.get(k, k) for k in keys]


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _label_bars(ax, bars, values, fmt="{:.1f}", horizontal=False, pad=0.015,
                skip_out_of_range=False):
    """
    Write each bar's value at its tip.

    `skip_out_of_range` matters whenever the axis has been clipped: a text
    artist is NOT clipped to the axes by default, so a label sitting at a bar
    tip far outside the view still counts towards `savefig(bbox_inches="tight")`
    and silently stretches the saved figure to include it.  One clipped bar in
    a Phase IV panel produced an 8,570-pixel-tall PNG before this was added.
    """
    lo, hi = (ax.get_xlim() if horizontal else ax.get_ylim())
    span = hi - lo
    for b, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        if skip_out_of_range and not (lo <= v <= hi):
            continue
        if horizontal:
            off = span * pad * (1 if b.get_width() >= 0 else -1)
            ax.text(b.get_width() + off, b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center",
                    ha="left" if b.get_width() >= 0 else "right",
                    fontsize=7, color=INK)
        else:
            off = span * pad * (1 if b.get_height() >= 0 else -1)
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                    fmt.format(v), ha="center",
                    va="bottom" if b.get_height() >= 0 else "top",
                    fontsize=7, color=INK)


def _save(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)
    info(f"wrote {path}")


def _heat(ax, piv, title, cbar_label, cmap, norm=None, fmt="{:.0f}",
          diagonal_box: bool = True):
    """Annotated heat map; the within-machine diagonal is outlined, not coloured."""
    data = piv.to_numpy(float)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(_short(piv.columns), rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(_short(piv.index), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("target machine")
    ax.set_ylabel("source (trained on)")
    finite = data[np.isfinite(data)]
    mid = (finite.min() + finite.max()) / 2 if len(finite) else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=6,
                        color=INK_MUTED)
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > abs(mid) * 1.3 else INK)
            if diagonal_box and piv.index[i] == piv.columns[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK, linewidth=1.4))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.grid(False)


# ── Task 8 ────────────────────────────────────────────────────────────────────

def figure_task8_states(state: pd.DataFrame, out_dir: str) -> None:
    """
    Per-machine state identification: what the model found and whether it is
    physically admissible.

    Six panels, one quantity each.  Standby hours are normalised per COVERED
    day rather than quoted as a total: the records differ in length by a factor
    of five, and the totals would rank the machines by acquisition uptime.

    Panels C and D are deliberately adjacent and must be read together.  On
    power-factor separation alone the exhaust fans look excellent; the no-load
    current ratio is what shows that their "STANDBY" cluster is not an
    energised idle state at all.  Presenting C without D would report a pass
    for four machines that fail.
    """
    m = list(state.index)
    x = np.arange(len(m))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4))

    ax = axes[0, 0]
    b = ax.bar(x, state["standby_h_per_day"], color=CATEGORICAL[0], zorder=3)
    _label_bars(ax, b, state["standby_h_per_day"].to_numpy(), "{:.2f}")
    ax.set_title("A  Energised idle per covered day")
    ax.set_ylabel("STANDBY hours / day")

    ax = axes[0, 1]
    b = ax.bar(x, state["standby_power_w"] / 1000.0, color=CATEGORICAL[1], zorder=3)
    _label_bars(ax, b, (state["standby_power_w"] / 1000.0).to_numpy(), "{:.2f}")
    ax.set_title("B  Standby power (cluster mean)")
    ax.set_ylabel("kW")

    ax = axes[0, 2]
    b = ax.bar(x, state["pf_separation"], color=CATEGORICAL[2], zorder=3)
    ax.axhline(0.40, color=INK, linewidth=1.0, linestyle="--")
    ax.text(len(m) - 0.4, 0.42, "acceptance >= 0.40", fontsize=6, ha="right", color=INK)
    _label_bars(ax, b, state["pf_separation"].to_numpy(), "{:.2f}")
    ax.set_title("C  Power-factor separation (check C3)")
    ax.set_ylabel("PF(productive) - PF(STANDBY)")

    ax = axes[1, 0]
    ratio = state["current_ratio"].to_numpy(float)
    ok = (ratio >= 0.25) & (ratio <= 0.50)
    b = ax.bar(x, ratio, color=[CATEGORICAL[2] if o else CATEGORICAL[1] for o in ok],
               zorder=3)
    ax.axhspan(0.25, 0.50, color=GRID, zorder=1)
    ax.text(len(m) - 0.4, 0.51, "motor no-load band [0.25, 0.50]", fontsize=6,
            ha="right", color=INK)
    _label_bars(ax, b, ratio, "{:.3f}")
    ax.set_title("D  No-load current ratio (check C4)")
    ax.set_ylabel("I(STANDBY) / I(productive)")

    ax = axes[1, 1]
    w = 0.38
    before = state.get("flicker_before_min_dwell_pct")
    if before is None:
        before = pd.Series(np.nan, index=state.index)
    b1 = ax.bar(x - w / 2, before, w, color=NEUTRAL, label="Viterbi only", zorder=3)
    b2 = ax.bar(x + w / 2, state["flicker_pct"], w, color=CATEGORICAL[3],
                label="+ min-dwell constraint", zorder=3)
    _label_bars(ax, b1, np.asarray(before, float), "{:.0f}")
    _label_bars(ax, b2, state["flicker_pct"].to_numpy(), "{:.2f}")
    ax.set_title("E  Flicker before and after the dwell constraint")
    ax.set_ylabel("% of state runs shorter than 10 s")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    b = ax.bar(x, state["m1_agreement_pct"], color=CATEGORICAL[0], zorder=3)
    _label_bars(ax, b, state["m1_agreement_pct"].to_numpy(), "{:.1f}")
    ax.set_ylim(50, 102)
    ax.set_title("F  Agreement with the Otsu PF classifier")
    ax.set_ylabel("% of energised rows (five-method proof, M1)")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(_short(m), rotation=30, ha="right")
        _grid(ax)
    fig.suptitle("Task 8 -- state identification across the eight IMDELD machines",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task8_states.png")


def figure_task8_difficulty(diff: pd.DataFrame, state: pd.DataFrame,
                            out_dir: str) -> None:
    """
    Difficulty against outcome.

    Panel A puts the separability of the no-load/load boundary on the x-axis
    and the agreement of the independent Otsu power-factor classifier on the
    y-axis: if the ordering is real, machines whose states overlap should be
    the ones where two independent methods disagree.  Panel B is the same
    ordering read through the record's own quality (fragmentation and spikes).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.scatter(diff["standby_working_d"], diff["m1_agreement_pct"],
               s=46, color=CATEGORICAL[0], zorder=3)
    for k, r in diff.iterrows():
        ax.annotate(SHORT.get(k, k), (r["standby_working_d"], r["m1_agreement_pct"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("STANDBY vs productive separability (Cohen's d, log power)")
    ax.set_ylabel("agreement with the Otsu PF classifier (%)")
    ax.set_title("A  Separability against independent agreement")
    _grid(ax, "both")

    ax = axes[1]
    ax.scatter(diff["coverage_fraction"] * 100.0, diff["spike_pct"],
               s=46, color=CATEGORICAL[1], zorder=3)
    for k, r in diff.iterrows():
        ax.annotate(SHORT.get(k, k), (r["coverage_fraction"] * 100.0, r["spike_pct"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("record coverage (% of span with data)")
    ax.set_ylabel("rows flagged as spikes (%)")
    ax.set_title("B  Record quality")
    _grid(ax, "both")

    fig.suptitle("Task 8 -- what makes a machine hard", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task8_difficulty.png")


def figure_task8_headroom(dec: pd.DataFrame, out_dir: str) -> None:
    """
    Where the decision layer has anything to win.

    Panel A splits each machine's idle time into the part the plant already
    de-energises and the part it leaves energised -- only the latter is
    recoverable, which is the distinction Phase III had to make on one machine
    and can now be made on eight.  Panel B is the attainable head-room against
    what the proposed policy actually captured, both annualised over covered
    time.
    """
    m = list(dec.index)
    x = np.arange(len(m))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    # Two different degeneracies, and they must not share a symbol: a machine
    # with no energised-idle band has nothing to recover, whereas one whose
    # restart cost was never observed has an unmeasurable price for recovering
    # it.  Hatching is reserved for the first; the second is marked on the tick.
    degen = (dec["standby_power_w"].to_numpy(float) < 1.0)
    unident = ((dec["decision_problem"] == "degenerate").to_numpy() & ~degen
               if "decision_problem" in dec.columns else np.zeros(len(dec), bool))
    ax.bar(x, dec["off_h"], color=NEUTRAL, label="already de-energised (OFF)", zorder=3)
    ax.bar(x, dec["standby_h_energised"], bottom=dec["off_h"],
           color=CATEGORICAL[0], label="left energised (STANDBY)", zorder=3)
    # The STANDBY hours of a machine with no measurable energised-idle state
    # are not recoverable hours -- the cluster sits at ~0 W.  Hatching them
    # keeps the panel from implying that exhaust-fan-I holds 371 recoverable
    # hours, which is the reading the bar heights alone would invite.
    ax.bar(x[degen], dec["standby_h_energised"].to_numpy()[degen],
           bottom=dec["off_h"].to_numpy()[degen], color="none", hatch="///",
           edgecolor="white", linewidth=0, zorder=4,
           label="no energised-idle state (P$_{standby}$ < 1 W)")
    for xi, (o, s) in enumerate(zip(dec["off_h"], dec["standby_h_energised"])):
        if np.isfinite(s):
            ax.text(xi, o + s, f"{s:.0f} h", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("idle hours in the record")
    ax.set_title("A  Idle time, and how much of it is recoverable")
    ax.set_ylim(0, float(np.nanmax((dec["off_h"] + dec["standby_h_energised"]).to_numpy())) * 1.28)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    w = 0.38
    b1 = ax.bar(x - w / 2, dec["head_room_usd_yr"], w, color=INK,
                label="attainable head-room (oracle)", zorder=3)
    b2 = ax.bar(x + w / 2, dec["proposed_usd_yr"], w, color=CATEGORICAL[0],
                label="captured by the proposed policy", zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_ylabel("USD / year vs the status quo")
    ax.set_title("B  Decision-layer value per machine")

    # A machine whose policy loses heavily compresses every other bar, so the
    # axis is clipped to the head-room scale.  The clipped values are then
    # written INSIDE the axes -- never at the bar tip, which would sit far off
    # the canvas and drag the saved figure's bounding box with it.
    hr = dec["head_room_usd_yr"].replace([np.inf, -np.inf], np.nan).dropna()
    lim = max(abs(hr).max() * 2.2, 10) if len(hr) else 10
    ax.set_ylim(-lim, lim)
    _label_bars(ax, b1, dec["head_room_usd_yr"].to_numpy(), "{:.0f}",
                skip_out_of_range=True)
    _label_bars(ax, b2, dec["proposed_usd_yr"].to_numpy(), "{:.0f}",
                skip_out_of_range=True)
    for xi, v in enumerate(dec["proposed_usd_yr"]):
        if np.isfinite(v) and abs(v) > lim:
            ax.annotate(f"{v:,.0f}", xy=(xi + w / 2, -lim * 0.88 if v < 0 else lim * 0.88),
                        ha="center", va="center", fontsize=6.5, rotation=90,
                        color=CATEGORICAL[1], annotation_clip=True)
    ax.legend(fontsize=7, loc="upper right")

    labels = [f"{s}†" if u else s for s, u in zip(_short(m), unident)]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        _grid(ax)
    if unident.any():
        fig.text(0.5, -0.01,
                 "†  no cold start observed in the record: the restart cost is "
                 "not identifiable, so no policy is scored",
                 ha="center", fontsize=7, color=INK_MUTED)
    fig.suptitle("Task 8 -- decision-layer head-room across machines",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task8_headroom.png")


# ── Task 9 ────────────────────────────────────────────────────────────────────

def figure_task9_part_a(tab: pd.DataFrame, out_dir: str) -> None:
    """Decision-forecaster transfer: error raw vs scaled, and the currency it earns."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, variant, title in ((axes[0], "raw", "A  Forecast MAE, raw transfer"),
                               (axes[1], "scaled", "B  Forecast MAE, target-scaled")):
        sub = tab[tab["variant"] == variant]
        piv = sub.pivot(index="source", columns="target", values="mae_s") / 3600.0
        _heat(ax, piv, title, "MAE (h)", SEQ, fmt="{:.1f}")

    sub = tab[(tab["variant"] == "scaled") & tab["savings_usd_yr"].notna()]
    if len(sub):
        piv = sub.pivot(index="source", columns="target", values="savings_usd_yr")
        v = np.nanmax(np.abs(piv.to_numpy(float)))
        _heat(axes[2], piv, "C  Annual saving vs status quo (target-scaled)",
              "USD / year", DIV, norm=TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v),
              fmt="{:.0f}")
    else:
        axes[2].axis("off")

    fig.suptitle("Task 9A -- zero-shot transfer of the decision forecaster",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task9_decision_transfer.png")


def figure_task9_part_b(tab: pd.DataFrame, out_dir: str) -> None:
    """Sequence-forecaster transfer, with the training-free reference beside it."""
    models = tab[tab["source"] != "persistence"]
    pers = tab[tab["source"] == "persistence"].set_index("target")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, variant, title in (
            (axes[0], "source_scaler", "A  STANDBY F1, source scaler (true zero-shot)"),
            (axes[1], "target_scaler", "B  STANDBY F1, target scaler")):
        sub = models[models["variant"] == variant]
        if not len(sub):
            ax.axis("off")
            continue
        piv = sub.pivot(index="source", columns="target", values="f1_STANDBY")
        _heat(ax, piv, title, "F1 (STANDBY)", SEQ, fmt="{:.2f}")

    ax = axes[2]
    sub = models[models["variant"] == "target_scaler"]
    if len(sub) and len(pers):
        tgts = list(pers.index)
        x = np.arange(len(tgts))
        within = [sub[(sub.target == t) & (sub.same_machine)]["f1_STANDBY"].mean()
                  for t in tgts]
        trans = [sub[(sub.target == t) & (~sub.same_machine)]["f1_STANDBY"].median()
                 for t in tgts]
        base = [pers.loc[t, "f1_STANDBY"] for t in tgts]
        w = 0.27
        ax.bar(x - w, within, w, color=CATEGORICAL[0], label="trained on this machine",
               zorder=3)
        ax.bar(x, trans, w, color=CATEGORICAL[1], label="transferred (median)", zorder=3)
        ax.bar(x + w, base, w, color=NEUTRAL, label="persistence (no training)",
               zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(_short(tgts), rotation=30, ha="right")
        ax.set_ylabel("F1 (STANDBY)")
        ax.set_title("C  Transfer against the training-free reference")
        ax.legend(fontsize=7)
        _grid(ax)
    else:
        ax.axis("off")

    fig.suptitle("Task 9B -- zero-shot transfer of the Seq2Seq LSTM",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task9_sequence_transfer.png")


# ── Task 10 ───────────────────────────────────────────────────────────────────

def figure_task10(chars: dict, out_dir: str) -> None:
    """
    Cross-dataset validation, and why the control record matters.

    Panel A is the physics battery: how many checks each record can even be
    scored on.  Panel B is the diurnal productive profile, which is the
    evidence that separates a scheduled machine from a solar generator -- and
    which the battery itself never looks at.
    """
    keys = list(chars)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    x = np.arange(len(keys))
    passed = [chars[k]["physics"]["n_passed"] for k in keys]
    applicable = [chars[k]["physics"]["n_applicable"] for k in keys]
    ax.bar(x, [5] * len(keys), color=GRID, label="checks defined for IMDELD", zorder=2)
    ax.bar(x, applicable, color=NEUTRAL, label="checks evaluable here", zorder=3)
    b = ax.bar(x, passed, color=CATEGORICAL[0], label="checks passed", zorder=4)
    _label_bars(ax, b, np.array(passed, float), "{:.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels(_short(keys), rotation=0)
    ax.set_ylabel("number of physics checks")
    ax.set_title("A  Physics validation on a single-channel record")
    ax.legend(fontsize=7, loc="lower right")
    _grid(ax)

    ax = axes[1]
    for i, k in enumerate(keys):
        prof = chars[k].get("productive_pct_by_hour", {})
        hours = sorted(int(h) for h in prof)
        ax.plot(hours, [prof[str(h)] if str(h) in prof else prof[h] for h in hours],
                marker="o", markersize=3, linewidth=1.6,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=SHORT.get(k, k))
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("% of readings labelled productive")
    ax.set_title("B  Diurnal profile of the productive states")
    ax.legend(fontsize=7)
    _grid(ax, "both")

    fig.suptitle("Task 10 -- cross-dataset validation and the negative control",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"{out_dir}/fig_task10_cross_dataset.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    section("PHASE IV FIGURES")
    out_dir = ensure_dir(FIG_DIR)

    t8 = f"{PHASE4_DIR}/task8"
    if os.path.exists(f"{t8}/table_state_identification.csv"):
        state = pd.read_csv(f"{t8}/table_state_identification.csv", index_col=0)
        figure_task8_states(state, out_dir)
        if os.path.exists(f"{t8}/table_difficulty.csv"):
            diff = pd.read_csv(f"{t8}/table_difficulty.csv", index_col=0)
            figure_task8_difficulty(diff, state, out_dir)
    if os.path.exists(f"{t8}/table_decision.csv"):
        dec = pd.read_csv(f"{t8}/table_decision.csv", index_col=0)
        if len(dec):
            figure_task8_headroom(dec, out_dir)

    t9a = f"{PHASE4_DIR}/task9/part_a/transfer_matrix.csv"
    if os.path.exists(t9a):
        figure_task9_part_a(pd.read_csv(t9a), out_dir)
    t9b = f"{PHASE4_DIR}/task9/part_b/transfer_matrix.csv"
    if os.path.exists(t9b):
        figure_task9_part_b(pd.read_csv(t9b), out_dir)

    t10 = f"{PHASE4_DIR}/task10/task10_results.json"
    if os.path.exists(t10):
        with open(t10, encoding="utf-8") as fh:
            res = json.load(fh)
        if res.get("characterisation"):
            figure_task10(res["characterisation"], out_dir)

    section("PHASE IV FIGURES COMPLETE")


if __name__ == "__main__":
    main()
