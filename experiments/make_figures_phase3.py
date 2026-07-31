"""
============================================================
PHASE III RESULT FIGURES  --  experiments/make_figures_phase3.py
============================================================

PURPOSE
-------
Render the figures for Tasks 6 and 7.  Reads only the CSV/JSON/NPZ
written by the experiment scripts, so it re-runs in seconds without
recomputing anything.

    python -m experiments.make_figures_phase3

DESIGN NOTES
------------
Same validated palette and rules as `make_figures.py`, so a method
keeps its colour across Phase II and Phase III figures.  Consequences:

  * Categorical hues are assigned in FIXED slot order, never cycled.
    Three of the six sit below 3:1 contrast on a white surface, which
    is legal only because every bar also carries a direct value label.
  * Break-even and duration are ORDERED quantities, so where they are
    encoded in colour it is a single-hue light-to-dark ramp, never a
    rainbow.  Savings are SIGNED, so they use a diverging ramp with a
    neutral grey midpoint at zero -- the sign of a saving is the whole
    point and a sequential ramp would hide it.
  * No chart has two y-scales.  Currency and energy are different
    units and therefore always different panels.
  * Duration axes are logarithmic.  Idle episodes span four orders of
    magnitude and a linear axis collapses the entire decision-relevant
    range into the first pixel column.
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
from experiments.common import DEFAULT_MACHINE, ensure_dir, info, section  # noqa: E402
from experiments.task6_optimization import PHASE3_DIR                      # noqa: E402


# ── Palette (identical to make_figures.py; validated) ─────────────────────────

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
NEUTRAL = "#9a9894"

# Sequential: one hue, light -> dark.  Diverging: two poles, neutral midpoint.
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eaf1fb", "#104281"])
DIV = LinearSegmentedColormap.from_list("div_or_bl", ["#c0512a", "#f2f1ee", "#1c5ba8"])

# Fixed colour per policy, held across every Phase III figure.
POLICY_COLOR = {
    "observed": NEUTRAL,
    "never_shutdown": CATEGORICAL[4],
    "immediate": CATEGORICAL[3],
    "static_break_even": CATEGORICAL[1],
    "ski_rental": CATEGORICAL[2],
    "forecast_opt_median": "#b9d0ee",
    "forecast_opt_nochance": "#7fa9e0",
    "forecast_opt": CATEGORICAL[0],
    "oracle": INK,
}
POLICY_LABEL = {
    "never_shutdown": "Never shut down",
    "observed": "Observed (status quo)",
    "immediate": "Immediate shutdown",
    "static_break_even": "Static break-even",
    "ski_rental": "Ski-rental (no forecast)",
    "forecast_opt_median": "Forecast opt. (median estimand)",
    "forecast_opt_nochance": "Forecast opt. (no chance constr.)",
    "forecast_opt": "Forecast optimisation (proposed)",
    "oracle": "Oracle (offline optimum)",
}

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


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _label_bars(ax, bars, values, fmt="{:.1f}", horizontal=False, pad=0.015):
    span = (ax.get_xlim()[1] - ax.get_xlim()[0]) if horizontal else \
           (ax.get_ylim()[1] - ax.get_ylim()[0])
    for b, v in zip(bars, values):
        if not np.isfinite(v):
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


def _load(machine):
    t6 = f"{PHASE3_DIR}/task6/{machine}"
    t7 = f"{PHASE3_DIR}/task7/{machine}"
    with open(f"{t6}/task6_results.json", encoding="utf-8") as fh:
        r6 = json.load(fh)
    r7, curve, grid, c1, c2 = None, None, None, None, None
    if os.path.exists(f"{t7}/task7_results.json"):
        with open(f"{t7}/task7_results.json", encoding="utf-8") as fh:
            r7 = json.load(fh)
        curve = pd.read_csv(f"{t7}/attainability_curve.csv")
        grid = pd.read_csv(f"{t7}/economic_grid.csv")
        c1 = pd.read_csv(f"{t7}/tariff_invariance.csv")
        c2 = pd.read_csv(f"{t7}/constraint_sensitivity.csv")
    return {
        "t6": t6, "t7": t7, "r6": r6, "r7": r7,
        "policy": pd.read_csv(f"{t6}/policy_comparison.csv", index_col=0),
        "episodes": pd.read_csv(f"{t6}/idle_episodes.csv"),
        "forecast": np.load(f"{t6}/forecast_test.npz"),
        "curve": curve, "grid": grid, "c1": c1, "c2": c2,
    }


# ── Figure 1: the decision resource ───────────────────────────────────────────

def figure_episodes(d, machine, out_dir):
    """
    What the decision layer has to work with, and what any policy could at best
    recover from it.

    Panel A is the survival function of idle-episode duration.  It is the
    binding constraint on the whole framework: an episode shorter than the
    break-even point cannot be made profitable by any algorithm, so where the
    scenario markers fall on this curve determines the ceiling before a single
    model is fitted.
    """
    ep = d["episodes"]
    scen = d["r6"]["scenarios"]["break_even_s"]
    curve = d["curve"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # -- A: survival function ------------------------------------------------
    ax = axes[0]
    dur = np.sort(ep["duration_s"].to_numpy(float))
    surv = 1.0 - np.arange(len(dur)) / len(dur)
    ax.step(dur, surv * 100, where="post", color=CATEGORICAL[0], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("idle episode duration (s, log scale)")
    ax.set_ylabel("% of episodes at least this long")
    ax.set_title("A  Idle-episode survival")
    for i, (name, be) in enumerate(scen.items()):
        ax.axvline(be, color=INK_MUTED, linestyle="--", linewidth=1)
        frac = (dur > be).mean() * 100
        ax.text(be, 96 - i * 11, f" {name.split('_')[0]}  {be/60:.0f} min\n {frac:.1f}% above",
                fontsize=7, color=INK, va="top")
    _grid(ax)

    # -- B: attainable energy above break-even --------------------------------
    ax = axes[1]
    ax.plot(curve["break_even_h"], curve["attainable_kwh"],
            color=CATEGORICAL[0], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("break-even duration (h, log scale)")
    ax.set_ylabel("standby energy in reachable episodes (kWh)")
    ax.set_title("B  Attainable standby energy")
    longest = d["r6"]["episodes"]["max_duration_h"]
    ax.axvline(longest, color=INK_MUTED, linestyle=":", linewidth=1)
    ax.text(longest * 0.92, ax.get_ylim()[1] * 0.5, "longest\nepisode ",
            fontsize=7, color=INK_MUTED, va="top", ha="right")
    _grid(ax)

    # -- C: savings vs break-even, by policy ----------------------------------
    # Symmetric-log y: the interesting differences between policies live within
    # a few tens of USD of zero, while the far right of the sweep runs to
    # thousands (where restarts are so dear that the winning move is simply to
    # stop de-energising).  A linear axis renders the decision-relevant region
    # as a flat line on the zero mark.
    ax = axes[2]
    for key in ("oracle", "forecast_opt", "ski_rental", "never_shutdown"):
        ax.plot(curve["break_even_h"], curve[f"{key}_annual"],
                color=POLICY_COLOR[key], linewidth=2 if key == "forecast_opt" else 1.5,
                label=POLICY_LABEL[key])
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=50)
    ax.set_xlabel("break-even duration (h, log scale)")
    ax.set_ylabel("annual saving vs status quo (USD/yr, symlog)")
    ax.set_title("C  Where each policy pays")
    ax.legend(fontsize=7, loc="upper left")
    _grid(ax)

    fig.suptitle(f"Phase III / Task 6 -- decision resource and policy regimes "
                 f"({machine})", y=1.03, fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.32)
    _save(fig, f"{out_dir}/fig_task6_episodes.png")


# ── Figure 2: policy comparison ───────────────────────────────────────────────

def figure_policies(d, machine, out_dir):
    """
    Savings and oracle-capture for every policy, under each scenario.

    Both panels are needed and neither substitutes for the other: the left is
    what the plant banks, the right is what the research claim rests on.  A
    policy can bank little because the head-room is small while still capturing
    most of what exists.

    Drawn as small multiples -- one column per scenario -- rather than as
    grouped bars.  Grouping would have to distinguish the scenarios by shading
    the policy colours, which encodes two variables in one channel and makes
    the legend contradict the bars.  Separate panels also give each scenario
    its own scale, so the differences between the good policies are not
    flattened by the one policy that loses several hundred dollars.
    """
    tab = d["policy"].copy()
    tab["scenario"] = tab["scenario"].astype(str)
    be_s = d["r6"]["scenarios"]["break_even_s"]
    scenarios = list(be_s)
    order = [p for p in POLICY_LABEL if p in set(tab.index)]
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(2, len(scenarios), figsize=(15.0, 8.4))
    for j, sc in enumerate(scenarios):
        sub = tab[tab["scenario"] == sc].reindex(order)
        for i, (col, ylab, fmt) in enumerate([
            ("annual_savings_usd", "annual saving vs status quo (USD/yr)", "{:.0f}"),
            ("pct_of_oracle", "% of oracle head-room captured (clipped to ±120)",
             "{:.0f}"),
        ]):
            ax = axes[i, j]
            vals = sub[col].to_numpy(float)
            if col == "pct_of_oracle":
                vals = np.clip(vals, -120, 120)
            bars = ax.barh(y, vals, 0.68,
                           color=[POLICY_COLOR[p] for p in order],
                           edgecolor="white", linewidth=1.2)
            pad = max(abs(np.nanmin(vals)), abs(np.nanmax(vals))) * 0.28
            ax.set_xlim(min(0, np.nanmin(vals)) - pad, max(0, np.nanmax(vals)) + pad)
            _label_bars(ax, bars, vals, fmt=fmt, horizontal=True)
            ax.axvline(0, color=INK_MUTED, linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels([POLICY_LABEL[p] for p in order] if j == 0 else [],
                               fontsize=7.5)
            ax.set_xlabel(ylab, fontsize=8)
            if i == 0:
                ax.set_title(f"{sc}  --  break-even {be_s[sc]/60:.0f} min",
                             fontsize=9.5)
            _grid(ax, axis="x")

    fig.suptitle(f"Phase III / Task 6 -- policy comparison on held-out episodes "
                 f"({machine}).  Top row: currency saved against what the plant "
                 f"actually did.  Bottom row: share of the attainable head-room.",
                 y=0.985, fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(hspace=0.30, wspace=0.10, top=0.90)
    _save(fig, f"{out_dir}/fig_task6_policies.png")


# ── Figure 3: forecaster ──────────────────────────────────────────────────────

def figure_forecaster(d, machine, out_dir):
    """
    Why the estimand matters.

    Panel A bins decision epochs by predicted remaining time and plots the mean
    realised remaining time in each bin: a calibrated mean forecast lies on the
    diagonal.  Panel B shows the same for the log-scale model, which tracks the
    conditional median and therefore sits far below the diagonal wherever the
    duration distribution is heavy-tailed -- the bias that made the policy wait
    too long before acting.
    """
    f = d["forecast"]
    actual = f["actual_remaining_s"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    for ax, key, title, colour in [
        (axes[0], "pred_mean_s", "A  Mean-targeting model (used)", CATEGORICAL[0]),
        (axes[1], "pred_median_s", "B  Log-scale model (ablation)", CATEGORICAL[1]),
    ]:
        pred = f[key]
        qs = np.unique(np.quantile(pred, np.linspace(0, 1, 13)))
        idx = np.clip(np.digitize(pred, qs[1:-1]), 0, len(qs) - 2)
        px, ay = [], []
        for b in range(len(qs) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            px.append(pred[m].mean())
            ay.append(actual[m].mean())
        lo = max(1.0, min(min(px), min(ay)) * 0.5)
        lim = [lo, max(max(px), max(ay)) * 2.0]
        ax.plot(lim, lim, color=INK_MUTED, linestyle="--", linewidth=1,
                label="perfect calibration")
        ax.scatter(px, ay, s=42, color=colour, edgecolor="white", linewidth=1.2,
                   zorder=3, label="decile of prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("mean predicted remaining idle time (s)")
        ax.set_ylabel("mean realised remaining idle time (s)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")
        _grid(ax, axis="both")

    # -- C: chance-constraint reliability -------------------------------------
    ax = axes[2]
    p_ok = f["p_feasible"]
    min_off = d["r6"]["forecaster"].get("feasibility", {}).get("min_off_s", 600.0)
    y_ok = (actual >= min_off).astype(float)
    if np.isfinite(p_ok).any():
        bins = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(p_ok, bins[1:-1]), 0, 9)
        xs, ys = [], []
        for k in range(10):
            m = b == k
            if m.sum() < 20:
                continue
            xs.append(p_ok[m].mean())
            ys.append(y_ok[m].mean())
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1,
                label="perfect reliability")
        ax.plot(xs, ys, marker="o", markersize=6, color=CATEGORICAL[2],
                linewidth=2, markeredgecolor="white", label="observed frequency")
        conf = d["r6"]["forecaster"].get("feasibility", {}).get("confidence_level", 0.9)
        ax.axvline(conf, color=INK_MUTED, linestyle=":", linewidth=1)
        ax.text(conf, 0.06, f" operating\n point {conf:.2f}", fontsize=7,
                color=INK_MUTED)
        ax.set_xlabel(f"predicted P(remaining >= {min_off:.0f}s)")
        ax.set_ylabel("observed frequency")
        ax.set_title("C  Chance-constraint reliability")
        ax.legend(fontsize=7, loc="upper left")
        _grid(ax, axis="both")

    fig.suptitle(f"Phase III / Task 6 -- forecaster calibration ({machine})",
                 y=1.03, fontsize=11, fontweight="bold")
    _save(fig, f"{out_dir}/fig_task6_forecaster.png")


# ── Figure 4: the economic plane ──────────────────────────────────────────────

def figure_economic_plane(d, machine, out_dir, tariff=0.12):
    """
    Break-even and realised saving over the two coefficients that actually
    carry information.

    Panel A uses a sequential ramp because break-even is a magnitude; panel B
    uses a diverging ramp centred on zero because a saving has a sign, and
    the boundary between "the policy helps" and "the policy hurts" is the
    single most decision-relevant contour on the map.
    """
    g = d["grid"]
    g = g[np.isclose(g["tariff_per_kwh"], tariff)]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.4))
    fig.subplots_adjust(wspace=0.46)

    piv_be = g.pivot(index="restart_energy_kwh", columns="delay_cost_per_hour",
                     values="break_even_h")
    piv_sav = g.pivot(index="restart_energy_kwh", columns="delay_cost_per_hour",
                      values="forecast_opt_annual")
    piv_kwh = g.pivot(index="restart_energy_kwh", columns="delay_cost_per_hour",
                      values="forecast_opt_energy_saved_kwh")

    def _heat(ax, piv, title, cbar_label, cmap, norm=None, fmt="{:.1f}"):
        im = ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r:g}" for r in piv.index], fontsize=7)
        ax.set_xlabel("production-delay valuation (USD/h)")
        ax.set_ylabel("restart energy (kWh)")
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=6, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=7)

    _heat(axes[0], piv_be, "A  Derived break-even duration",
          "hours", SEQ, fmt="{:.1f}")
    lim = float(np.nanmax(np.abs(piv_sav.values))) or 1.0
    _heat(axes[1], piv_sav, "B  Proposed policy: annual saving",
          "USD/yr", DIV, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
          fmt="{:.0f}")
    lim2 = float(np.nanmax(np.abs(piv_kwh.values))) or 1.0
    _heat(axes[2], piv_kwh, "C  Proposed policy: energy saved",
          "kWh over test window", DIV,
          norm=TwoSlopeNorm(vmin=-lim2, vcenter=0, vmax=lim2), fmt="{:.0f}")

    fig.suptitle(f"Phase III / Task 7 -- economic plane at tariff "
                 f"{tariff:.2f} USD/kWh ({machine}).  Panels B and C can "
                 f"disagree in sign: avoiding restarts saves money while "
                 f"spending standby energy.",
                 y=1.04, fontsize=10.5, fontweight="bold")
    _save(fig, f"{out_dir}/fig_task7_economic_plane.png")


# ── Figure 5: tariff invariance and constraint sensitivity ────────────────────

def figure_sensitivity(d, machine, out_dir):
    """
    Panel A is the correction to the planned analysis: with a purely electrical
    restart cost the tariff cancels out of the break-even expression, so the
    line is flat and a tariff-vs-break-even heat-map carries no information.
    Non-energy restart costs are what make price matter, and the curvature of
    the other two lines is that dependence.
    """
    c1, c2 = d["c1"], d["c2"]
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.8))

    ax = axes[0]
    for i, (lab, sub) in enumerate(c1.groupby("restart_labour_cost")):
        sub = sub.sort_values("tariff_per_kwh")
        ax.plot(sub["tariff_per_kwh"], sub["break_even_h"], marker="o",
                markersize=5, linewidth=2, color=CATEGORICAL[i],
                markeredgecolor="white",
                label=f"non-energy restart cost {lab:.2f} USD")
    ax.set_xlabel("electricity tariff (USD/kWh)")
    ax.set_ylabel("break-even duration (h)")
    ax.set_title("A  Break-even vs tariff")
    ax.legend(fontsize=7)
    _grid(ax)

    for k, (knob, title, xlab, fmt) in enumerate([
        ("chance_confidence", "B  Chance-constraint confidence",
         "required P(remaining >= min-off)", "{:g}"),
        ("max_restarts_per_day", "C  Restarts-per-day cap", "cap", "{:g}"),
        ("min_off_s", "D  Minimum off time", "seconds", "{:g}"),
    ]):
        ax = axes[k + 1]
        sub = c2[c2["knob"] == knob]
        xs = np.arange(len(sub))
        bars = ax.bar(xs, sub["annual"].to_numpy(float), 0.62,
                      color=CATEGORICAL[0], edgecolor="white", linewidth=1.2)
        _label_bars(ax, bars, sub["annual"].to_numpy(float), fmt="{:.0f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([fmt.format(v) for v in sub["value"]], fontsize=7)
        ax.set_xlabel(xlab)
        ax.set_ylabel("annual saving (USD/yr)")
        ax.set_title(title)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        _grid(ax)

    fig.suptitle(f"Phase III / Task 7 -- sensitivity to the stated constants "
                 f"({machine})", y=1.04, fontsize=11, fontweight="bold")
    _save(fig, f"{out_dir}/fig_task7_sensitivity.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = DEFAULT_MACHINE):
    section(f"PHASE III FIGURES -- {machine}")
    out_dir = ensure_dir(f"{PHASE3_DIR}/figures/{machine}")
    d = _load(machine)

    figure_policies(d, machine, out_dir)
    figure_forecaster(d, machine, out_dir)
    if d["r7"] is not None:
        figure_episodes(d, machine, out_dir)
        figure_economic_plane(d, machine, out_dir)
        figure_sensitivity(d, machine, out_dir)
    else:
        info("Task 7 outputs not found -- skipping the sensitivity figures")
    section("FIGURES COMPLETE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase III figures")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    main(**vars(ap.parse_args()))
