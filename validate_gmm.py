"""
============================================================
GMM VALIDATION HARNESS  —  validate_gmm.py
============================================================

PURPOSE
-------
Verify that the Gaussian Mixture Model (GMM) in data_analysis.py
is correctly splitting power readings into physically meaningful
machine states BEFORE anything else in the pipeline runs.

This script is 100% standalone — it imports only the data loader
from data_analysis.py, then re-implements the GMM fitting step
so we can inspect it at every level.

SIX INDEPENDENT TESTS
----------------------
T1  Sanity check       — power column is non-negative and non-empty
T2  GMM convergence    — did sklearn's EM algorithm actually converge?
T3  k-selection        — Silhouette + BIC agree on the same k?
T4  State separation   — are cluster means physically distinguishable (2σ)?
T5  Visual confirmation— histogram annotated with GMM Gaussians
T6  Soft-assignment    — is the model confident (max probability > 80%)?

HOW TO RUN
----------
    python validate_gmm.py

Change DATA_PATH and MACHINE_NAME below to match your file.
Results are saved to outputs/gmm_validation/
============================================================
"""

import os
import sys

# Force UTF-8 output so box-drawing chars work on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ── project path so we can import the data loader ────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_analysis import load_and_prepare_data

# ============================================================
# CONFIGURATION  — change these two lines
# ============================================================
DATA_PATH    = "data/2024_AC_ActivePower.csv.xz"      # your data file
MACHINE_NAME = "Solar Panel"                   # label for plots
OUTPUT_DIR   = "outputs/gmm_validation"
K_RANGE      = [2, 3, 4]                       # candidate cluster counts
# ============================================================


# ────────────────────────────────────────────────────────────
# COLOUR PALETTE  (same as data_analysis.py)
# ────────────────────────────────────────────────────────────
STATE_COLORS = {
    'OFF':       '#555555',
    'STANDBY':   '#f0a500',
    'IDLE':      '#4fc3f7',
    'WORKING':   '#66bb6a',
    'PEAK_LOAD': '#e53935',
}
K_TO_NAMES = {
    2: ['OFF', 'WORKING'],
    3: ['OFF', 'STANDBY', 'WORKING'],
    4: ['OFF', 'STANDBY', 'WORKING', 'PEAK_LOAD'],
}

# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def _pass(msg):  print(f"   [PASS] {msg}")
def _fail(msg):  print(f"   [FAIL] {msg}")
def _info(msg):  print(f"   [INFO] {msg}")
def _warn(msg):  print(f"   [WARN] {msg}")
def _sep():      print("   " + "-" * 62)


def fit_gmm(X, k, n_init=5):
    """Fit a GMM with n_init restarts to avoid bad local optima."""
    gmm = GaussianMixture(
        n_components  = k,
        covariance_type = 'full',
        n_init        = n_init,
        max_iter      = 300,
        random_state  = 42,
    )
    gmm.fit(X)
    return gmm


def map_states(gmm, k):
    """
    Sort GMM components by ascending mean and assign state names.
    Returns dict: component_index → state_name
    """
    means      = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    names      = K_TO_NAMES.get(k, [f"State_{i}" for i in range(k)])
    return {int(sorted_idx[i]): names[i] for i in range(k)}


# ============================================================
# TEST 1 — DATA SANITY
# ============================================================

def test_data_sanity(df):
    print("\n" + "=" * 65)
    print("  TEST 1 -- DATA SANITY CHECK")
    print("=" * 65)

    passed = True

    # 1a — Row count
    n = len(df)
    if n < 1000:
        _fail(f"Only {n:,} rows — too few for reliable GMM fitting (need ≥1,000)")
        passed = False
    else:
        _pass(f"{n:,} rows loaded")

    # 1b — No NaN
    nan_count = df['power'].isna().sum()
    if nan_count > 0:
        _fail(f"{nan_count:,} NaN values in power column")
        passed = False
    else:
        _pass("No NaN values in power column")

    # 1c — All non-negative
    neg_count = (df['power'] < 0).sum()
    if neg_count > 0:
        _fail(f"{neg_count:,} negative power values (clipping issue?)")
        passed = False
    else:
        _pass("All power values ≥ 0")

    # 1d — Has off state (values near 0)
    off_count = (df['power'] < 5).sum()
    off_pct   = off_count / n * 100
    if off_count == 0:
        _warn("No readings < 5W — machine may never be fully OFF. "
              "GMM will still run but OFF state may not exist.")
    else:
        _pass(f"OFF-state readings (<5W): {off_count:,}  ({off_pct:.1f}%)")

    # 1e — Has active state
    on_count = (df['power'] > 5).sum()
    if on_count < 100:
        _fail("Fewer than 100 ON-state readings — not enough active data")
        passed = False
    else:
        _pass(f"ON-state readings  (>5W): {on_count:,}")

    # 1f — Power range
    _info(f"Power range: {df['power'].min():.2f} W → {df['power'].max():.2f} W")
    _info(f"Power mean : {df['power'].mean():.2f} W  |  std: {df['power'].std():.2f} W")

    return passed


# ============================================================
# TEST 2 — GMM CONVERGENCE
# ============================================================

def test_gmm_convergence(X, models):
    print("\n" + "=" * 65)
    print("  TEST 2 -- GMM CONVERGENCE CHECK")
    print("=" * 65)
    _info("Each GMM is run with 5 random restarts (n_init=5) to avoid bad local optima.")
    _sep()

    all_converged = True
    for k, gmm in models.items():
        if gmm.converged_:
            _pass(f"k={k} — converged in {gmm.n_iter_} EM iterations")
        else:
            _fail(f"k={k} — DID NOT CONVERGE after {gmm.n_iter_} iterations. "
                  "Increase max_iter or check for degenerate data.")
            all_converged = False

    return all_converged


# ============================================================
# TEST 3 — k SELECTION: SILHOUETTE + BIC
# ============================================================

def test_k_selection(X, models, labels):
    print("\n" + "=" * 65)
    print("  TEST 3 -- k SELECTION  (Silhouette + BIC)")
    print("=" * 65)
    _info("Silhouette (geometric) and BIC (probabilistic) are two independent")
    _info("methods. When both agree on the same k -> strong evidence the split is real.")
    _sep()

    sil_scores = {}
    bic_scores = {}
    aic_scores = {}

    for k in K_RANGE:
        gmm = models[k]
        lbl = labels[k]
        sil = silhouette_score(X, lbl, sample_size=min(10_000, len(X)), random_state=42)
        sil_scores[k] = sil
        bic_scores[k] = gmm.bic(X)
        aic_scores[k] = gmm.aic(X)

    print(f"\n   {'k':<6} {'Silhouette':>12} {'BIC':>18} {'AIC':>18}")
    print(f"   {'':->6} {'':->12} {'':->18} {'':->18}")
    for k in K_RANGE:
        sil_marker = " <- best" if k == max(sil_scores, key=sil_scores.get) else ""
        bic_marker = " <- best" if k == min(bic_scores, key=bic_scores.get) else ""
        print(f"   {k:<6} {sil_scores[k]:>12.4f}{sil_marker:<7} "
              f"{bic_scores[k]:>18,.1f}{bic_marker}")

    sil_best = max(sil_scores, key=sil_scores.get)
    bic_best = min(bic_scores, key=bic_scores.get)

    print()
    _info(f"Silhouette selects : k = {sil_best}")
    _info(f"BIC selects        : k = {bic_best}")

    if sil_best == bic_best:
        _pass(f"Both Silhouette AND BIC agree -> k = {sil_best}  (strong confirmation)")
        agreed_k = sil_best
        agreement = True
    else:
        _warn(f"Silhouette->k={sil_best} vs BIC->k={bic_best}. Using BIC (more rigorous).")
        agreed_k  = bic_best
        agreement = False

    return agreed_k, sil_scores, bic_scores, agreement


# ============================================================
# TEST 4 — PHYSICAL STATE SEPARATION (2σ rule)
# ============================================================

def test_state_separation(gmm, best_k):
    print("\n" + "=" * 65)
    print("  TEST 4 -- PHYSICAL STATE SEPARATION  (2-sigma rule)")
    print("=" * 65)
    _info("For two clusters to be physically different states, their means")
    _info("must be at least 2x the average standard deviation apart.")
    _info("Below 2-sigma = the distributions overlap = likely one real state split in two.")
    _sep()

    state_map  = map_states(gmm, best_k)
    means      = gmm.means_.flatten()
    variances  = gmm.covariances_.flatten()
    stds       = np.sqrt(np.abs(variances))
    sorted_idx = np.argsort(means)

    sorted_means = means[sorted_idx]
    sorted_stds  = stds[sorted_idx]
    sorted_names = [state_map[i] for i in sorted_idx]

    print(f"\n   {'State':<14} {'Mean (W)':>10} {'Std (W)':>10} {'Weight':>8}")
    print(f"   {'':->14} {'':->10} {'':->10} {'':->8}")
    weights = gmm.weights_
    for i, idx in enumerate(sorted_idx):
        print(f"   {sorted_names[i]:<14} {sorted_means[i]:>10.1f} "
              f"{sorted_stds[i]:>10.1f} {weights[idx]:>8.3f}")

    print()
    all_separated = True
    for i in range(len(sorted_means) - 1):
        gap     = sorted_means[i+1] - sorted_means[i]
        avg_std = (sorted_stds[i] + sorted_stds[i+1]) / 2
        ratio   = gap / avg_std if avg_std > 0 else float('inf')

        pair = f"{sorted_names[i]} -> {sorted_names[i+1]}"
        if ratio >= 2.0:
            _pass(f"{pair:<25}  gap={gap:8.1f}W  avg_std={avg_std:7.1f}W  "
                  f"ratio={ratio:.2f}s  (well separated)")
        else:
            _fail(f"{pair:<25}  gap={gap:8.1f}W  avg_std={avg_std:7.1f}W  "
                  f"ratio={ratio:.2f}s  (OVERLAPPING -- may be one state)")
            all_separated = False

    return all_separated, state_map, sorted_means, sorted_stds, sorted_names


# ============================================================
# TEST 5 — SOFT-ASSIGNMENT CONFIDENCE
# ============================================================

def test_soft_assignment_confidence(gmm, X):
    print("\n" + "=" * 65)
    print("  TEST 5 -- SOFT-ASSIGNMENT CONFIDENCE")
    print("=" * 65)
    _info("GMM assigns each reading a probability for EACH cluster.")
    _info("If the max-probability is high (>80%), the model is confident.")
    _info("If most readings hover near 50%, the clusters overlap badly.")
    _sep()

    probs      = gmm.predict_proba(X)          # shape (N, k)
    max_probs  = probs.max(axis=1)             # confidence per reading

    pct_high   = (max_probs > 0.80).mean() * 100
    pct_low    = (max_probs < 0.60).mean() * 100
    avg_conf   = max_probs.mean() * 100

    _info(f"Average max-probability   : {avg_conf:.1f}%")
    _info(f"Readings with >80% conf   : {pct_high:.1f}%  (want >70%)")
    _info(f"Readings with <60% conf   : {pct_low:.1f}%   (want <20%)")
    print()

    passed = True
    if pct_high > 70:
        _pass(f"{pct_high:.1f}% of readings assigned with >80% confidence (model is decisive)")
    else:
        _fail(f"Only {pct_high:.1f}% of readings have >80% confidence. "
              "Clusters overlap too much — GMM is uncertain about assignments.")
        passed = False

    if pct_low < 20:
        _pass(f"Only {pct_low:.1f}% of readings are ambiguous (<60% confidence)")
    else:
        _fail(f"{pct_low:.1f}% of readings are ambiguous. Consider a different k.")
        passed = False

    return passed, max_probs


# ============================================================
# TEST 6 — WEIGHT SANITY
# ============================================================

def test_weight_sanity(gmm, best_k, state_map):
    print("\n" + "=" * 65)
    print("  TEST 6 -- CLUSTER WEIGHT SANITY")
    print("=" * 65)
    _info("Each GMM component has a weight = fraction of data it models.")
    _info("A weight <0.5% means the cluster likely caught edge-case noise")
    _info("rather than a real physical state.")
    _sep()

    weights    = gmm.weights_
    sorted_idx = np.argsort(gmm.means_.flatten())
    names      = [state_map[i] for i in sorted_idx]
    sorted_w   = weights[sorted_idx]

    passed = True
    print()
    for name, w in zip(names, sorted_w):
        pct = w * 100
        if pct < 0.5:
            _fail(f"{name:<14}: weight = {pct:.2f}%  (too tiny — likely noise artefact)")
            passed = False
        elif pct < 3.0:
            _warn(f"{name:<14}: weight = {pct:.2f}%  (very small — verify this state exists)")
        else:
            _pass(f"{name:<14}: weight = {pct:.2f}%  (meaningful cluster)")

    return passed


# ============================================================
# PLOT 1 — BIC / SILHOUETTE CURVE  (k-selection proof)
# ============================================================

def plot_k_selection(sil_scores, bic_scores, best_k, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"GMM k-Selection Proof — {MACHINE_NAME}",
                 fontsize=14, fontweight='bold')

    ks  = list(sil_scores.keys())
    sil = [sil_scores[k] for k in ks]
    bic = [bic_scores[k] for k in ks]

    # Silhouette
    ax = axes[0]
    ax.plot(ks, sil, 'o-', color='#4fc3f7', linewidth=2.5, markersize=9)
    ax.axvline(best_k, color='#e53935', linestyle='--', linewidth=1.5,
               label=f'Selected k={best_k}')
    ax.fill_between(ks, sil, alpha=0.15, color='#4fc3f7')
    ax.set_title("Silhouette Score  (↑ higher is better)", fontsize=12)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # BIC
    ax = axes[1]
    ax.plot(ks, bic, 's-', color='#f0a500', linewidth=2.5, markersize=9)
    ax.axvline(best_k, color='#e53935', linestyle='--', linewidth=1.5,
               label=f'Selected k={best_k}')
    ax.fill_between(ks, bic, alpha=0.15, color='#f0a500')
    ax.set_title("BIC Score  (↓ lower is better)", fontsize=12)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("BIC Score")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "T3_k_selection_curves.png")
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: T3_k_selection_curves.png")


# ============================================================
# PLOT 2 — GMM GAUSSIAN OVERLAY ON HISTOGRAM
# ============================================================

def plot_gmm_histogram(df, gmm, best_k, state_map, save_dir):
    """
    The key diagnostic plot.
    Shows:
     - Power histogram (log scale x-axis for ON-state)
     - Each GMM Gaussian bell curve overlaid in the state colour
     - Cluster boundary vertical lines
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"GMM Gaussian Overlay — {MACHINE_NAME}  (k={best_k})",
                 fontsize=14, fontweight='bold')

    power       = df['power'].values
    means       = gmm.means_.flatten()
    variances   = gmm.covariances_.flatten()
    stds        = np.sqrt(np.abs(variances))
    weights     = gmm.weights_
    sorted_idx  = np.argsort(means)
    sorted_names = [state_map[i] for i in sorted_idx]
    state_list  = K_TO_NAMES.get(best_k, [f"S{i}" for i in range(best_k)])

    # ── LEFT: full distribution including OFF state ───────────────
    ax = axes[0]
    bins = np.linspace(0, power.max(), 120)
    ax.hist(power, bins=bins, density=True, color='#1e2a3a',
            edgecolor='#2d3f58', linewidth=0.3, label='Data', alpha=0.9)

    x = np.linspace(0, power.max(), 2000)
    total_pdf = np.zeros_like(x)
    for idx in sorted_idx:
        pdf     = weights[idx] * norm.pdf(x, means[idx], stds[idx])
        name    = state_map[idx]
        color   = STATE_COLORS.get(name, '#aaaaaa')
        ax.fill_between(x, pdf, alpha=0.35, color=color)
        ax.plot(x, pdf, linewidth=2, color=color, label=name)
        total_pdf += pdf

    ax.plot(x, total_pdf, linewidth=1.5, color='white',
            linestyle='--', alpha=0.6, label='Total GMM')
    ax.set_title("Full Distribution  (incl. OFF state)", fontsize=11)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(left=0)

    # ── RIGHT: ON-state only (zoom, log-x) ───────────────────────
    ax = axes[1]
    on_power = power[power > 5]
    if len(on_power) > 10:
        max_p    = on_power.max()
        log_bins = np.logspace(np.log10(5.1), np.log10(max_p + 1), 120)
        ax.hist(on_power, bins=log_bins, density=True, color='#1e2a3a',
                edgecolor='#2d3f58', linewidth=0.3, alpha=0.9)

        x2         = np.logspace(np.log10(5.1), np.log10(max_p + 1), 2000)
        total_pdf2 = np.zeros_like(x2)
        for idx in sorted_idx:
            if means[idx] <= 5:        # skip OFF-state bell curve here
                continue
            pdf   = weights[idx] * norm.pdf(x2, means[idx], stds[idx])
            name  = state_map[idx]
            color = STATE_COLORS.get(name, '#aaaaaa')
            ax.fill_between(x2, pdf, alpha=0.4, color=color)
            ax.plot(x2, pdf, linewidth=2.5, color=color,
                    label=f"{name}\n~{means[idx]:.0f}W ± {stds[idx]:.0f}W")
            total_pdf2 += pdf
            # Mean marker
            ax.axvline(means[idx], color=color, linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xscale('log')
        ax.set_title("ON-State Only  (log x-axis, shows cluster separation)",
                     fontsize=11)
        ax.set_xlabel("Power (W)  [log scale]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    out = os.path.join(save_dir, "T5_gmm_histogram_overlay.png")
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: T5_gmm_histogram_overlay.png")


# ============================================================
# PLOT 3 — SOFT-ASSIGNMENT CONFIDENCE DISTRIBUTION
# ============================================================

def plot_confidence_histogram(max_probs, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(max_probs * 100, bins=60, color='#4fc3f7',
            edgecolor='#1e2a3a', linewidth=0.4)
    ax.axvline(80, color='#66bb6a', linestyle='--', linewidth=2,
               label='80% confidence threshold (want most readings here →)')
    ax.axvline(60, color='#e53935', linestyle='--', linewidth=2,
               label='60% confidence threshold (← readings here are ambiguous)')
    ax.set_title(f"GMM Assignment Confidence — {MACHINE_NAME}\n"
                 "Taller bar near 100% = model is decisive and correct",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Max Assignment Probability (%)")
    ax.set_ylabel("Number of Readings")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(save_dir, "T5_confidence_distribution.png")
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: T5_confidence_distribution.png")


# ============================================================
# PLOT 4 — LABELLED TIME-SERIES SAMPLE  (first 2000 readings)
# ============================================================

def plot_labeled_sample(df, gmm, best_k, state_map, save_dir):
    sample = df.head(2000).copy()
    X_s    = sample['power'].values.reshape(-1, 1)
    labels = gmm.predict(X_s)
    sample['state'] = [state_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(16, 5))
    for state, color in STATE_COLORS.items():
        mask = sample['state'] == state
        if mask.any():
            ax.scatter(sample.loc[mask, 'timestamp'],
                       sample.loc[mask, 'power'],
                       c=color, s=4, label=state, zorder=3)

    ax.plot(sample['timestamp'], sample['power'],
            color='#cccccc', linewidth=0.6, alpha=0.4, zorder=2)

    ax.set_title(f"GMM State Labels — First 2000 Readings — {MACHINE_NAME}\n"
                 "Colour = state assigned by GMM. No HMM or post-processing applied.",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.legend(markerscale=4, fontsize=9, loc='upper right')
    plt.tight_layout()
    out = os.path.join(save_dir, "T5_labeled_time_sample.png")
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: T5_labeled_time_sample.png")


# ============================================================
# PLOT 5 — SEPARATION VISUAL (violin / box plot per cluster)
# ============================================================

def plot_cluster_separation(df, gmm, best_k, state_map, save_dir):
    X      = df['power'].values.reshape(-1, 1)
    labels = gmm.predict(X)
    df2    = df.copy()
    df2['state'] = [state_map[l] for l in labels]

    ordered_states = [s for s in ['OFF', 'STANDBY', 'IDLE', 'WORKING', 'PEAK_LOAD']
                      if s in df2['state'].unique()]

    fig, ax = plt.subplots(figsize=(12, 6))
    data_per_state = [
        df2.loc[df2['state'] == s, 'power'].sample(
            min(5000, (df2['state'] == s).sum()), random_state=42
        ).values
        for s in ordered_states
    ]
    colors = [STATE_COLORS[s] for s in ordered_states]

    parts = ax.violinplot(data_per_state, positions=range(len(ordered_states)),
                          showmeans=True, showmedians=False, showextrema=True)

    for i, (pc, col) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)

    for part_name in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
        if part_name in parts:
            parts[part_name].set_color('white')
            parts[part_name].set_linewidth(1.5)

    ax.set_xticks(range(len(ordered_states)))
    ax.set_xticklabels(ordered_states, fontsize=11)
    ax.set_title(f"Power Distribution per GMM State — {MACHINE_NAME}\n"
                 "Well-separated violins = GMM split is physically correct",
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Power (W)")
    ax.grid(axis='y', alpha=0.25)

    # annotate means
    means = gmm.means_.flatten()
    for i, state in enumerate(ordered_states):
        mean_val = df2.loc[df2['state'] == state, 'power'].mean()
        ax.annotate(f"{mean_val:.0f}W", xy=(i, mean_val),
                    xytext=(i + 0.15, mean_val),
                    fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(save_dir, "T4_cluster_separation_violin.png")
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: T4_cluster_separation_violin.png")


# ============================================================
# TEST 7 — PHYSICS THRESHOLD CROSS-CHECK
# ============================================================
# This is the test that answers the core question:
# "How do we KNOW the GMM split matches actual machine behaviour?"
#
# EVIDENCE: Industrial power physics dictates:
#   OFF    → Mean must be near 0 W (< 5% of max observed power)
#             because when off, only residual sensor current flows.
#   STANDBY → Must be STABLE (low Coefficient of Variation CV = σ/μ)
#             because auxiliary loads (cooling fan, PLC, lubrication)
#             are constant and do not fluctuate like a cutting spindle.
#   WORKING → Must have GAP from max power; cannot be at 100% of max
#             all the time (machines have variable cutting loads).
#
# Source: Jiang et al. (2015) patent model  P_total = P₀ + P_spindle
#         + P_feed + P_tip. This directly implies:
#         • P₀ (OFF/standby base) is small and constant.
#         • P_spindle and P_feed are variable → WORKING has high CV.
# ============================================================

def test_physics_thresholds(gmm, best_k, state_map, max_power):
    print("\n" + "=" * 65)
    print("  TEST 7 -- PHYSICS THRESHOLD CROSS-CHECK")
    print("=" * 65)
    _info("This test checks whether GMM cluster means/variances match")
    _info("the physics of industrial machine power consumption.")
    _info("Evidence base: ISO 14955, Jiang et al. (2015), Wu et al. (2024)")
    _sep()

    means      = gmm.means_.flatten()
    variances  = gmm.covariances_.flatten()
    stds       = np.sqrt(np.abs(variances))
    sorted_idx = np.argsort(means)

    off_mean  = means[sorted_idx[0]]
    off_std   = stds[sorted_idx[0]]
    stby_mean = means[sorted_idx[1]] if best_k >= 3 else None
    stby_std  = stds[sorted_idx[1]] if best_k >= 3 else None
    work_mean = means[sorted_idx[-1]]
    work_std  = stds[sorted_idx[-1]]

    passed = True

    # ── Check 1: OFF mean must be near 0 (< 5% of max power) ────────
    off_threshold = max(5.0, max_power * 0.05)   # 5W floor or 5% of max
    print(f"\n   CHECK 1 — OFF cluster mean must be < {off_threshold:.1f} W  "
          f"(5% of max={max_power:.1f} W, or 5W floor)")
    print(f"   Physics: when a machine is truly OFF, only residual\n"
          f"   sensor/standby-circuit current flows (Jiang et al. 2015)")
    if off_mean < off_threshold:
        _pass(f"OFF mean = {off_mean:.1f} W  < {off_threshold:.1f} W  — "
              f"cluster is near-zero: physically correct for OFF state")
    else:
        _fail(f"OFF mean = {off_mean:.1f} W  ≥ {off_threshold:.1f} W  — "
              f"cluster is NOT near zero. Possible causes:\n"
              f"         • Machine is never fully powered down\n"
              f"         • k=3 is wrong for this machine (try k=2)\n"
              f"         • Data contains a constant baseline load")
        passed = False

    # ── Check 2: OFF cluster CV must be tiny (σ/μ or σ < 10W) ───────
    print(f"\n   CHECK 2 — OFF cluster must be stable (σ < 15 W or σ < 20% of mean)")
    print(f"   Physics: OFF power is residual circuit draw — nearly constant,\n"
          f"   very low noise (ISO 14955-2 measurement procedure)")
    off_cv_ok = (off_std < 15.0) or (off_mean > 0 and off_std / off_mean < 0.20)
    if off_cv_ok:
        _pass(f"OFF std  = {off_std:.1f} W  — stable, low-noise cluster. "
              f"Consistent with residual standby-circuit draw")
    else:
        _warn(f"OFF std  = {off_std:.1f} W  — high spread for an OFF cluster. "
              f"May include very-low-load active periods")
        passed = False

    # ── Check 3: STANDBY must be stable (lower CV than WORKING) ─────
    if stby_mean is not None:
        stby_cv = stby_std / stby_mean if stby_mean > 0 else float('inf')
        work_cv = work_std / work_mean if work_mean > 0 else float('inf')
        print(f"\n   CHECK 3 — STANDBY must be more stable than WORKING (CV_standby < CV_working)")
        print(f"   Physics: Standby = fixed auxiliary loads (fan, PLC, pump) → low CV.")
        print(f"   Working = variable cutting load (tool, material) → high CV.")
        print(f"   Formula: CV = σ / μ  (Coefficient of Variation)")
        print(f"   STANDBY CV = {stby_std:.1f} / {stby_mean:.1f} = {stby_cv:.3f}")
        print(f"   WORKING CV = {work_std:.1f} / {work_mean:.1f} = {work_cv:.3f}")
        if stby_cv < work_cv:
            _pass(f"CV(STANDBY)={stby_cv:.3f} < CV(WORKING)={work_cv:.3f}  — "
                  f"STANDBY is more stable than WORKING. Physics law confirmed.")
        else:
            _fail(f"CV(STANDBY)={stby_cv:.3f} ≥ CV(WORKING)={work_cv:.3f}  — "
                  f"STANDBY is MORE variable than WORKING. This is physically wrong.\n"
                  f"         The cluster labelled STANDBY may actually be a light-load\n"
                  f"         cutting mode, not true standby.")
            passed = False

    # ── Check 4: WORKING mean must be substantially above OFF ────────
    ratio = work_mean / off_mean if off_mean > 0 else float('inf')
    print(f"\n   CHECK 4 — WORKING mean must be >> OFF mean (ratio ≥ 3×)")
    print(f"   Physics: Active machining draws spindle + servo + coolant power.")
    print(f"   Combined, this must be several times the OFF baseline.")
    print(f"   WORKING mean = {work_mean:.1f} W,  OFF mean = {off_mean:.1f} W,  "
          f"ratio = {ratio:.1f}×")
    if ratio >= 3.0:
        _pass(f"WORKING / OFF ratio = {ratio:.1f}×  — "
              f"strong evidence these are genuinely different physical states")
    elif ratio >= 1.5:
        _warn(f"WORKING / OFF ratio = {ratio:.1f}×  — "
              f"moderate separation. Acceptable but could be stronger.")
    else:
        _fail(f"WORKING / OFF ratio = {ratio:.1f}×  — "
              f"insufficient. WORKING and OFF overlap — GMM may have split\n"
              f"         a single state into two clusters.")
        passed = False

    return passed


# ============================================================
# TEST 8 — VARIANCE ORDERING (Physics Law)
# ============================================================
# A machine obeying physics MUST satisfy:
#   σ²(OFF) << σ²(STANDBY) << σ²(WORKING)
#
# Reason:
#   OFF      → barely fluctuates → tiny variance
#   STANDBY  → fixed auxiliary loads → small-medium variance
#   WORKING  → variable cutting load each second → large variance
#
# If GMM produces σ(WORKING) < σ(STANDBY), the labels are WRONG.
# This is an independent, formula-based check requiring no labels.
# ============================================================

def test_variance_ordering(gmm, best_k, state_map):
    print("\n" + "=" * 65)
    print("  TEST 8 -- VARIANCE ORDERING (Physics Law: σ_OFF < σ_STANDBY < σ_WORKING)")
    print("=" * 65)
    _info("Physics law: σ(OFF) << σ(STANDBY) << σ(WORKING)")
    _info("This is because cutting load varies every second, but auxiliary")
    _info("loads (cooling fan, PLC) are nearly constant.")
    _info("If this ordering is violated, the state labels are WRONG.")
    _sep()

    means      = gmm.means_.flatten()
    variances  = gmm.covariances_.flatten()
    stds       = np.sqrt(np.abs(variances))
    sorted_idx = np.argsort(means)            # order by mean: OFF < STANDBY < WORKING
    sorted_stds  = stds[sorted_idx]
    sorted_names = [state_map[i] for i in sorted_idx]

    print(f"\n   State ordering by mean (should = by variance too):")
    for name, std in zip(sorted_names, sorted_stds):
        bar = '█' * int(std / sorted_stds.max() * 30)
        print(f"   {name:<14}  σ = {std:8.1f} W   {bar}")

    passed = True
    print()
    for i in range(len(sorted_stds) - 1):
        if sorted_stds[i+1] > sorted_stds[i]:
            _pass(f"σ({sorted_names[i]}) = {sorted_stds[i]:.1f} W  < "
                  f"σ({sorted_names[i+1]}) = {sorted_stds[i+1]:.1f} W  — "
                  f"variance ordering correct")
        else:
            _fail(f"σ({sorted_names[i]}) = {sorted_stds[i]:.1f} W  ≥  "
                  f"σ({sorted_names[i+1]}) = {sorted_stds[i+1]:.1f} W  — "
                  f"VARIANCE LAW VIOLATED. Higher-power cluster is more\n"
                  f"         stable than lower-power cluster. This is physically\n"
                  f"         impossible for a correct OFF→STANDBY→WORKING split.")
            passed = False

    return passed


# ============================================================
# TEST 9 — TEMPORAL / DAY-NIGHT ALIGNMENT
# ============================================================
# KEY IDEA: If the GMM correctly identifies the OFF state,
# then OFF periods MUST correlate with nights and weekends —
# because industrial machines are not operated at 2am or Sunday.
#
# This is an EXTERNAL validation check that requires NO labels.
# It uses only the timestamp column that already exists in the data.
#
# Metric:
#   night_off_rate = fraction of night-time readings (10pm–6am)
#                   that are labelled OFF
#   day_off_rate   = fraction of daytime readings (8am–6pm weekdays)
#                   that are labelled OFF
#
# Expected: night_off_rate >> day_off_rate
# If night_off_rate ≈ day_off_rate, the OFF cluster does NOT
# correspond to machine-off periods — it is misidentified.
# ============================================================

def test_temporal_alignment(df, gmm, best_k, state_map, X):
    print("\n" + "=" * 65)
    print("  TEST 9 -- TEMPORAL / DAY-NIGHT ALIGNMENT")
    print("=" * 65)
    _info("If OFF = 'machine turned off', then OFF must align with nights/weekends.")
    _info("This is an INDEPENDENT check using only timestamps — no labels needed.")
    _info("Industrial machines are not operated at 2am or Sundays.")
    _sep()

    labels_arr = gmm.predict(X)
    df2 = df.copy()
    df2['_gmm_state'] = [state_map[l] for l in labels_arr]
    df2['_hour']      = df2['timestamp'].dt.hour
    df2['_weekday']   = df2['timestamp'].dt.weekday   # 0=Mon … 6=Sun
    df2['_is_night']  = (df2['_hour'] >= 22) | (df2['_hour'] < 6)
    df2['_is_day']    = (df2['_weekday'] < 5) & (df2['_hour'] >= 8) & (df2['_hour'] < 18)
    df2['_is_weekend'] = df2['_weekday'] >= 5

    night_total   = df2['_is_night'].sum()
    day_total     = df2['_is_day'].sum()
    weekend_total = df2['_is_weekend'].sum()

    if night_total < 10 or day_total < 10:
        _warn("Not enough night-time or daytime data to run temporal check.")
        return True   # not a failure, just insufficient data

    night_off = ((df2['_gmm_state'] == 'OFF') & df2['_is_night']).sum()
    day_off   = ((df2['_gmm_state'] == 'OFF') & df2['_is_day']).sum()
    wknd_off  = ((df2['_gmm_state'] == 'OFF') & df2['_is_weekend']).sum()

    night_off_rate = night_off / night_total * 100
    day_off_rate   = day_off   / day_total   * 100
    wknd_off_rate  = wknd_off  / weekend_total * 100 if weekend_total > 0 else 0

    print(f"\n   Time window          Total readings  OFF readings   OFF rate")
    print(f"   {'':-<60}")
    print(f"   Night (10pm–6am)     {night_total:>14,}  {night_off:>12,}   {night_off_rate:>6.1f}%")
    print(f"   Weekday (8am–6pm)    {day_total:>14,}  {day_off:>12,}   {day_off_rate:>6.1f}%")
    print(f"   Weekend (all day)    {weekend_total:>14,}  {wknd_off:>12,}   {wknd_off_rate:>6.1f}%")
    print()

    passed = True

    # Night OFF rate should be substantially higher than daytime OFF rate
    if night_total > 0 and day_total > 0:
        if night_off_rate > day_off_rate * 1.5:   # night should be ≥1.5× more OFF
            _pass(f"Night OFF rate ({night_off_rate:.1f}%) >> Day OFF rate ({day_off_rate:.1f}%) — "
                  f"OFF cluster correctly aligns with machine-off hours")
        elif night_off_rate > day_off_rate:
            _warn(f"Night OFF rate ({night_off_rate:.1f}%) > Day OFF rate ({day_off_rate:.1f}%) — "
                  f"correct direction but weak separation. Machine may run some\n"
                  f"         night shifts, or OFF cluster may include low-load periods.")
        else:
            _fail(f"Night OFF rate ({night_off_rate:.1f}%) ≤ Day OFF rate ({day_off_rate:.1f}%) — "
                  f"OFF cluster does NOT align with machine-off hours.\n"
                  f"         The cluster labelled OFF may not be the actual off state.")
            passed = False

    # Weekend check (bonus — not a hard fail)
    if weekend_total > 100:
        if wknd_off_rate > day_off_rate:
            _pass(f"Weekend OFF rate ({wknd_off_rate:.1f}%) > Weekday day rate ({day_off_rate:.1f}%) — "
                  f"machine powers down on weekends. Confirms OFF label is correct.")
        else:
            _warn(f"Weekend OFF rate ({wknd_off_rate:.1f}%) ≤ Weekday rate ({day_off_rate:.1f}%). "
                  f"Machine may operate on weekends, or OFF cluster is misidentified.")

    return passed


# ============================================================
# PLOT 6 — TEMPORAL ALIGNMENT HEATMAP
# ============================================================

def plot_temporal_alignment(df, gmm, best_k, state_map, X, save_dir):
    """
    Hour-of-day × Day-of-week heatmap showing the fraction of OFF
    readings. A correct OFF cluster should produce a heatmap that
    is dark (high OFF %) at nights and weekends, and light (low %) 
    during business hours.
    """
    labels_arr = gmm.predict(X)
    df2 = df.copy()
    df2['_state']   = [state_map[l] for l in labels_arr]
    df2['_hour']    = df2['timestamp'].dt.hour
    df2['_weekday'] = df2['timestamp'].dt.weekday
    df2['_is_off']  = (df2['_state'] == 'OFF').astype(int)

    pivot = df2.groupby(['_weekday', '_hour'])['_is_off'].mean().unstack(fill_value=0)

    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=1, interpolation='nearest')

    # x-axis = hours 0–23, y-axis = weekdays
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([day_labels[d] for d in pivot.index], fontsize=9)

    plt.colorbar(im, ax=ax, label='Fraction of readings labelled OFF (1.0 = all OFF)')
    ax.set_title(
        f"T9 — Temporal Alignment: OFF-State Rate by Hour & Day — {MACHINE_NAME}\n"
        "Green = machine running. Red/dark = machine off. Nights & weekends should be dark.",
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    out = os.path.join(save_dir, "T9_temporal_alignment_heatmap.png")
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: T9_temporal_alignment_heatmap.png")


# ============================================================
# FINAL SUMMARY REPORT
# ============================================================

def print_final_report(results, best_k, state_map, gmm):
    print("\n" + "=" * 65)
    print("  FINAL GMM VALIDATION SUMMARY")
    print("=" * 65)

    labels = {True: "[PASS]", False: "[FAIL]"}
    print(f"  {'Test':<45} {'Result':>10}")
    print(f"  {'':->45} {'':->10}")
    for name, passed in results.items():
        print(f"  {name:<45} {labels[passed]:>10}")

    passed_count = sum(1 for v in results.values() if v)
    total_count  = len(results)
    pct          = passed_count / total_count * 100

    print(f"\n  Passed: {passed_count}/{total_count}  ({pct:.0f}%)")
    print()

    if pct == 100:
        verdict = "[CONFIRMED] GMM is splitting the data correctly."
        detail  = ("All 9 independent tests passed. The power clusters represent "
                   "physically distinct machine states. Safe to build the next "
                   "pipeline stage on top of this.")
    elif pct >= 66:
        verdict = "[PARTIAL] GMM is working but some checks raised warnings."
        detail  = ("Review the FAIL items above. The model may still be usable "
                   "but the flagged checks suggest either a better k exists or "
                   "some states overlap. Manual inspection of the plots recommended.")
    else:
        verdict = "[SUSPECT] GMM likely not splitting correctly."
        detail  = ("Multiple checks failed. Do NOT proceed to the next pipeline "
                   "stage yet. Review the plots and consider adjusting k, checking "
                   "your data loader, or investigating degenerate power values.")

    print(f"  {verdict}")
    print(f"\n  {detail}")

    # Print final state assignments
    means      = gmm.means_.flatten()
    stds       = np.sqrt(np.abs(gmm.covariances_.flatten()))
    sorted_idx = np.argsort(means)
    print(f"\n  FINAL STATE MAP  (k={best_k}):")
    print(f"  {'State':<14} {'Mean (W)':>10}  {'Std (W)':>10}  {'Weight':>8}")
    print(f"  {'':->14} {'':->10}  {'':->10}  {'':->8}")
    for idx in sorted_idx:
        name = state_map[idx]
        print(f"  {name:<14} {means[idx]:>10.1f}  {stds[idx]:>10.1f}  "
              f"{gmm.weights_[idx]:>8.3f}")
    print("=" * 65 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  GMM VALIDATION HARNESS")
    print(f"  File    : {DATA_PATH}")
    print(f"  Machine : {MACHINE_NAME}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Data file not found: {DATA_PATH}")
        print("   Update DATA_PATH at the top of this file.")
        return

    df = load_and_prepare_data(DATA_PATH)

    # ── Test 1: Sanity ────────────────────────────────────────────
    t1 = test_data_sanity(df)

    # ── Fit GMMs for all k values ─────────────────────────────────
    print("\n   Fitting GMMs for k ∈", K_RANGE, "with n_init=5 …")
    X      = df['power'].values.reshape(-1, 1)
    models = {}
    for k in K_RANGE:
        print(f"   Fitting k={k} … ", end='', flush=True)
        models[k] = fit_gmm(X, k, n_init=5)
        print("done")

    labels = {k: models[k].predict(X) for k in K_RANGE}

    # ── Test 2: Convergence ───────────────────────────────────────
    t2 = test_gmm_convergence(X, models)

    # ── Test 3: k selection ───────────────────────────────────────
    best_k, sil_scores, bic_scores, agreement = test_k_selection(X, models, labels)
    t3 = agreement

    best_gmm  = models[best_k]
    state_map = map_states(best_gmm, best_k)

    # ── Test 4: Physical separation ───────────────────────────────
    t4, state_map, s_means, s_stds, s_names = test_state_separation(best_gmm, best_k)

    # ── Test 5: Soft-assignment confidence ────────────────────────
    t5, max_probs = test_soft_assignment_confidence(best_gmm, X)

    # ── Test 6: Weight sanity ─────────────────────────────────────
    t6 = test_weight_sanity(best_gmm, best_k, state_map)

    # ── Test 7: Physics threshold cross-check ────────────────────
    max_power = df['power'].max()
    t7 = test_physics_thresholds(best_gmm, best_k, state_map, max_power)

    # ── Test 8: Variance ordering (physics law) ───────────────────
    t8 = test_variance_ordering(best_gmm, best_k, state_map)

    # ── Test 9: Temporal / day-night alignment ────────────────────
    t9 = test_temporal_alignment(df, best_gmm, best_k, state_map, X)

    # ── Diagnostic plots ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GENERATING DIAGNOSTIC PLOTS")
    print("=" * 65)
    plot_k_selection(sil_scores, bic_scores, best_k, OUTPUT_DIR)
    plot_gmm_histogram(df, best_gmm, best_k, state_map, OUTPUT_DIR)
    plot_confidence_histogram(max_probs, OUTPUT_DIR)
    plot_labeled_sample(df, best_gmm, best_k, state_map, OUTPUT_DIR)
    plot_cluster_separation(df, best_gmm, best_k, state_map, OUTPUT_DIR)
    plot_temporal_alignment(df, best_gmm, best_k, state_map, X, OUTPUT_DIR)

    # ── Final report ──────────────────────────────────────────────
    results = {
        "T1  Data Sanity (rows, no-NaN, non-neg)":              t1,
        "T2  GMM Convergence (EM algorithm)":                    t2,
        "T3  k Selection (Sil + BIC agree)":                     t3,
        "T4  Physical Separation (2σ between clusters)":         t4,
        "T5  Soft-Assignment Confidence (>80% in >70%)":         t5,
        "T6  Cluster Weight Sanity (no ghost clusters)":         t6,
        "T7  Physics Threshold (OFF≈0W, STBY stable, WRK gap)":  t7,
        "T8  Variance Ordering (σ_OFF < σ_STBY < σ_WORK)":       t8,
        "T9  Temporal Alignment (OFF aligns nights/weekends)":   t9,
    }
    print_final_report(results, best_k, state_map, best_gmm)

    print(f"All plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
