"""
============================================================
GMM VALIDATION HARNESS -- validate_gmm.py
============================================================

PURPOSE
-------
Validate whether a Gaussian Mixture Model (GMM) splits machine
power readings into physically meaningful machine states before
later pipeline stages are used.

This script is standalone. It imports only the data loader from:
    src.data_analysis -> load_and_prepare_data

VALIDATION TESTS
----------------
T1  Data sanity
T2  GMM convergence
T3  k-selection using BIC, AIC, and Silhouette
T4  Physical separation of adjacent states
T5  Soft-assignment confidence
T6  Cluster weight sanity

Each test is classified by the KIND of evidence it produces:
    FORMAL     -> BIC / AIC              (model-selection theory)
    GEOMETRIC  -> Silhouette / separation d / intersection boundary
    HEURISTIC  -> confidence thresholds, tiny-weight thresholds
Heuristic results are never treated as proof; they are reported
as engineering guidance only.

HOW TO RUN
----------
    python validate_gmm.py

Update DATA_PATH and MACHINE_NAME below.
Outputs are saved to outputs/gmm_validation
============================================================
"""

import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Project path so we can import the data loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_analysis import load_and_prepare_data


# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/2024_P_total_VacuumSoldering.csv.xz"
MACHINE_NAME = "VacuumSoldering"
OUTPUT_DIR = "outputs/gmm_validation"
K_RANGE = [2, 3, 4]

MIN_ROWS = 1000
OFF_THRESHOLD_W = 5.0

# Heuristic thresholds (engineering judgement, not formal statistics)
HIGH_CONF_THRESHOLD = 0.80
LOW_CONF_THRESHOLD = 0.60
HIGH_CONF_TARGET_PCT = 70.0
LOW_CONF_MAX_PCT = 20.0
TINY_WEIGHT_FAIL_PCT = 0.5
SMALL_WEIGHT_WARN_PCT = 3.0


# ============================================================
# STATE METADATA
# ============================================================
STATE_COLORS = {
    "OFF": "#555555",
    "STANDBY": "#f0a500",
    "IDLE": "#4fc3f7",
    "WORKING": "#66bb6a",
}

K_TO_NAMES = {
    2: ["OFF", "WORKING"],
    3: ["OFF", "STANDBY", "WORKING"],
    4: ["OFF", "STANDBY", "IDLE", "WORKING"],
}


# ============================================================
# PRINT HELPERS (ASCII only -- safe for Windows terminals)
# ============================================================
def _pass(msg):
    print(f"   [PASS] {msg}")


def _fail(msg):
    print(f"   [FAIL] {msg}")


def _warn(msg):
    print(f"   [WARN] {msg}")


def _info(msg):
    print(f"   [INFO] {msg}")


def _sep():
    print("   " + "-" * 68)


# ============================================================
# CORE HELPERS
# ============================================================
def fit_gmm(X, k, n_init=10, max_iter=500, random_state=42):
    """
    Fit a 1D Gaussian Mixture Model.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 1)
    k : int
        Number of mixture components
    """
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    gmm.fit(X)
    return gmm


def map_states(gmm, k):
    """
    Sort components by ascending mean and assign safe semantic state names.

    k=2 -> OFF, WORKING
    k=3 -> OFF, STANDBY, WORKING
    k=4 -> OFF, STANDBY, IDLE, WORKING

    We deliberately never emit a "PEAK_LOAD" style label here -- the
    fixed name lists above are the only vocabulary this function is
    allowed to produce, since nothing in this script proves a distinct
    high-power state exists beyond "WORKING".
    """
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    names = K_TO_NAMES.get(k, [f"STATE_{i}" for i in range(k)])
    return {int(sorted_idx[i]): names[i] for i in range(k)}


def compute_model_metrics(X, gmm, labels):
    """
    Compute core model-comparison metrics for a fitted GMM.

    Formal criteria (model-selection theory)
    -----------------------------------------
        BIC = -2 log(L) + p log(n)
        AIC = -2 log(L) + 2p
    where L is the model likelihood, p is the number of free
    parameters, and n is the number of samples. Both are provided
    directly by sklearn's GaussianMixture (gmm.bic / gmm.aic).

    Lower BIC/AIC = better trade-off between fit and complexity.
    BIC penalizes extra parameters more heavily than AIC, so it is
    the primary selector used in test_k_selection().

    Geometric diagnostic
    ---------------------
    Silhouette score measures how well-separated the discovered
    clusters are in feature space (higher is better, range [-1, 1]).
    It says nothing about parsimony, so it is reported alongside
    BIC/AIC rather than replacing them.
    """
    n = len(X)
    sil = np.nan
    unique_labels = np.unique(labels)

    if len(unique_labels) > 1 and len(unique_labels) < n:
        sample_size = min(10000, n)
        try:
            sil = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        except ValueError:
            # Can happen with pathological/degenerate clusterings
            sil = np.nan

    weights = gmm.weights_.flatten()
    return {
        "bic": gmm.bic(X),
        "aic": gmm.aic(X),
        "silhouette": sil,
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "lower_bound": float(gmm.lower_bound_),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
    }


def solve_gaussian_intersections(mu1, sigma1, w1, mu2, sigma2, w2):
    """
    Solve for x where two weighted 1D Gaussians cross:

        w1 * N(x | mu1, sigma1^2) = w2 * N(x | mu2, sigma2^2)

    Taking logs of both sides:

        log(w1) - log(sigma1) - (x-mu1)^2 / (2*sigma1^2)
            = log(w2) - log(sigma2) - (x-mu2)^2 / (2*sigma2^2)

    Rearranging into standard quadratic form a*x^2 + b*x + c = 0:

        a = 1/(2*sigma2^2) - 1/(2*sigma1^2)
        b = mu1/sigma1^2 - mu2/sigma2^2
        c = mu2^2/(2*sigma2^2) - mu1^2/(2*sigma1^2)
            - log( (w2/sigma2) / (w1/sigma1) )

    NOTE: the log-ratio term is SUBTRACTED here. (An earlier version
    of this function added it by mistake, which shifted the computed
    boundary away from its true location -- fixed below.)

    Returns a sorted list of real roots (0, 1, or 2 values).
    """
    sigma1 = max(float(sigma1), 1e-9)
    sigma2 = max(float(sigma2), 1e-9)
    w1 = max(float(w1), 1e-12)
    w2 = max(float(w2), 1e-12)

    a = (1.0 / (2.0 * sigma2 ** 2)) - (1.0 / (2.0 * sigma1 ** 2))
    b = (mu1 / (sigma1 ** 2)) - (mu2 / (sigma2 ** 2))
    log_ratio = math.log((w2 / sigma2) / (w1 / sigma1))
    c = (
        (mu2 ** 2) / (2.0 * sigma2 ** 2)
        - (mu1 ** 2) / (2.0 * sigma1 ** 2)
        - log_ratio
    )

    roots = []

    if abs(a) < 1e-12:
        # Degenerates to a linear equation (near-equal variances)
        if abs(b) < 1e-12:
            return []
        roots.append(-c / b)
        return sorted(roots)

    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        # No real crossing point (can happen with extreme weight/variance
        # imbalance) -- caller falls back to reporting "None" for boundary.
        return []

    sqrt_disc = math.sqrt(max(disc, 0.0))
    roots.append((-b + sqrt_disc) / (2.0 * a))
    roots.append((-b - sqrt_disc) / (2.0 * a))
    roots = sorted([float(r) for r in roots if np.isfinite(r)])
    return roots


def choose_between_means(roots, left_mean, right_mean):
    """
    Prefer an intersection root that lies between the two adjacent means,
    since that is the physically meaningful decision boundary.
    """
    for r in roots:
        if left_mean <= r <= right_mean:
            return r
    return None


def analyze_adjacent_separation(gmm, state_map):
    """
    Analyze adjacent Gaussian components after sorting by mean.

    Separation score (geometric diagnostic, effect-size style):

        d = (mu_(i+1) - mu_i) / sqrt((sigma_i^2 + sigma_(i+1)^2) / 2)

    Interpretation (heuristic bands, not formal thresholds):
        d >= 2.0    : well separated
        1.0 <= d<2.0: moderately separated
        d < 1.0     : overlapping
    """
    means = gmm.means_.flatten()
    variances = np.abs(gmm.covariances_.flatten())
    stds = np.sqrt(variances)
    weights = gmm.weights_.flatten()

    sorted_idx = np.argsort(means)
    rows = []

    for i in range(len(sorted_idx) - 1):
        i1 = sorted_idx[i]
        i2 = sorted_idx[i + 1]

        mu1 = float(means[i1])
        mu2 = float(means[i2])
        s1 = float(stds[i1])
        s2 = float(stds[i2])
        w1 = float(weights[i1])
        w2 = float(weights[i2])

        gap = mu2 - mu1
        pooled_std = math.sqrt((s1 ** 2 + s2 ** 2) / 2.0) if (s1 > 0 or s2 > 0) else np.nan
        separation_d = gap / pooled_std if (pooled_std and pooled_std > 0) else np.inf

        roots = solve_gaussian_intersections(mu1, s1, w1, mu2, s2, w2)
        boundary = choose_between_means(roots, mu1, mu2)

        if separation_d >= 2.0:
            verdict = "well separated"
            passed = True
        elif separation_d >= 1.0:
            verdict = "moderately separated"
            passed = True
        else:
            verdict = "overlapping"
            passed = False

        rows.append({
            "left_state": state_map[i1],
            "right_state": state_map[i2],
            "mu_left": mu1,
            "mu_right": mu2,
            "std_left": s1,
            "std_right": s2,
            "gap": gap,
            "pooled_std": pooled_std,
            "separation_d": separation_d,
            "boundary_w": boundary,
            "verdict": verdict,
            "passed": passed,
        })

    return rows


# ============================================================
# TEST 1 -- DATA SANITY
# ============================================================
def test_data_sanity(df):
    print("\n" + "=" * 72)
    print("  TEST 1 -- DATA SANITY")
    print("=" * 72)

    if df is None or len(df) == 0:
        _fail("Dataframe is empty or None.")
        return False

    if "power" not in df.columns:
        _fail("Column 'power' not found in dataframe.")
        return False

    passed = True
    n = len(df)

    if n < MIN_ROWS:
        _fail(f"Only {n:,} rows found; need at least {MIN_ROWS:,} for stable GMM fitting.")
        passed = False
    else:
        _pass(f"{n:,} rows loaded.")

    non_null = df["power"].notna().sum()
    if non_null == 0:
        _fail("Power column is empty after loading.")
        return False
    else:
        _pass(f"Power column is non-empty ({non_null:,} non-null values).")

    nan_count = int(df["power"].isna().sum())
    if nan_count > 0:
        _fail(f"{nan_count:,} NaN values found in power column.")
        passed = False
    else:
        _pass("No NaN values in power column.")

    neg_count = int((df["power"] < 0).sum())
    if neg_count > 0:
        _fail(f"{neg_count:,} negative power readings found.")
        passed = False
    else:
        _pass("All power values are non-negative.")

    off_count = int((df["power"] < OFF_THRESHOLD_W).sum())
    on_count = int((df["power"] > OFF_THRESHOLD_W).sum())
    off_pct = off_count / n * 100.0
    on_pct = on_count / n * 100.0

    if off_count == 0:
        _warn(f"No readings below {OFF_THRESHOLD_W:.1f} W; a true OFF state may not exist.")
    else:
        _pass(f"OFF-like readings (<{OFF_THRESHOLD_W:.1f} W): {off_count:,} ({off_pct:.1f}%).")

    if on_count < 100:
        _fail(f"Only {on_count:,} readings above {OFF_THRESHOLD_W:.1f} W; too few active readings.")
        passed = False
    else:
        _pass(f"ON-like readings (>{OFF_THRESHOLD_W:.1f} W): {on_count:,} ({on_pct:.1f}%).")

    _info(f"Power range: {df['power'].min():.2f} W to {df['power'].max():.2f} W")
    _info(f"Power mean : {df['power'].mean():.2f} W")
    _info(f"Power std  : {df['power'].std():.2f} W")

    return passed


# ============================================================
# TEST 2 -- GMM CONVERGENCE
# ============================================================
def test_gmm_convergence(models):
    print("\n" + "=" * 72)
    print("  TEST 2 -- GMM CONVERGENCE")
    print("=" * 72)
    _info("Each model is fit with multiple random restarts (n_init).")
    _info("This checks whether EM actually converged for each candidate k.")
    _sep()

    all_converged = True
    for k, gmm in models.items():
        if gmm.converged_:
            _pass(f"k={k}: converged in {gmm.n_iter_} iterations; lower bound={gmm.lower_bound_:.6f}")
        else:
            _fail(f"k={k}: did NOT converge after {gmm.n_iter_} iterations.")
            all_converged = False

    return all_converged


# ============================================================
# TEST 3 -- K SELECTION
# ============================================================
def test_k_selection(X, models, labels_by_k, k_range=None):
    print("\n" + "=" * 72)
    print("  TEST 3 -- K SELECTION")
    print("=" * 72)
    _info("Formal criteria (model-selection theory):")
    _info("  BIC = -2 log(L) + p log(n)")
    _info("  AIC = -2 log(L) + 2p")
    _info("Lower BIC/AIC is better. Higher silhouette is better.")
    _info("BIC is used as the primary selector because it penalizes extra")
    _info("complexity more strongly than AIC.")
    _sep()

    if k_range is None:
        k_range = sorted(models.keys())

    metrics_by_k = {}
    for k in k_range:
        metrics_by_k[k] = compute_model_metrics(X, models[k], labels_by_k[k])

    print(
        f"\n   {'k':<4} {'Conv':<6} {'Iter':>6} {'Silhouette':>12} "
        f"{'BIC':>16} {'AIC':>16} {'MinWt%':>10} {'MaxWt%':>10}"
    )
    print("   " + "-" * 90)

    for k in k_range:
        m = metrics_by_k[k]
        sil_text = f"{m['silhouette']:.4f}" if np.isfinite(m["silhouette"]) else "nan"
        print(
            f"   {k:<4} {str(m['converged']):<6} {m['n_iter']:>6} {sil_text:>12} "
            f"{m['bic']:>16,.1f} {m['aic']:>16,.1f} "
            f"{100*m['min_weight']:>9.2f} {100*m['max_weight']:>9.2f}"
        )

    bic_best = min(k_range, key=lambda k: metrics_by_k[k]["bic"])
    aic_best = min(k_range, key=lambda k: metrics_by_k[k]["aic"])

    valid_sil = {k: metrics_by_k[k]["silhouette"] for k in k_range if np.isfinite(metrics_by_k[k]["silhouette"])}
    sil_best = max(valid_sil, key=valid_sil.get) if valid_sil else None

    print()
    _info(f"BIC selects k = {bic_best}")
    _info(f"AIC selects k = {aic_best}")
    _info(f"Silhouette selects k = {sil_best}")

    if sil_best is not None and sil_best == bic_best:
        _pass(f"Silhouette and BIC agree on k = {bic_best}.")
        agreement = True
    else:
        _warn(f"Silhouette and BIC do not agree. Proceeding with BIC-selected k = {bic_best}.")
        agreement = False

    return bic_best, metrics_by_k, agreement


# ============================================================
# TEST 4 -- PHYSICAL SEPARATION
# ============================================================
def test_state_separation(gmm, best_k):
    print("\n" + "=" * 72)
    print("  TEST 4 -- PHYSICAL SEPARATION")
    print("=" * 72)
    _info("Adjacent-state separation score (geometric diagnostic):")
    _info("  d = (mu_(i+1) - mu_i) / sqrt((sigma_i^2 + sigma_(i+1)^2)/2)")
    _info("This compares the mean gap to pooled spread, like an effect size.")
    _info("We also solve for the Gaussian intersection boundary between")
    _info("adjacent states: w1*N(x|mu1,s1^2) = w2*N(x|mu2,s2^2).")
    _sep()

    state_map = map_states(gmm, best_k)
    means = gmm.means_.flatten()
    stds = np.sqrt(np.abs(gmm.covariances_.flatten()))
    weights = gmm.weights_.flatten()
    sorted_idx = np.argsort(means)

    print(f"\n   {'State':<12} {'Mean(W)':>12} {'Std(W)':>12} {'Weight%':>10}")
    print("   " + "-" * 52)
    for idx in sorted_idx:
        print(
            f"   {state_map[idx]:<12} "
            f"{means[idx]:>12.2f} {stds[idx]:>12.2f} {100*weights[idx]:>9.2f}"
        )

    rows = analyze_adjacent_separation(gmm, state_map)

    print(f"\n   {'Pair':<24} {'Gap(W)':>10} {'PooledStd':>12} {'d-score':>10} {'Boundary(W)':>14} {'Verdict':>16}")
    print("   " + "-" * 96)

    all_ok = True
    for row in rows:
        pair = f"{row['left_state']} -> {row['right_state']}"
        boundary_text = f"{row['boundary_w']:.2f}" if row["boundary_w"] is not None else "None"
        print(
            f"   {pair:<24} {row['gap']:>10.2f} {row['pooled_std']:>12.2f} "
            f"{row['separation_d']:>10.2f} {boundary_text:>14} {row['verdict']:>16}"
        )
        if row["passed"]:
            _pass(f"{pair}: {row['verdict']}.")
        else:
            _fail(f"{pair}: distributions overlap strongly.")
            all_ok = False

    return all_ok, state_map, rows


# ============================================================
# TEST 5 -- SOFT ASSIGNMENT
# ============================================================
def test_soft_assignment_confidence(gmm, X):
    print("\n" + "=" * 72)
    print("  TEST 5 -- SOFT-ASSIGNMENT CONFIDENCE")
    print("=" * 72)
    _info("For each sample i, confidence is c_i = max_k gamma_ik.")
    _info("gamma_ik is the posterior probability sample i belongs to component k.")
    _info("Thresholds below (80% / 60%) are engineering heuristics, not proof.")
    _sep()

    probs = gmm.predict_proba(X)
    max_probs = probs.max(axis=1)

    avg_conf = float(max_probs.mean() * 100.0)
    pct_high = float((max_probs > HIGH_CONF_THRESHOLD).mean() * 100.0)
    pct_low = float((max_probs < LOW_CONF_THRESHOLD).mean() * 100.0)

    _info(f"Average max probability : {avg_conf:.1f}%")
    _info(f"Readings above 80%      : {pct_high:.1f}%")
    _info(f"Readings below 60%      : {pct_low:.1f}%")

    passed = True
    if pct_high >= HIGH_CONF_TARGET_PCT:
        _pass(f"{pct_high:.1f}% of readings exceed 80% confidence.")
    else:
        _fail(f"Only {pct_high:.1f}% of readings exceed 80% confidence.")
        passed = False

    if pct_low <= LOW_CONF_MAX_PCT:
        _pass(f"Only {pct_low:.1f}% of readings are below 60% confidence.")
    else:
        _fail(f"{pct_low:.1f}% of readings are below 60% confidence.")
        passed = False

    return passed, max_probs


# ============================================================
# TEST 6 -- WEIGHT SANITY
# ============================================================
def test_weight_sanity(gmm, state_map):
    print("\n" + "=" * 72)
    print("  TEST 6 -- CLUSTER WEIGHT SANITY")
    print("=" * 72)
    _info("Mixture weights should sum to 1 (sum_k w_k = 1).")
    _info("Very tiny components often indicate noise-catching clusters")
    _info("rather than a genuine machine state.")
    _sep()

    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    sorted_idx = np.argsort(means)

    passed = True
    total_weight = weights.sum()
    _info(f"Sum of weights: {total_weight:.6f}")

    for idx in sorted_idx:
        name = state_map[idx]
        pct = 100.0 * weights[idx]
        if pct < TINY_WEIGHT_FAIL_PCT:
            _fail(f"{name:<12}: {pct:.2f}% weight -- too tiny, likely an artifact.")
            passed = False
        elif pct < SMALL_WEIGHT_WARN_PCT:
            _warn(f"{name:<12}: {pct:.2f}% weight -- very small, verify manually.")
        else:
            _pass(f"{name:<12}: {pct:.2f}% weight -- substantial.")

    return passed


# ============================================================
# PLOTTING
# ============================================================
def plot_k_selection(metrics_by_k, best_k, save_dir):
    try:
        ks = sorted(metrics_by_k.keys())
        sil = [metrics_by_k[k]["silhouette"] for k in ks]
        bic = [metrics_by_k[k]["bic"] for k in ks]
        aic = [metrics_by_k[k]["aic"] for k in ks]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"GMM Model Selection - {MACHINE_NAME}", fontsize=14, fontweight="bold")

        axes[0].plot(ks, sil, marker="o", linewidth=2)
        axes[0].axvline(best_k, linestyle="--", color="red")
        axes[0].set_title("Silhouette (higher is better)")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Silhouette")
        axes[0].set_xticks(ks)
        axes[0].grid(alpha=0.3)

        axes[1].plot(ks, bic, marker="o", linewidth=2, color="#f0a500")
        axes[1].axvline(best_k, linestyle="--", color="red")
        axes[1].set_title("BIC (lower is better)")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("BIC")
        axes[1].set_xticks(ks)
        axes[1].grid(alpha=0.3)

        axes[2].plot(ks, aic, marker="o", linewidth=2, color="#4fc3f7")
        axes[2].axvline(best_k, linestyle="--", color="red")
        axes[2].set_title("AIC (lower is better)")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("AIC")
        axes[2].set_xticks(ks)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(save_dir, "T3_k_selection_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out}")
    except Exception as e:
        _warn(f"Could not generate k-selection plot: {e}")


def plot_gmm_histogram(df, gmm, state_map, save_dir):
    try:
        power = df["power"].values
        means = gmm.means_.flatten()
        stds = np.sqrt(np.abs(gmm.covariances_.flatten()))
        weights = gmm.weights_.flatten()
        sorted_idx = np.argsort(means)

        fig, ax = plt.subplots(figsize=(12, 6))
        # Fixed bin count keeps bin width consistent across machines with
        # very different power ranges, avoiding misleading spike artifacts.
        n_bins = min(120, max(30, int(np.sqrt(len(power)))))
        bins = np.linspace(power.min(), power.max(), n_bins)
        ax.hist(power, bins=bins, density=True, alpha=0.5, color="#334155", edgecolor="white", linewidth=0.3)

        x = np.linspace(power.min(), power.max(), 2000)
        total_pdf = np.zeros_like(x)

        for idx in sorted_idx:
            state = state_map[idx]
            color = STATE_COLORS.get(state, "#999999")
            pdf = weights[idx] * norm.pdf(x, means[idx], stds[idx])
            total_pdf += pdf
            ax.fill_between(x, pdf, alpha=0.25, color=color)
            ax.plot(x, pdf, color=color, linewidth=2, label=f"{state} ({means[idx]:.1f}W)")
            ax.axvline(means[idx], color=color, linestyle=":", linewidth=1.2, alpha=0.8)

        ax.plot(x, total_pdf, color="black", linestyle="--", linewidth=1.5, label="Total mixture")
        ax.set_title(f"GMM Gaussian Overlay - {MACHINE_NAME}")
        ax.set_xlabel("Power (W)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.2)

        plt.tight_layout()
        out = os.path.join(save_dir, "T4_gmm_histogram_overlay.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out}")
    except Exception as e:
        _warn(f"Could not generate histogram overlay plot: {e}")


def plot_confidence_histogram(max_probs, save_dir):
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(max_probs * 100.0, bins=60, color="#4fc3f7", edgecolor="#1e293b", linewidth=0.4)
        ax.axvline(80, color="#16a34a", linestyle="--", linewidth=2, label="80% threshold")
        ax.axvline(60, color="#dc2626", linestyle="--", linewidth=2, label="60% threshold")
        ax.set_title(f"GMM Assignment Confidence - {MACHINE_NAME}")
        ax.set_xlabel("Max posterior probability (%)")
        ax.set_ylabel("Number of readings")
        ax.legend()
        ax.grid(alpha=0.2)

        plt.tight_layout()
        out = os.path.join(save_dir, "T5_confidence_distribution.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out}")
    except Exception as e:
        _warn(f"Could not generate confidence histogram: {e}")


def plot_labeled_sample(df, gmm, state_map, save_dir, n_points=2000):
    try:
        sample = df.head(n_points).copy()
        Xs = sample["power"].values.reshape(-1, 1)
        labels = gmm.predict(Xs)
        sample["state"] = [state_map[l] for l in labels]

        fig, ax = plt.subplots(figsize=(15, 5))
        for state, color in STATE_COLORS.items():
            mask = sample["state"] == state
            if mask.any():
                ax.scatter(
                    sample.loc[mask, "timestamp"],
                    sample.loc[mask, "power"],
                    s=6,
                    c=color,
                    label=state,
                    alpha=0.8,
                )

        ax.plot(sample["timestamp"], sample["power"], color="#cbd5e1", linewidth=0.7, alpha=0.6)
        ax.set_title(f"Labeled Time Sample - First {min(n_points, len(sample))} Readings")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (W)")
        ax.legend()
        ax.grid(alpha=0.2)

        plt.tight_layout()
        out = os.path.join(save_dir, "T5_labeled_time_sample.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out}")
    except Exception as e:
        _warn(f"Could not generate labeled time-series plot: {e}")


def plot_cluster_separation(df, gmm, state_map, save_dir):
    try:
        df2 = df.copy()
        X = df2["power"].values.reshape(-1, 1)
        labels = gmm.predict(X)
        df2["state"] = [state_map[l] for l in labels]

        ordered_states = [s for s in ["OFF", "STANDBY", "IDLE", "WORKING"] if s in df2["state"].unique()]
        data_per_state = []
        for s in ordered_states:
            subset = df2.loc[df2["state"] == s, "power"]
            n_sample = min(5000, len(subset))
            if n_sample == 0:
                continue
            data_per_state.append(subset.sample(n_sample, random_state=42).values)

        if not data_per_state:
            _warn("No states with data available for cluster separation plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        parts = ax.violinplot(data_per_state, showmeans=True, showmedians=False, showextrema=True)

        for body, state in zip(parts["bodies"], ordered_states):
            body.set_facecolor(STATE_COLORS[state])
            body.set_alpha(0.65)

        for part_name in ["cmeans", "cbars", "cmins", "cmaxes"]:
            if part_name in parts:
                parts[part_name].set_color("black")
                parts[part_name].set_linewidth(1.0)

        ax.set_xticks(range(1, len(ordered_states) + 1))
        ax.set_xticklabels(ordered_states)
        ax.set_ylabel("Power (W)")
        ax.set_title(f"Power Distribution by Assigned GMM State - {MACHINE_NAME}")
        ax.grid(axis="y", alpha=0.2)

        plt.tight_layout()
        out = os.path.join(save_dir, "T4_cluster_separation_violin.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {out}")
    except Exception as e:
        _warn(f"Could not generate cluster separation plot: {e}")


# ============================================================
# FINAL REPORT
# ============================================================
def print_final_report(results, best_k, state_map, gmm):
    print("\n" + "=" * 72)
    print("  FINAL GMM VALIDATION SUMMARY")
    print("=" * 72)

    print(f"  {'Test':<52} {'Result':>10}")
    print(f"  {'-' * 52} {'-' * 10}")

    for name, passed in results.items():
        label = "[PASS]" if passed else "[FAIL]"
        print(f"  {name:<52} {label:>10}")

    passed_count = sum(bool(v) for v in results.values())
    total_count = len(results)
    pct = 100.0 * passed_count / total_count if total_count else 0.0

    print(f"\n  Passed: {passed_count}/{total_count} ({pct:.0f}%)")

    if pct >= 83:
        verdict = "CONFIRMED"
        detail = "The GMM appears to split the power readings into meaningful states."
    elif pct >= 50:
        verdict = "PARTIAL"
        detail = "The GMM is usable, but some checks indicate caution and manual review."
    else:
        verdict = "SUSPECT"
        detail = "The GMM likely does not separate the machine states reliably yet."

    print(f"\n  Verdict: {verdict}")
    print(f"  {detail}")
    print("  (Heuristic checks above are engineering guidance, not formal proof.)")

    means = gmm.means_.flatten()
    stds = np.sqrt(np.abs(gmm.covariances_.flatten()))
    weights = gmm.weights_.flatten()
    sorted_idx = np.argsort(means)

    print(f"\n  Final state map (k={best_k}):")
    print(f"  {'State':<12} {'Mean(W)':>12} {'Std(W)':>12} {'Weight%':>10}")
    print("  " + "-" * 50)
    for idx in sorted_idx:
        print(f"  {state_map[idx]:<12} {means[idx]:>12.2f} {stds[idx]:>12.2f} {100*weights[idx]:>9.2f}")
    print("=" * 72 + "\n")


# ============================================================
# MAIN
# ============================================================
def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        print(f"ERROR: could not create output directory '{OUTPUT_DIR}': {e}")
        return

    print("=" * 72)
    print("  GMM VALIDATION HARNESS")
    print(f"  File    : {DATA_PATH}")
    print(f"  Machine : {MACHINE_NAME}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 72)

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: data file not found: {DATA_PATH}")
        print("Update DATA_PATH at the top of this script.")
        return

    try:
        df = load_and_prepare_data(DATA_PATH)
    except Exception as e:
        print(f"\nERROR: load_and_prepare_data() failed: {e}")
        return

    if df is None or len(df) == 0:
        print("\nERROR: load_and_prepare_data() returned an empty dataframe.")
        return

    if "power" not in df.columns:
        print("\nERROR: 'power' column missing from loaded dataframe. Cannot continue.")
        return

    if "timestamp" not in df.columns:
        _warn("Column 'timestamp' not found; labeled time plot may fail.")

    t1 = test_data_sanity(df)
    if not t1:
        _warn("Data sanity test failed; continuing so you can inspect diagnostics.")

    X = df["power"].values.reshape(-1, 1)

    if len(np.unique(X)) < max(K_RANGE):
        print(f"\nERROR: power column has fewer unique values than the largest k in {K_RANGE}.")
        print("Cannot fit that many mixture components on this data.")
        return

    print("\n   Fitting candidate GMMs...")
    models = {}
    labels_by_k = {}
    for k in K_RANGE:
        print(f"   Fitting k={k}...", end=" ")
        try:
            gmm = fit_gmm(X, k)
        except Exception as e:
            print("failed")
            _fail(f"GMM fit failed for k={k}: {e}")
            continue
        models[k] = gmm
        labels_by_k[k] = gmm.predict(X)
        print("done")

    if not models:
        print("\nERROR: no candidate GMMs could be fit. Aborting.")
        return

    active_k_range = sorted(models.keys())
    if active_k_range != K_RANGE:
        _warn(f"Only fit k in {active_k_range} (some candidates failed).")

    t2 = test_gmm_convergence(models)

    # Use whichever k values actually fit successfully
    best_k, metrics_by_k, t3 = test_k_selection(X, models, labels_by_k, k_range=active_k_range)

    best_gmm = models[best_k]
    state_map = map_states(best_gmm, best_k)

    t4, state_map, separation_rows = test_state_separation(best_gmm, best_k)
    t5, max_probs = test_soft_assignment_confidence(best_gmm, X)
    t6 = test_weight_sanity(best_gmm, state_map)

    print("\n" + "=" * 72)
    print("  GENERATING DIAGNOSTIC PLOTS")
    print("=" * 72)

    plot_k_selection(metrics_by_k, best_k, OUTPUT_DIR)
    plot_gmm_histogram(df, best_gmm, state_map, OUTPUT_DIR)
    plot_confidence_histogram(max_probs, OUTPUT_DIR)

    if "timestamp" in df.columns:
        plot_labeled_sample(df, best_gmm, state_map, OUTPUT_DIR)
    else:
        _warn("Skipping labeled time-series plot because 'timestamp' is missing.")

    plot_cluster_separation(df, best_gmm, state_map, OUTPUT_DIR)

    results = {
        "T1 Data sanity": t1,
        "T2 GMM convergence": t2,
        "T3 k-selection support": t3,
        "T4 Physical separation": t4,
        "T5 Soft-assignment confidence": t5,
        "T6 Cluster weight sanity": t6,
    }

    print_final_report(results, best_k, state_map, best_gmm)
    print(f"All outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()