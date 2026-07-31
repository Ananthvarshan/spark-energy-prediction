"""
============================================================
TASK 4 -- STATE DETECTION METHOD COMPARISON
experiments/task4_state_detection.py
============================================================

PURPOSE
-------
Phase II / Task 4.  Compare five unsupervised state-detection
methods on identical data, identical features, and identical
metrics, to establish (a) that a mixture model beats simpler
clustering on this problem and (b) how much of the benefit comes
specifically from modelling TIME rather than from the emission
model.

METHODS
-------
  kmeans     K-Means                     Lloyd (1982)
  gmm        Gaussian mixture, i.i.d.    McLachlan & Peel (2000)
  dbscan     DBSCAN                      Ester et al. (1996)
  spectral   Spectral clustering         Ng, Jordan & Weiss (2001)
  gmm_hmm    Gaussian HMM (PROPOSED)     Rabiner (1989); Fox et al. (2011)

`gmm` and `gmm_hmm` are the ablation pair that isolates the
contribution of temporal modelling: both use full-covariance
Gaussian emissions over the same k, and differ ONLY in whether the
state sequence is decoded jointly (Viterbi over a learned
transition matrix) or independently per sample.

EVALUATION PROTOCOL
-------------------
Because two of the metrics (flicker rate, dwell distribution) are
only defined on a contiguous series, the evaluation set is a set of
contiguous BLOCKS, not a random row sample.  Blocks are spread
evenly across the 155-day record and never cross an acquisition gap.
Alternate blocks form the FIT set; the remainder form the EVAL set.

Every method is fitted on FIT and produces labels on EVAL:
  * kmeans / gmm / gmm_hmm have a native out-of-sample rule
    (nearest centroid, posterior argmax, Viterbi decode).
  * dbscan and spectral are transductive -- neither defines a
    predict() for unseen points -- so labels are extended to EVAL by
    1-nearest-neighbour in the same standardised feature space.
    This is the conventional out-of-sample extension and is applied
    identically to both, so it cannot favour either.

METRICS (no ground truth exists -- see experiments/common.py)
------------------------------------------------------------
  flicker_rate_pct          Method 4 of the five-method proof
  physics_score             Methods 1, 2 + variance/ordering checks
  balanced_schedule_score   Test T10, factory closed-window agreement
  median_dwell_s            dwell plausibility
  ari/ami vs reference      how far each partition sits from the
                            pipeline's exported labels (context, not
                            correctness)

USAGE
-----
    python -m experiments.task4_state_detection
    python -m experiments.task4_state_detection --n-blocks 64 --k 4
============================================================
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (            # noqa: E402
    DEFAULT_MACHINE, PHASE2_DIR, FLICKER_TAU_S,
    load_labelled, make_blocks, split_blocks, build_cluster_matrix,
    map_clusters_to_states, labels_to_states,
    flicker_rate, physics_compliance, schedule_accuracy,
    agreement_with_reference, standby_hours,
    ensure_dir, save_json, info, section,
)


# ── Configuration ─────────────────────────────────────────────────────────────

N_BLOCKS = 64                 # contiguous evaluation blocks
BLOCK_SECONDS = 7200.0        # 2 hours each -> 128 h total, ~460k rows
K_STATES = 4                  # OFF / STANDBY / WORKING / PEAK_LOAD

# Rows drawn from the FIT blocks to fit each emission model.  DBSCAN is
# O(n log n) with a spatial index but spectral clustering builds an n x n
# affinity matrix, so the two density/graph methods get a much smaller
# fitting sample.  This is a genuine practical property of those methods on
# a 4.4M-row series and is reported as such rather than hidden.
N_FIT_ROWS = 200_000
N_FIT_ROWS_DBSCAN = 60_000
N_FIT_ROWS_SPECTRAL = 15_000

# Sticky self-transition prior for the HMM, matching validate_gmm.py.
STICKY_TARGET_DWELL_S = 30.0
HMM_N_ITER = 200

RANDOM_STATE = 42


# ── Method implementations ────────────────────────────────────────────────────
# Each returns (labels_per_eval_block, extra_info_dict).

def _subsample(n: int, cap: int, rng) -> np.ndarray:
    return (rng.choice(n, size=cap, replace=False) if n > cap
            else np.arange(n))


def _nn_extend(X_fit, y_fit, X_eval, n_jobs: int = -1) -> np.ndarray:
    """1-NN out-of-sample extension for transductive clusterers."""
    from sklearn.neighbors import KNeighborsClassifier
    keep = y_fit >= 0                      # never propagate a noise label
    if keep.sum() == 0:
        return np.full(len(X_eval), -1, dtype=int)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=n_jobs)
    knn.fit(X_fit[keep], y_fit[keep])
    return knn.predict(X_eval).astype(int)


def run_kmeans(X_fit, X_eval_blocks, k, rng):
    from sklearn.cluster import KMeans
    m = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_fit)
    return [m.predict(Xb) for Xb in X_eval_blocks], {
        "n_clusters_found": k, "inertia": float(m.inertia_)
    }


def run_gmm(X_fit, X_eval_blocks, k, rng):
    from sklearn.mixture import GaussianMixture
    m = GaussianMixture(n_components=k, covariance_type="full",
                        n_init=5, random_state=RANDOM_STATE).fit(X_fit)
    return [m.predict(Xb) for Xb in X_eval_blocks], {
        "n_clusters_found": k,
        "converged": bool(m.converged_),
        "bic": float(m.bic(X_fit)),
        "aic": float(m.aic(X_fit)),
    }


def run_dbscan(X_fit, X_eval_blocks, k, rng):
    """
    DBSCAN with eps chosen by the standard k-distance elbow heuristic
    (Ester et al., 1996, Sec. 4.2): sort every point's distance to its
    min_samples-th neighbour and take a high percentile as eps.

    DBSCAN chooses its own cluster count; we do NOT force it to k.  If it
    returns a different number, that is a finding about the method, and
    the state mapping handles it by power rank.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    sel = _subsample(len(X_fit), N_FIT_ROWS_DBSCAN, rng)
    Xs = X_fit[sel]

    min_samples = 2 * Xs.shape[1]          # 2 * dimensionality, common default
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    eps = float(np.percentile(dists[:, -1], 95))

    m = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(Xs)
    y_fit = m.labels_
    found = int(len(set(y_fit)) - (1 if -1 in y_fit else 0))
    noise_frac = float((y_fit < 0).mean())

    return [_nn_extend(Xs, y_fit, Xb) for Xb in X_eval_blocks], {
        "n_clusters_found": found,
        "eps": eps,
        "min_samples": int(min_samples),
        "noise_fraction_fit": noise_frac,
        "n_fit_rows": int(len(Xs)),
    }


def run_spectral(X_fit, X_eval_blocks, k, rng):
    """
    Spectral clustering on a subsample, extended by 1-NN.

    The affinity matrix is n x n, so n is capped at N_FIT_ROWS_SPECTRAL
    (15k -> a 1.8 GB float64 matrix already).  Nearest-neighbour affinity
    is used rather than a dense RBF kernel for the same reason.
    """
    from sklearn.cluster import SpectralClustering

    sel = _subsample(len(X_fit), N_FIT_ROWS_SPECTRAL, rng)
    Xs = X_fit[sel]
    m = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=15,
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(Xs)
    y_fit = m.labels_.astype(int)

    return [_nn_extend(Xs, y_fit, Xb) for Xb in X_eval_blocks], {
        "n_clusters_found": k,
        "n_fit_rows": int(len(Xs)),
        "affinity": "nearest_neighbors(15)",
    }


def run_gmm_hmm(X_fit_blocks, X_eval_blocks, k, rng, sample_interval_s,
                X_fit_sample=None, min_dwell_s: float = 0.0):
    """
    PROPOSED METHOD.  Gaussian HMM decoded by Viterbi over the same
    emission model the i.i.d. GMM baseline uses.

    EMISSIONS ARE WARM-STARTED FROM THE FITTED GMM AND HELD FIXED
    (`params="t"` -- only the transition matrix is estimated).  This is a
    deliberate experimental-design choice and it matters:

    Fitting the HMM's emissions from scratch by Baum-Welch was tried
    first and converged to a materially worse local optimum than the
    GMM's -- its "STANDBY" component absorbed part of the productive
    range (mean 10.3 kW, sd 15.4 kW) instead of finding the true no-load
    band at 3.7 kW, and it consequently failed the variance-ordering
    check.  Comparing that fit against the GMM would have confounded two
    different things: the value of temporal decoding, and the luck of the
    EM initialisation.

    Holding emissions identical makes `gmm` and `gmm_hmm` a clean
    ablation pair.  Any difference between their rows is attributable to
    the decoding rule alone -- posterior argmax per sample versus Viterbi
    over a learned transition matrix -- which is the question the row is
    supposed to answer.

    `min_dwell_s > 0` additionally enforces a minimum state duration on
    the decoded path.  See experiments/task4_flicker_diagnosis.py for why
    this, and not the transition prior, is what actually controls flicker.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.mixture import GaussianMixture
    from experiments.task4_flicker_diagnosis import enforce_min_dwell

    X = np.vstack(X_fit_blocks)
    lengths = [len(b) for b in X_fit_blocks]

    # -- emissions: identical to the `gmm` baseline row -------------------
    base = X_fit_sample if X_fit_sample is not None else X
    g = GaussianMixture(n_components=k, covariance_type="full",
                        n_init=5, random_state=RANDOM_STATE).fit(base)

    target_dwell_samples = max(STICKY_TARGET_DWELL_S / sample_interval_s, 1.0)
    transmat_prior = np.ones((k, k)) + np.eye(k) * target_dwell_samples

    m = GaussianHMM(
        n_components=k,
        covariance_type="full",
        n_iter=HMM_N_ITER,
        random_state=RANDOM_STATE,
        init_params="",          # nothing auto-initialised
        params="t",              # only the transition matrix is learned
        transmat_prior=transmat_prior,
    )
    m.startprob_ = g.weights_.copy()
    m.means_ = g.means_.copy()
    m.covars_ = g.covariances_.copy()
    m.transmat_ = np.full((k, k), 1.0 / k)
    m.fit(X, lengths)

    labels = [m.predict(Xb) for Xb in X_eval_blocks]

    min_rows = max(1, int(round(min_dwell_s / sample_interval_s)))
    if min_dwell_s > 0:
        labels = [enforce_min_dwell(lb, min_rows) for lb in labels]

    return labels, {
        "n_clusters_found": k,
        "converged": bool(m.monitor_.converged),
        "n_iter": int(m.monitor_.iter),
        "transmat": m.transmat_.tolist(),
        "startprob": m.startprob_.tolist(),
        "sticky_target_dwell_s": STICKY_TARGET_DWELL_S,
        "mean_self_transition": float(np.mean(np.diag(m.transmat_))),
        "emissions": "warm-started from GMM, held fixed (params='t')",
        "min_dwell_s": min_dwell_s,
    }


METHODS = {
    "kmeans":   ("K-Means",              "Lloyd (1982)"),
    "gmm":      ("GMM (i.i.d.)",         "McLachlan & Peel (2000)"),
    "dbscan":   ("DBSCAN",               "Ester et al. (1996)"),
    "spectral": ("Spectral clustering",  "Ng et al. (2001)"),
    "gmm_hmm":  ("GMM-HMM (proposed)",   "Rabiner (1989); Fox et al. (2011)"),
    "gmm_hmm_mindwell": ("GMM-HMM + min-dwell decode (proposed)",
                         "Rabiner (1989); Yu (2010)"),
}

# Minimum state duration enforced by the `gmm_hmm_mindwell` variant.  Chosen to
# equal the flicker threshold itself (FLICKER_TAU_S), so the constraint is not
# tuned to flatter the metric -- it removes exactly what the metric defines as
# implausible, and nothing more.
MIN_DWELL_S = FLICKER_TAU_S


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_partition(
    df: pd.DataFrame,
    eval_blocks: list[np.ndarray],
    label_blocks: list[np.ndarray],
    sample_interval_s: float,
) -> dict:
    """Score one method's EVAL-block labelling on every Task 4 metric."""
    flat_idx = np.concatenate(eval_blocks)
    flat_labels = np.concatenate(label_blocks)

    power = df["active_power"].to_numpy(float)[flat_idx]
    mapping = map_clusters_to_states(flat_labels, power)
    state_blocks = [labels_to_states(lb, mapping) for lb in label_blocks]
    flat_states = np.concatenate(state_blocks)

    res: dict = {}
    res.update(flicker_rate(state_blocks, sample_interval_s, FLICKER_TAU_S))

    phys = physics_compliance(df, flat_idx, flat_states)
    res["physics_score"] = phys["score"]
    res["physics_checks"] = phys["checks"]
    res["physics_values"] = phys["values"]
    res["per_state"] = phys["per_state"]

    res.update(schedule_accuracy(df, flat_idx, flat_states))

    if "state" in df.columns:
        ref = df["state"].to_numpy()[flat_idx]
        res.update(agreement_with_reference(flat_states, ref))

    res["standby_hours"] = standby_hours(state_blocks, sample_interval_s)
    res["noise_fraction"] = float((flat_states == "NOISE").mean())
    res["cluster_to_state"] = {str(c): s for c, s in mapping.items()}
    res["n_eval_rows"] = int(len(flat_idx))
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    machine: str = DEFAULT_MACHINE,
    n_blocks: int = N_BLOCKS,
    block_seconds: float = BLOCK_SECONDS,
    k: int = K_STATES,
    methods: list[str] | None = None,
    drop_spikes: bool = False,
    tag: str = "",
):
    """
    NOTE ON SPIKE ROWS -- why the default is to KEEP them here.

    lstm_pipeline.py drops MAD-flagged spike rows before training, and
    Task 3 follows it.  Task 4 must NOT, because its headline metrics are
    dwell times.  Removing 19.5% of rows from a 1 Hz series makes the
    retained rows non-uniformly spaced, so a run of N rows no longer
    spans N seconds -- every dwell is silently compressed and every
    flicker rate correspondingly inflated.  Keeping every row preserves
    the 1 Hz grid, so run length in rows equals dwell in seconds and the
    metric means what it says.

    Pass drop_spikes=True to reproduce the de-spiked variant as a
    sensitivity check; both are reported in the Phase II notes.
    """
    section("PHASE II / TASK 4 -- STATE DETECTION METHOD COMPARISON")
    suffix = tag or ("_despiked" if drop_spikes else "")
    out_dir = ensure_dir(f"{PHASE2_DIR}/task4/{machine}")
    rng = np.random.default_rng(RANDOM_STATE)
    methods = methods or list(METHODS)

    df, meta = load_labelled(machine, drop_spikes=drop_spikes, add_features=True)
    meta["drop_spikes"] = drop_spikes
    dt = meta["sample_interval_s"]

    section("Evaluation blocks")
    blocks = make_blocks(df, block_seconds, n_blocks, dt,
                         random_state=RANDOM_STATE)
    fit_blocks, eval_blocks = split_blocks(blocks, fit_fraction=0.5)
    info(f"FIT  : {len(fit_blocks)} blocks, {sum(len(b) for b in fit_blocks):,} rows")
    info(f"EVAL : {len(eval_blocks)} blocks, {sum(len(b) for b in eval_blocks):,} rows")

    # -- Shared feature representation -----------------------------------
    fit_idx = np.concatenate(fit_blocks)
    X_fit_all, scaler = build_cluster_matrix(df, fit_idx)
    X_fit_blocks = []
    pos = 0
    for b in fit_blocks:
        X_fit_blocks.append(X_fit_all[pos:pos + len(b)])
        pos += len(b)

    sel = _subsample(len(X_fit_all), N_FIT_ROWS, rng)
    X_fit = X_fit_all[sel]
    info(f"emission models fitted on {len(X_fit):,} of {len(X_fit_all):,} FIT rows")

    X_eval_blocks = [build_cluster_matrix(df, b, scaler=scaler)[0]
                     for b in eval_blocks]

    # -- Run every method -------------------------------------------------
    results: dict = {}
    for key in methods:
        name, citation = METHODS[key]
        section(f"Method: {name}  [{citation}]")
        t0 = time.time()
        try:
            if key == "kmeans":
                labels, extra = run_kmeans(X_fit, X_eval_blocks, k, rng)
            elif key == "gmm":
                labels, extra = run_gmm(X_fit, X_eval_blocks, k, rng)
            elif key == "dbscan":
                labels, extra = run_dbscan(X_fit, X_eval_blocks, k, rng)
            elif key == "spectral":
                labels, extra = run_spectral(X_fit, X_eval_blocks, k, rng)
            elif key == "gmm_hmm":
                labels, extra = run_gmm_hmm(X_fit_blocks, X_eval_blocks, k,
                                            rng, dt, X_fit_sample=X_fit)
            elif key == "gmm_hmm_mindwell":
                labels, extra = run_gmm_hmm(X_fit_blocks, X_eval_blocks, k,
                                            rng, dt, X_fit_sample=X_fit,
                                            min_dwell_s=MIN_DWELL_S)
            else:
                raise KeyError(key)
        except Exception as exc:            # a failed baseline is a result
            info(f"FAILED: {type(exc).__name__}: {exc}")
            results[key] = {"method": name, "citation": citation,
                            "error": f"{type(exc).__name__}: {exc}"}
            continue

        fit_s = time.time() - t0
        scores = evaluate_partition(df, eval_blocks, labels, dt)
        scores.update({"method": name, "citation": citation,
                       "fit_predict_seconds": round(fit_s, 1), **extra})
        results[key] = scores

        info(f"flicker {scores['flicker_rate_pct']:.2f}% "
             f"(STANDBY-binary {scores['flicker_rate_standby_binary_pct']:.2f}%)  "
             f"physics {scores['physics_score']:.2f}  "
             f"schedule {scores['balanced_schedule_score']:.1f}%  "
             f"median dwell {scores['median_dwell_s']:.0f}s  "
             f"({fit_s:.1f}s)")

        np.savez_compressed(
            f"{out_dir}/labels_{key}{suffix}.npz",
            **{f"block_{i}": lb for i, lb in enumerate(labels)},
        )

    # -- Reference row: the pipeline's own exported labels ----------------
    ref_blocks = [df["state"].to_numpy()[b] for b in eval_blocks]
    flat_idx = np.concatenate(eval_blocks)
    ref_flat = np.concatenate(ref_blocks)
    ref_scores = {"method": "Exported pipeline labels (reference)",
                  "citation": "validate_gmm.py export"}
    ref_scores.update(flicker_rate(ref_blocks, dt, FLICKER_TAU_S))
    _p = physics_compliance(df, flat_idx, ref_flat)
    ref_scores["physics_score"] = _p["score"]
    ref_scores["physics_checks"] = _p["checks"]
    ref_scores["physics_values"] = _p["values"]
    ref_scores["per_state"] = _p["per_state"]
    ref_scores.update(schedule_accuracy(df, flat_idx, ref_flat))
    ref_scores["standby_hours"] = standby_hours(ref_blocks, dt)
    results["_reference_export"] = ref_scores

    # -- Comparison table --------------------------------------------------
    section("TASK 4 -- COMPARISON TABLE")
    rows = []
    for key, r in results.items():
        if "error" in r:
            rows.append({"key": key, "method": r["method"], "status": "FAILED"})
            continue
        rows.append({
            "key": key,
            "method": r["method"],
            "citation": r.get("citation", ""),
            "n_clusters": r.get("n_clusters_found", np.nan),
            "flicker_rate_pct": r["flicker_rate_pct"],
            "flicker_standby_binary_pct": r["flicker_rate_standby_binary_pct"],
            "median_dwell_s": r["median_dwell_s"],
            "median_standby_run_s": r["median_standby_run_s"],
            "n_standby_runs": r["n_standby_runs"],
            "physics_score": r["physics_score"],
            "n_physics_passed": sum(1 for v in r["physics_checks"].values() if v),
            "n_physics_applicable": sum(1 for v in r["physics_checks"].values() if v is not None),
            "pf_separation": r["physics_values"].get("pf_separation", np.nan),
            "current_ratio": r["physics_values"].get("current_ratio", np.nan),
            "closed_window_pct": r["closed_window_correct_pct"],
            "open_window_pct": r["open_window_productive_pct"],
            "schedule_score": r["balanced_schedule_score"],
            "standby_hours": r["standby_hours"],
            "ari_vs_reference": r.get("ari_vs_reference", np.nan),
            "fit_seconds": r.get("fit_predict_seconds", np.nan),
        })
    tbl = pd.DataFrame(rows).set_index("key")
    tbl.to_csv(f"{out_dir}/comparison_table{suffix}.csv")

    show = ["method", "n_clusters", "flicker_rate_pct",
            "flicker_standby_binary_pct", "median_dwell_s",
            "median_standby_run_s", "physics_score", "pf_separation",
            "current_ratio", "schedule_score", "standby_hours"]
    print(tbl[[c for c in show if c in tbl.columns]].round(3).to_string())

    save_json({"meta": meta,
               "config": {"n_blocks": n_blocks, "block_seconds": block_seconds,
                          "k": k, "n_fit_rows": N_FIT_ROWS,
                          "n_fit_rows_dbscan": N_FIT_ROWS_DBSCAN,
                          "n_fit_rows_spectral": N_FIT_ROWS_SPECTRAL,
                          "sticky_target_dwell_s": STICKY_TARGET_DWELL_S,
                          "flicker_tau_s": FLICKER_TAU_S,
                          "random_state": RANDOM_STATE,
                          "drop_spikes": drop_spikes},
               "results": results},
              f"{out_dir}/task4_results{suffix}.json")
    info(f"wrote {out_dir}/comparison_table.csv")
    section("TASK 4 COMPLETE")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase II Task 4 -- state detection comparison")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--block-seconds", type=float, default=BLOCK_SECONDS)
    ap.add_argument("--k", type=int, default=K_STATES)
    ap.add_argument("--methods", nargs="*", default=None,
                    help=f"subset of {list(METHODS)}")
    ap.add_argument("--drop-spikes", action="store_true",
                    help="de-spiked sensitivity variant (see main() docstring)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    a = ap.parse_args()
    main(a.machine, a.n_blocks, a.block_seconds, a.k, a.methods,
         a.drop_spikes, a.tag)
