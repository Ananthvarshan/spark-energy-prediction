"""
============================================================
TASK 4c -- WHY THE TRANSITION PRIOR CANNOT FIX FLICKER
experiments/task4_flicker_diagnosis.py
============================================================

CONTEXT
-------
Task 4 found that the Gaussian HMM barely improves on the i.i.d.
GMM, and Task 4b found that the sticky prior in validate_gmm.py is
numerically inert (kappa = 30 pseudo-counts against ~5.8e5
observed transitions) -- but ALSO that fixing it does not help
much: raising the prior until the self-transition probability
reaches 0.9994 (an implied mean dwell of 1,605 s) still leaves the
flicker rate at 50%.

That second result rules out the obvious explanation.  This script
tests the remaining one.

THE HYPOTHESIS
--------------
Viterbi chooses the state path maximising

    sum_t [ log P(x_t | s_t)  +  log P(s_t | s_{t-1}) ]

The transition term can only influence the path if it is
comparable in magnitude to the emission term.  Switching state
costs

    log(p_self / p_switch)

which at p_self = 0.999 is about 7 nats -- and CANNOT exceed
~14 nats even at p_self = 0.999999.  Meanwhile a full-covariance
Gaussian fitted to six-dimensional electrical data with very small
within-state variance can produce emission log-likelihood
differences of hundreds or thousands of nats between adjacent
states.

If that is what is happening, the transition matrix is
arithmetically irrelevant, the HMM degenerates to per-sample
argmax, and no amount of prior strength will change it.  The
remedy is not a stronger prior but an explicit constraint on
duration.

WHAT THIS SCRIPT MEASURES
-------------------------
  1. The distribution of the emission log-likelihood gap between
     the best and second-best state at each timestep, against the
     maximum switching cost the transition matrix can impose.
  2. The fraction of timesteps where the emission gap exceeds that
     switching cost -- i.e. where the transition model provably
     cannot alter the decision.
  3. A minimum-dwell decode as the actual remedy, evaluated on the
     same metrics so it can be compared like for like.

USAGE
-----
    python -m experiments.task4_flicker_diagnosis
============================================================
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (            # noqa: E402
    DEFAULT_MACHINE, PHASE2_DIR, FLICKER_TAU_S,
    load_labelled, make_blocks, split_blocks, build_cluster_matrix,
    map_clusters_to_states, labels_to_states, run_lengths,
    flicker_rate, physics_compliance, schedule_accuracy, standby_hours,
    ensure_dir, save_json, info, section,
)

N_BLOCKS = 64
BLOCK_SECONDS = 7200.0
K_STATES = 4
RANDOM_STATE = 42

# Minimum dwell enforced by the post-decode remedy, in seconds.
MIN_DWELL_GRID_S = [10.0, 30.0, 60.0, 120.0]


def enforce_min_dwell(labels: np.ndarray, min_rows: int) -> np.ndarray:
    """
    Absorb every run shorter than `min_rows` into a neighbouring run.

    Repeatedly finds the shortest sub-threshold run and relabels it with
    whichever neighbour is longer (ties go to the left), until no short
    run remains.  This is the discrete analogue of imposing a minimum
    state duration -- the property a hidden semi-Markov model would give
    by construction, applied here as a decode-time constraint so that it
    can be compared against the plain HMM on identical emissions.

    Note this is exactly the operation lstm_pipeline.py already performs
    via smooth_short_standby_labels(); measuring it here makes explicit
    that the pipeline's low flicker comes from THIS step and not from the
    HMM.
    """
    out = labels.copy()
    while True:
        vals, lens = run_lengths(out)
        if len(lens) <= 1:
            break
        short = np.flatnonzero(lens < min_rows)
        if len(short) == 0:
            break
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        i = short[np.argmin(lens[short])]
        left_len = lens[i - 1] if i > 0 else -1
        right_len = lens[i + 1] if i + 1 < len(lens) else -1
        take = vals[i - 1] if left_len >= right_len and i > 0 else (
            vals[i + 1] if right_len > 0 else vals[i - 1])
        out[starts[i]:starts[i] + lens[i]] = take
    return out


def main(machine: str = DEFAULT_MACHINE, k: int = K_STATES,
         n_blocks: int = N_BLOCKS, block_seconds: float = BLOCK_SECONDS):
    from hmmlearn.hmm import GaussianHMM

    section("PHASE II / TASK 4c -- FLICKER DIAGNOSIS")
    out_dir = ensure_dir(f"{PHASE2_DIR}/task4/{machine}")

    df, meta = load_labelled(machine, drop_spikes=False, add_features=True)
    dt = meta["sample_interval_s"]

    blocks = make_blocks(df, block_seconds, n_blocks, dt,
                         random_state=RANDOM_STATE)
    fit_blocks, eval_blocks = split_blocks(blocks, fit_fraction=0.5)

    fit_idx = np.concatenate(fit_blocks)
    X_fit_all, scaler = build_cluster_matrix(df, fit_idx)
    X_fit_blocks, pos = [], 0
    for b in fit_blocks:
        X_fit_blocks.append(X_fit_all[pos:pos + len(b)])
        pos += len(b)
    X_eval_blocks = [build_cluster_matrix(df, b, scaler=scaler)[0]
                     for b in eval_blocks]

    # -- Fit the HMM exactly as validate_gmm.py does ----------------------
    section("Fitting Gaussian HMM (validate_gmm.py settings)")
    kappa = 30.0
    m = GaussianHMM(n_components=k, covariance_type="full", n_iter=200,
                    random_state=RANDOM_STATE, init_params="mc",
                    transmat_prior=np.ones((k, k)) + np.eye(k) * kappa)
    m.fit(np.vstack(X_fit_blocks), [len(b) for b in X_fit_blocks])

    T = m.transmat_
    p_self = np.diag(T)
    off = T - np.diag(p_self)
    p_switch_best = off.max(axis=1)
    max_switch_cost = float(np.max(np.log(p_self / np.clip(p_switch_best, 1e-300, None))))
    info(f"self-transition probabilities: {np.round(p_self, 6).tolist()}")
    info(f"maximum switching cost the transition matrix can impose: "
         f"{max_switch_cost:.2f} nats")

    # -- Emission log-likelihood gaps -------------------------------------
    section("Emission dominance")
    X_eval = np.vstack(X_eval_blocks)
    sample = X_eval[::37]                       # thin for speed; ~12k rows
    logprob = m._compute_log_likelihood(sample)
    part = np.partition(logprob, -2, axis=1)
    gap = part[:, -1] - part[:, -2]

    frac_dominant = float((gap > max_switch_cost).mean() * 100.0)
    stats = {
        "max_switch_cost_nats": max_switch_cost,
        "emission_gap_median_nats": float(np.median(gap)),
        "emission_gap_p10_nats": float(np.percentile(gap, 10)),
        "emission_gap_p90_nats": float(np.percentile(gap, 90)),
        "pct_timesteps_emission_dominates": frac_dominant,
        "n_sampled": int(len(sample)),
    }
    for key, val in stats.items():
        info(f"{key:>38}: {val:,.2f}" if isinstance(val, float) else
             f"{key:>38}: {val}")
    info(f"=> at {frac_dominant:.1f}% of timesteps the emission term exceeds "
         f"the largest penalty the transition matrix can apply, so the "
         f"transition model cannot change the decision there")

    # -- Baseline decode + minimum-dwell remedy ---------------------------
    section("Minimum-dwell decode as the remedy")
    base_labels = [m.predict(Xb) for Xb in X_eval_blocks]
    flat_idx = np.concatenate(eval_blocks)
    power = df["active_power"].to_numpy(float)[flat_idx]

    rows = []
    for min_dwell_s in [0.0] + MIN_DWELL_GRID_S:
        min_rows = max(1, int(round(min_dwell_s / dt)))
        labs = (base_labels if min_dwell_s == 0 else
                [enforce_min_dwell(lb, min_rows) for lb in base_labels])

        flat = np.concatenate(labs)
        mapping = map_clusters_to_states(flat, power)
        state_blocks = [labels_to_states(lb, mapping) for lb in labs]
        flat_states = np.concatenate(state_blocks)

        fl = flicker_rate(state_blocks, dt, FLICKER_TAU_S)
        ph = physics_compliance(df, flat_idx, flat_states)
        sc = schedule_accuracy(df, flat_idx, flat_states)
        rows.append({
            "min_dwell_s": min_dwell_s,
            "flicker_rate_pct": fl["flicker_rate_pct"],
            "flicker_standby_binary_pct": fl["flicker_rate_standby_binary_pct"],
            "median_dwell_s": fl["median_dwell_s"],
            "median_standby_run_s": fl["median_standby_run_s"],
            "n_standby_runs": fl["n_standby_runs"],
            "physics_score": ph["score"],
            "pf_separation": ph["values"].get("pf_separation", np.nan),
            "current_ratio": ph["values"].get("current_ratio", np.nan),
            "schedule_score": sc["balanced_schedule_score"],
            "standby_hours": standby_hours(state_blocks, dt),
        })
        info(f"min dwell {min_dwell_s:>6.0f}s -> flicker "
             f"{rows[-1]['flicker_rate_pct']:6.2f}%  "
             f"STANDBY-binary {rows[-1]['flicker_standby_binary_pct']:6.2f}%  "
             f"physics {rows[-1]['physics_score']:.2f}  "
             f"standby {rows[-1]['standby_hours']:.2f}h")

    tbl = pd.DataFrame(rows)
    section("MINIMUM-DWELL DECODE RESULTS")
    print(tbl.round(4).to_string(index=False))
    tbl.to_csv(f"{out_dir}/min_dwell_decode.csv", index=False)

    save_json({"meta": meta, "emission_dominance": stats,
               "transmat": T.tolist(), "min_dwell_decode": rows},
              f"{out_dir}/flicker_diagnosis.json")
    section("TASK 4c COMPLETE")
    return stats, tbl


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase II Task 4c -- flicker diagnosis")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--k", type=int, default=K_STATES)
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--block-seconds", type=float, default=BLOCK_SECONDS)
    a = ap.parse_args()
    main(a.machine, a.k, a.n_blocks, a.block_seconds)
