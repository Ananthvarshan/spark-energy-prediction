"""
============================================================
TASK 4b -- STICKY-PRIOR STRENGTH SWEEP
experiments/task4_sticky_prior_sweep.py
============================================================

WHY THIS EXPERIMENT EXISTS
--------------------------
The main Task 4 run produced a result that contradicts the
project's own design intent: the Gaussian HMM barely improved on
the i.i.d. GMM (flicker 64% vs 67%), when the whole point of the
temporal model is to eliminate flicker.

Inspecting the learned transition matrix explains it.  The HMM
learns a highly persistent OFF state (mean dwell ~106 s) and an
extremely persistent STANDBY state (~25,000 s), but a WORKING
state with a self-transition of only 0.864 -- a mean dwell of
7.4 samples.  The residual flicker is almost entirely churn along
the WORKING <-> PEAK_LOAD boundary, which is not a real physical
boundary but an artefact of splitting a continuous productive
load range into two states.

The second cause is the sticky prior itself.  validate_gmm.py
sets

    kappa = target_dwell_s / sample_interval_s          (= 30)
    transmat_prior = ones(k,k) + eye(k) * kappa

A Dirichlet prior contributes kappa PSEUDO-counts against the
real transition counts.  With ~10^5 observed transitions per
state, 30 pseudo-counts move the MAP self-transition probability
by ~3e-4.  The prior is numerically inert: it is present in the
code, documented in the comments, cited to Fox et al. (2011), and
has no measurable effect on the fitted model.

This script tests that diagnosis directly by sweeping the prior
strength as a FRACTION of the evidence,

    kappa = sticky_scale * (n_transitions / k)

so that sticky_scale is scale-free: 0 reproduces a flat prior,
and larger values hold their influence no matter how much data is
fitted.  If the diagnosis is right, flicker should fall
monotonically with sticky_scale while the physics checks stay
satisfied -- and the fixed-kappa setting should sit
indistinguishably next to sticky_scale = 0.

USAGE
-----
    python -m experiments.task4_sticky_prior_sweep
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
    flicker_rate, physics_compliance, schedule_accuracy, standby_hours,
    ensure_dir, save_json, info, section,
)

N_BLOCKS = 64
BLOCK_SECONDS = 7200.0
K_STATES = 4
HMM_N_ITER = 200
RANDOM_STATE = 42

# Prior strength as a fraction of the observed transition evidence.
SWEEP = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def fit_hmm(X_blocks, k, sticky_scale, n_iter=HMM_N_ITER, fixed_kappa=None):
    """
    Fit a Gaussian HMM with a self-transition Dirichlet prior whose
    strength is either a fixed pseudo-count (`fixed_kappa`, reproducing
    validate_gmm.py) or a fraction of the evidence (`sticky_scale`).
    """
    from hmmlearn.hmm import GaussianHMM

    X = np.vstack(X_blocks)
    lengths = [len(b) for b in X_blocks]
    n_transitions = len(X) - len(lengths)

    if fixed_kappa is not None:
        kappa = float(fixed_kappa)
    else:
        kappa = float(sticky_scale) * (n_transitions / k)

    transmat_prior = np.ones((k, k)) + np.eye(k) * kappa

    m = GaussianHMM(
        n_components=k, covariance_type="full", n_iter=n_iter,
        random_state=RANDOM_STATE, init_params="mc",
        transmat_prior=transmat_prior,
    )
    m.fit(X, lengths)
    return m, kappa, n_transitions


def main(machine: str = DEFAULT_MACHINE, k: int = K_STATES,
         n_blocks: int = N_BLOCKS, block_seconds: float = BLOCK_SECONDS):
    section("PHASE II / TASK 4b -- STICKY-PRIOR STRENGTH SWEEP")
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

    settings = [("fixed_kappa_30 (as implemented)", None, 30.0)]
    settings += [(f"sticky_scale={s:g}", s, None) for s in SWEEP]

    rows = []
    detail = {}
    for name, scale, fixed in settings:
        section(name)
        t0 = time.time()
        try:
            m, kappa, n_tr = fit_hmm(X_fit_blocks, k, scale, fixed_kappa=fixed)
        except Exception as exc:
            info(f"FAILED: {type(exc).__name__}: {exc}")
            continue

        label_blocks = [m.predict(Xb) for Xb in X_eval_blocks]
        flat_idx = np.concatenate(eval_blocks)
        flat_labels = np.concatenate(label_blocks)
        power = df["active_power"].to_numpy(float)[flat_idx]
        mapping = map_clusters_to_states(flat_labels, power)
        state_blocks = [labels_to_states(lb, mapping) for lb in label_blocks]
        flat_states = np.concatenate(state_blocks)

        fl = flicker_rate(state_blocks, dt, FLICKER_TAU_S)
        ph = physics_compliance(df, flat_idx, flat_states)
        sc = schedule_accuracy(df, flat_idx, flat_states)

        self_p = np.diag(m.transmat_)
        row = {
            "setting": name,
            "kappa": kappa,
            "kappa_over_evidence": kappa / max(n_tr / k, 1),
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
            "min_self_transition": float(self_p.min()),
            "mean_self_transition": float(self_p.mean()),
            "implied_min_dwell_s": float(1.0 / (1.0 - self_p.min()) * dt),
            "converged": bool(m.monitor_.converged),
            "fit_seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        detail[name] = {
            "transmat": m.transmat_.tolist(),
            "cluster_to_state": {str(c): s for c, s in mapping.items()},
            "per_state_dwell": fl["per_state_dwell"],
            "physics_checks": ph["checks"],
        }
        info(f"kappa {kappa:>12.1f}  flicker {row['flicker_rate_pct']:6.2f}%  "
             f"STANDBY-binary {row['flicker_standby_binary_pct']:6.2f}%  "
             f"min p_self {row['min_self_transition']:.4f}  "
             f"physics {row['physics_score']:.2f}  "
             f"standby {row['standby_hours']:.2f}h")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{out_dir}/sticky_prior_sweep.csv", index=False)
    section("STICKY-PRIOR SWEEP RESULTS")
    print(tbl.round(4).to_string(index=False))
    save_json({"meta": meta, "sweep": rows, "detail": detail},
              f"{out_dir}/sticky_prior_sweep.json")
    info(f"wrote {out_dir}/sticky_prior_sweep.csv")
    return tbl


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase II Task 4b -- sticky prior sweep")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--k", type=int, default=K_STATES)
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--block-seconds", type=float, default=BLOCK_SECONDS)
    a = ap.parse_args()
    main(a.machine, a.k, a.n_blocks, a.block_seconds)
