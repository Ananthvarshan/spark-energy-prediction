"""
============================================================
TASK 12 -- REPEATED RUNS
experiments/task12_multiple_runs.py
============================================================

WHAT THIS PRODUCES
------------------
Every headline number in Phases I-IV came from one run at one seed.
This script re-runs the three places where a seed can move a result
and stores the per-run values so Task 11 can test them:

  A  forecasting models   Task 5's protocol, repeated over seeds.
                          Gives mean +/- sd per metric and, more
                          importantly, per-window predictions for
                          every run so the comparisons can be PAIRED.

  B  labelling            the Phase IV labeller re-fitted at ten
                          seeds on the same preprocessed record.
                          Preprocessing is deterministic, so this
                          isolates exactly what the seed controls:
                          the GMM's EM initialisation and the blocks
                          the transition matrix is fitted on.

  C  decision layer       Phase III's policy comparison re-run at ten
                          seeds, which move the forecaster's fit and
                          the restart-signature bootstrap.

WHY THE SEED COUNTS DIFFER BY MODEL
-----------------------------------
The plan asks for 20-30 runs of everything.  That is affordable for
the tree and trivial for the untrained baseline, and it is not
affordable for five neural architectures on this hardware: one seed
of the five costs ~42 min, so 25 seeds each would be 87 h.  The
seed counts are therefore stated per model (`SEED_PLAN`) rather
than uniform, and -- this is the substantive point -- the
CONFIRMATORY tests in Task 11 are paired over the ~5,400 test
WINDOWS, not over seeds.  Seeds quantify training variability; they
are not the sample the significance tests use, so a smaller number
of them costs precision in the reported standard deviation, not
validity in the test.

    python -m experiments.task12_multiple_runs                 # everything
    python -m experiments.task12_multiple_runs --only labelling
    python -m experiments.task12_multiple_runs --neural-seeds 3
============================================================
"""

from __future__ import annotations

import os
import sys
import gc
import time
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from experiments.common import (                    # noqa: E402
    DEFAULT_MACHINE, MACHINES, PHASE4_DIR, FLICKER_TAU_S,
    load_labelled, physics_compliance, flicker_rate,
    ensure_dir, save_json, info, section,
)
from src.stats import describe_runs                 # noqa: E402

PHASE5_DIR = "outputs/phase5"
TASK12_DIR = f"{PHASE5_DIR}/task12"

# Task 5's protocol, reproduced exactly so the repeated runs are comparable
# with the Phase II table (which was run at --decimate 10 --epochs 30).
DECIMATE = 10
LOOKBACK_S, HORIZON_S, STRIDE_S = 600.0, 300.0, 120.0
EPOCHS, BATCH_SIZE, PATIENCE = 30, 64, 5

# Runs per model.  Stated, not uniform -- see the module docstring.
SEED_PLAN = {
    "persistence": 1,        # deterministic: no fitted parameters
    "xgboost": 15,
    "seq2seq_lstm": 5,
    "gru": 5,
    "vanilla_lstm": 5,
    "tcn": 5,
    "transformer": 5,
}

LABEL_SEEDS = 10
DECISION_SEEDS = 10


# ══════════════════════════════════════════════════════════════════════════════
# A -- forecasting models over seeds
# ══════════════════════════════════════════════════════════════════════════════

def forecasting_seeds(
    machine: str = DEFAULT_MACHINE,
    models: list[str] | None = None,
    seed_plan: dict | None = None,
    source: str = "auto",
    epochs: int = EPOCHS,
) -> dict:
    """
    Run Task 5's six models plus the persistence reference over seeds.

    The windows are built ONCE and shared by every run, which is what makes
    the comparison paired: run i of model A and run j of model B are scored on
    the identical 5,449 test windows in the identical order, so Task 11 can
    difference them window by window.  Task 5 rebuilt the windows per
    invocation; that is fine for a table of means and useless for a paired
    test.
    """
    from experiments.task5_forecasting_baselines import (
        decimate, make_windows, standardise, standby_seconds,
        sequence_metrics, duration_metrics, train_neural, train_xgboost,
        TRAIN_PCT, VAL_PCT,
    )
    from experiments.baseline_models import MODEL_INFO
    from src.features import all_feature_names
    from sklearn.utils.class_weight import compute_class_weight

    section("TASK 12A -- forecasting models over seeds")
    out_dir = ensure_dir(f"{TASK12_DIR}/{machine}/forecasting")
    seed_plan = seed_plan or SEED_PLAN
    models = models or list(seed_plan)

    df, meta = load_labelled(machine, drop_spikes=True, add_features=True,
                             source=source)
    feature_cols = [f for f in all_feature_names() if f in df.columns]
    state_names = sorted(df["state"].unique().tolist())
    encoder = {s: i for i, s in enumerate(state_names)}
    df["state_id"] = df["state"].map(encoder).astype(np.int16)
    standby_id = encoder["STANDBY"]

    df = decimate(df, DECIMATE)
    step_s = meta["sample_interval_s"] * DECIMATE
    lb = max(2, int(round(LOOKBACK_S / step_s)))
    hz = max(1, int(round(HORIZON_S / step_s)))
    st = max(1, int(round(STRIDE_S / step_s)))

    n = len(df)
    c1, c2 = int(n * TRAIN_PCT), int(n * (TRAIN_PCT + VAL_PCT))
    W = {}
    for name, part in (("train", df.iloc[:c1]), ("val", df.iloc[c1:c2]),
                       ("test", df.iloc[c2:])):
        W[name] = make_windows(part.reset_index(drop=True), feature_cols,
                               lb, hz, st, return_last_state=True)
    del df
    gc.collect()

    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, l_te) = (
        W["train"], W["val"], W["test"])
    (X_tr, X_va, X_te), _ = standardise(X_tr, X_va, X_te)
    info(f"{len(X_tr):,} train / {len(X_va):,} val / {len(X_te):,} test windows, "
         f"step {step_s:.0f}s")

    flat = y_tr.ravel()
    present = np.unique(flat)
    class_weight = np.ones(len(state_names), dtype=np.float32)
    class_weight[present] = compute_class_weight("balanced", classes=present,
                                                 y=flat)
    w_tr = class_weight[y_tr].astype(np.float32)
    w_va = class_weight[y_va].astype(np.float32)
    true_s = standby_seconds(y_te, standby_id, step_s)

    # A model already run is kept rather than dropped: this step costs hours,
    # and a partial re-run (say, to add seeds for one architecture) must not
    # silently delete the models it did not touch.  The stored per-window
    # predictions are what Task 11 pairs over, and they live in their own npz
    # per model, so merging the summary here is enough to keep the set
    # consistent.
    results: dict = {}
    out_path = f"{out_dir}/forecasting_seeds.json"
    prev_path = out_path
    prev_all: dict = {}
    if os.path.exists(prev_path):
        with open(prev_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        prev_all = prev.get("results", {})
        results = {k: v for k, v in prev_all.items() if k not in models}
        if results:
            info(f"keeping {len(results)} model(s) from the previous run: "
                 f"{list(results)}")

    def _persist() -> None:
        """
        Write the summary after every SEED, not merely after every model.

        One model can cost two hours here, so checkpointing per model still
        throws away most of a run that is interrupted part-way through a seed
        loop -- which is exactly how two attempts at this step have now been
        lost.  The unit of work that survives an interruption is therefore the
        seed.
        """
        save_json({"machine": machine, "label_source": meta["label_source"],
                   "config": {"decimate": DECIMATE, "step_seconds": step_s,
                              "lookback_rows": lb, "horizon_rows": hz,
                              "stride_rows": st, "epochs": epochs,
                              "seed_plan": seed_plan,
                              "n_test_windows": int(len(X_te)),
                              "state_encoder": encoder,
                              "feature_cols": feature_cols},
                   "results": results},
                  out_path)

    def _checkpoint(key, name, citation, per_seed, preds) -> None:
        """Fold this model's finished seeds into `results` and write both files."""
        if preds:
            np.savez_compressed(f"{out_dir}/preds_{key}.npz",
                                y_pred=np.stack(preds), y_true=y_te,
                                true_standby_s=true_s, step_s=step_s,
                                standby_id=standby_id)
        ok = [s for s in per_seed if "error" not in s]
        agg = {}
        if ok:
            for metric in ("accuracy", "macro_f1", "f1_STANDBY", "standby_mae_s",
                           "standby_rmse_s", "standby_r2", "seconds"):
                if metric in ok[0]:
                    agg[metric] = describe_runs([s[metric] for s in ok])
        results[key] = {"model": name, "citation": citation,
                        "n_seeds": len(ok), "runs": per_seed, "aggregate": agg}
        _dump_seed_table(results, out_dir)
        _persist()

    def _resume(key):
        """
        Recover this model's already-finished seeds, if they are on the same
        windows.

        The stored `y_true` is compared element-wise against the windows just
        built.  That is the check that matters: Task 11 pairs models window by
        window, so reusing seeds computed on a different test set would
        silently invalidate every paired test downstream.  Anything that does
        not match exactly is discarded and recomputed.
        """
        old = prev_all.get(key)
        npz_path = f"{out_dir}/preds_{key}.npz"
        if not old or not os.path.exists(npz_path):
            return [], [], set()
        try:
            z = np.load(npz_path)
            if z["y_true"].shape != y_te.shape or not np.array_equal(z["y_true"], y_te):
                info(f"{key}: stored predictions are on different windows; recomputing")
                return [], [], set()
            stored = z["y_pred"]
            runs = list(old.get("runs", []))
            ok = [r for r in runs if "error" not in r]
            if len(ok) != len(stored):
                info(f"{key}: {len(ok)} stored run(s) but {len(stored)} prediction "
                     f"array(s); recomputing")
                return [], [], set()
            done = {int(r["seed"]) for r in ok}
            info(f"{key}: resuming, {len(done)} seed(s) already done {sorted(done)}")
            return runs, [stored[i] for i in range(len(stored))], done
        except Exception as exc:
            info(f"{key}: could not reuse stored runs ({type(exc).__name__}: {exc}); "
                 f"recomputing")
            return [], [], set()

    for key in models:
        name, citation = MODEL_INFO[key]
        n_seeds = seed_plan.get(key, 1)
        section(f"{name} -- {n_seeds} run(s)")
        per_seed, preds, done_seeds = _resume(key)
        for seed in range(n_seeds):
            if seed in done_seeds:
                continue
            t0 = time.time()
            try:
                if key == "persistence":
                    y_pred = np.repeat(l_te[:, None], hz, axis=1).astype(np.int16)
                    extra = {"n_params": 0, "train_seconds": 0.0}
                elif key == "xgboost":
                    y_pred, extra = train_xgboost(
                        X_tr, y_tr, X_va, y_va, X_te, len(state_names),
                        class_weight, seed, out_dir)
                else:
                    y_pred, extra = train_neural(
                        key, X_tr, y_tr, X_va, y_va, X_te, len(state_names),
                        w_tr, w_va, epochs, BATCH_SIZE, PATIENCE, seed, out_dir)
            except Exception as exc:
                info(f"seed {seed} FAILED: {type(exc).__name__}: {exc}")
                per_seed.append({"seed": seed, "error": str(exc)})
                _checkpoint(key, name, citation, per_seed, preds)
                gc.collect()
                continue

            m = sequence_metrics(y_te, y_pred, state_names)
            m.update(duration_metrics(true_s,
                                      standby_seconds(y_pred, standby_id, step_s)))
            m.update({k: v for k, v in extra.items()
                      if isinstance(v, (int, float))})
            m["seed"] = seed
            m["seconds"] = round(time.time() - t0, 1)
            per_seed.append(m)
            preds.append(y_pred)
            info(f"seed {seed}: macroF1 {m['macro_f1']:.4f}  "
                 f"STANDBY-F1 {m.get('f1_STANDBY', float('nan')):.4f}  "
                 f"MAE {m['standby_mae_s']:.2f}s  ({m['seconds']:.0f}s)")

            # Checkpoint here, not after the seed loop: this is the line that
            # decides how much a kill costs, and on this hardware it costs one
            # seed.
            _checkpoint(key, name, citation, per_seed, preds)
            del y_pred
            gc.collect()

        _checkpoint(key, name, citation, per_seed, preds)
        gc.collect()

    _persist()
    section("TASK 12A COMPLETE")
    return results


def _dump_seed_table(results: dict, out_dir: str) -> pd.DataFrame:
    rows = []
    for key, r in results.items():
        a = r.get("aggregate", {})
        row = {"key": key, "model": r["model"], "n_seeds": r["n_seeds"]}
        for metric, d in a.items():
            row[f"{metric}_mean"] = d.get("mean")
            row[f"{metric}_std"] = d.get("std")
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("key")
    tbl.to_csv(f"{out_dir}/seed_table.csv")
    return tbl


# ══════════════════════════════════════════════════════════════════════════════
# B -- labelling seeds
# ══════════════════════════════════════════════════════════════════════════════

def labelling_seeds(machine: str = DEFAULT_MACHINE, n_seeds: int = LABEL_SEEDS,
                    k: int = 4) -> dict:
    """
    Re-fit the state model at `n_seeds` seeds on ONE preprocessed record.

    Preprocessing -- loading, the spike mask, the OFF denoise and the feature
    matrix -- is deterministic, so it is done once and shared.  What the seed
    changes is precisely the two stochastic steps: the GMM's EM initialisation
    (k-means init at that seed) and which contiguous blocks the transition
    matrix is estimated from.  Sharing the deterministic part is not just a
    saving; it makes the reported spread attributable to those two steps
    instead of to an unknown mixture of everything.

    Reports per seed: STANDBY hours over the whole record, the standby cluster's
    mean power, the physics score, flicker after the dwell constraint, and the
    adjusted Rand index against seed 0's partition.
    """
    from src import labelling as L
    from sklearn.metrics import adjusted_rand_score

    section(f"TASK 12B -- labelling seeds on {machine}")
    out_dir = ensure_dir(f"{TASK12_DIR}/{machine}/labelling")
    cfg = MACHINES[machine]

    df = L.load_raw(cfg["raw"])
    df, dt = L.add_segments(df)
    df["is_spike"] = L.compute_spike_mask(df)
    df = L.denoise_off_state(df)
    X, _, cols = L.build_feature_matrix(df)
    bounds = L.segment_bounds(df["segment_id"].to_numpy())
    power = df["active_power"].to_numpy(np.float64)
    min_rows = max(1, int(round(L.MIN_DWELL_S / dt)))

    # One strided sample for the physics checks, shared by every seed so the
    # comparison across seeds is on identical rows.
    stride = max(1, len(df) // 2_000_000)
    idx = np.arange(0, len(df), stride)
    sub = df.iloc[idx][[c for c in ("timestamp", "active_power", "current",
                                    "power_factor") if c in df.columns]
                       ].reset_index(drop=True)

    seg = df["segment_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(df)]
    del df
    gc.collect()

    runs, ref_states = [], None
    for seed in range(n_seeds):
        t0 = time.time()
        model, gmm = fit_quiet(L, X, bounds, dt, k, seed)
        raw, smooth = L.decode(model, X, bounds, min_rows, verbose=False)
        mapping = L.map_labels_to_states(smooth, power, k=k)
        states = np.array([mapping[int(c)] for c in smooth], dtype=object)

        phys = physics_compliance(sub, np.arange(len(sub)), states[idx])
        fl = flicker_rate([states[s:e] for s, e in zip(starts, ends) if e - s > 1],
                          dt, FLICKER_TAU_S)
        sby = states == "STANDBY"
        row = {
            "seed": seed,
            "standby_hours": float(sby.sum() * dt / 3600.0),
            "standby_power_w": float(power[sby].mean()) if sby.any() else np.nan,
            "off_hours": float((states == "OFF").sum() * dt / 3600.0),
            "physics_score": phys["score"],
            "pf_separation": phys["values"].get("pf_separation", np.nan),
            "current_ratio": phys["values"].get("current_ratio", np.nan),
            "flicker_pct": fl["flicker_rate_pct"],
            "median_dwell_s": fl["median_dwell_s"],
            "rows_changed_by_min_dwell_pct": float((raw != smooth).mean() * 100),
            "seconds": round(time.time() - t0, 1),
        }
        if ref_states is None:
            ref_states = states[idx]
            row["ari_vs_seed0"] = 1.0
        else:
            row["ari_vs_seed0"] = float(adjusted_rand_score(ref_states, states[idx]))
        runs.append(row)
        info(f"seed {seed}: STANDBY {row['standby_hours']:.1f} h @ "
             f"{row['standby_power_w']:.0f} W, physics {row['physics_score']:.2f}, "
             f"flicker {row['flicker_pct']:.2f}%, ARI vs seed0 "
             f"{row['ari_vs_seed0']:.3f} ({row['seconds']:.0f}s)")
        del states, smooth, raw
        gc.collect()

    tab = pd.DataFrame(runs)
    tab.to_csv(f"{out_dir}/labelling_seeds.csv", index=False)
    summary = {c: describe_runs(tab[c]) for c in
               ("standby_hours", "standby_power_w", "physics_score",
                "pf_separation", "current_ratio", "flicker_pct",
                "ari_vs_seed0")}
    save_json({"machine": machine, "n_seeds": n_seeds, "k": k,
               "sample_interval_s": dt, "channels": cols,
               "runs": runs, "summary": summary},
              f"{out_dir}/labelling_seeds.json")
    print(tab.round(3).to_string(index=False))
    section("TASK 12B COMPLETE")
    return {"runs": runs, "summary": summary}


def fit_quiet(L, X, bounds, dt, k, seed):
    """`fit_state_model` without its per-seed logging."""
    return L.fit_state_model(X, bounds, dt, k=k, seed=seed, verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
# C -- decision layer seeds
# ══════════════════════════════════════════════════════════════════════════════

def decision_seeds(machine: str = DEFAULT_MACHINE, n_seeds: int = DECISION_SEEDS,
                   source: str = "phase4", scenario: str = "S2_moderate") -> dict:
    """
    Re-run Phase III's policy comparison at `n_seeds` seeds.

    The seed moves two things: the gradient-boosted forecaster's fit (subsample
    and column-subsample draws) and the restart-signature bootstrap.  It does
    NOT move the episodes, the split or the economics, so the spread reported
    here is model-fitting variance alone -- the sampling variance of the
    episodes themselves is a different question and is answered by the
    day-block bootstrap in Task 11.
    """
    from experiments.task6_optimization import (
        prepare, build_policies, BREAK_EVEN_SCENARIOS, DECISION_EPOCH_S,
    )
    from src.decision import compare_policies, params_for_break_even, annualise

    section(f"TASK 12C -- decision layer seeds on {machine}")
    out_dir = ensure_dir(f"{TASK12_DIR}/{machine}/decision")
    be = BREAK_EVEN_SCENARIOS[scenario]

    runs = []
    for seed in range(n_seeds):
        t0 = time.time()
        prep = prepare(machine, seed=seed, source=source)
        params = params_for_break_even(prep["base_params"], be)
        pol = build_policies(prep["ep_test"], params, prep["predictors"],
                             prep["feasible"], DECISION_EPOCH_S)
        tab = compare_policies(prep["ep_test"], pol, params)
        span = prep["test_span_s"]
        row = {
            "seed": seed,
            "forecast_mae_s": float(np.mean(np.abs(
                np.clip(prep["forecaster_models"]["mean"].predict(prep["Xte"]),
                        0, None) - prep["yte"]))),
            "proposed_usd_yr": float(annualise(
                tab.loc["forecast_opt", "savings_vs_baseline"], span)),
            "proposed_pct_oracle": float(tab.loc["forecast_opt", "pct_of_oracle"]),
            "ski_rental_usd_yr": float(annualise(
                tab.loc["ski_rental", "savings_vs_baseline"], span)),
            "head_room_usd_yr": float(annualise(
                tab.loc["observed", "cost_total"] - tab.loc["oracle", "cost_total"],
                span)),
            "n_shutdowns": int(tab.loc["forecast_opt", "n_shutdowns"]),
            "min_off_violations": int(tab.loc["forecast_opt", "min_off_violations"]),
            "restart_delay_s": float(prep["restart_signature"]["restart_delay_s"]),
            "seconds": round(time.time() - t0, 1),
        }
        runs.append(row)
        info(f"seed {seed}: proposed {row['proposed_usd_yr']:+.2f} USD/yr "
             f"({row['proposed_pct_oracle']:.1f}% of oracle), "
             f"ski-rental {row['ski_rental_usd_yr']:+.2f}, "
             f"MAE {row['forecast_mae_s']:.0f}s ({row['seconds']:.0f}s)")
        del prep
        gc.collect()

    tab = pd.DataFrame(runs)
    tab.to_csv(f"{out_dir}/decision_seeds.csv", index=False)
    summary = {c: describe_runs(tab[c]) for c in
               ("proposed_usd_yr", "proposed_pct_oracle", "ski_rental_usd_yr",
                "head_room_usd_yr", "forecast_mae_s", "n_shutdowns",
                "min_off_violations")}
    save_json({"machine": machine, "scenario": scenario, "n_seeds": n_seeds,
               "runs": runs, "summary": summary},
              f"{out_dir}/decision_seeds.json")
    print(tab.round(3).to_string(index=False))
    section("TASK 12C COMPLETE")
    return {"runs": runs, "summary": summary}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machine: str = DEFAULT_MACHINE,
         steps: tuple[str, ...] = ("forecasting", "labelling", "decision"),
         neural_seeds: int | None = None,
         models: list[str] | None = None,
         label_seeds: int = LABEL_SEEDS,
         decision_seed_count: int = DECISION_SEEDS,
         source: str = "auto") -> dict:
    ensure_dir(TASK12_DIR)
    section(f"PHASE V / TASK 12 -- repeated runs on {machine}")
    out = {}

    if "forecasting" in steps:
        plan = dict(SEED_PLAN)
        if neural_seeds:
            for k in ("seq2seq_lstm", "gru", "vanilla_lstm", "tcn", "transformer"):
                plan[k] = neural_seeds
        out["forecasting"] = forecasting_seeds(machine, models=models,
                                               seed_plan=plan, source=source)
    if "labelling" in steps:
        out["labelling"] = labelling_seeds(machine, n_seeds=label_seeds)
    if "decision" in steps:
        out["decision"] = decision_seeds(machine, n_seeds=decision_seed_count)

    section("TASK 12 COMPLETE")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase V Task 12 -- repeated runs")
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--only", nargs="*", default=None,
                    help="steps: forecasting labelling decision")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--neural-seeds", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of the forecasting models to (re-)run")
    ap.add_argument("--label-seeds", type=int, default=LABEL_SEEDS)
    ap.add_argument("--decision-seeds", type=int, default=DECISION_SEEDS)
    ap.add_argument("--source", default="auto",
                    choices=["auto", "reference", "phase4"])
    a = ap.parse_args()
    all_steps = ("forecasting", "labelling", "decision")
    steps = tuple(s for s in (a.only or all_steps) if s not in a.skip)
    main(a.machine, steps, a.neural_seeds, a.models, a.label_seeds,
         a.decision_seeds, a.source)
