"""
============================================================
TASK 9 -- CROSS-MACHINE GENERALISATION
experiments/task9_cross_machine.py
============================================================

THE QUESTION
------------
Every model in Phases II and III was fitted and tested on the same
machine.  A plant that has instrumented one machine will not refit
per asset, so the question that decides whether the framework is
deployable is: does a model trained on machine A do anything useful
on machine B, which it has never seen?

TWO TRANSFERS, BECAUSE THERE ARE TWO FORECASTERS
------------------------------------------------
Part A  the DECISION forecaster (Phase III): remaining idle time at
        every 60 s decision epoch, and the shutdown policy it drives.
        This is the model the decision layer actually consumes, so
        its transfer is measured in currency, not only in MAE.

Part B  the SEQUENCE forecaster (Phase II, Task 5): the proposed
        Seq2Seq LSTM predicting the state at every step of the next
        300 s, evaluated by per-step F1 and by the STANDBY-seconds
        duration error.

WHAT COUNTS AS SUCCESS
----------------------
Not "the transferred model is worse than the local one" -- it will
be.  The deployment-relevant bar is the FORECAST-FREE alternative
already available on the target machine:

  Part A   the ski-rental rule, which is 2-competitive using only a
           clock (Karlin et al. 1994).  A transferred forecaster
           that cannot beat it adds nothing to a new machine.
  Part B   a persistence baseline (the horizon continues the last
           observed state), which needs no training at all.

Both are computed on every target machine and reported alongside.

THE SCALE PROBLEM, AND WHY IT IS AN EXPERIMENT RATHER THAN A FIX
----------------------------------------------------------------
Two of the transfers are between machines whose rated power differs
by orders of magnitude (an exhaust fan against a pelletizer), and
both forecasters take absolute electrical quantities as inputs.  A
model trained on one will read a normal load on the other as an
extreme value.  Each part therefore reports two variants:

  raw     the model applied exactly as fitted -- true zero-shot;
  scaled  inputs rescaled by a constant taken from the TARGET
          machine's own unlabelled history (Part A: its median
          productive power; Part B: its own feature means and
          standard deviations).

The scaled variant requires no labels on the target machine, so it
is available in deployment; the pair measures how much of the
transfer loss is a units mismatch rather than a behavioural
difference.

USAGE
-----
    python -m experiments.task9_cross_machine                  # both parts
    python -m experiments.task9_cross_machine --parts a
    python -m experiments.task9_cross_machine --machines pelletizer-I pelletizer-II
============================================================
"""

from __future__ import annotations

import os
import sys
import gc
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from experiments.common import (                    # noqa: E402
    MACHINES, IMDELD_MACHINES, PHASE4_DIR, STATE_ORDER,
    load_labelled, degenerate_reason, ensure_dir, save_json, info, section,
)
from src.decision import (                          # noqa: E402
    compare_policies, params_for_break_even, annualise,
    policy_observed, policy_elapsed_time, policy_forecast_optimising,
    optimal_offline, SECONDS_PER_HOUR,
)

TASK9_DIR = f"{PHASE4_DIR}/task9"

# Fixed state encoding shared by every machine.  Task 5 built its encoder from
# the states present in one record; across machines that would silently permute
# the class ids and make a transferred classifier's output meaningless.
STATE_ENCODER = {s: i for i, s in enumerate(STATE_ORDER)}

# Part B protocol -- identical to Task 5's so the diagonal of the transfer
# matrix is directly comparable with the Phase II result.
DECIMATE = 10
LOOKBACK_S = 600.0
HORIZON_S = 300.0
STRIDE_S = 120.0
TRAIN_PCT, VAL_PCT = 0.70, 0.15
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5
SEQ_MODEL = "seq2seq_lstm"


# ══════════════════════════════════════════════════════════════════════════════
# PART A -- decision forecaster
# ══════════════════════════════════════════════════════════════════════════════

# Index of `prev_productive_power_w` in task6.FEATURE_NAMES: the only feature
# carried in absolute Watts, and therefore the only one the scale correction
# touches.  Asserted at run time rather than hard-coded blindly.
def _power_feature_index() -> int:
    from experiments.task6_optimization import FEATURE_NAMES
    j = FEATURE_NAMES.index("prev_productive_power_w")
    assert FEATURE_NAMES[j] == "prev_productive_power_w"
    return j


def machine_power_scale(ep_train: pd.DataFrame) -> float:
    """
    The constant used to put one machine's power features on a common scale:
    the median power of the productive runs that PRECEDE its idle episodes.

    Taken from the training episodes only, and computed from the same column
    the model consumes, so it is available on a target machine from unlabelled
    history alone -- no state labels, no rating plate, nothing the plant would
    have to supply.
    """
    v = ep_train["prev_productive_power_w"].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else 1.0


def fit_scaled_models(bundle: dict, scale: float, seed: int = 0):
    """Refit the mean forecaster and the feasibility classifier on scaled inputs."""
    from experiments.task6_optimization import fit_forecaster, fit_feasibility
    from experiments.task6_optimization import MIN_OFF_S

    j = _power_feature_index()
    Xtr = bundle["Xtr"].copy()
    Xtr[:, j] /= max(scale, 1e-9)
    mdl, spec = fit_forecaster(Xtr, bundle["ytr"], seed=seed, mode="mean")
    clf = fit_feasibility(Xtr, bundle["ytr"], MIN_OFF_S, seed=seed)
    return mdl, spec, clf


def make_scaled_predictor(model, close_hour: int, open_hour: int, scale: float):
    """Predictor closure that rescales the one absolute-power feature."""
    from experiments.task6_optimization import _feature_matrix
    j = _power_feature_index()

    def predict_remaining_s(ep_rows: pd.DataFrame, elapsed_s: float) -> np.ndarray:
        X = _feature_matrix(ep_rows, elapsed_s, close_hour, open_hour)
        X[:, j] /= max(scale, 1e-9)
        return np.clip(model.predict(X), 0.0, None)
    return predict_remaining_s


def make_scaled_feasibility(model, close_hour: int, open_hour: int, scale: float):
    if model is None:
        return None
    from experiments.task6_optimization import _feature_matrix
    j = _power_feature_index()

    def predict_feasible(ep_rows: pd.DataFrame, elapsed_s: float) -> np.ndarray:
        X = _feature_matrix(ep_rows, elapsed_s, close_hour, open_hour)
        X[:, j] /= max(scale, 1e-9)
        return model.predict_proba(X)[:, 1]
    return predict_feasible


def scaled_X(X: np.ndarray, scale: float) -> np.ndarray:
    j = _power_feature_index()
    Y = X.copy()
    Y[:, j] /= max(scale, 1e-9)
    return Y


def part_a(machines: list[str], source: str = "phase4", seed: int = 0,
           scenario: str = "S2_moderate") -> dict:
    """
    Transfer matrix for the decision forecaster, in forecast error and in
    currency.

    The economics are always the TARGET machine's: its measured standby power,
    its measured restart delay, its own break-even scenario.  Only the fitted
    forecaster travels.  Mixing the source machine's economics in would make
    the currency column unreadable, since it is bounded by the target's
    head-room whatever model is used.
    """
    from experiments.task6_optimization import (
        prepare, BREAK_EVEN_SCENARIOS, DECISION_EPOCH_S, MAX_DECISION_HOURS,
        CHANCE_CONFIDENCE, make_predictor, make_feasibility,
    )

    section("TASK 9 / PART A -- decision-forecaster transfer")
    out_dir = ensure_dir(f"{TASK9_DIR}/part_a")

    bundles: dict[str, dict] = {}
    for m in machines:
        info(f"preparing {m}")
        b = prepare(m, seed=seed, source=source)
        b["scale"] = machine_power_scale(b["ep_train"])
        b["scaled"] = fit_scaled_models(b, b["scale"], seed=seed)
        # Only the small objects are kept; the epoch tables are what dominate.
        info(f"{m}: power scale {b['scale']:.0f} W, "
             f"{len(b['ep_train']):,} train / {len(b['ep_test']):,} test episodes, "
             f"restart delay {b['restart_signature'].get('restart_delay_s', float('nan')):.0f}s "
             f"(n_cold={b['restart_signature'].get('n_cold')})")
        bundles[m] = b
        gc.collect()

    be_s = BREAK_EVEN_SCENARIOS[scenario]
    steps = int(MAX_DECISION_HOURS * SECONDS_PER_HOUR / DECISION_EPOCH_S)
    rows: list[dict] = []

    for tgt in machines:
        B = bundles[tgt]
        cfg = MACHINES[tgt]
        ch, oh = cfg["close_hour"], cfg["open_hour"]
        ep_te, yte = B["ep_test"], B["yte"]
        Xte_raw = B["Xte"]
        Xte_scaled = scaled_X(Xte_raw, B["scale"])

        # Economics of the TARGET machine.  Where the decision problem is
        # degenerate -- no energised-idle band, or no observed cold restart --
        # the currency columns stay NaN and only the forecast columns are
        # reported; a machine that cannot be saved on is not evidence about
        # whether a forecaster transferred.
        params = None
        degen = degenerate_reason(B["standby_power_w"], B["restart_signature"])
        if degen:
            info(f"{tgt}: no decision problem -- {degen}")
        else:
            params = params_for_break_even(B["base_params"], be_s)
            base_policies = {
                "observed": policy_observed(ep_te, params),
                "ski_rental": policy_elapsed_time(ep_te, params),
                "oracle": optimal_offline(ep_te, params),
            }
            test_span_s = B["test_span_s"]

        for src in machines:
            A = bundles[src]
            for variant in ("raw", "scaled"):
                if variant == "raw":
                    model = A["forecaster_models"]["mean"]
                    clf = A["feasibility_model"]
                    Xeval = Xte_raw
                    predict = make_predictor(model, A["forecaster_specs"]["mean"], ch, oh)
                    feasible = make_feasibility(clf, ch, oh)
                else:
                    model, spec, clf = A["scaled"]
                    Xeval = Xte_scaled
                    predict = make_scaled_predictor(model, ch, oh, B["scale"])
                    feasible = make_scaled_feasibility(clf, ch, oh, B["scale"])

                p = np.clip(model.predict(Xeval), 0, None)
                row = {
                    "source": src, "target": tgt, "variant": variant,
                    "same_machine": src == tgt,
                    "same_type": _same_type(src, tgt),
                    "mae_s": float(np.mean(np.abs(p - yte))),
                    "spearman": float(pd.Series(p).corr(pd.Series(yte),
                                                        method="spearman")),
                    "mean_ratio": float(p.mean() / max(yte.mean(), 1e-9)),
                    "n_test_epochs": int(len(yte)),
                }

                if params is not None:
                    pol = dict(base_policies)
                    pol["transferred"] = policy_forecast_optimising(
                        ep_te, params, predict, epoch_s=DECISION_EPOCH_S,
                        max_epochs=steps, predict_feasible=feasible,
                        confidence=CHANCE_CONFIDENCE)
                    tab = compare_policies(ep_te, pol, params,
                                           baseline="observed", oracle="oracle")
                    row.update({
                        "savings_usd_yr": float(annualise(
                            tab.loc["transferred", "savings_vs_baseline"], test_span_s)),
                        "pct_of_oracle": float(tab.loc["transferred", "pct_of_oracle"]),
                        "ski_rental_usd_yr": float(annualise(
                            tab.loc["ski_rental", "savings_vs_baseline"], test_span_s)),
                        "head_room_usd_yr": float(annualise(
                            tab.loc["observed", "cost_total"]
                            - tab.loc["oracle", "cost_total"], test_span_s)),
                        "n_shutdowns": int(tab.loc["transferred", "n_shutdowns"]),
                        "min_off_violations": int(
                            tab.loc["transferred", "min_off_violations"]),
                        "beats_ski_rental": bool(
                            tab.loc["transferred", "savings_vs_baseline"]
                            > tab.loc["ski_rental", "savings_vs_baseline"]),
                    })
                rows.append(row)
                info(f"  {src:>14} -> {tgt:<14} [{variant:>6}] "
                     f"MAE {row['mae_s']:9.0f}s  rho {row['spearman']:+.3f}"
                     + (f"  {row['savings_usd_yr']:+8.0f} USD/yr "
                        f"({row['pct_of_oracle']:6.1f}% of oracle)"
                        if params is not None else "  [no scenario]"))
        del Xte_scaled
        gc.collect()

    tab = pd.DataFrame(rows)
    tab.to_csv(f"{out_dir}/transfer_matrix.csv", index=False)

    summary = _summarise_part_a(tab)
    save_json({"scenario": scenario, "break_even_s": be_s, "source": source,
               "machines": machines, "summary": summary,
               "power_scales": {m: bundles[m]["scale"] for m in machines}},
              f"{out_dir}/part_a_summary.json")
    _print_matrix(tab, "mae_s", "raw", "Part A -- forecast MAE (s), raw transfer")
    _print_matrix(tab, "mae_s", "scaled", "Part A -- forecast MAE (s), scaled transfer")
    if "savings_usd_yr" in tab.columns:
        _print_matrix(tab, "savings_usd_yr", "scaled",
                      "Part A -- annual saving vs status quo (USD/yr), scaled transfer")
    return {"table": tab, "summary": summary}


def _same_type(a: str, b: str) -> bool:
    """Same machine type (the two pelletizers, the two fans, ...)."""
    def fam(k):
        return k.rsplit("-", 1)[0]
    return fam(a) == fam(b)


def _summarise_part_a(tab: pd.DataFrame) -> dict:
    """
    Reduce the matrix to the three comparisons the paper makes: local versus
    transferred, same-type versus cross-type, and raw versus scaled inputs.
    """
    out: dict = {}
    for variant, sub in tab.groupby("variant"):
        diag = sub[sub["same_machine"]]
        same = sub[(~sub["same_machine"]) & (sub["same_type"])]
        cross = sub[(~sub["same_machine"]) & (~sub["same_type"])]
        block = {}
        for name, part in (("within", diag), ("same_type", same), ("cross_type", cross)):
            if not len(part):
                continue
            block[name] = {
                "n": int(len(part)),
                "mae_s_median": float(part["mae_s"].median()),
                "spearman_median": float(part["spearman"].median()),
            }
            if "savings_usd_yr" in part.columns:
                ok = part["savings_usd_yr"].notna()
                if ok.any():
                    block[name].update({
                        "savings_usd_yr_median": float(part.loc[ok, "savings_usd_yr"].median()),
                        "pct_of_oracle_median": float(part.loc[ok, "pct_of_oracle"].median()),
                        "beats_ski_rental_pct": float(
                            part.loc[ok, "beats_ski_rental"].mean() * 100.0),
                    })
        out[variant] = block
    return out


def _print_matrix(tab: pd.DataFrame, value: str, variant: str, title: str) -> None:
    sub = tab[tab["variant"] == variant]
    if value not in sub.columns or sub[value].isna().all():
        return
    piv = sub.pivot(index="source", columns="target", values=value)
    section(title)
    print(piv.round(1).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PART B -- sequence forecaster
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(machine: str, source: str = "phase4") -> dict:
    """
    Window one machine exactly as Task 5 did, with one deliberate change: the
    state encoder is fixed across machines (see STATE_ENCODER).

    Returns the train/val windows (for fitting), the raw -- unstandardised --
    test windows (so any machine's scaler can be applied to them later), and
    the training-split feature statistics.
    """
    from experiments.task5_forecasting_baselines import (
        decimate, make_windows, TRAIN_PCT as T5_TRAIN, VAL_PCT as T5_VAL,
    )
    from src.features import all_feature_names

    df, meta = load_labelled(machine, drop_spikes=True, add_features=True,
                             source=source)
    feature_cols = [f for f in all_feature_names() if f in df.columns]
    df["state_id"] = df["state"].map(STATE_ENCODER).astype(np.int16)
    if df["state_id"].isna().any():
        raise ValueError(f"{machine}: unmapped state label")

    df = decimate(df, DECIMATE)
    step_s = meta["sample_interval_s"] * DECIMATE
    lb = max(2, int(round(LOOKBACK_S / step_s)))
    hz = max(1, int(round(HORIZON_S / step_s)))
    st = max(1, int(round(STRIDE_S / step_s)))

    n = len(df)
    c1, c2 = int(n * T5_TRAIN), int(n * (T5_TRAIN + T5_VAL))
    parts = {}
    for name, part in (("train", df.iloc[:c1]), ("val", df.iloc[c1:c2]),
                       ("test", df.iloc[c2:])):
        parts[name] = make_windows(part.reset_index(drop=True), feature_cols,
                                   lb, hz, st)
    del df
    gc.collect()

    Xtr = parts["train"][0]
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0)
    sd[sd < 1e-8] = 1.0

    info(f"{machine}: {len(parts['train'][0]):,} train / {len(parts['val'][0]):,} val / "
         f"{len(parts['test'][0]):,} test windows, step {step_s:.0f}s, "
         f"{len(feature_cols)} features")
    return {"machine": machine, "parts": parts, "mu": mu, "sd": sd,
            "step_s": step_s, "feature_cols": feature_cols,
            "lookback_rows": lb, "horizon_rows": hz, "stride_rows": st}


def _standardise(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def part_b(machines: list[str], source: str = "phase4", seed: int = 0,
           epochs: int = EPOCHS) -> dict:
    """
    Train the proposed Seq2Seq LSTM once per machine and evaluate every model
    on every machine's held-out windows.

    Two scaler variants (see the module docstring) and a persistence reference
    per target.  Models are written to disk and the target windows are cached,
    so the 8x8 matrix costs eight trainings rather than sixty-four.
    """
    import tensorflow as tf
    from experiments.baseline_models import NEURAL_BUILDERS
    from experiments.task5_forecasting_baselines import (
        sequence_metrics, duration_metrics, standby_seconds,
    )
    from sklearn.utils.class_weight import compute_class_weight

    section("TASK 9 / PART B -- sequence-forecaster transfer")
    out_dir = ensure_dir(f"{TASK9_DIR}/part_b")
    cache_dir = ensure_dir(f"{out_dir}/cache")
    standby_id = STATE_ENCODER["STANDBY"]
    state_names = list(STATE_ORDER)

    # -- one pass per machine: window, cache the test split, train, free ------
    meta: dict[str, dict] = {}
    for m in machines:
        model_path = f"{out_dir}/model_{m}.keras"
        cache_path = f"{cache_dir}/{m}_test.npz"
        if os.path.exists(model_path) and os.path.exists(cache_path):
            info(f"{m}: model and cache present, skipping training")
            z = np.load(cache_path, allow_pickle=False)
            meta[m] = {"mu": z["mu"], "sd": z["sd"], "step_s": float(z["step_s"]),
                       "n_test": int(z["y"].shape[0])}
            continue

        section(f"windowing and training on {m}")
        W = build_windows(m, source=source)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = (W["parts"]["train"],
                                              W["parts"]["val"], W["parts"]["test"])
        np.savez_compressed(cache_path, X=Xte, y=yte,
                            mu=W["mu"], sd=W["sd"], step_s=W["step_s"])
        info(f"cached {cache_path} ({os.path.getsize(cache_path)/1e6:.0f} MB)")

        flat = ytr.ravel()
        present = np.unique(flat)
        cw = np.ones(len(state_names), dtype=np.float32)
        cw[present] = compute_class_weight("balanced", classes=present, y=flat)
        w_tr, w_va = cw[ytr].astype(np.float32), cw[yva].astype(np.float32)

        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.clear_session()
        model = NEURAL_BUILDERS[SEQ_MODEL](
            n_features=Xtr.shape[2], lookback=Xtr.shape[1],
            horizon=ytr.shape[1], n_states=len(state_names))
        cbs = [tf.keras.callbacks.EarlyStopping(
                   monitor="val_loss", patience=PATIENCE,
                   restore_best_weights=True, verbose=1),
               tf.keras.callbacks.ReduceLROnPlateau(
                   monitor="val_loss", factor=0.5,
                   patience=max(2, PATIENCE // 2), verbose=0)]
        t0 = time.time()
        hist = model.fit(_standardise(Xtr, W["mu"], W["sd"]), ytr, sample_weight=w_tr,
                         validation_data=(_standardise(Xva, W["mu"], W["sd"]), yva, w_va),
                         epochs=epochs, batch_size=BATCH_SIZE, callbacks=cbs,
                         verbose=2, shuffle=True)
        model.save(model_path)
        pd.DataFrame(hist.history).to_csv(f"{out_dir}/history_{m}.csv", index=False)
        meta[m] = {"mu": W["mu"], "sd": W["sd"], "step_s": W["step_s"],
                   "n_test": int(len(yte)),
                   "train_seconds": round(time.time() - t0, 1),
                   "epochs_run": len(hist.history["loss"]),
                   "class_weights": cw.tolist()}
        info(f"{m}: trained in {meta[m]['train_seconds']/60:.1f} min "
             f"over {meta[m]['epochs_run']} epochs")
        del Xtr, ytr, Xva, yva, Xte, yte, W, model
        tf.keras.backend.clear_session()
        gc.collect()

    # -- evaluation: every model on every cached target ----------------------
    rows: list[dict] = []
    for tgt in machines:
        z = np.load(f"{cache_dir}/{tgt}_test.npz", allow_pickle=False)
        X_raw, y_te = z["X"], z["y"]
        step_s = float(z["step_s"])
        true_s = standby_seconds(y_te, standby_id, step_s)

        # Forecast-free reference: repeat the last observed state.  The last
        # lookback step is not in y, so it is reconstructed as the state of the
        # step immediately before the horizon -- available to any controller.
        last_state = _last_lookback_state(X_raw, tgt)
        y_pers = np.repeat(last_state[:, None], y_te.shape[1], axis=1).astype(np.int16)
        pers = sequence_metrics(y_te, y_pers, state_names)
        pers.update(duration_metrics(true_s,
                                     standby_seconds(y_pers, standby_id, step_s)))
        rows.append({"source": "persistence", "target": tgt, "variant": "none",
                     "same_machine": False, "same_type": False, **pers})
        info(f"  persistence -> {tgt:<14} macroF1 {pers['macro_f1']:.3f}  "
             f"STANDBY-F1 {pers.get('f1_STANDBY', float('nan')):.3f}  "
             f"MAE {pers['standby_mae_s']:.1f}s")

        for src in machines:
            import tensorflow as tf
            model = tf.keras.models.load_model(f"{out_dir}/model_{src}.keras")
            for variant, (mu, sd) in (("source_scaler", (meta[src]["mu"], meta[src]["sd"])),
                                      ("target_scaler", (z["mu"], z["sd"]))):
                y_pred = model.predict(_standardise(X_raw, mu, sd),
                                       batch_size=256, verbose=0).argmax(-1).astype(np.int16)
                mset = sequence_metrics(y_te, y_pred, state_names)
                mset.update(duration_metrics(
                    true_s, standby_seconds(y_pred, standby_id, step_s)))
                rows.append({"source": src, "target": tgt, "variant": variant,
                             "same_machine": src == tgt,
                             "same_type": _same_type(src, tgt), **mset})
                info(f"  {src:>14} -> {tgt:<14} [{variant:>13}] "
                     f"macroF1 {mset['macro_f1']:.3f}  "
                     f"STANDBY-F1 {mset.get('f1_STANDBY', float('nan')):.3f}  "
                     f"MAE {mset['standby_mae_s']:.1f}s")
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        del X_raw, y_te
        gc.collect()

    tab = pd.DataFrame(rows)
    tab.to_csv(f"{out_dir}/transfer_matrix.csv", index=False)
    summary = _summarise_part_b(tab)
    save_json({"config": {"decimate": DECIMATE, "lookback_s": LOOKBACK_S,
                          "horizon_s": HORIZON_S, "stride_s": STRIDE_S,
                          "epochs": epochs, "model": SEQ_MODEL,
                          "state_encoder": STATE_ENCODER},
               "per_machine": {m: {k: v for k, v in d.items()
                                   if k not in ("mu", "sd")} for m, d in meta.items()},
               "summary": summary},
              f"{out_dir}/part_b_summary.json")
    _print_matrix(tab, "f1_STANDBY", "target_scaler",
                  "Part B -- STANDBY F1, target-scaler transfer")
    _print_matrix(tab, "standby_mae_s", "target_scaler",
                  "Part B -- STANDBY-seconds MAE, target-scaler transfer")
    return {"table": tab, "summary": summary}


def _last_lookback_state(X_raw: np.ndarray, tgt: str) -> np.ndarray:
    """
    State at the final lookback step, recovered from the raw feature window.

    The windows carry features, not labels, so the persistence reference needs
    the state at the last observed step.  `active_power` is feature 0 and the
    state boundaries are the fitted cluster boundaries, which are not carried
    here -- so the state is recovered by the same power ranking the labels use:
    the nearest of the target machine's own per-state mean powers.  This is a
    reference baseline, not a proposed method, and it is deliberately generous:
    it is given the target machine's state means.
    """
    import json
    path = f"{PHASE4_DIR}/task8/{tgt}/characterisation.json"
    p_last = X_raw[:, -1, 0].astype(float)          # active_power, last step
    if not os.path.exists(path):
        # Fall back to quartiles of the observed power distribution.
        edges = np.quantile(p_last, [0.25, 0.5, 0.75])
        return np.digitize(p_last, edges).astype(np.int16)
    with open(path, encoding="utf-8") as fh:
        per_state = json.load(fh)["per_state"]
    means = np.array([per_state.get(s, {}).get("mean_active_power_w", np.inf)
                      for s in STATE_ORDER], dtype=float)
    finite = np.isfinite(means)
    idx = np.argmin(np.abs(p_last[:, None] - means[None, finite]), axis=1)
    return np.flatnonzero(finite)[idx].astype(np.int16)


def _summarise_part_b(tab: pd.DataFrame) -> dict:
    out: dict = {}
    for variant, sub in tab[tab["source"] != "persistence"].groupby("variant"):
        block = {}
        for name, part in (("within", sub[sub["same_machine"]]),
                           ("same_type", sub[(~sub["same_machine"]) & sub["same_type"]]),
                           ("cross_type", sub[(~sub["same_machine"]) & ~sub["same_type"]])):
            if not len(part):
                continue
            block[name] = {
                "n": int(len(part)),
                "macro_f1_median": float(part["macro_f1"].median()),
                "f1_standby_median": float(part["f1_STANDBY"].median()),
                "standby_mae_s_median": float(part["standby_mae_s"].median()),
            }
        out[variant] = block
    pers = tab[tab["source"] == "persistence"]
    if len(pers):
        out["persistence"] = {
            "macro_f1_median": float(pers["macro_f1"].median()),
            "f1_standby_median": float(pers["f1_STANDBY"].median()),
            "standby_mae_s_median": float(pers["standby_mae_s"].median()),
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(machines: list[str] | None = None, parts: tuple[str, ...] = ("a", "b"),
         source: str = "phase4", seed: int = 0, epochs: int = EPOCHS) -> dict:
    machines = machines or IMDELD_MACHINES
    ensure_dir(TASK9_DIR)
    section(f"PHASE IV / TASK 9 -- cross-machine generalisation, "
            f"{len(machines)} machines")
    out = {}
    if "a" in parts:
        out["part_a"] = part_a(machines, source=source, seed=seed)
    if "b" in parts:
        out["part_b"] = part_b(machines, source=source, seed=seed, epochs=epochs)
    section("TASK 9 COMPLETE")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase IV Task 9 -- cross-machine transfer")
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--parts", nargs="*", default=["a", "b"], choices=["a", "b"])
    ap.add_argument("--source", default="phase4", choices=["phase4", "reference", "auto"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    main(machines=a.machines, parts=tuple(a.parts), source=a.source,
         seed=a.seed, epochs=a.epochs)
