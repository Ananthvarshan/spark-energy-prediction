"""
============================================================
TASK 10 -- CROSS-DATASET VALIDATION
experiments/task10_cross_dataset.py
============================================================

THE QUESTION
------------
Everything so far is IMDELD.  A framework validated on one dataset
from one Brazilian facility has not been shown to be a framework.
This task applies the pipeline unchanged to a second dataset -- a
full year of 5 s active-power readings from a different site and a
different acquisition system -- and reports what survives.

TWO RECORDS, AND THE SECOND ONE IS A CONTROL
--------------------------------------------
`spark-cnc`    a CNC machining centre.  A genuine driven industrial
               load, so the framework's claims should hold.

`spark-solar`  a photovoltaic inverter's AC output.  This is NOT a
               machine: it is a generator whose "off" is nightfall
               and whose "load" is irradiance, and no shutdown
               decision exists for it.  It is included precisely
               because the pipeline will happily label it anyway.
               A validation battery is only worth quoting if it can
               reject something, so the honest test of the ten
               physics tests is whether they fire on a record where
               the modelling assumption is false by construction.

WHAT CHANGES WHEN THE DATA ARE SINGLE-CHANNEL
---------------------------------------------
IMDELD supplies six electrical channels; these records supply one.
Two of the five physics checks -- power-factor separation (C3) and
the no-load current ratio (C4) -- are then undefined, and those are
exactly the two that do not depend on power magnitude, i.e. the two
that actually test whether STANDBY has been separated from a lightly
loaded WORKING state rather than merely from a lower-power one.  The
remaining three (power ordering, OFF near zero, variance ordering)
are satisfiable by any monotone partition of a power histogram.

This is the central finding of Task 10 and it is reported as a
limitation of the METHOD, not of the data: on a single-channel
record the framework can still produce states and still drive the
decision layer, but its physics validation has no discriminating
power, and the control record demonstrates that concretely.

USAGE
-----
    python -m experiments.task10_cross_dataset
    python -m experiments.task10_cross_dataset --datasets spark-cnc
    python -m experiments.task10_cross_dataset --skip label
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

from experiments.common import (                    # noqa: E402
    MACHINES, SPARK_DATASETS, PHASE4_DIR, FLICKER_TAU_S, STATE_ORDER,
    load_labelled, flicker_rate, physics_compliance, degenerate_reason,
    ensure_dir, save_json, info, section,
)
from src import labelling as L                      # noqa: E402
from src.decision import (                          # noqa: E402
    extract_idle_episodes, attach_local_calendar, measure_restart_signature,
    EconomicParams, params_for_break_even, compare_policies, annualise,
    policy_observed, policy_elapsed_time, policy_forecast_optimising,
    optimal_offline, SECONDS_PER_HOUR,
)

TASK10_DIR = f"{PHASE4_DIR}/task10"

# Same economic constants Phase III stated for IMDELD.  Not re-tuned: the point
# of the exercise is to run the pipeline unchanged.
TARIFF_USD_PER_KWH = 0.12
MIN_OFF_S = 600.0
MAX_RESTARTS_PER_DAY = 4
BREAK_EVEN_S = 3600.0            # the headline S2_moderate scenario


# ── Loading ───────────────────────────────────────────────────────────────────

def load_spark(key: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load one single-channel record into the frame `src.labelling` expects.

    Two properties of these files need stating because both affect the result.

    THE SERIES IS SIGNED.  The CNC record's median is -18 W: the meter carries
    a small negative offset, and the PV record is negative whenever the
    inverter draws its own housekeeping power.  `src.labelling.load_raw` clips
    active power at zero, as validate_gmm.py and proof_5methods.py do, so the
    same rule is applied here and the number of clipped rows is reported --
    on a machine that is off most of the year, that clip is what creates the
    OFF cluster's near-zero variance.

    THE GRID IS COMPLETE.  Unlike IMDELD, which has 31 acquisition gaps, these
    records have one row every 5 s for the whole of 2024 with no gap at all, so
    the whole year is one contiguous segment.  Missing readings appear as NaN
    rather than as absent rows, and those are dropped, which does introduce
    gaps -- reported below.
    """
    cfg = MACHINES[key]
    t0 = time.time()
    df = pd.read_csv(cfg["raw"])
    col = cfg["column"]
    df = df.rename(columns={"WsDateTime": "timestamp", col: "active_power"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df["active_power"] = pd.to_numeric(df["active_power"], errors="coerce")

    n_raw = len(df)
    n_nan = int(df["active_power"].isna().sum())
    n_neg = int((df["active_power"] < 0).sum())
    df = df.dropna(subset=["active_power"]).reset_index(drop=True)
    df["active_power"] = df["active_power"].clip(lower=0).astype(np.float32)
    df["power"] = df["active_power"]

    if verbose:
        info(f"{key}: {len(df):,} rows of {n_raw:,} in {time.time()-t0:.0f}s "
             f"({n_nan:,} NaN dropped, {n_neg:,} negative readings clipped to 0); "
             f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
    df.attrs.update({"n_raw": n_raw, "n_nan": n_nan, "n_negative_clipped": n_neg})
    return df


def label_dataset(key: str, force: bool = False, seed: int = 42) -> dict:
    """Label one SPARK record with the identical Phase IV procedure."""
    path = MACHINES[key]["labelled_phase4"]
    if os.path.exists(path) and not force:
        info(f"{key}: {path} exists, skipping")
        return {}
    df = load_spark(key)
    attrs = dict(df.attrs)
    df, meta = L.label_record(df, k=L.K_STATES, min_dwell_s=L.MIN_DWELL_S, seed=seed)
    meta.update({"machine_key": key, "raw_path": MACHINES[key]["raw"],
                 "n_rows": int(len(df)), **attrs})
    meta["labelled_path"] = L.save_labelled(df, path)
    save_json(meta, f"{TASK10_DIR}/{key}/label_info.json")
    del df
    gc.collect()
    return meta


# ── Characterisation ──────────────────────────────────────────────────────────

def characterise(key: str, seed: int = 42) -> dict:
    """
    The Task 8 read-outs that a single-channel record can support, plus the
    two that it cannot -- reported as unavailable rather than omitted.
    """
    cfg = MACHINES[key]
    section(f"TASK 10 -- characterising {key}")
    df, meta = load_labelled(key, drop_spikes=False, add_features=False,
                             source="phase4")
    dt = float(meta["sample_interval_s"])
    n = len(df)
    states = df["state"].to_numpy()
    covered_s = n * dt
    span_s = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

    counts = pd.Series(states).value_counts()
    hours = {s: float(counts.get(s, 0) * dt / SECONDS_PER_HOUR) for s in STATE_ORDER}

    rec: dict = {
        "dataset": key,
        "name": cfg["name"],
        "role": cfg["role"],
        "n_rows": n,
        "sample_interval_s": dt,
        "coverage_days": covered_s / 86400.0,
        "span_days": span_s / 86400.0,
        "n_segments": int(df["segment_id"].nunique()),
        "spike_pct": float(df["is_spike"].mean() * 100.0),
        "state_hours": hours,
        "state_share_pct": {s: h / max(sum(hours.values()), 1e-9) * 100.0
                            for s, h in hours.items()},
        "standby_hours": hours["STANDBY"],
        "standby_power_w": (float(df.loc[states == "STANDBY", "active_power"].mean())
                            if (states == "STANDBY").any() else float("nan")),
        "channels_available": [c for c in L.IMDELD_FEATURES if c in df.columns],
    }

    seg = df["segment_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], n]
    blocks = [states[s:e] for s, e in zip(starts, ends) if e - s > 1]
    rec["flicker"] = flicker_rate(blocks, dt, FLICKER_TAU_S)
    rec["flicker"].pop("per_state_dwell", None)
    del blocks

    stride = max(1, n // 2_000_000)
    idx = np.arange(0, n, stride)
    sub = df.iloc[idx][["timestamp", "active_power"]].reset_index(drop=True)
    phys = physics_compliance(sub, np.arange(len(sub)), states[idx])
    rec["physics"] = {
        "score": phys["score"],
        "n_passed": phys["n_checks_passed"],
        "n_applicable": phys["n_checks_applicable"],
        "checks": phys["checks"],
        "values": phys["values"],
        "unavailable_checks": [k for k, v in phys["checks"].items() if v is None],
    }
    rec["per_state"] = phys["per_state"]

    # Diurnal profile: the shape that distinguishes a scheduled machine from a
    # solar generator, and the only external evidence available here since the
    # site publishes no shift calendar.
    hour = df["timestamp"].dt.hour.to_numpy()[idx]
    prof = pd.crosstab(hour, states[idx], normalize="index") * 100.0
    rec["diurnal_state_pct"] = prof.round(2).to_dict()
    prod = np.isin(states[idx], ("WORKING", "PEAK_LOAD"))
    by_hour = pd.Series(prod).groupby(hour).mean() * 100.0
    rec["productive_pct_by_hour"] = {int(h): float(v) for h, v in by_hour.items()}
    rec["productive_night_pct"] = float(
        by_hour.reindex(range(0, 5)).mean())          # 00:00-05:00
    rec["productive_midday_pct"] = float(
        by_hour.reindex(range(10, 15)).mean())        # 10:00-15:00
    rec["schedule_known"] = bool(cfg.get("schedule_known", True))
    del sub

    info(f"{key}: physics {rec['physics']['n_passed']}/{rec['physics']['n_applicable']} "
         f"(unavailable: {rec['physics']['unavailable_checks']}), "
         f"flicker {rec['flicker']['flicker_rate_pct']:.2f}%, "
         f"STANDBY {rec['standby_hours']:.0f} h @ {rec['standby_power_w']:.0f} W, "
         f"productive at night {rec['productive_night_pct']:.1f}% vs "
         f"midday {rec['productive_midday_pct']:.1f}%")

    del df, states
    gc.collect()
    save_json(rec, f"{TASK10_DIR}/{key}/characterisation.json")
    return rec


# ── Decision layer and zero-shot transfer ─────────────────────────────────────

def decision_and_transfer(key: str, source_machine: str = "pelletizer-I",
                          seed: int = 0) -> dict:
    """
    Run the decision layer on the held-out dataset, and test the IMDELD-trained
    forecaster on it zero-shot.

    Three policies are compared on the same held-out episodes: the plant's
    observed behaviour, the forecast-free ski-rental rule, and the decision
    rule driven by (a) a forecaster fitted on this record and (b) one fitted on
    pelletizer-I and never shown this dataset.  The comparison against
    ski-rental is the one that matters -- a transferred forecaster that cannot
    beat a clock is not transferring anything.
    """
    from experiments.task6_optimization import (
        build_epoch_table, fit_forecaster, fit_feasibility, make_predictor,
        make_feasibility, DECISION_EPOCH_S, MAX_DECISION_HOURS,
        CHANCE_CONFIDENCE, prepare,
    )
    from experiments.task9_cross_machine import (
        machine_power_scale, make_scaled_predictor, make_scaled_feasibility,
        fit_scaled_models,
    )

    cfg = MACHINES[key]
    section(f"TASK 10 -- decision layer and zero-shot transfer on {key}")

    df, meta = load_labelled(key, drop_spikes=False, add_features=False,
                             source="phase4")
    dt = float(meta["sample_interval_s"])
    covered_s = len(df) * dt
    sig = measure_restart_signature(df, sample_interval_s=dt, random_state=seed)
    ep_all = extract_idle_episodes(df, sample_interval_s=dt)
    ep_all = attach_local_calendar(ep_all, cfg["factory_tz"],
                                   cfg["close_hour"], cfg["open_hour"])
    ep = ep_all[~ep_all["is_truncated"]].reset_index(drop=True)
    standby_power_w = float(df.loc[df["state"] == "STANDBY", "active_power"].mean())
    del df, ep_all
    gc.collect()

    ep = ep.sort_values("start_ts").reset_index(drop=True)
    cut = int(len(ep) * 0.70)
    ep_tr, ep_te = ep.iloc[:cut].reset_index(drop=True), ep.iloc[cut:].reset_index(drop=True)
    Xtr, ytr = build_epoch_table(ep_tr, DECISION_EPOCH_S, MAX_DECISION_HOURS,
                                 cfg["close_hour"], cfg["open_hour"])
    Xte, yte = build_epoch_table(ep_te, DECISION_EPOCH_S, MAX_DECISION_HOURS,
                                 cfg["close_hour"], cfg["open_hour"])

    out: dict = {
        "dataset": key,
        "coverage_days": covered_s / 86400.0,
        "standby_power_w": standby_power_w,
        "restart_delay_s": sig.get("restart_delay_s"),
        "restart_delay_ci95_s": sig.get("restart_delay_ci95_s"),
        "restart_energy_kwh": sig.get("restart_energy_kwh"),
        "n_cold": sig.get("n_cold"), "n_warm": sig.get("n_warm"),
        "n_episodes_usable": int(len(ep)),
        "idle_hours": float(ep["duration_s"].sum() / SECONDS_PER_HOUR),
        "standby_hours_observed": float(ep["standby_s"].sum() / SECONDS_PER_HOUR),
        "off_hours_observed": float(ep["off_s"].sum() / SECONDS_PER_HOUR),
        "median_episode_s": float(ep["duration_s"].median()),
        "max_episode_h": float(ep["duration_s"].max() / SECONDS_PER_HOUR),
        "n_train_epochs": int(len(ytr)), "n_test_epochs": int(len(yte)),
    }

    # -- local forecaster ------------------------------------------------------
    local_model, local_spec = fit_forecaster(Xtr, ytr, seed=seed, mode="mean")
    local_clf = fit_feasibility(Xtr, ytr, MIN_OFF_S, seed=seed)
    p_local = np.clip(local_model.predict(Xte), 0, None)
    out["local_forecaster"] = _forecast_metrics(p_local, yte)

    # -- transferred forecaster ------------------------------------------------
    src = prepare(source_machine, seed=seed, source="phase4")
    scale_src = machine_power_scale(src["ep_train"])
    scale_tgt = machine_power_scale(ep_tr)
    src_scaled_model, _, src_scaled_clf = fit_scaled_models(src, scale_src, seed=seed)

    from experiments.task9_cross_machine import scaled_X
    p_raw = np.clip(src["forecaster_models"]["mean"].predict(Xte), 0, None)
    p_scaled = np.clip(src_scaled_model.predict(scaled_X(Xte, scale_tgt)), 0, None)
    out["transfer_raw"] = _forecast_metrics(p_raw, yte)
    out["transfer_scaled"] = _forecast_metrics(p_scaled, yte)
    out["power_scale_source_w"] = scale_src
    out["power_scale_target_w"] = scale_tgt
    out["source_machine"] = source_machine

    # -- policies --------------------------------------------------------------
    ch, oh = cfg["close_hour"], cfg["open_hour"]
    degen = degenerate_reason(standby_power_w, sig)
    out["degenerate_reason"] = degen
    if degen is None:
        base = EconomicParams(
            standby_power_w=standby_power_w, tariff_per_kwh=TARIFF_USD_PER_KWH,
            restart_energy_kwh=float(max(0.0, sig["restart_energy_kwh"])),
            restart_delay_s=float(max(0.0, sig["restart_delay_s"])),
            min_off_s=MIN_OFF_S, max_restarts_per_day=MAX_RESTARTS_PER_DAY)
        params = params_for_break_even(base, BREAK_EVEN_S)
        steps = int(MAX_DECISION_HOURS * SECONDS_PER_HOUR / DECISION_EPOCH_S)
        test_span_s = float((ep_te["end_ts"].max()
                             - ep_te["start_ts"].min()).total_seconds())

        policies = {
            "observed": policy_observed(ep_te, params),
            "ski_rental": policy_elapsed_time(ep_te, params),
            "local": policy_forecast_optimising(
                ep_te, params,
                make_predictor(local_model, local_spec, ch, oh),
                epoch_s=DECISION_EPOCH_S, max_epochs=steps,
                predict_feasible=make_feasibility(local_clf, ch, oh),
                confidence=CHANCE_CONFIDENCE),
            "transferred_raw": policy_forecast_optimising(
                ep_te, params,
                make_predictor(src["forecaster_models"]["mean"],
                               src["forecaster_specs"]["mean"], ch, oh),
                epoch_s=DECISION_EPOCH_S, max_epochs=steps,
                predict_feasible=make_feasibility(src["feasibility_model"], ch, oh),
                confidence=CHANCE_CONFIDENCE),
            "transferred_scaled": policy_forecast_optimising(
                ep_te, params,
                make_scaled_predictor(src_scaled_model, ch, oh, scale_tgt),
                epoch_s=DECISION_EPOCH_S, max_epochs=steps,
                predict_feasible=make_scaled_feasibility(src_scaled_clf, ch, oh,
                                                         scale_tgt),
                confidence=CHANCE_CONFIDENCE),
            "oracle": optimal_offline(ep_te, params),
        }
        tab = compare_policies(ep_te, policies, params)
        tab["annual_savings_usd"] = [annualise(v, test_span_s)
                                     for v in tab["savings_vs_baseline"]]
        tab.to_csv(f"{TASK10_DIR}/{key}/policy_comparison.csv")
        out["economics"] = params.to_dict()
        out["policies"] = tab.to_dict(orient="index")
        out["test_span_days"] = test_span_s / 86400.0
        print(tab[["n_shutdowns", "standby_h", "off_h", "cost_total",
                   "savings_vs_baseline", "pct_of_oracle", "min_off_violations",
                   "annual_savings_usd"]].to_string(
                       float_format=lambda v: f"{v:9.3f}"))
    else:
        out["policies_unavailable"] = degen
        info(f"no decision problem on this record: {degen}")

    save_json(out, f"{TASK10_DIR}/{key}/decision_transfer.json")
    gc.collect()
    return out


def _forecast_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "mae_s": float(np.mean(np.abs(p - y))),
        "rmse_s": float(np.sqrt(np.mean((p - y) ** 2))),
        "spearman": float(pd.Series(p).corr(pd.Series(y), method="spearman")),
        "mean_ratio": float(p.mean() / max(y.mean(), 1e-9)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(datasets: list[str] | None = None,
         steps: tuple[str, ...] = ("label", "characterise", "decision"),
         source_machine: str = "pelletizer-I", relabel: bool = False,
         seed: int = 42) -> dict:
    datasets = datasets or list(SPARK_DATASETS)
    ensure_dir(TASK10_DIR)
    section(f"PHASE IV / TASK 10 -- cross-dataset validation: {datasets}")

    labels = {}
    if "label" in steps:
        for k in datasets:
            labels[k] = label_dataset(k, force=relabel, seed=seed)

    chars = {k: characterise(k, seed=seed) for k in datasets} if "characterise" in steps else {}

    decisions = {}
    if "decision" in steps:
        for k in datasets:
            try:
                decisions[k] = decision_and_transfer(k, source_machine, seed=0)
            except Exception as exc:
                info(f"{k}: decision/transfer FAILED -- {type(exc).__name__}: {exc}")
                decisions[k] = {"dataset": k, "error": f"{type(exc).__name__}: {exc}"}

    if chars:
        rows = []
        for k, c in chars.items():
            rows.append({
                "dataset": k,
                "role": c["role"],
                "channels": len(c["channels_available"]),
                "coverage_days": c["coverage_days"],
                "physics_passed": f"{c['physics']['n_passed']}/{c['physics']['n_applicable']}",
                "physics_unavailable": ",".join(c["physics"]["unavailable_checks"]),
                "flicker_pct": c["flicker"]["flicker_rate_pct"],
                "median_dwell_s": c["flicker"]["median_dwell_s"],
                "standby_h": c["standby_hours"],
                "standby_power_w": c["standby_power_w"],
                "productive_night_pct": c["productive_night_pct"],
                "productive_midday_pct": c["productive_midday_pct"],
            })
        tab = pd.DataFrame(rows).set_index("dataset")
        tab.to_csv(f"{TASK10_DIR}/table_cross_dataset.csv")
        section("TASK 10 -- CROSS-DATASET SUMMARY")
        print(tab.round(3).to_string())

    save_json({"datasets": datasets, "labels": labels,
               "characterisation": chars, "decision": decisions},
              f"{TASK10_DIR}/task10_results.json")
    section("TASK 10 COMPLETE")
    return {"characterisation": chars, "decision": decisions}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase IV Task 10 -- cross-dataset validation")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help=f"subset of {list(SPARK_DATASETS)}")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--source-machine", default="pelletizer-I")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    all_steps = ("label", "characterise", "decision")
    steps = tuple(s for s in (a.only or all_steps) if s not in a.skip)
    main(datasets=a.datasets, steps=steps, source_machine=a.source_machine,
         relabel=a.relabel, seed=a.seed)
