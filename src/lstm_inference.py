"""
============================================================
LSTM INFERENCE + DECISION LOGIC  --  src/lstm_inference.py
============================================================

PURPOSE
-------
Run the trained LSTM model on live/new data to:
  1. Predict future machine states (Step 8)
  2. Decide whether to recommend a machine shutdown based on
     a physics-based energy break-even calculation (Step 9)

The decision threshold is NOT a magic number — it is the
minimum predicted STANDBY duration at which the energy saved
by shutting down the machine exceeds the energy cost of the
subsequent restart. This directly uses the GMM cluster's own
mean standby power estimate as a physical input parameter,
closing the loop between state detection and decision logic.

HOW TO USE
----------
    from src.lstm_inference import predict_future_states, should_recommend_shutdown

    state_names, probs = predict_future_states(
        model, recent_window, state_decoder
    )
    recommend, savings_wh = should_recommend_shutdown(
        predicted_standby_seconds=sum_standby_in_prediction,
        standby_power_w=gmm_standby_cluster_mean_w,
        restart_energy_wh=machine_restart_energy_wh,
        electricity_price_per_kwh=0.15,
    )
============================================================
"""

import numpy as np


# ── State prediction ──────────────────────────────────────────────────────────

def predict_future_states(
    model,
    recent_window: np.ndarray,
    state_decoder: dict,
) -> tuple[list[str], np.ndarray]:
    """
    Predict the future state sequence given a recent observation window.

    Parameters
    ----------
    model         : trained Keras Model (from lstm_model.build_lstm_model)
    recent_window : numpy array of shape (1, lookback, n_features) OR
                    (lookback, n_features) — the most recent M readings.
    state_decoder : {int: state_name} e.g. {0:"OFF", 1:"STANDBY", 2:"WORKING"}

    Returns
    -------
    (state_names, probs)
      state_names : list of predicted state name strings for each future step
                    e.g. ["STANDBY", "STANDBY", "WORKING", ...]
      probs       : float32 array (horizon, n_states) — full probability
                    distribution per step (use for confidence scores)
    """
    window = np.asarray(recent_window, dtype=np.float32)
    if window.ndim == 2:
        window = window[np.newaxis, ...]       # add batch dimension: (1, L, F)

    probs    = model.predict(window, verbose=0)   # (1, horizon, n_states)
    probs    = probs[0]                            # (horizon, n_states)
    pred_ids = probs.argmax(axis=-1)               # (horizon,)

    state_names = [state_decoder.get(int(i), f"class_{i}") for i in pred_ids]
    return state_names, probs


def extract_predicted_standby_seconds(
    state_names: list[str],
    probs: np.ndarray,
    sample_interval_s: float,
    confidence_threshold: float = 0.0,
    standby_label: str = "STANDBY",
) -> tuple[float, float]:
    """
    From the raw prediction output, compute:
    (a) how many seconds of STANDBY are predicted in the horizon, and
    (b) the mean prediction confidence for those STANDBY steps.

    Parameters
    ----------
    state_names          : list of predicted state names (from predict_future_states)
    probs                : (horizon, n_states) probability array
    sample_interval_s    : seconds per row
    confidence_threshold : only count STANDBY steps where max_prob > this
    standby_label        : name of the STANDBY state (default "STANDBY")

    Returns
    -------
    (predicted_standby_seconds, mean_standby_confidence)
    """
    standby_seconds    = 0.0
    standby_confidences = []

    for i, name in enumerate(state_names):
        if name == standby_label:
            conf = float(probs[i].max())
            if conf >= confidence_threshold:
                standby_seconds += sample_interval_s
                standby_confidences.append(conf)

    mean_confidence = float(np.mean(standby_confidences)) if standby_confidences else 0.0
    return standby_seconds, mean_confidence


# ── Decision logic ────────────────────────────────────────────────────────────

def should_recommend_shutdown(
    predicted_standby_seconds: float,
    standby_power_w: float,
    restart_energy_wh: float,
    electricity_price_per_kwh: float,
) -> tuple[bool, float]:
    """
    Physics-based break-even shutdown recommendation (Step 9).

    The threshold is NOT arbitrary — it is the minimum predicted STANDBY
    duration at which the energy saved by shutting down the machine
    EXCEEDS the energy cost of the subsequent restart.

    Formula
    -------
    energy_saved_wh = (standby_power_w × predicted_standby_seconds) / 3600
    net_savings_wh  = energy_saved_wh - restart_energy_wh
    Recommend = True  iff  net_savings_wh > 0
    Savings   = net_savings_wh / 1000 × electricity_price_per_kwh  (£/$)

    Parameters
    ----------
    predicted_standby_seconds : predicted duration of the upcoming STANDBY
                                period (from extract_predicted_standby_seconds)
    standby_power_w           : mean power draw during STANDBY, in Watts.
                                USE the GMM cluster's own mean (inverse-scaled)
                                so this parameter is data-derived, not guessed.
    restart_energy_wh         : energy consumed by a cold restart of the machine,
                                in Watt-hours. Machine-specific constant; obtain
                                from machine spec sheet or measured restart trace.
    electricity_price_per_kwh : local electricity tariff ($/£/€ per kWh)

    Returns
    -------
    (recommend, savings_currency)
      recommend         : True if shutdown saves more than restart costs
      savings_currency  : estimated net monetary saving (same currency as price param)
                          negative value means shutdown would COST money
    """
    energy_saved_wh = (standby_power_w * predicted_standby_seconds) / 3600.0
    net_savings_wh  = energy_saved_wh - restart_energy_wh
    savings         = (net_savings_wh / 1000.0) * electricity_price_per_kwh

    recommend = net_savings_wh > 0

    return recommend, savings


def compute_break_even_duration_s(
    standby_power_w: float,
    restart_energy_wh: float,
) -> float:
    """
    Compute the minimum standby duration (in seconds) at which shutdown
    becomes energy-positive — i.e. the break-even point.

    Useful for reporting: "This machine should be shut down if predicted
    standby duration exceeds X minutes."

    Parameters
    ----------
    standby_power_w   : mean STANDBY power draw in Watts
    restart_energy_wh : restart energy cost in Watt-hours

    Returns
    -------
    float — break-even duration in seconds
    """
    if standby_power_w <= 0:
        return float("inf")   # standby draws no power → never worth shutting down
    break_even_s = (restart_energy_wh * 3600.0) / standby_power_w
    return break_even_s
