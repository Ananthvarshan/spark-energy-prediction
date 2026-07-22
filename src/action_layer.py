"""
============================================================
ENERGY ACTION LAYER  --  src/action_layer.py
============================================================

PURPOSE
-------
Execute energy-saving recommendations from the decision logic
(Step 10) in either manual or automatic mode.

Two modes
---------
"manual"    : log the recommendation to a CSV file + print to console.
              No machine is ever physically controlled.
              All recommendations are recorded for post-hoc accuracy
              measurement (see feedback_loop below).

"automatic" : if prediction confidence >= AUTO_CONFIDENCE_THRESHOLD,
              send a PLC control signal (placeholder — wire up your
              actual PLC interface here).
              If confidence is too low, falls back to "manual" mode
              automatically rather than acting on an uncertain prediction.

Feedback loop
-------------
Every recommendation is logged with its timestamp, machine ID, action,
confidence, estimated savings, and a 'acted_on' flag. After the
recommendation window elapses, a separate monitoring process can
log what actually happened — this creates a labelled dataset of
{prediction: outcome} pairs for tracking real-world accuracy over
time (i.e. detecting concept drift when factory schedules change).

HOW TO USE
----------
    from src.action_layer import EnergyAction

    action_layer = EnergyAction(mode="manual", output_dir="outputs/action_log")
    action_layer.execute(
        machine_id="pelletizer-I",
        action="SHUTDOWN_RECOMMENDED",
        confidence=0.85,
        savings_estimate=0.42,
        predicted_standby_s=1800,
    )
============================================================
"""

import os
import csv
import json
from datetime import datetime, timezone


# ── Constants ─────────────────────────────────────────────────────────────────
AUTO_CONFIDENCE_THRESHOLD = 0.70   # minimum confidence for automatic action
LOG_FILENAME              = "recommendations.csv"
LOG_COLUMNS = [
    "timestamp_utc",
    "machine_id",
    "mode",
    "action",
    "confidence",
    "predicted_standby_s",
    "savings_estimate_currency",
    "auto_executed",
    "downgraded_to_manual",
    "notes",
]


# ── PLC stub ──────────────────────────────────────────────────────────────────

def _send_plc_signal(machine_id: str, action: str) -> bool:
    """
    Placeholder for the actual PLC / SCADA control interface.

    Replace this function body with your real PLC communication code
    (e.g., Modbus TCP, OPC-UA, MQTT) when deploying to production.

    Returns True on success, False on failure.
    """
    print(
        f"[action_layer] 🔌 PLC SIGNAL (STUB): machine={machine_id}  action={action}\n"
        f"   ⚠️  Wire up _send_plc_signal() with your actual PLC interface before production."
    )
    return True   # stub always succeeds


# ── Action layer class ────────────────────────────────────────────────────────

class EnergyAction:
    """
    Execute energy-saving recommendations in manual or automatic mode.

    Parameters
    ----------
    mode       : "manual" (log only) or "automatic" (log + PLC signal if confident)
    output_dir : directory to write the recommendation log CSV
    confidence_threshold : override AUTO_CONFIDENCE_THRESHOLD if needed
    """

    def __init__(
        self,
        mode: str = "manual",
        output_dir: str = "outputs/action_log",
        confidence_threshold: float = AUTO_CONFIDENCE_THRESHOLD,
    ):
        if mode not in ("manual", "automatic"):
            raise ValueError(f"mode must be 'manual' or 'automatic', got '{mode}'")

        self.mode                 = mode
        self.output_dir           = output_dir
        self.confidence_threshold = confidence_threshold
        self._log_path            = os.path.join(output_dir, LOG_FILENAME)

        os.makedirs(output_dir, exist_ok=True)
        self._ensure_log_header()

        print(
            f"[action_layer] Initialized in '{mode}' mode\n"
            f"  Log file: {self._log_path}\n"
            f"  Auto-execute threshold: confidence >= {confidence_threshold:.2f}"
        )

    def _ensure_log_header(self) -> None:
        """Write CSV header if the log file doesn't exist yet."""
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
                writer.writeheader()

    def execute(
        self,
        machine_id: str,
        action: str,
        confidence: float,
        savings_estimate: float = 0.0,
        predicted_standby_s: float = 0.0,
        notes: str = "",
    ) -> dict:
        """
        Execute or log an energy recommendation.

        Parameters
        ----------
        machine_id          : unique identifier for the machine
        action              : recommendation string, e.g. "SHUTDOWN_RECOMMENDED"
                              or "NO_ACTION" or "STANDBY_DETECTED"
        confidence          : mean prediction confidence from the LSTM (0–1)
        savings_estimate    : estimated monetary savings (from should_recommend_shutdown)
        predicted_standby_s : predicted STANDBY duration in seconds
        notes               : any additional context for the log

        Returns
        -------
        dict with execution result metadata
        """
        ts_utc        = datetime.now(timezone.utc).isoformat()
        auto_executed = False
        downgraded    = False

        # ── Mode dispatch ─────────────────────────────────────────────────────
        if self.mode == "automatic":
            if confidence >= self.confidence_threshold:
                success = _send_plc_signal(machine_id, action)
                auto_executed = success
                status_msg = (
                    f"✅ AUTO-EXECUTED: {action} on {machine_id}  "
                    f"(confidence={confidence:.2f}, savings≈{savings_estimate:.3f})"
                )
            else:
                downgraded = True
                status_msg = (
                    f"⚠️  AUTO DOWNGRADED TO MANUAL: confidence {confidence:.2f} "
                    f"< threshold {self.confidence_threshold:.2f}\n"
                    f"   → Logging recommendation only: {action} on {machine_id}"
                )
        else:
            status_msg = (
                f"📋 MANUAL RECOMMENDATION: {action} on {machine_id}  "
                f"(confidence={confidence:.2f}, "
                f"predicted_standby={predicted_standby_s/60:.1f}min, "
                f"savings≈{savings_estimate:.3f})"
            )

        print(f"\n[action_layer] {status_msg}")

        # ── Write to log ──────────────────────────────────────────────────────
        row = {
            "timestamp_utc":              ts_utc,
            "machine_id":                 machine_id,
            "mode":                       self.mode,
            "action":                     action,
            "confidence":                 f"{confidence:.4f}",
            "predicted_standby_s":        f"{predicted_standby_s:.1f}",
            "savings_estimate_currency":  f"{savings_estimate:.4f}",
            "auto_executed":              str(auto_executed),
            "downgraded_to_manual":       str(downgraded),
            "notes":                      notes,
        }
        with open(self._log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            writer.writerow(row)

        return {
            "timestamp_utc":   ts_utc,
            "action":          action,
            "auto_executed":   auto_executed,
            "downgraded":      downgraded,
            "savings_estimate": savings_estimate,
        }

    def log_outcome(
        self,
        machine_id: str,
        recommendation_ts_utc: str,
        actual_standby_s: float,
        outcome_notes: str = "",
    ) -> None:
        """
        Log the ACTUAL outcome of a previous recommendation.

        Call this after the prediction window has elapsed to record what
        actually happened — enabling real-world accuracy tracking and
        concept-drift detection without a separate monitoring system.

        Parameters
        ----------
        machine_id             : must match the original recommendation
        recommendation_ts_utc  : ISO timestamp of the original recommendation
        actual_standby_s       : how long the machine was actually in STANDBY
        outcome_notes          : free text, e.g. "operator rejected shutdown"
        """
        outcome_log = os.path.join(self.output_dir, "outcomes.csv")
        write_header = not os.path.exists(outcome_log)

        outcome_cols = [
            "logged_at_utc",
            "machine_id",
            "recommendation_ts_utc",
            "actual_standby_s",
            "outcome_notes",
        ]
        with open(outcome_log, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=outcome_cols)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "logged_at_utc":          datetime.now(timezone.utc).isoformat(),
                "machine_id":             machine_id,
                "recommendation_ts_utc":  recommendation_ts_utc,
                "actual_standby_s":       f"{actual_standby_s:.1f}",
                "outcome_notes":          outcome_notes,
            })

        print(f"[action_layer] Outcome logged → {outcome_log}")
