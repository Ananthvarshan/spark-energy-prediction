"""
============================================================
MASTER PIPELINE RUNNER — main.py
============================================================

PURPOSE
-------
Single master entry point for the entire Industrial Energy State 
Optimization pipeline.

Running this single script executes all stages end-to-end:
  1. GMM State Detection & Validation (validate_gmm.py)
  2. 5-Method Physics Proof Harness (proof_5methods.py)
  3. Feature Engineering & Processed Dataset Export (outputs/processed_datasets/)
  4. Seq2Seq LSTM Model Training, Diagnostics, & Action Layer (lstm_pipeline.py)

HOW TO RUN
----------
  python main.py                     # Runs default machine: pelletizer-I
  python main.py pelletizer-II       # Runs specified machine
  python main.py --mode automatic    # Runs with automatic PLC action mode

OUTPUTS GENERATED
-----------------
  1. Base GMM Labelled CSV  → outputs/imdeld_labelled/<machine>_labelled.csv
  2. Separate Processed CSV → outputs/processed_datasets/<machine>_processed_lstm_ready.csv
  3. Model & Diagnostics   → outputs/models/<machine>/
       - best_model.keras (Trained LSTM weights)
       - state_encoder.json
       - training_config.json
       - training_history.csv
       - test_evaluation.txt
       - confusion_matrix.csv
       - test_predictions_detailed.csv (Step-by-step diagnostic prediction log)
  4. Action Recommendations → outputs/action_log/recommendations.csv
============================================================
"""

import os
import sys
import subprocess
import argparse

# Force UTF-8 encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MACHINE = "pelletizer-I"

def run_command_stage(stage_num: int, title: str, cmd_args: list):
    print("\n" + "=" * 70)
    print(f"  STAGE {stage_num}: {title}")
    print("=" * 70)
    print(f"Executing: {' '.join(cmd_args)}\n")
    
    result = subprocess.run([sys.executable] + cmd_args)
    if result.returncode != 0:
        print(f"\n❌ STAGE {stage_num} FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n✅ STAGE {stage_num} COMPLETE: {title}")

def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for GMM -> LSTM Industrial Energy Optimization Pipeline"
    )
    parser.add_argument(
        "machine",
        nargs="?",
        default=DEFAULT_MACHINE,
        help=f"Machine key to analyze (default: {DEFAULT_MACHINE})",
    )
    parser.add_argument(
        "--mode",
        default="manual",
        choices=["manual", "automatic"],
        help="Action layer mode: 'manual' (log only) or 'automatic' (PLC control)",
    )

    args = parser.parse_args()
    machine_key = args.machine

    print("=" * 70)
    print(" 🚀 INDUSTRIAL ENERGY PIPELINE MASTER EXECUTION")
    print(f" Target Machine : {machine_key}")
    print(f" Action Mode    : {args.mode}")
    print("=" * 70)

    # ── Stage 1: GMM Validation & Baseline State Labeling ───────────────────
    run_command_stage(
        1,
        "GMM State Clustering & Validation",
        ["validate_gmm.py", machine_key]
    )

    # ── Stage 2: 5-Method Physics Validation Harness ─────────────────────────
    run_command_stage(
        2,
        "5-Method Independent Physics Validation",
        ["proof_5methods.py", machine_key]
    )

    # ── Stage 3 & 4: Feature Engineering, Processed Dataset Export, & LSTM Pipeline ──
    run_command_stage(
        3,
        "Feature Engineering, Processed Dataset Export & LSTM Training/Evaluation",
        ["lstm_pipeline.py", machine_key, "--mode", args.mode]
    )

    print("\n" + "=" * 70)
    print(" 🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  1. GMM Validation Artifacts  → outputs/gmm_validation_imdeld/{machine_key}/")
    print(f"  2. Base Labelled CSV         → outputs/imdeld_labelled/{machine_key}_labelled.csv")
    print(f"  3. Processed LSTM Dataset    → outputs/processed_datasets/{machine_key}_processed_lstm_ready.csv")
    print(f"  4. Trained Model & Logs      → outputs/models/{machine_key}/")
    print(f"  5. Action Recommendations    → outputs/action_log/recommendations.csv")

if __name__ == "__main__":
    main()
