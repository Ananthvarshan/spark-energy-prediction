"""
============================================================
SPARK DATASET ANALYSER — main.py
============================================================

SPARK Dataset:
  - 22 industrial machines (mills, lathes, chip presses, pumps)
  - 1 photovoltaic (solar) system
  - 5-second sampling rate
  - 1 to 7 years of data per machine
  - File format: .csv.xz (compressed)

HOW TO USE:
-----------
OPTION A — Test with a SINGLE machine file:
    Set MODE = "single"
    Set DATA_PATH to your .csv or .csv.xz file
    Run: python main.py

OPTION B — Run on ALL 22 machines at once:
    Set MODE = "multi"
    Set SPARK_ROOT to your extracted SPARK dataset folder
    Run: python main.py

OPTION C — Quick test with a small CSV (no download needed):
    Set MODE = "single"
    Place any small test CSV in data/dataset.csv
    Run: python main.py

DO I NEED TO DOWNLOAD THE FULL 76 GB?
    NO! You can download just ONE machine file from SPARK.
    Each machine file is typically 50MB–200MB compressed.
    Just pick one .csv.xz file and test with that.
============================================================
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_analysis import run_analysis

# ============================================================
# CONFIGURATION — CHANGE THIS
# ============================================================

# Choose mode: "single" or "multi"
MODE = "single"

# --- OPTION A: Single file path ---
# For SPARK: e.g., "data/MILL_01/power.i1/2023.csv.xz"
# For quick test: "data/dataset.csv"
DATA_PATH   = "data/2024_P_total_PickAndPlace.csv.xz"
MACHINE_NAME = "EPI_PickAndPlace Robot" # Give it a name for the report
OUTPUT_PATH = "outputs/plots"

# --- OPTION B: Multi-machine SPARK root folder ---
# Set this to where you extracted the full SPARK dataset
# e.g., "C:/Downloads/SPARK/" or "/home/user/SPARK/"
SPARK_ROOT  = "data/SPARK/"

# ============================================================


def run_single():
    """Analyse a single machine file."""
    print(f"\n🔬 Running analysis on: {DATA_PATH}")
    print(f"   Machine: {MACHINE_NAME}")
    print(f"   Output:  {OUTPUT_PATH}\n")

    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: File not found → {DATA_PATH}")
        print("\nPlease either:")
        print("  1. Place your CSV file at:", DATA_PATH)
        print("  2. Or update DATA_PATH in main.py to your actual file path")
        return

    summary = run_analysis(DATA_PATH, OUTPUT_PATH, MACHINE_NAME)
    print("\n🎉 Single machine analysis complete!")
    return summary


def run_multi():
    """
    Scan the SPARK_ROOT folder and run analysis on EVERY machine found.
    Saves results in separate subfolders per machine.
    Prints a combined comparison table at the end.
    """
    import glob

    if not os.path.exists(SPARK_ROOT):
        print(f"❌ ERROR: SPARK root folder not found → {SPARK_ROOT}")
        print("Please update SPARK_ROOT in main.py to your SPARK dataset folder.")
        return

    # Find all csv.xz files inside SPARK_ROOT
    all_files = glob.glob(os.path.join(SPARK_ROOT, "**", "*.csv.xz"), recursive=True)

    if len(all_files) == 0:
        # Also try plain csv
        all_files = glob.glob(os.path.join(SPARK_ROOT, "**", "*.csv"), recursive=True)

    if len(all_files) == 0:
        print(f"❌ No .csv.xz or .csv files found in: {SPARK_ROOT}")
        return

    print(f"\n📦 Found {len(all_files)} data files in SPARK dataset")
    print("="*60)

    all_summaries = {}

    for i, fpath in enumerate(sorted(all_files)):
        # Use relative path as machine name
        rel = os.path.relpath(fpath, SPARK_ROOT)
        parts = rel.replace("\\", "/").split("/")
        machine_name = parts[0] if len(parts) > 1 else f"Machine_{i+1:02d}"
        measurement  = parts[1] if len(parts) > 2 else "power"

        label      = f"{machine_name}_{measurement}"
        out_folder = os.path.join("outputs", "plots", machine_name, measurement)

        print(f"\n[{i+1}/{len(all_files)}] Processing: {label}")

        try:
            summary = run_analysis(fpath, out_folder, label)
            all_summaries[label] = summary
        except Exception as e:
            print(f"  ⚠️  Skipped due to error: {e}")

    # Print combined comparison table
    if all_summaries:
        print_comparison_table(all_summaries)


def print_comparison_table(all_summaries):
    """Print a side-by-side comparison of all 22 machines."""
    print("\n")
    print("="*90)
    print("  ALL MACHINES — STATE COMPARISON TABLE")
    print("="*90)
    print(f"  {'MACHINE':<30} {'OFF':>8} {'STANDBY':>10} {'IDLE':>8} {'WORKING':>10}")
    print("-"*90)

    for machine, summary in all_summaries.items():
        off_h  = summary.get('OFF',     {}).get('hours', 0)
        stby_h = summary.get('STANDBY', {}).get('hours', 0)
        idle_h = summary.get('IDLE',    {}).get('hours', 0)
        work_h = summary.get('WORKING', {}).get('hours', 0)
        print(f"  {machine:<30} {off_h:>7.1f}h {stby_h:>9.1f}h {idle_h:>7.1f}h {work_h:>9.1f}h")

    print("="*90)

    # Save comparison to CSV
    import csv
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/machine_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Machine", "OFF_hours", "STANDBY_hours", "IDLE_hours", "WORKING_hours",
                         "OFF_%", "STANDBY_%", "IDLE_%", "WORKING_%"])
        for machine, summary in all_summaries.items():
            writer.writerow([
                machine,
                summary.get('OFF',     {}).get('hours', 0),
                summary.get('STANDBY', {}).get('hours', 0),
                summary.get('IDLE',    {}).get('hours', 0),
                summary.get('WORKING', {}).get('hours', 0),
                summary.get('OFF',     {}).get('percent', 0),
                summary.get('STANDBY', {}).get('percent', 0),
                summary.get('IDLE',    {}).get('percent', 0),
                summary.get('WORKING', {}).get('percent', 0),
            ])
    print(f"\n✅ Comparison table saved: outputs/machine_comparison.csv")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("  SPARK INDUSTRIAL ENERGY DATASET ANALYSER")
    print("="*60)

    if MODE == "single":
        run_single()
    elif MODE == "multi":
        run_multi()
    else:
        print(f"❌ Unknown MODE: '{MODE}'. Use 'single' or 'multi'.")