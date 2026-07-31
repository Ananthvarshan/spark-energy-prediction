"""
============================================================
EQUIVALENCE TESTS FOR THE PHASE IV LABELLER
tests/test_labelling.py
============================================================

`src/labelling.py` exists so that one procedure can be applied to
every machine.  It only earns that role if it is the SAME procedure
Phases I-III used, so each place where it departs from the original
code for speed or memory is checked here against the original:

  T1  timestamp fast path   == pandas' general parser
  T2  raw loader            == validate_gmm.load_and_prepare_data
  T3  min-dwell operator    == task4_flicker_diagnosis.enforce_min_dwell
  T4  feature matrix        == validate_gmm.build_feature_matrix
  T5  spike mask            == validate_gmm.compute_spike_mask

Run:  python -m tests.test_labelling
============================================================
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import labelling as L                      # noqa: E402
from experiments.task4_flicker_diagnosis import enforce_min_dwell as ref_min_dwell  # noqa: E402


# The clean sample carries no missing cells, so every transform must agree
# with validate_gmm.py to floating-point tolerance.  The blank-current sample
# exercises the one documented departure (T2).
SAMPLE_CSV = "data/Appliances/pelletizer-I.csv"
BLANK_CURRENT_CSV = "data/Appliances/millingmachine-I.csv"
N_SAMPLE_ROWS = 40_000


def _import_validate_gmm():
    """Import validate_gmm with a clean argv -- it reads sys.argv at import."""
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        import validate_gmm as vg
        return vg
    finally:
        sys.argv = saved


def _sample_csv(path: str, n: int, tag: str) -> str:
    """Write the first `n` data rows of `path` to a temp CSV, header kept."""
    df = pd.read_csv(path, nrows=n)
    tmp = os.path.join(tempfile.gettempdir(), f"phase4_sample_{tag}.csv")
    df.to_csv(tmp, index=False)
    return tmp


def t1_timestamps(tmp_csv: str) -> bool:
    s = pd.read_csv(tmp_csv, usecols=["timestamp"])["timestamp"].astype(str)
    fast = L.parse_timestamps(s)
    ref = pd.to_datetime(s, utc=True)
    ok = bool((fast == ref).all())
    print(f"  T1 timestamp fast path        : {'PASS' if ok else 'FAIL'} "
          f"({len(s):,} stamps)")
    return ok


def t2_loader(tmp_csv: str, vg) -> bool:
    """
    Equal to validate_gmm's loader everywhere the latter produces a number.

    The one deliberate difference is the blank-current rule documented in
    `load_raw`: validate_gmm leaves those cells NaN (and a GMM cannot be
    fitted through them), Phase IV fills them with the only value consistent
    with S = 0 VA.  The test asserts that this is exactly what happened --
    that every filled cell was NaN in the reference and had zero apparent
    power -- rather than waiving the comparison.
    """
    mine = L.load_raw(tmp_csv, verbose=False)
    theirs = vg.load_and_prepare_data(tmp_csv)
    ok = len(mine) == len(theirs) and bool((mine["timestamp"] == theirs["timestamp"]).all())
    for col in L.IMDELD_FEATURES:
        if col not in theirs.columns:
            continue
        a = mine[col].to_numpy(np.float64)
        b = theirs[col].to_numpy(np.float64)
        filled = np.isnan(b) & ~np.isnan(a)
        if filled.any():
            legit = bool(np.all(a[filled] == 0.0)
                         and np.all(theirs["apparent_power"].to_numpy()[filled] == 0.0))
            ok &= legit
            print(f"     column {col}: {filled.sum():,} NaN filled with 0 "
                  f"at S=0 VA -- {'as documented' if legit else 'UNEXPECTED'}")
        rest = ~filled
        close = np.allclose(a[rest], b[rest], rtol=1e-6, atol=1e-4, equal_nan=True)
        ok &= close
        if not close:
            print(f"     column {col} differs "
                  f"(max |d| = {np.nanmax(np.abs(a[rest] - b[rest])):.3g})")
    print(f"  T2 raw loader vs validate_gmm : {'PASS' if ok else 'FAIL'}")
    return ok


def t3_min_dwell() -> bool:
    """
    The linear-time operator must reproduce the Phase II one exactly, including
    its tie-breaking: shortest run first, leftmost on a tie, absorbed into the
    longer neighbour with ties going left.
    """
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(200):
        n = int(rng.integers(5, 400))
        k = int(rng.integers(2, 5))
        # Mix of i.i.d. noise and sticky runs so both extremes are covered.
        if trial % 2:
            lab = rng.integers(0, k, size=n)
        else:
            lab = np.repeat(rng.integers(0, k, size=n // 3 + 1),
                            rng.integers(1, 8, size=n // 3 + 1))[:n]
        min_rows = int(rng.integers(2, 12))
        a = L.enforce_min_dwell(lab.copy(), min_rows)
        b = ref_min_dwell(lab.copy(), min_rows)
        if not np.array_equal(a, b):
            ok = False
            print(f"     mismatch: n={n} k={k} min_rows={min_rows}")
            break
    print(f"  T3 min-dwell vs Phase II      : {'PASS' if ok else 'FAIL'} "
          f"(200 random sequences)")
    return ok


def t2b_blank_current(tmp_csv: str, vg) -> bool:
    """The blank-current rule: every filled cell was NaN at exactly S = 0 VA."""
    mine = L.load_raw(tmp_csv, verbose=False)
    theirs = vg.load_and_prepare_data(tmp_csv)
    was_nan = theirs["current"].isna().to_numpy()
    n = int(was_nan.sum())
    ok = n > 0 and bool(
        np.all(mine["current"].to_numpy()[was_nan] == 0.0)
        and np.all(theirs["apparent_power"].to_numpy()[was_nan] == 0.0)
        and np.all(theirs["active_power"].to_numpy()[was_nan] == 0.0)
        and not mine["current"].isna().any()
    )
    print(f"  T2b blank-current rule        : {'PASS' if ok else 'FAIL'} "
          f"({n:,} blanks, all at S=0 VA and P=0 W, filled with 0 A)")
    return ok


def t4_features(tmp_csv: str, vg) -> bool:
    theirs_df = vg.load_and_prepare_data(tmp_csv)
    theirs_df = vg.denoise_off_state(theirs_df)
    Xt, _ = vg.build_feature_matrix(theirs_df)

    mine_df = L.load_raw(tmp_csv, verbose=False)
    mine_df, _ = L.add_segments(mine_df)
    mine_df = L.denoise_off_state(mine_df, verbose=False)
    Xm, _, cols = L.build_feature_matrix(mine_df)

    # Rows the reference cannot represent (blank current -> NaN, see T2) are
    # compared only where the reference is finite.
    finite = np.isfinite(Xt).all(axis=1)
    ok = Xm.shape == Xt.shape and np.allclose(Xm[finite], Xt[finite], atol=2e-3)
    print(f"  T4 feature matrix             : {'PASS' if ok else 'FAIL'} "
          f"(max |d| = {np.max(np.abs(Xm[finite] - Xt[finite])):.3g} on "
          f"{finite.sum():,} comparable rows, channels={cols})")
    return ok


def t5_spike_mask(tmp_csv: str, vg) -> bool:
    """
    Agreement is checked to 99.9%, not to the row: Phase IV stores the
    electrical channels as float32 (float64 for eight 5.5M-row records does
    not fit in this machine's memory), so a row whose deviation sits within
    one ulp of the 5x MAD threshold can fall on either side.  Measured at
    ~2 rows per 10^5 -- immaterial to hour totals, and stated rather than
    silently tolerated.
    """
    theirs_df = vg.load_and_prepare_data(tmp_csv)
    ref = vg.compute_spike_mask(theirs_df).to_numpy()
    mine_df = L.load_raw(tmp_csv, verbose=False)
    mine = L.compute_spike_mask(mine_df, verbose=False)
    agree = float((ref == mine).mean())
    ok = agree > 0.999
    print(f"  T5 spike mask                 : {'PASS' if ok else 'FAIL'} "
          f"(agreement {agree*100:.3f}%, {ref.sum():,} vs {mine.sum():,} flagged)")
    return ok


def main() -> int:
    print("=" * 68)
    print("  PHASE IV LABELLER -- EQUIVALENCE TESTS")
    print("=" * 68)
    vg = _import_validate_gmm()
    tmp = _sample_csv(SAMPLE_CSV, N_SAMPLE_ROWS, "clean")
    tmp_nan = _sample_csv(BLANK_CURRENT_CSV, N_SAMPLE_ROWS, "blank")
    results = [
        t1_timestamps(tmp),
        t2_loader(tmp, vg),
        t2b_blank_current(tmp_nan, vg),
        t3_min_dwell(),
        t4_features(tmp, vg),
        t5_spike_mask(tmp, vg),
    ]
    print("-" * 68)
    print(f"  {sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
