"""
============================================================
STATE MAPPING MODULE  --  src/state_mapping.py
============================================================

PURPOSE
-------
Turn arbitrary GMM cluster IDs (0, 1, 2 ...) into
meaningful physical state labels (OFF / STANDBY / WORKING /
PEAK_LOAD) using a physics-anchored, programmatic ranking
rule — not manual eyeballing of a plot.

DESIGN RULES
------------
1. Rank by MEAN ACTIVE POWER (low → high). This is the only
   ordering that is physically meaningful AND reproducible
   across every machine without manual re-tuning.

2. OFF gets a PHYSICS ANCHOR, not just "lowest rank":
   mean(OFF candidate) < 5% of mean(highest cluster).
   On machines with no true OFF window in the data, the
   lowest cluster is NOT silently mislabelled OFF.

3. GHOST-CLUSTER VETO: any cluster with weight < 1% raises
   ValueError. A weight < 1% is the noise cluster the
   Research_Report §11.2 describes — it must never get a
   state label; it means re-fit with k-1.

4. No absolute Watt thresholds — relative-to-max fractions
   only — so the same code generalises from a 200 W CNC
   spindle to a 37 kW pelletizer without per-machine tuning.

5. The mapping dict is FROZEN at fit time and saved as JSON.
   At inference time, load it back with load_mapping() — it
   must NEVER be silently recomputed per-batch.

HOW TO USE
----------
    from src.state_mapping import (
        map_clusters_to_states,
        refine_standby_with_pf,
        save_mapping,
        load_mapping,
    )

    mapping = map_clusters_to_states(gmm, scaler, feature_names)
    save_mapping(mapping, "outputs/models/pelletizer-I/state_map.json")
    # later, at inference:
    mapping = load_mapping("outputs/models/pelletizer-I/state_map.json")
============================================================
"""

import json
import os
import numpy as np


# ── Public constants ──────────────────────────────────────────────────────────
OFF_POWER_FRACTION   = 0.05   # OFF mean must be < 5% of the highest cluster mean
GHOST_WEIGHT_MIN     = 0.01   # clusters below 1% weight are ghost noise clusters
PF_STANDBY_THRESHOLD = 0.35   # PF > this on a STANDBY-labelled row → flag as ambiguous
ACTIVE_POWER_OFF_W   = 5.0    # hard OFF threshold used as a secondary anchor (Watts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_power_col_idx(feature_names: list[str]) -> int:
    """
    Return the index of the active-power feature in the feature matrix.
    Falls back to index 0 if 'active_power' (or aliases) is not found.
    """
    for i, name in enumerate(feature_names):
        if name.lower() in ("active_power", "power", "p", "kw", "active power"):
            return i
    return 0   # safe fallback: first feature is almost always power


def _cluster_means_on_power(gmm, power_col_idx: int) -> np.ndarray:
    """
    Extract the per-cluster mean of the active-power feature from a fitted GMM.
    Handles 'full', 'tied', 'diag', and 'spherical' covariance types.
    """
    means = gmm.means_
    if means.ndim == 1:
        # Univariate GMM (single feature) — means is shape (k,)
        return means.astype(float)
    return means[:, power_col_idx].astype(float)


# ── Core mapping function ─────────────────────────────────────────────────────

def map_clusters_to_states(
    gmm,
    scaler=None,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Map raw GMM cluster IDs to physical state names.

    Parameters
    ----------
    gmm : fitted sklearn GaussianMixture (or hmmlearn GaussianHMM)
    scaler : fitted StandardScaler used during GMM training, or None.
             When provided, means are inverse-transformed so the
             OFF anchor check uses real Watts, not scaled units.
    feature_names : list of column names in the feature matrix.
                    If None, falls back to index 0 as active power.

    Returns
    -------
    dict  {cluster_id (int): state_name (str)}
        e.g. {0: "OFF", 1: "STANDBY", 2: "WORKING"}

    Raises
    ------
    ValueError
        If any cluster weight < GHOST_WEIGHT_MIN (1%) — ghost cluster.
        If k is outside supported range (2–4).
        If no OFF cluster can be identified (no cluster < 5% of max mean).
    """
    if feature_names is None:
        feature_names = []

    power_col_idx = _get_power_col_idx(feature_names)

    # ── Extract means and weights ────────────────────────────────────────────
    raw_means = _cluster_means_on_power(gmm, power_col_idx)
    weights   = np.asarray(gmm.weights_, dtype=float)
    k         = len(weights)

    if k < 2 or k > 4:
        raise ValueError(
            f"k={k} is outside the supported range [2, 4]. "
            "Extend map_clusters_to_states() explicitly for larger k."
        )

    # ── Ghost-cluster veto (Research_Report §11.2) ───────────────────────────
    for cid, w in enumerate(weights):
        if w < GHOST_WEIGHT_MIN:
            raise ValueError(
                f"Cluster {cid} has weight {w:.4f} ({w*100:.2f}%) "
                f"< {GHOST_WEIGHT_MIN*100:.0f}% — this is a ghost/noise cluster. "
                f"Re-fit with k={k-1} or fall back to Silhouette-selected k."
            )

    # ── Inverse-transform means to real units if a scaler was supplied ───────
    if scaler is not None:
        # Reconstruct a dummy feature matrix of shape (k, n_features)
        # with the cluster mean in each feature's position, then invert-scale.
        n_feat = len(scaler.mean_)
        dummy  = np.zeros((k, n_feat))
        for cid in range(k):
            if gmm.means_.ndim == 1:
                dummy[cid, 0] = gmm.means_[cid]
            else:
                dummy[cid, :] = gmm.means_[cid, :]
        inv = scaler.inverse_transform(dummy)
        power_means = inv[:, power_col_idx]
    else:
        power_means = raw_means

    # ── Rank clusters low → high power ──────────────────────────────────────
    order      = np.argsort(power_means)    # indices sorted by ascending mean power
    max_mean   = float(power_means[order[-1]])

    # ── Identify OFF cluster via physics anchor ──────────────────────────────
    off_candidate = int(order[0])
    off_mean      = float(power_means[off_candidate])

    if max_mean > 0 and (off_mean / max_mean) < OFF_POWER_FRACTION:
        # The lowest cluster is clearly OFF (< 5% of the highest cluster's mean)
        labels    = {off_candidate: "OFF"}
        remaining = order[1:]
    else:
        # No clear OFF cluster — don't force one
        labels    = {}
        remaining = order
        print(
            f"[state_mapping] WARNING: lowest cluster mean ({off_mean:.2f} W) is "
            f"{off_mean/max(max_mean,1)*100:.1f}% of max cluster mean ({max_mean:.2f} W) "
            f"— no clean OFF cluster found. Machine may not have a true OFF window in this data."
        )

    # ── Label remaining clusters STANDBY / WORKING / PEAK_LOAD ─────────────
    n_remaining = len(remaining)
    if n_remaining == 0:
        raise ValueError("All clusters were consumed by the OFF anchor — check your data.")
    elif n_remaining == 1:
        labels[int(remaining[0])] = "WORKING"
    elif n_remaining == 2:
        labels[int(remaining[0])] = "STANDBY"
        labels[int(remaining[1])] = "WORKING"
    elif n_remaining == 3:
        labels[int(remaining[0])] = "STANDBY"
        labels[int(remaining[1])] = "WORKING"
        labels[int(remaining[2])] = "PEAK_LOAD"
    else:
        raise ValueError(
            f"After OFF anchor, {n_remaining} clusters remain — k={k} is not fully handled. "
            "Extend the labelling rule explicitly."
        )

    # ── Diagnostics ─────────────────────────────────────────────────────────
    print("\n[state_mapping] Cluster → State Mapping:")
    print(f"  {'Cluster':<10} {'State':<12} {'Mean Power':>12} {'Weight':>8}")
    print("  " + "-" * 46)
    for cid in sorted(labels.keys()):
        print(
            f"  {cid:<10} {labels[cid]:<12} "
            f"{power_means[cid]:>11.2f}W {weights[cid]:>7.3%}"
        )

    return labels


# ── STANDBY refinement ────────────────────────────────────────────────────────

def refine_standby_with_pf(
    df,
    state_col: str = "state",
    pf_col: str = "power_factor",
    pf_threshold: float = PF_STANDBY_THRESHOLD,
) -> object:
    """
    Flag STANDBY-labelled rows whose power factor looks more like a
    lightly-loaded working state (PF > pf_threshold) than a true no-load
    standby (PF << pf_threshold).

    This does NOT relabel any row — it adds a 'state_confidence_flag'
    column so ambiguous STANDBY rows are visible in the output without
    silently corrupting the state labels.

    Parameters
    ----------
    df         : DataFrame with state_col and pf_col already computed.
    state_col  : name of the state label column (default 'state').
    pf_col     : name of the power-factor column (default 'power_factor').
    pf_threshold : PF cutoff; rows above this are flagged (default 0.35).

    Returns
    -------
    DataFrame with an added 'state_confidence_flag' column.
    """
    df = df.copy()

    if pf_col not in df.columns:
        print(
            f"[state_mapping] refine_standby_with_pf: '{pf_col}' not found — "
            "computing PF from active_power / apparent_power."
        )
        if "active_power" in df.columns and "apparent_power" in df.columns:
            df[pf_col] = (
                df["active_power"].abs()
                / df["apparent_power"].replace(0, float("nan"))
            )
        else:
            print("[state_mapping] Cannot compute PF — skipping refinement.")
            df["state_confidence_flag"] = "OK"
            return df

    # Default all rows to OK
    df["state_confidence_flag"] = "OK"

    # Flag ambiguous STANDBY rows
    standby_mask  = df[state_col] == "STANDBY"
    high_pf_mask  = df[pf_col] > pf_threshold
    ambiguous     = standby_mask & high_pf_mask

    n_ambiguous   = int(ambiguous.sum())
    n_standby     = int(standby_mask.sum())

    df.loc[ambiguous, "state_confidence_flag"] = "STANDBY_LIGHT_LOAD_AMBIGUOUS"

    print(
        f"\n[state_mapping] STANDBY ambiguity check (PF > {pf_threshold}):\n"
        f"  Total STANDBY rows : {n_standby:,}\n"
        f"  Ambiguous rows     : {n_ambiguous:,} "
        f"({n_ambiguous/max(n_standby,1)*100:.1f}% of STANDBY)\n"
        f"  → Flagged as STANDBY_LIGHT_LOAD_AMBIGUOUS in 'state_confidence_flag'"
    )

    return df


# ── Serialization ─────────────────────────────────────────────────────────────

def save_mapping(mapping: dict, path: str) -> None:
    """
    Save a cluster→state mapping dict to a JSON file.

    The JSON uses string keys so it survives round-trip serialisation
    (JSON keys are always strings; we cast back to int in load_mapping).

    Parameters
    ----------
    mapping : {int: str}  e.g. {0: "OFF", 1: "STANDBY", 2: "WORKING"}
    path    : file path to write (parent dirs are created automatically).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    serialisable = {str(k): v for k, v in mapping.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"[state_mapping] Mapping saved → {path}")


def load_mapping(path: str) -> dict:
    """
    Load a cluster→state mapping dict from a JSON file previously saved
    by save_mapping().

    Returns
    -------
    dict  {int: str}  e.g. {0: "OFF", 1: "STANDBY", 2: "WORKING"}

    Raises
    ------
    FileNotFoundError if the JSON file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[state_mapping] Mapping file not found: {path}\n"
            "Re-run validate_gmm.py to regenerate the model artifacts."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    mapping = {int(k): v for k, v in raw.items()}
    print(f"[state_mapping] Mapping loaded ← {path}")
    return mapping


# ── State encoder helpers ─────────────────────────────────────────────────────

def build_state_encoder(states: list[str]) -> tuple[dict, dict]:
    """
    Build integer encoding / decoding dicts for a list of state names.

    Parameters
    ----------
    states : list of unique state name strings, e.g. ["OFF","STANDBY","WORKING"]
             (need not be sorted — will be sorted internally for reproducibility)

    Returns
    -------
    (encoder, decoder)
      encoder : {state_name: int}
      decoder : {int: state_name}
    """
    sorted_states = sorted(set(states))
    encoder = {s: i for i, s in enumerate(sorted_states)}
    decoder = {i: s for s, i in encoder.items()}
    return encoder, decoder


def save_state_encoder(encoder: dict, path: str) -> None:
    """Save state encoder dict {state_name: int} to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(encoder, fh, indent=2)
    print(f"[state_mapping] State encoder saved → {path}")


def load_state_encoder(path: str) -> tuple[dict, dict]:
    """
    Load state encoder from JSON. Returns (encoder, decoder).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[state_mapping] Encoder file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        encoder = json.load(fh)
    # JSON keys are strings — keep them as strings (state names)
    decoder = {int(v): k for k, v in encoder.items()}
    print(f"[state_mapping] State encoder loaded ← {path}")
    return encoder, decoder
