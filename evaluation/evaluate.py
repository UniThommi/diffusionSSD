#!/usr/bin/env python3
"""
evaluate.py — Quantitative comparison of generated vs simulated
neutron capture detector events.

Metrics
-------
1.  Per-region Wasserstein W₁ (hit-count distributions across events)
2.  Per-voxel Wasserstein W₁ (hit-count distribution per voxel across events)
3.  Spatial W₁ on r/z projections and hit-weighted center-of-gravity
4.  Support-based metrics: IoU, precision, recall, mismatch (per event)
5.  MSE / RMSE on raw voxel hit counts (per event)
6.  Poisson negative log-likelihood (per event)
7.  Jensen-Shannon divergence (population-level and per event)

Usage
-----
    python evaluation/evaluate.py \\
        --sim  /path/to/MLFormatSSD.hdf5 \\
        --gen  /path/to/generated.hdf5 \\
        [--output_dir results/] [--n_events N] [--seed 42]
        [--no_plots] [--chunk_size 10000]

Author: Thomas Buerger (University of Tübingen)
"""

import argparse
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
import matplotlib
matplotlib.use("Agg")  # headless backend for HPC / server environments
import matplotlib.pyplot as plt

# ─── Region prefix constants (from config.toml [regions]) ──────────────────────
_PIT_PREFIX  = "00"
_BOT_PREFIX  = "01"
_TOP_PREFIX  = "99"
# region_matrix column order: [pit=0, bot=1, wall=2, top=3]
_REGION_COLS = [("PIT", 0), ("BOT", 1), ("WALL", 2), ("TOP", 3)]


# =============================================================================
# I/O and Geometry
# =============================================================================

def load_hdf5(path: str, load_target: bool = True) -> dict:
    """
    Load an ML-format HDF5 file.

    Parameters
    ----------
    path         : Path to the HDF5 file.
    load_target  : If True (default), load the full target_matrix into RAM.
                   Set to False for large files where only event_ids,
                   region_matrix, and phi_matrix are needed upfront (e.g. to
                   pair events before streaming target_matrix in batches via
                   the batch loop in main()).

    Returns
    -------
    dict with:
        phi_matrix       : (N, F)    float32 — conditioning features
        target_matrix    : (N, 9583) int32   — voxel hit counts, or None if
                           load_target=False
        region_matrix    : (N, 4)    int32   — [pit, bot, wall, top] hits
        target_columns   : list[str], length 9583 — voxel index strings
        event_ids        : (N, k)    int64   — [run_id, (muon_id,) nc_id], or
                           None if the file predates ID writing.
        event_id_columns : list[str] of length k, or None.
        n_events         : int — total number of events in the file.
    """
    with h5py.File(path, "r") as f:
        phi_matrix    = f["phi_matrix"][:]
        region_matrix = f["region_matrix"][:].astype(np.int32)
        target_columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in f["target_columns"][:]
        ]
        if "event_ids" in f:
            event_ids = f["event_ids"][:]
            event_id_columns = [
                c.decode() if isinstance(c, bytes) else c
                for c in f["event_id_columns"][:]
            ]
        else:
            event_ids        = None
            event_id_columns = None

        n_events = int(region_matrix.shape[0])

        if load_target:
            target_matrix = f["target_matrix"][:].astype(np.int32)
            n_vox = target_matrix.shape[1]
            if n_vox != 9583:
                raise ValueError(
                    f"Expected 9583 voxels, got {n_vox} in '{path}'. "
                    "Ensure this is an ML-format HDF5 from the NC simulation pipeline."
                )
        else:
            target_matrix = None

    return {
        "phi_matrix":        phi_matrix,
        "target_matrix":     target_matrix,
        "region_matrix":     region_matrix,
        "target_columns":    target_columns,
        "event_ids":         event_ids,
        "event_id_columns":  event_id_columns,
        "n_events":          n_events,
    }


def load_target_rows(path: str, row_indices: np.ndarray) -> np.ndarray:
    """
    Load only selected rows from target_matrix in an HDF5 file.

    h5py fancy indexing requires a sorted index array for efficient I/O
    (a single contiguous read pass rather than one seek per row).  We sort
    the requested indices, load, then invert the permutation so the returned
    array aligns with the original (unsorted) row_indices ordering.

    Parameters
    ----------
    path        : Path to the HDF5 file.
    row_indices : 1-D int array of row positions to load (any order, no repeats).

    Returns
    -------
    (len(row_indices), 9583) int32 array — rows in the same order as row_indices.
    """
    sort_order = np.argsort(row_indices)
    sorted_idx = row_indices[sort_order]
    inv_order  = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(len(sort_order))

    with h5py.File(path, "r") as f:
        data = f["target_matrix"][sorted_idx].astype(np.int32)

    return data[inv_order]


def build_voxel_centers(gen_path: str, target_columns: list) -> np.ndarray:
    """
    Build a (9583, 3) float32 array of voxel center coordinates [x, y, z] in mm,
    ordered to match the column axis of target_matrix.

    Reads from the generated HDF5 /voxels/{idx}/center groups written by
    generate.py::write_output_hdf5 from the external voxel JSON geometry file.

    Args
    ----
    gen_path       : Path to the generated HDF5 (contains /voxels/).
    target_columns : Ordered list of 9583 voxel index strings from load_hdf5.
    """
    centers = np.zeros((len(target_columns), 3), dtype=np.float32)
    with h5py.File(gen_path, "r") as f:
        if "voxels" not in f:
            raise KeyError(
                f"'{gen_path}' is missing /voxels/ groups. "
                "Pass the generated file (not a raw simulation file) as --gen."
            )
        vg = f["voxels"]
        for i, key in enumerate(target_columns):
            centers[i] = vg[key]["center"][:]
    return centers


def build_region_masks(target_columns: list) -> Dict[str, np.ndarray]:
    """
    Build boolean index masks for the 4 detector regions using voxel index prefixes.

    Prefix convention (config.toml [regions]):
        "00" → PIT  (1,261 voxels)
        "01" → BOT  (  400 voxels)
        "99" → TOP  (1,528 voxels)
        else → WALL (6,394 voxels)

    Returns
    -------
    dict: {"PIT": bool ndarray(9583,), "BOT": ..., "WALL": ..., "TOP": ...}
    """
    n = len(target_columns)
    masks = {r: np.zeros(n, dtype=bool) for r in ("PIT", "BOT", "WALL", "TOP")}
    for i, key in enumerate(target_columns):
        if key.startswith(_PIT_PREFIX):
            masks["PIT"][i]  = True
        elif key.startswith(_BOT_PREFIX):
            masks["BOT"][i]  = True
        elif key.startswith(_TOP_PREFIX):
            masks["TOP"][i]  = True
        else:
            masks["WALL"][i] = True

    expected = {"PIT": 1261, "BOT": 400, "WALL": 6394, "TOP": 1528}
    for r, exp in expected.items():
        got = int(masks[r].sum())
        if got != exp:
            print(f"  WARNING: Region {r}: expected {exp} voxels, got {got}.")
    return masks


# =============================================================================
# Event Pairing
# =============================================================================

def pair_events(sim: dict, gen: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match simulation events to generated events by (run_id, muon_id, nc_id).

    Both files must contain event_ids / event_id_columns datasets written by
    generate.py.  Raises RuntimeError if either file is missing them.

    Returns
    -------
    sim_idx, gen_idx : int32 arrays of shape (N_matched,) — index pairs.

    Raises
    ------
    RuntimeError    if event_ids is missing from either file.
    ValueError      if fewer than 10 % of gen events are matched.
    """
    if sim["event_ids"] is None or gen["event_ids"] is None:
        missing = []
        if sim["event_ids"] is None:
            missing.append("--sim")
        if gen["event_ids"] is None:
            missing.append("--gen")
        raise RuntimeError(
            f"event_ids missing in: {', '.join(missing)}. "
            "Re-generate the file(s) with a current version of generate.py "
            "that writes run_id and nc_id."
        )
    return _pair_by_ids(sim, gen)


def _pair_by_ids(sim: dict, gen: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Match events by (run_id, muon_id, nc_id) triple."""
    def _key_cols(ids, cols):
        run_idx  = cols.index("run_id")
        muon_idx = cols.index("muon_id")
        nc_idx   = cols.index("nc_id")
        return ids[:, run_idx], ids[:, muon_idx], ids[:, nc_idx]

    sim_run, sim_muon, sim_nc = _key_cols(sim["event_ids"], sim["event_id_columns"])
    gen_run, gen_muon, gen_nc = _key_cols(gen["event_ids"], gen["event_id_columns"])

    # Build lookup: (run_id, muon_id, nc_id) → sim row index (-1 marks collision)
    sim_lookup: dict = {}
    n_collisions = 0
    for i in range(len(sim_run)):
        key = (int(sim_run[i]), int(sim_muon[i]), int(sim_nc[i]))
        if key in sim_lookup:
            sim_lookup[key] = -1
            n_collisions += 1
        else:
            sim_lookup[key] = i

    if n_collisions > 0:
        pct = n_collisions / len(sim_run) * 100
        print(f"  WARNING: {n_collisions} duplicate (run_id, muon_id, nc_id) triples in sim "
              f"({pct:.2f}%) — ambiguous matches excluded.")

    sim_indices, gen_indices = [], []
    n_unmatched = 0
    for j in range(len(gen_run)):
        key = (int(gen_run[j]), int(gen_muon[j]), int(gen_nc[j]))
        s = sim_lookup.get(key, None)
        if s is not None and s != -1:
            sim_indices.append(s)
            gen_indices.append(j)
        else:
            n_unmatched += 1

    n_gen     = len(gen_run)
    n_matched = len(sim_indices)
    if n_matched < 0.1 * n_gen:
        raise ValueError(
            f"Only {n_matched}/{n_gen} gen events matched ({n_matched/n_gen:.1%}) "
            "by (run_id, nc_id). Verify that --sim is the source file used for generation."
        )
    print(f"  Paired {n_matched:,}/{n_gen:,} events by ID "
          f"({n_matched/n_gen:.1%} matched, {n_unmatched} unmatched).")
    return np.array(sim_indices, np.int32), np.array(gen_indices, np.int32)


# =============================================================================
# Metric 1 — Per-Region Wasserstein
# =============================================================================

def wasserstein_region(
    sim_region: np.ndarray,
    gen_region: np.ndarray,
) -> dict:
    """
    1D Wasserstein distance W₁ between per-region hit-count distributions.

    For each region r, the empirical distribution is the vector of per-event
    total hit counts: sim_region[:, r] and gen_region[:, r].

    Operates on (N, 4) region matrices which are small even at large N and
    are loaded fully into RAM before the batch loop.

    Args
    ----
    sim_region, gen_region : (N_matched, 4) int32 — columns [pit, bot, wall, top].
    """
    result = {}
    for r_name, r_idx in _REGION_COLS:
        s = sim_region[:, r_idx].astype(np.float64)
        g = gen_region[:, r_idx].astype(np.float64)
        result[r_name] = {
            "sim_mean":       float(s.mean()),
            "sim_std":        float(s.std()),
            "gen_mean":       float(g.mean()),
            "gen_std":        float(g.std()),
            "wasserstein_1d": float(wasserstein_distance(s, g)),
        }
    s_tot = sim_region.astype(np.float64).sum(axis=1)
    g_tot = gen_region.astype(np.float64).sum(axis=1)
    result["TOTAL_HITS"] = {
        "sim_mean":       float(s_tot.mean()),
        "sim_std":        float(s_tot.std()),
        "gen_mean":       float(g_tot.mean()),
        "gen_std":        float(g_tot.std()),
        "wasserstein_1d": float(wasserstein_distance(s_tot, g_tot)),
    }
    return result


# =============================================================================
# Batch Infrastructure
# =============================================================================

def _find_kmax(
    sim_path:   str,
    gen_path:   str,
    sim_idx:    np.ndarray,
    gen_idx:    np.ndarray,
    chunk_size: int,
) -> int:
    """
    First pass: find the global maximum hit count across all matched sim and gen
    events.  Used to size the per-voxel integer histograms for M2.

    Reads rows in globally sorted order for efficient sequential HDF5 I/O.
    """
    N = len(sim_idx)
    kmax = 0

    sort_s  = np.argsort(sim_idx)
    sort_g  = np.argsort(gen_idx)
    s_sorted = sim_idx[sort_s]
    g_sorted = gen_idx[sort_g]

    with h5py.File(sim_path, "r") as f_sim, h5py.File(gen_path, "r") as f_gen:
        ds_sim = f_sim["target_matrix"]
        ds_gen = f_gen["target_matrix"]
        for start in range(0, N, chunk_size):
            end  = min(start + chunk_size, N)
            kmax = max(kmax, int(ds_sim[s_sorted[start:end]].max()))
            kmax = max(kmax, int(ds_gen[g_sorted[start:end]].max()))

    return kmax


def _init_accumulators(
    n_matched:    int,
    kmax:         int,
    region_masks: Dict[str, np.ndarray],
) -> dict:
    """
    Allocate all arrays that _process_batch fills incrementally.

    Pre-allocating avoids repeated list.append() and enables direct slice
    assignment (acc["arr"][start:end] = chunk_result), keeping memory bounded
    regardless of N.

    Memory layout
    -------------
    Per-voxel histograms : 2 × (9583, kmax+1) int64  — O(V × K), N-independent
    Population totals    : 2 × (9583,) float64        — O(V),   N-independent
    Per-event scalars    : ~20 × (N,) float64          — O(N),   grows with N
    """
    V  = 9583
    K1 = kmax + 1
    return {
        # M2: per-voxel integer hit-count histograms
        "hist_sim":      np.zeros((V, K1), dtype=np.int64),
        "hist_gen":      np.zeros((V, K1), dtype=np.int64),

        # M3 + M7-pop + spatial profiles: population hit totals (shared)
        "tot_sim":       np.zeros(V, dtype=np.float64),
        "tot_gen":       np.zeros(V, dtype=np.float64),

        # M3: per-event center-of-gravity and per-event projection W₁
        "cog_r_sim":     np.full(n_matched, np.nan),
        "cog_r_gen":     np.full(n_matched, np.nan),
        "cog_z_sim":     np.full(n_matched, np.nan),
        "cog_z_gen":     np.full(n_matched, np.nan),
        "ev_tot_sim":    np.zeros(n_matched, dtype=np.float64),
        "ev_tot_gen":    np.zeros(n_matched, dtype=np.float64),
        "w1_z_ev":       np.full(n_matched, np.nan),
        "w1_r_ev":       np.full(n_matched, np.nan),

        # M4: per-event support metrics
        "iou_ev":        np.zeros(n_matched, dtype=np.float64),
        "prec_ev":       np.zeros(n_matched, dtype=np.float64),
        "rec_ev":        np.zeros(n_matched, dtype=np.float64),
        "mismatch_ev":   np.zeros(n_matched, dtype=np.float64),
        "wmismatch_ev":  np.zeros(n_matched, dtype=np.float64),
        "iou_ev_r":      {r: np.zeros(n_matched) for r in region_masks},
        "prec_ev_r":     {r: np.zeros(n_matched) for r in region_masks},
        "rec_ev_r":      {r: np.zeros(n_matched) for r in region_masks},

        # M5: per-event MSE
        "mse_ev":        np.zeros(n_matched, dtype=np.float64),

        # M6: per-event Poisson NLL
        "nll_ev":        np.zeros(n_matched, dtype=np.float64),
        "nll_ev_r":      {r: np.zeros(n_matched) for r in region_masks},
        "penalised_sum": 0,   # total penalised (voxel × event) count; ÷N at finalize

        # M7: per-event JS divergence
        "js_ev":         np.zeros(n_matched, dtype=np.float64),
        "js_ev_r":       {r: np.zeros(n_matched) for r in region_masks},
    }


def _process_batch(
    acc:          dict,
    sc:           np.ndarray,
    gc:           np.ndarray,
    chunk_start:  int,
    r_coords:     np.ndarray,
    z_coords:     np.ndarray,
    region_masks: Dict[str, np.ndarray],
) -> None:
    """
    Accumulate statistics for one chunk of matched events into *acc* (in place).

    Parameters
    ----------
    acc         : Accumulator dict from _init_accumulators().
    sc, gc      : (C, 9583) int32 — sim/gen target_matrix rows for this chunk.
    chunk_start : Index of the first event of this chunk in acc's per-event arrays.
    r_coords    : (9583,) float64 — radial voxel coordinates [mm].
    z_coords    : (9583,) float64 — z voxel coordinates [mm].
    region_masks: Dict of bool masks (9583,) from build_region_masks().
    """
    C   = sc.shape[0]
    V   = sc.shape[1]
    K   = acc["hist_sim"].shape[1] - 1
    sl  = slice(chunk_start, chunk_start + C)

    # ── M2: histogram accumulation ────────────────────────────────────────────
    # Map each (voxel v, hit_count k) pair to a unique linear index v*(K+1)+k,
    # then use a single np.bincount call over all C×V pairs per chunk.
    # This avoids a Python loop over 9583 voxels per batch.
    vox_offsets = np.arange(V, dtype=np.int64) * (K + 1)           # (V,)
    sc_c = sc.clip(0, K).astype(np.int64)                           # (C, V)
    gc_c = gc.clip(0, K).astype(np.int64)
    acc["hist_sim"] += np.bincount(
        (sc_c + vox_offsets).ravel(), minlength=V * (K + 1)
    ).reshape(V, K + 1)
    acc["hist_gen"] += np.bincount(
        (gc_c + vox_offsets).ravel(), minlength=V * (K + 1)
    ).reshape(V, K + 1)

    # ── Float arrays shared by M3–M7 ─────────────────────────────────────────
    sf = sc.astype(np.float64)                                       # (C, V)
    gf = gc.astype(np.float64)

    # ── M3 + shared: population hit totals ───────────────────────────────────
    # Also used for M7 population JS and spatial profiles — same arrays.
    acc["tot_sim"] += sf.sum(axis=0)
    acc["tot_gen"] += gf.sum(axis=0)

    # ── M3: center-of-gravity and per-event projection W₁ ────────────────────
    ev_tot_s = sf.sum(axis=1)                                        # (C,)
    ev_tot_g = gf.sum(axis=1)
    acc["ev_tot_sim"][sl] = ev_tot_s
    acc["ev_tot_gen"][sl] = ev_tot_g

    with np.errstate(invalid="ignore", divide="ignore"):
        cog_r_s = np.where(ev_tot_s > 0, sf @ r_coords / ev_tot_s, np.nan)
        cog_r_g = np.where(ev_tot_g > 0, gf @ r_coords / ev_tot_g, np.nan)
        cog_z_s = np.where(ev_tot_s > 0, sf @ z_coords / ev_tot_s, np.nan)
        cog_z_g = np.where(ev_tot_g > 0, gf @ z_coords / ev_tot_g, np.nan)
    acc["cog_r_sim"][sl] = cog_r_s
    acc["cog_r_gen"][sl] = cog_r_g
    acc["cog_z_sim"][sl] = cog_z_s
    acc["cog_z_gen"][sl] = cog_z_g

    # Per-event projection W₁: scipy call per event — O(V log V) each.
    for i in range(C):
        if ev_tot_s[i] > 0 and ev_tot_g[i] > 0:
            gi = chunk_start + i
            acc["w1_z_ev"][gi] = wasserstein_distance(
                z_coords, z_coords, u_weights=sf[i], v_weights=gf[i])
            acc["w1_r_ev"][gi] = wasserstein_distance(
                r_coords, r_coords, u_weights=sf[i], v_weights=gf[i])

    # ── M4: support metrics ───────────────────────────────────────────────────
    S        = sc > 0                                                # (C, V) bool
    G        = gc > 0
    inter    = (S & G).sum(axis=1).astype(np.float64)
    union_sz = (S | G).sum(axis=1).astype(np.float64)
    S_size   = S.sum(axis=1).astype(np.float64)
    G_size   = G.sum(axis=1).astype(np.float64)
    sym_diff = (S ^ G).sum(axis=1).astype(np.float64)

    iou      = np.where(union_sz > 0, inter / union_sz, 1.0)
    precision = np.where(G_size > 0, inter / G_size,
                         np.where(S_size == 0, 1.0, 0.0))
    recall   = np.where(S_size > 0, inter / S_size,
                         np.where(G_size == 0, 1.0, 0.0))
    mismatch = sym_diff / np.maximum(1.0, S_size)

    total_int = sf + gf
    w_num     = (total_int * (S ^ G)).sum(axis=1)
    w_den     = total_int.sum(axis=1)
    wmismatch = np.where(w_den > 0, w_num / w_den, 0.0)

    acc["iou_ev"][sl]       = iou
    acc["prec_ev"][sl]      = precision
    acc["rec_ev"][sl]       = recall
    acc["mismatch_ev"][sl]  = mismatch
    acc["wmismatch_ev"][sl] = wmismatch

    for r_name, mask in region_masks.items():
        S_r = S[:, mask];  G_r = G[:, mask]
        i_r = (S_r & G_r).sum(axis=1).astype(np.float64)
        u_r = (S_r | G_r).sum(axis=1).astype(np.float64)
        s_r = S_r.sum(axis=1).astype(np.float64)
        g_r = G_r.sum(axis=1).astype(np.float64)
        acc["iou_ev_r"][r_name][sl]  = np.where(u_r > 0, i_r / u_r, 1.0)
        acc["prec_ev_r"][r_name][sl] = np.where(g_r > 0, i_r / g_r,
                                                 np.where(s_r == 0, 1.0, 0.0))
        acc["rec_ev_r"][r_name][sl]  = np.where(s_r > 0, i_r / s_r,
                                                 np.where(g_r == 0, 1.0, 0.0))

    # ── M5: MSE ───────────────────────────────────────────────────────────────
    acc["mse_ev"][sl] = ((sf - gf) ** 2).mean(axis=1)

    # ── M6: Poisson NLL ───────────────────────────────────────────────────────
    eps     = 1e-6
    log_gen = np.log(gf + eps)                                       # (C, V)
    nll_mat = gf - sf * log_gen                                      # (C, V)
    acc["nll_ev"][sl]     = nll_mat.sum(axis=1)
    acc["penalised_sum"] += int(((gc == 0) & (sc > 0)).sum())
    for r_name, mask in region_masks.items():
        acc["nll_ev_r"][r_name][sl] = nll_mat[:, mask].sum(axis=1)

    # ── M7: Jensen-Shannon divergence ─────────────────────────────────────────
    s_sum = sf.sum(axis=1, keepdims=True).clip(min=1)
    g_sum = gf.sum(axis=1, keepdims=True).clip(min=1)
    sn    = sf / s_sum;  gn = gf / g_sum
    Mn    = 0.5 * (sn + gn)
    acc["js_ev"][sl] = (0.5 * rel_entr(sn, Mn).sum(axis=1)
                      + 0.5 * rel_entr(gn, Mn).sum(axis=1))
    for r_name, mask in region_masks.items():
        sr   = sf[:, mask];  gr = gf[:, mask]
        sr_n = sr / sr.sum(axis=1, keepdims=True).clip(min=1)
        gr_n = gr / gr.sum(axis=1, keepdims=True).clip(min=1)
        Mr   = 0.5 * (sr_n + gr_n)
        acc["js_ev_r"][r_name][sl] = (0.5 * rel_entr(sr_n, Mr).sum(axis=1)
                                    + 0.5 * rel_entr(gr_n, Mr).sum(axis=1))


# =============================================================================
# Metric 2 — Per-Voxel Wasserstein (histogram-based, exact for integer data)
# =============================================================================

def wasserstein_per_voxel(acc: dict) -> dict:
    """
    Per-voxel W₁ from accumulated integer hit-count histograms.

    For integer-valued data on {0, 1, ..., K} the 1-D Wasserstein distance has
    the exact closed form:

        W₁(P, Q) = Σ_{k=0}^{K-1} |F_P(k) − F_Q(k)|

    where F_P(k) = P(X ≤ k) is the empirical CDF.  Computing W₁ from the
    accumulated count histograms is algebraically identical to calling
    scipy.wasserstein_distance on the raw event columns — no approximation.

    Memory is O(V × K) for the histograms, independent of N (number of events).

    Args
    ----
    acc : Accumulator from _init_accumulators() after the batch loop.
    """
    hist_s = acc["hist_sim"].astype(np.float64)     # (V, K+1)
    hist_g = acc["hist_gen"].astype(np.float64)

    n_s = hist_s.sum(axis=1)                        # total event count per voxel
    n_g = hist_g.sum(axis=1)

    # Normalize to probability mass functions
    n_s_safe = np.where(n_s > 0, n_s, 1.0)[:, np.newaxis]
    n_g_safe = np.where(n_g > 0, n_g, 1.0)[:, np.newaxis]
    p_s = hist_s / n_s_safe                         # (V, K+1)
    p_g = hist_g / n_g_safe

    # CDFs: F[v, k] = P(hits ≤ k | voxel v)
    F_s = np.cumsum(p_s, axis=1)                    # (V, K+1)
    F_g = np.cumsum(p_g, axis=1)

    # W₁ = Σ_{k=0}^{K-1} |F_s − F_g|; last column is identically 1 for both
    w1 = np.abs(F_s - F_g)[:, :-1].sum(axis=1)     # (V,)
    # Voxels with zero hits in both are trivially identical
    w1 = np.where((n_s == 0) & (n_g == 0), 0.0, w1)

    # Per-voxel mean hits and hit rates from histograms
    k_vals   = np.arange(p_s.shape[1], dtype=np.float64)
    mean_sim = (p_s * k_vals).sum(axis=1)
    mean_gen = (p_g * k_vals).sum(axis=1)
    rate_sim = 1.0 - p_s[:, 0]
    rate_gen = 1.0 - p_g[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(mean_sim > 0, mean_gen / mean_sim, np.nan)

    return {
        "_mean_hits_sim":    mean_sim,
        "_mean_hits_gen":    mean_gen,
        "_hit_rate_sim":     rate_sim,
        "_hit_rate_gen":     rate_gen,
        "_ratio_gen_to_sim": ratio,
        "_w1_per_voxel":     w1,
        "w1_mean":           float(w1.mean()),
        "w1_median":         float(np.median(w1)),
        "w1_max":            float(w1.max()),
        "w1_p95":            float(np.percentile(w1, 95)),
        "n_zero_voxels_sim": int((mean_sim == 0).sum()),
        "n_zero_voxels_gen": int((mean_gen == 0).sum()),
    }


# =============================================================================
# Metric 3 — Spatial Wasserstein
# =============================================================================

def wasserstein_spatial(
    acc:           dict,
    voxel_centers: np.ndarray,
) -> dict:
    """
    Wasserstein distances on 1-D spatial projections of the hit pattern.

    Three levels of analysis:
    (A) Population-level r/z marginals: total hits per voxel across all events
        (from acc["tot_sim"] / acc["tot_gen"]).
    (B) Per-event hit-weighted center-of-gravity (r, z), accumulated per batch.
    (C) Per-event projection W₁, accumulated per batch.

    Args
    ----
    acc           : Accumulator from _init_accumulators() after the batch loop.
    voxel_centers : (9583, 3) float32 — [x, y, z] in mm.
    """
    x, y, z = (voxel_centers[:, k].astype(np.float64) for k in range(3))
    r = np.sqrt(x**2 + y**2)

    tot_sim = acc["tot_sim"]
    tot_gen = acc["tot_gen"]

    # (A) Population marginals
    w1_r_pop = float(wasserstein_distance(r, r, u_weights=tot_sim, v_weights=tot_gen))
    w1_z_pop = float(wasserstein_distance(z, z, u_weights=tot_sim, v_weights=tot_gen))

    # (B) Per-event CoG
    cog_r_sim = acc["cog_r_sim"]
    cog_r_gen = acc["cog_r_gen"]
    cog_z_sim = acc["cog_z_sim"]
    cog_z_gen = acc["cog_z_gen"]
    valid_r   = ~(np.isnan(cog_r_sim) | np.isnan(cog_r_gen))
    valid_z   = ~(np.isnan(cog_z_sim) | np.isnan(cog_z_gen))
    w1_cog_r  = float(wasserstein_distance(cog_r_sim[valid_r], cog_r_gen[valid_r]))
    w1_cog_z  = float(wasserstein_distance(cog_z_sim[valid_z], cog_z_gen[valid_z]))

    # (C) Per-event projection W₁
    w1_z_ev  = acc["w1_z_ev"]
    w1_r_ev  = acc["w1_r_ev"]
    valid_ev = ~np.isnan(w1_z_ev)
    arr_z    = w1_z_ev[valid_ev]
    arr_r    = w1_r_ev[valid_ev]
    N        = len(w1_z_ev)

    return {
        "_r_voxel":    r,
        "_z_voxel":    z,
        "_cog_r_sim":  cog_r_sim,
        "_cog_r_gen":  cog_r_gen,
        "_cog_z_sim":  cog_z_sim,
        "_cog_z_gen":  cog_z_gen,
        "w1_r_population":      w1_r_pop,
        "w1_z_population":      w1_z_pop,
        "w1_cog_r":             w1_cog_r,
        "w1_cog_z":             w1_cog_z,
        "w1_r_per_event_mean":  float(arr_r.mean()) if len(arr_r) else float("nan"),
        "w1_r_per_event_std":   float(arr_r.std())  if len(arr_r) else float("nan"),
        "w1_z_per_event_mean":  float(arr_z.mean()) if len(arr_z) else float("nan"),
        "w1_z_per_event_std":   float(arr_z.std())  if len(arr_z) else float("nan"),
        "n_skipped_per_event":  int(N - int(valid_ev.sum())),
    }


# =============================================================================
# Metric 4 — Support-Based (IoU, Precision, Recall, Mismatch)
# =============================================================================

def support_metrics_per_event(
    acc:          dict,
    region_masks: Dict[str, np.ndarray],
) -> dict:
    """
    Binary voxel-support overlap metrics, finalized from accumulator arrays.

    Args
    ----
    acc          : Accumulator from _init_accumulators() after the batch loop.
    region_masks : Dict from build_region_masks().
    """
    iou       = acc["iou_ev"]
    precision = acc["prec_ev"]
    recall    = acc["rec_ev"]
    mismatch  = acc["mismatch_ev"]
    wmismatch = acc["wmismatch_ev"]

    def _s(arr: np.ndarray) -> dict:
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "median": float(np.median(arr)),
            "p95":    float(np.percentile(arr, 95)),
        }

    per_region: dict = {}
    for r_name in region_masks:
        per_region[r_name] = {
            "iou_mean":       float(acc["iou_ev_r"][r_name].mean()),
            "iou_std":        float(acc["iou_ev_r"][r_name].std()),
            "precision_mean": float(acc["prec_ev_r"][r_name].mean()),
            "recall_mean":    float(acc["rec_ev_r"][r_name].mean()),
        }

    return {
        "_iou_per_event":       iou,
        "_precision_per_event": precision,
        "_recall_per_event":    recall,
        "iou":               _s(iou),
        "precision":         _s(precision),
        "recall":            _s(recall),
        "mismatch":          _s(mismatch),
        "weighted_mismatch": _s(wmismatch),
        "per_region":        per_region,
    }


# =============================================================================
# Metric 5 — MSE / RMSE
# =============================================================================

def mse_per_event(acc: dict, n_matched: int) -> dict:
    """
    L2 / Mean Squared Error on raw voxel hit counts, finalized from accumulator.

    Population MSE is derived from the accumulated per-voxel hit totals:
        pop_MSE = (1/V) Σ_i (tot_sim[i]/N − tot_gen[i]/N)²

    Args
    ----
    acc       : Accumulator from _init_accumulators() after the batch loop.
    n_matched : Number of matched events (= N).
    """
    mse  = acc["mse_ev"]
    rmse = np.sqrt(mse)

    mean_s   = acc["tot_sim"] / n_matched
    mean_g   = acc["tot_gen"] / n_matched
    pop_mse  = float(((mean_s - mean_g) ** 2).mean())

    return {
        "_mse_per_event":  mse,
        "_rmse_per_event": rmse,
        "mse_mean":        float(mse.mean()),
        "mse_std":         float(mse.std()),
        "mse_median":      float(np.median(mse)),
        "rmse_mean":       float(rmse.mean()),
        "rmse_std":        float(rmse.std()),
        "rmse_median":     float(np.median(rmse)),
        "population_mse":  pop_mse,
        "population_rmse": float(np.sqrt(pop_mse)),
    }


# =============================================================================
# Metric 6 — Poisson Negative Log-Likelihood
# =============================================================================

def poisson_nll_per_event(
    acc:          dict,
    n_matched:    int,
    region_masks: Dict[str, np.ndarray],
) -> dict:
    """
    Poisson NLL treating gen[e,i] as rate λ and sim[e,i] as observed count k,
    finalized from accumulator arrays.

    Args
    ----
    acc          : Accumulator from _init_accumulators() after the batch loop.
    n_matched    : Number of matched events.
    region_masks : Dict from build_region_masks().
    """
    nll_ev   = acc["nll_ev"]
    mean_pen = float(acc["penalised_sum"]) / n_matched

    per_region: dict = {}
    for r_name in region_masks:
        nll_r = acc["nll_ev_r"][r_name]
        per_region[r_name] = {
            "mean":   float(nll_r.mean()),
            "std":    float(nll_r.std()),
            "median": float(np.median(nll_r)),
        }

    return {
        "_nll_per_event":                    nll_ev,
        "mean":                              float(nll_ev.mean()),
        "std":                               float(nll_ev.std()),
        "median":                            float(np.median(nll_ev)),
        "p95":                               float(np.percentile(nll_ev, 95)),
        "mean_penalised_voxels_per_event":   mean_pen,
        "per_region": per_region,
    }


# =============================================================================
# Metric 7 — Jensen-Shannon Divergence
# =============================================================================

def js_divergence_population(acc: dict) -> dict:
    """
    Population-level JS divergence from accumulated per-voxel hit totals.

    Args
    ----
    acc : Accumulator from _init_accumulators() after the batch loop.
    """
    tot_sim = acc["tot_sim"]
    tot_gen = acc["tot_gen"]
    p_sim   = tot_sim / tot_sim.sum()
    p_gen   = tot_gen / tot_gen.sum()
    M       = 0.5 * (p_sim + p_gen)

    js_val = float(0.5 * rel_entr(p_sim, M).sum() + 0.5 * rel_entr(p_gen, M).sum())

    kl_sg          = rel_entr(p_sim, p_gen)
    kl_gs          = rel_entr(p_gen, p_sim)
    kl_sg_finite   = bool(np.all(np.isfinite(kl_sg)))
    kl_gs_finite   = bool(np.all(np.isfinite(kl_gs)))

    return {
        "_p_sim": p_sim,
        "_p_gen": p_gen,
        "js_divergence":          js_val,
        "kl_sim_gen":             float(kl_sg.sum()) if kl_sg_finite else float("inf"),
        "kl_sim_gen_is_finite":   kl_sg_finite,
        "kl_gen_sim":             float(kl_gs.sum()) if kl_gs_finite else float("inf"),
        "kl_gen_sim_is_finite":   kl_gs_finite,
        "n_zero_voxels_sim":      int((tot_sim == 0).sum()),
        "n_zero_voxels_gen":      int((tot_gen == 0).sum()),
    }


def js_divergence_per_event(
    acc:          dict,
    region_masks: Dict[str, np.ndarray],
) -> dict:
    """
    Per-event JS divergence, finalized from accumulator arrays.

    Args
    ----
    acc          : Accumulator from _init_accumulators() after the batch loop.
    region_masks : Dict from build_region_masks().
    """
    js_ev = acc["js_ev"]

    per_region: dict = {}
    for r_name in region_masks:
        arr = acc["js_ev_r"][r_name]
        per_region[r_name] = {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "median": float(np.median(arr)),
        }

    return {
        "_js_per_event": js_ev,
        "mean":          float(js_ev.mean()),
        "std":           float(js_ev.std()),
        "median":        float(np.median(js_ev)),
        "p95":           float(np.percentile(js_ev, 95)),
        "per_region":    per_region,
    }


# =============================================================================
# Spatial Profiles (for plotting only)
# =============================================================================

def _spatial_profiles(
    acc:           dict,
    voxel_centers: np.ndarray,
    n_bins_r:      int = 50,
    n_bins_z:      int = 80,
) -> dict:
    """
    Hit-count-weighted radial and z-axis probability density histograms.
    Uses the population hit totals from acc (same arrays as M3/M7-pop).

    Args
    ----
    acc           : Accumulator from _init_accumulators() after the batch loop.
    voxel_centers : (9583, 3) float32 — [x, y, z] in mm.
    """
    x, y, z = (voxel_centers[:, k].astype(np.float64) for k in range(3))
    r = np.sqrt(x**2 + y**2)

    w_sim = acc["tot_sim"]
    w_gen = acc["tot_gen"]

    def _hist1d(coord, n_bins):
        bins = np.linspace(coord.min(), coord.max(), n_bins + 1)
        hs, _ = np.histogram(coord, bins=bins, weights=w_sim)
        hg, _ = np.histogram(coord, bins=bins, weights=w_gen)
        bw    = bins[1] - bins[0]
        hs    = hs / (hs.sum() * bw + 1e-30)
        hg    = hg / (hg.sum() * bw + 1e-30)
        return 0.5 * (bins[:-1] + bins[1:]), hs, hg

    r_ctrs, r_sim_h, r_gen_h = _hist1d(r, n_bins_r)
    z_ctrs, z_sim_h, z_gen_h = _hist1d(z, n_bins_z)

    return {
        "r_bin_centers": r_ctrs,
        "z_bin_centers": z_ctrs,
        "r_hist_sim":    r_sim_h,
        "r_hist_gen":    r_gen_h,
        "z_hist_sim":    z_sim_h,
        "z_hist_gen":    z_gen_h,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_all(metrics: dict, output_dir: str) -> None:
    """Save all diagnostic plots to {output_dir}/plots/."""
    pd = Path(output_dir) / "plots"
    pd.mkdir(parents=True, exist_ok=True)
    _plot_region_hits(metrics, pd)
    _plot_voxel_mean_hits(metrics, pd)
    _plot_spatial_r_profile(metrics, pd)
    _plot_spatial_z_profile(metrics, pd)
    _plot_per_event_js(metrics, pd)
    _plot_per_event_mse(metrics, pd)
    _plot_per_event_nll(metrics, pd)
    _plot_support_iou(metrics, pd)
    _plot_support_precision_recall(metrics, pd)
    _plot_wasserstein_summary(metrics, pd)
    print(f"  Plots saved to {pd}/")


def _plot_region_hits(metrics: dict, pd: Path) -> None:
    sim_region  = metrics["_sim_region"]
    gen_region  = metrics["_gen_region"]
    region_data = metrics["region_hits"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    panels = [("PIT", 0), ("BOT", 1), ("WALL", 2), ("TOP", 3)]

    for ax_idx, (r_name, r_idx) in enumerate(panels):
        ax = axes[ax_idx]
        s  = sim_region[:, r_idx]
        g  = gen_region[:, r_idx]
        hi = max(int(max(s.max(), g.max())) + 1, 2)
        bins = np.arange(0, hi + 1)
        ax.hist(s, bins=bins, density=True, alpha=0.6, label="sim", color="steelblue")
        ax.hist(g, bins=bins, density=True, alpha=0.6, label="gen", color="darkorange")
        rd = region_data[r_name]
        ax.set_title(f"{r_name}  W₁={rd['wasserstein_1d']:.2f}")
        ax.set_xlabel("Hits per event"); ax.set_ylabel("Density"); ax.legend(fontsize=8)

    ax = axes[4]
    s_tot = sim_region.sum(axis=1)
    g_tot = gen_region.sum(axis=1)
    hi    = max(int(max(s_tot.max(), g_tot.max())) + 1, 2)
    bins  = np.linspace(0, hi, 60)
    ax.hist(s_tot, bins=bins, density=True, alpha=0.6, label="sim", color="steelblue")
    ax.hist(g_tot, bins=bins, density=True, alpha=0.6, label="gen", color="darkorange")
    rd = region_data["TOTAL_HITS"]
    ax.set_title(f"TOTAL HITS  W₁={rd['wasserstein_1d']:.2f}")
    ax.set_xlabel("Total hits per event"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    axes[5].set_visible(False)

    fig.suptitle("Per-Region Hit Distributions (sim vs generated)", fontsize=12)
    fig.tight_layout()
    fig.savefig(pd / "region_hits.png", dpi=150)
    plt.close(fig)


def _plot_voxel_mean_hits(metrics: dict, pd: Path) -> None:
    vm  = metrics["_per_voxel"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    idx = np.arange(len(vm["_mean_hits_sim"]))
    axes[0].scatter(idx, vm["_mean_hits_sim"], s=0.3, alpha=0.5, color="steelblue")
    axes[0].set_title("Mean hits/voxel (sim)")
    axes[0].set_xlabel("Voxel index"); axes[0].set_ylabel("Mean hits")
    axes[0].set_yscale("symlog", linthresh=0.1)

    axes[1].scatter(idx, vm["_mean_hits_gen"], s=0.3, alpha=0.5, color="darkorange")
    axes[1].set_title("Mean hits/voxel (gen)")
    axes[1].set_xlabel("Voxel index")
    axes[1].set_yscale("symlog", linthresh=0.1)

    ratio = vm["_ratio_gen_to_sim"]
    valid = np.isfinite(ratio)
    sc    = axes[2].scatter(idx[valid], ratio[valid], s=0.3, alpha=0.5,
                            c=ratio[valid], cmap="RdBu_r", vmin=0.5, vmax=2.0)
    plt.colorbar(sc, ax=axes[2], label="gen/sim")
    axes[2].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[2].set_title("gen/sim hit ratio per voxel")
    axes[2].set_xlabel("Voxel index")

    fig.tight_layout()
    fig.savefig(pd / "voxel_mean_hits.png", dpi=150)
    plt.close(fig)


def _plot_spatial_r_profile(metrics: dict, pd: Path) -> None:
    sp = metrics["_spatial_profiles"]
    ws = metrics["spatial"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(sp["r_bin_centers"], sp["r_hist_sim"], where="mid",
            color="steelblue", label="sim", lw=1.5)
    ax.step(sp["r_bin_centers"], sp["r_hist_gen"], where="mid",
            color="darkorange", label="gen", lw=1.5, ls="--")
    ax.set_xlabel("r [mm]"); ax.set_ylabel("Hit density [1/mm]")
    ax.set_title("Radial hit profile")
    ax.text(0.98, 0.95, f"W₁(pop.) = {ws['w1_r_population']:.1f} mm",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pd / "spatial_r_profile.png", dpi=150)
    plt.close(fig)


def _plot_spatial_z_profile(metrics: dict, pd: Path) -> None:
    sp = metrics["_spatial_profiles"]
    ws = metrics["spatial"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(sp["z_bin_centers"], sp["z_hist_sim"], where="mid",
            color="steelblue", label="sim", lw=1.5)
    ax.step(sp["z_bin_centers"], sp["z_hist_gen"], where="mid",
            color="darkorange", label="gen", lw=1.5, ls="--")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("Hit density [1/mm]")
    ax.set_title("Longitudinal hit profile (z-axis)")
    ax.text(0.98, 0.95, f"W₁(pop.) = {ws['w1_z_population']:.1f} mm",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pd / "spatial_z_profile.png", dpi=150)
    plt.close(fig)


def _hist_plot(ax, arr, color, n_bins=60, label="", stats=None):
    ax.hist(arr, bins=n_bins, density=True, color=color, alpha=0.75, edgecolor="none",
            label=label)
    if stats:
        ax.axvline(stats.get("mean",   float("nan")), color="black", ls="-",  lw=1.2,
                   label=f"mean={stats['mean']:.4g}")
        ax.axvline(stats.get("median", float("nan")), color="black", ls="--", lw=1.2,
                   label=f"median={stats['median']:.4g}")
        if "p95" in stats:
            ax.axvline(stats["p95"], color="gray", ls=":", lw=1.0,
                       label=f"p95={stats['p95']:.4g}")
    ax.legend(fontsize=8)


def _plot_per_event_js(metrics: dict, pd: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist_plot(ax, metrics["_js_per_event_arr"], "mediumpurple",
               stats=metrics["js_per_event"])
    ax.set_xlabel("JS divergence per event (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Per-event Jensen-Shannon Divergence (sim ↔ gen)")
    fig.tight_layout()
    fig.savefig(pd / "per_event_js.png", dpi=150)
    plt.close(fig)


def _plot_per_event_mse(metrics: dict, pd: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    stats = {
        "mean":   metrics["mse_per_event"]["rmse_mean"],
        "median": metrics["mse_per_event"]["rmse_median"],
    }
    _hist_plot(ax, metrics["_rmse_per_event_arr"], "teal", stats=stats)
    ax.set_xlabel("RMSE per event (hit counts)")
    ax.set_ylabel("Density")
    ax.set_title("Per-event RMSE on raw voxel hit counts")
    fig.tight_layout()
    fig.savefig(pd / "per_event_mse.png", dpi=150)
    plt.close(fig)


def _plot_per_event_nll(metrics: dict, pd: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist_plot(ax, metrics["_nll_per_event_arr"], "coral",
               stats=metrics["poisson_nll"])
    ax.set_xlabel("Poisson NLL per event (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Per-event Poisson Negative Log-Likelihood")
    fig.tight_layout()
    fig.savefig(pd / "per_event_poisson_nll.png", dpi=150)
    plt.close(fig)


def _plot_support_iou(metrics: dict, pd: Path) -> None:
    iou_arr = metrics["_iou_per_event_arr"]
    per_reg = metrics["support_metrics"]["per_region"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    _hist_plot(axes[0], iou_arr, "seagreen",
               stats=metrics["support_metrics"]["iou"])
    axes[0].set_xlabel("IoU per event"); axes[0].set_ylabel("Density")
    axes[0].set_title("Per-event IoU (voxel support overlap)")

    regions = list(per_reg.keys())
    means   = [per_reg[r]["iou_mean"] for r in regions]
    stds    = [per_reg[r]["iou_std"]  for r in regions]
    axes[1].bar(regions, means, yerr=stds, color="seagreen", alpha=0.7,
                capsize=4, edgecolor="black")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Mean IoU")
    axes[1].set_title("Mean IoU per detector region")

    fig.tight_layout()
    fig.savefig(pd / "support_iou.png", dpi=150)
    plt.close(fig)


def _plot_support_precision_recall(metrics: dict, pd: Path) -> None:
    prec = metrics["_precision_per_event_arr"]
    rec  = metrics["_recall_per_event_arr"]
    pm   = metrics["support_metrics"]["precision"]["mean"]
    rm   = metrics["support_metrics"]["recall"]["mean"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rec, prec, s=1, alpha=0.3, color="steelblue")
    ax.scatter(rm, pm, s=120, color="red", marker="*", zorder=5,
               label=f"mean: P={pm:.3f}, R={rm:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Per-event Precision vs Recall (voxel support)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(pd / "support_precision_recall.png", dpi=150)
    plt.close(fig)


def _plot_wasserstein_summary(metrics: dict, pd: Path) -> None:
    labels = [
        "W₁ total hits", "W₁ PIT", "W₁ BOT", "W₁ WALL", "W₁ TOP",
        "W₁ r (pop.)", "W₁ z (pop.)", "W₁ CoG-r", "W₁ CoG-z",
        "W₁ voxel (mean)",
    ]
    values = [
        metrics["region_hits"]["TOTAL_HITS"]["wasserstein_1d"],
        metrics["region_hits"]["PIT"]["wasserstein_1d"],
        metrics["region_hits"]["BOT"]["wasserstein_1d"],
        metrics["region_hits"]["WALL"]["wasserstein_1d"],
        metrics["region_hits"]["TOP"]["wasserstein_1d"],
        metrics["spatial"]["w1_r_population"],
        metrics["spatial"]["w1_z_population"],
        metrics["spatial"]["w1_cog_r"],
        metrics["spatial"]["w1_cog_z"],
        metrics["per_voxel"]["w1_mean"],
    ]
    units = ["hits"] * 5 + ["mm"] * 4 + ["hits"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="steelblue", alpha=0.75,
                   edgecolor="black")
    for bar, unit in zip(bars, units[::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                unit, va="center", fontsize=8, color="gray")
    ax.set_xlabel("W₁ distance")
    ax.set_title("Wasserstein Distance Summary (note: mixed units)")
    fig.tight_layout()
    fig.savefig(pd / "wasserstein_summary.png", dpi=150)
    plt.close(fig)


# =============================================================================
# JSON serialization
# =============================================================================

class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars, inf, and nan."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_metrics_json(metrics: dict, path: str) -> None:
    """Serialise scalar metrics to JSON, stripping private '_*' keys."""
    def _strip(d):
        if isinstance(d, dict):
            return {k: _strip(v) for k, v in d.items() if not k.startswith("_")}
        return d

    with open(path, "w") as f:
        json.dump(_strip(metrics), f, indent=2, cls=_NpEncoder)
    print(f"  Metrics → {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative comparison of generated vs simulated NC detector events."
    )
    parser.add_argument("--sim",        required=True,        help="Path to simulation HDF5.")
    parser.add_argument("--gen",        required=True,        help="Path to generated HDF5.")
    parser.add_argument("--output_dir", default="results/",   help="Output directory.")
    parser.add_argument("--n_events",   type=int, default=None,
                        help="Max matched events to analyse (default: all).")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed.")
    parser.add_argument("--no_plots",   action="store_true",  help="Skip plot generation.")
    parser.add_argument("--chunk_size", type=int, default=10_000,
                        help="Events per batch in the streaming pipeline (default: 10000). "
                             "Reduce if RAM is tight; increase if I/O is the bottleneck.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    plt.style.use("seaborn-v0_8-darkgrid")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load metadata ─────────────────────────────────────────────────────────
    # Neither file's target_matrix is loaded here — they are streamed in the
    # batch loop (step 5).  Only event_ids / region_matrix / phi_matrix are
    # needed upfront.
    print("\n[1/9] Loading HDF5 metadata...")
    sim = load_hdf5(args.sim, load_target=False)
    gen = load_hdf5(args.gen, load_target=False)
    print(f"  Sim: {sim['n_events']:,} events | Gen: {gen['n_events']:,} events")
    print(f"  target_matrix deferred — will be streamed in chunks of {args.chunk_size:,}")

    # 2. Geometry ──────────────────────────────────────────────────────────────
    print("\n[2/9] Building voxel geometry and region masks...")
    voxel_centers = build_voxel_centers(args.gen, gen["target_columns"])
    region_masks  = build_region_masks(gen["target_columns"])
    x_vc = voxel_centers[:, 0].astype(np.float64)
    y_vc = voxel_centers[:, 1].astype(np.float64)
    z_vc = voxel_centers[:, 2].astype(np.float64)
    r_vc = np.sqrt(x_vc**2 + y_vc**2)

    # 3. Pair and subsample events ─────────────────────────────────────────────
    print("\n[3/9] Pairing events...")
    sim_idx, gen_idx = pair_events(sim, gen)

    n_matched = len(sim_idx)
    if args.n_events is not None and args.n_events < n_matched:
        chosen  = np.random.choice(n_matched, args.n_events, replace=False)
        sim_idx = sim_idx[chosen]
        gen_idx = gen_idx[chosen]
        print(f"  Subsampled to {len(sim_idx):,} events (--n_events {args.n_events}).")
    n_matched = len(sim_idx)

    # Region matrices are (N, 4) — trivially small even at large N; load now.
    sim_region = sim["region_matrix"][sim_idx]
    gen_region = gen["region_matrix"][gen_idx]

    # 4. First pass: find kmax ─────────────────────────────────────────────────
    print(f"\n[4/9] First pass — finding global max hit count "
          f"({n_matched:,} events, chunk={args.chunk_size:,})...")
    t = time.time()
    kmax = _find_kmax(args.sim, args.gen, sim_idx, gen_idx, args.chunk_size)
    print(f"  kmax = {kmax}  ({time.time()-t:.1f}s)")
    hist_mem_mb = 2 * 9583 * (kmax + 1) * 8 / 1e6
    print(f"  Histogram memory: {hist_mem_mb:.1f} MB (2 × 9583 × {kmax+1} int64)")

    # 5. Batch accumulation loop ───────────────────────────────────────────────
    # Single pass over all N matched events in chunks of chunk_size.
    # All 7 metrics are accumulated simultaneously; no dense (N, 9583) matrix
    # is ever held in RAM — only the current chunk (chunk_size × 9583).
    print(f"\n[5/9] Batch accumulation — all metrics "
          f"({n_matched:,} events in {-(-n_matched // args.chunk_size)} batches)...")
    acc = _init_accumulators(n_matched, kmax, region_masks)

    # Pre-sort global indices for efficient sequential HDF5 reads.
    # Within each batch we sort locally to restore the pairing order.
    t = time.time()
    n_batches = -(-n_matched // args.chunk_size)   # ceil division

    with h5py.File(args.sim, "r") as f_sim, h5py.File(args.gen, "r") as f_gen:
        ds_sim = f_sim["target_matrix"]
        ds_gen = f_gen["target_matrix"]

        for batch_num, batch_start in enumerate(range(0, n_matched, args.chunk_size)):
            batch_end = min(batch_start + args.chunk_size, n_matched)

            # Sort this batch's indices for efficient I/O, then un-sort to
            # restore the sim↔gen pairing (sc[i] corresponds to gc[i]).
            b_sim = sim_idx[batch_start:batch_end]
            b_gen = gen_idx[batch_start:batch_end]
            sort_s = np.argsort(b_sim);  inv_s = np.argsort(sort_s)
            sort_g = np.argsort(b_gen);  inv_g = np.argsort(sort_g)

            sc = ds_sim[b_sim[sort_s]][inv_s].astype(np.int32)
            gc = ds_gen[b_gen[sort_g]][inv_g].astype(np.int32)

            _process_batch(acc, sc, gc, batch_start, r_vc, z_vc, region_masks)

            if (batch_num + 1) % max(1, n_batches // 10) == 0 or batch_num == 0:
                elapsed = time.time() - t
                done    = batch_end / n_matched
                eta     = elapsed / done * (1 - done) if done > 0 else 0
                print(f"  Batch {batch_num+1:>4}/{n_batches}  "
                      f"{batch_end:>{len(str(n_matched))},}/{n_matched:,}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"  Batch loop done: {time.time()-t:.1f}s")

    # 6–8. Finalize metrics from accumulators ──────────────────────────────────
    print("\n[6/9] Wasserstein distances (finalize from accumulators)...")
    t = time.time()
    reg_m  = wasserstein_region(sim_region, gen_region)
    print(f"  Region W₁: done ({time.time()-t:.1f}s)")

    t = time.time()
    vox_m  = wasserstein_per_voxel(acc)
    print(f"  Per-voxel W₁: done ({time.time()-t:.1f}s)  mean={vox_m['w1_mean']:.3f}")

    t = time.time()
    spat_m  = wasserstein_spatial(acc, voxel_centers)
    spat_pr = _spatial_profiles(acc, voxel_centers)
    print(f"  Spatial W₁ + profiles: done ({time.time()-t:.1f}s)")

    print("\n[7/9] Support / MSE / NLL / JS (finalize from accumulators)...")
    t = time.time()
    supp_m = support_metrics_per_event(acc, region_masks)
    mse_m  = mse_per_event(acc, n_matched)
    nll_m  = poisson_nll_per_event(acc, n_matched, region_masks)
    js_pop = js_divergence_population(acc)
    js_ev  = js_divergence_per_event(acc, region_masks)
    print(f"  done ({time.time()-t:.1f}s) | "
          f"IoU={supp_m['iou']['mean']:.3f}  "
          f"RMSE={mse_m['rmse_mean']:.3f}  "
          f"JS(pop)={js_pop['js_divergence']:.5f}")

    # 8. Save ──────────────────────────────────────────────────────────────────
    print("\n[8/9] Saving results...")
    metrics = {
        "metadata": {
            "sim_file":         args.sim,
            "gen_file":         args.gen,
            "n_sim_events":     sim["n_events"],
            "n_gen_events":     gen["n_events"],
            "n_matched_events": int(n_matched),
            "n_unmatched_gen":  gen["n_events"] - n_matched,
            "chunk_size":       args.chunk_size,
            "kmax":             kmax,
            "seed":             args.seed,
            "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
            "runtime_s":        round(time.time() - t0, 1),
        },
        # Public (serialised) metric dicts
        "region_hits":    reg_m,
        "per_voxel":      {k: v for k, v in vox_m.items()  if not k.startswith("_")},
        "spatial":        {k: v for k, v in spat_m.items() if not k.startswith("_")},
        "support_metrics":{k: v for k, v in supp_m.items() if not k.startswith("_")},
        "mse_per_event":  {k: v for k, v in mse_m.items()  if not k.startswith("_")},
        "poisson_nll":    {k: v for k, v in nll_m.items()  if not k.startswith("_")},
        "js_population":  {k: v for k, v in js_pop.items() if not k.startswith("_")},
        "js_per_event":   {k: v for k, v in js_ev.items()  if not k.startswith("_")},
        # Private arrays for plotting (stripped from JSON by save_metrics_json)
        "_per_voxel":               vox_m,
        "_spatial_profiles":        spat_pr,
        "_js_per_event_arr":        js_ev["_js_per_event"],
        "_rmse_per_event_arr":      mse_m["_rmse_per_event"],
        "_nll_per_event_arr":       nll_m["_nll_per_event"],
        "_iou_per_event_arr":       supp_m["_iou_per_event"],
        "_precision_per_event_arr": supp_m["_precision_per_event"],
        "_recall_per_event_arr":    supp_m["_recall_per_event"],
        "_sim_region":              sim_region,
        "_gen_region":              gen_region,
    }

    save_metrics_json(metrics, str(Path(args.output_dir) / "metrics.json"))

    # 9. Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n[9/9] Generating plots...")
        t = time.time()
        plot_all(metrics, args.output_dir)
        print(f"  done ({time.time()-t:.1f}s)")
    else:
        print("\n[9/9] Plots skipped (--no_plots).")

    elapsed = time.time() - t0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"\nTotal runtime: {h:02d}:{m:02d}:{s:05.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
