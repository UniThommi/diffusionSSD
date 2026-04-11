#!/usr/bin/env python3
# generate.py
"""
NC-Score Generative Data Pipeline

Generates optical detector hits from neutron capture simulation data
using a trained score-based diffusion model (NC-Score).

Pipeline:
  Sim-HDF5 → NC extraction → φ normalization → Model inference
  → Denormalization → Grid-to-Voxel remap → ML-format HDF5

Usage:
  # Single directory
  python generate.py --config config.toml \
      --checkpoint_dir ./checkpoints/nc_score_baseline_GPU_final \
      --input_dir /path/to/sim_output \
      --output_file generated_ML_format.hdf5 \
      --voxel_json /path/to/currentDistZylVoxelsPMTSize.json

  # Nested directories (run_001, run_002, ...)
  python generate.py --config config.toml \
      --checkpoint_dir ./checkpoints/nc_score_baseline_GPU_final \
      --input_dir /path/to/runs \
      --output_file generated_ML_format.hdf5 \
      --voxel_json /path/to/currentDistZylVoxelsPMTSize.json \
      --nested

  # CPU-only
  python generate.py ... --device cpu
"""

import argparse
import json
import glob
import os
import re
import sys
import time
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import toml


def _load_mapping(file_path: str) -> dict:
    """Load JSON mapping file."""
    with open(file_path, "r") as f:
        return json.load(f)

def _remap_material_ids_to_global(
    glob_mapping_path: str, local_mat_map: dict, local_ids: np.ndarray
) -> np.ndarray:
    """Remap local material IDs to global IDs."""
    glob_map = _load_mapping(glob_mapping_path)
    local_ids = np.array(local_ids)
    mapping_dict = {}
    for local_id in np.unique(local_ids):
        local_name = local_mat_map[local_id]
        if local_name not in glob_map:
            raise RuntimeError(f"Material '{local_name}' not in global mapping")
        mapping_dict[local_id] = glob_map[local_name]
    return np.vectorize(mapping_dict.get)(local_ids).astype(np.float32)


def _remap_volume_ids_to_global(
    glob_mapping_path: str, local_vol_map: dict, local_ids: np.ndarray
) -> np.ndarray:
    """Remap local volume IDs to global IDs."""
    glob_map = _load_mapping(glob_mapping_path)
    local_ids = np.array(local_ids)
    mapping_dict = {}
    for local_id in np.unique(local_ids):
        local_name = local_vol_map[local_id]
        local_name = "noVolume" if local_name == "" else local_name
        if local_name not in glob_map:
            raise RuntimeError(f"Volume '{local_name}' not in global mapping")
        mapping_dict[local_id] = glob_map[local_name]
    return np.vectorize(mapping_dict.get)(local_ids).astype(np.float32)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate optical hits from NC simulation data using NC-Score"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.toml (same as used for training)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to checkpoint directory containing EMA weights"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing output_t*.hdf5 simulation files"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path for output HDF5 file in ML format"
    )
    parser.add_argument(
        "--voxel_json", type=str, required=True,
        help="Path to currentDistZylVoxelsPMTSize.json"
    )
    parser.add_argument(
        "--nested", action="store_true", default=False,
        help="Input dir contains subdirectories (run_*) with output_t*.hdf5"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for model inference (default: 256)"
    )
    parser.add_argument(
        "--device", type=str, default="gpu", choices=["cpu", "gpu"],
        help="Device for inference (default: gpu)"
    )
    parser.add_argument(
        "--max_events", type=int, default=None,
        help="Maximum number of NC events to generate (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible generation (default: None = non-deterministic)"
    )
    return parser.parse_args()


# =============================================================================
# Step 1: Discover simulation files
# =============================================================================

def discover_sim_files(input_dir: str, nested: bool) -> List[str]:
    """
    Find all simulation HDF5 files.

    Args:
        input_dir: Root directory
        nested: If True, search in subdirectories matching run_*

    Returns:
        Sorted list of absolute paths to output_t*.hdf5 files
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if nested:
        pattern = str(input_path / "run_*" / "output_t*.hdf5")
    else:
        pattern = str(input_path / "output_t*.hdf5")

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No simulation files found with pattern: {pattern}"
        )
    return files


# =============================================================================
# Step 2: Extract NC events from simulation HDF5
# =============================================================================

def _parse_run_id(fpath: str) -> int:
    """Extract integer run ID from a 'run_NNN' component in the file path."""
    for part in reversed(Path(fpath).parts):
        m = re.search(r'run[_\-]?(\d+)', part, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def extract_nc_events(
    sim_files: List[str],
    config: dict,
    max_events: Optional[int] = None,
) -> Tuple[dict, np.ndarray, List[str]]:
    """
    Extract neutron capture events from simulation HDF5 files.

    Each NC event is identified by unique (evtid, nC_track_id) tuple.

    Args:
        sim_files: List of simulation HDF5 paths
        config: Full config dict

    Returns:
        phi_data     : Dict with keys matching ML-format phi fields, each value
                       is np.ndarray of shape (n_nc_events,)
        event_ids    : int64 array of shape (N, 2) with columns [run_id, nc_id],
                       or shape (N, 3) with columns [run_id, muon_id, nc_id] for
                       muon-seeded simulations.
        id_columns   : list of column name strings for event_ids.
    """
    norm_cfg = config["normalization"]
    r_cyl = norm_cfg["r_cylinder"]
    z_min = norm_cfg["z_min"]
    z_max = norm_cfg["z_max"]

    # Accumulate per-field lists
    fields = {
        "xNC_mm": [], "yNC_mm": [], "zNC_mm": [],
        "r_NC_mm": [], "phi_NC_rad": [],
        "matID": [], "volID": [],
        "#gamma": [], "E_gamma_tot_keV": [],
        "dist_to_wall_mm": [], "dist_to_bot_mm": [], "dist_to_top_mm": [],
        "gammaE1_keV": [], "gammapx1": [], "gammapy1": [], "gammapz1": [],
        "gammaE2_keV": [], "gammapx2": [], "gammapy2": [], "gammapz2": [],
        "gammaE3_keV": [], "gammapx3": [], "gammapy3": [], "gammapz3": [],
        "gammaE4_keV": [], "gammapx4": [], "gammapy4": [], "gammapz4": [],
        "p_mean_r": [], "p_mean_z": [],
    }

    total_ncs = 0
    id_run: List[np.ndarray] = []
    id_nc: List[np.ndarray] = []
    id_muon: List[np.ndarray] = []   # only populated for muon-seeded sims
    is_muon_sim: Optional[bool] = None  # detected from first file

    for fpath in sim_files:
        with h5py.File(fpath, "r") as f:
            nc_group = f["hit"]["MyNeutronCaptureOutput"]

            # Read NC-level data (pages format)
            evtid = np.array(nc_group["evtid"]["pages"])
            track_id = np.array(nc_group["nC_track_id"]["pages"])

            # Detect simulation type once from the first file
            if is_muon_sim is None:
                is_muon_sim = bool(np.any(evtid != track_id))

            # Positions (meters → mm)
            x_m = np.array(nc_group["nC_x_position_in_m"]["pages"])
            y_m = np.array(nc_group["nC_y_position_in_m"]["pages"])
            z_m = np.array(nc_group["nC_z_position_in_m"]["pages"])

            x_mm = x_m * 1000.0
            y_mm = y_m * 1000.0
            z_mm = z_m * 1000.0

            # Derived coordinates
            r_mm = np.sqrt(x_mm**2 + y_mm**2)
            phi_rad = np.mod(np.arctan2(y_mm, x_mm), 2.0 * np.pi)

            # Distances
            dist_wall = r_cyl - r_mm
            dist_bot = z_mm - z_min
            dist_top = z_max - z_mm

            # NC properties
            # NC properties (local IDs → global IDs)
            local_mat_ids = np.array(nc_group["nC_material_id"]["pages"])
            local_vol_ids = np.array(nc_group["nC_phys_vol_id"]["pages"])

            # Build local→global mapping from per-file material/volume tables
            mat_names = [x.decode() for x in f["hit/materials/materialNames"]["pages"][:]]
            mat_ids_table = f["hit/materials/materialsID"]["pages"][:]
            local_mat_map = dict(zip(mat_ids_table, mat_names))

            vol_names = [x.decode() for x in f["hit/physVolumes/physVolumeNames"]["pages"][:]]
            vol_ids_table = f["hit/physVolumes/physVolumesID"]["pages"][:]
            local_vol_map = dict(zip(vol_ids_table, vol_names))

            mat_id = _remap_material_ids_to_global(
                config["mapping"]["material_mapping_file"], local_mat_map, local_mat_ids
            )
            vol_id = _remap_volume_ids_to_global(
                config["mapping"]["volume_mapping_file"], local_vol_map, local_vol_ids
            )

            n_gamma = np.array(nc_group["nC_gamma_amount"]["pages"]).astype(np.float32)
            e_gamma_tot = np.array(nc_group["nC_gamma_total_energy_in_keV"]["pages"])

            # Individual gamma data (top-4 from NC output scheme)
            g1_e = np.array(nc_group["gamma1_E_in_keV"]["pages"])
            g1_px = np.array(nc_group["gamma1_px"]["pages"])
            g1_py = np.array(nc_group["gamma1_py"]["pages"])
            g1_pz = np.array(nc_group["gamma1_pz"]["pages"])

            g2_e = np.array(nc_group["gamma2_E_in_keV"]["pages"])
            g2_px = np.array(nc_group["gamma2_px"]["pages"])
            g2_py = np.array(nc_group["gamma2_py"]["pages"])
            g2_pz = np.array(nc_group["gamma2_pz"]["pages"])

            g3_e = np.array(nc_group["gamma3_E_in_keV"]["pages"])
            g3_px = np.array(nc_group["gamma3_px"]["pages"])
            g3_py = np.array(nc_group["gamma3_py"]["pages"])
            g3_pz = np.array(nc_group["gamma3_pz"]["pages"])

            g4_e = np.array(nc_group["gamma4_E_in_keV"]["pages"])
            g4_px = np.array(nc_group["gamma4_px"]["pages"])
            g4_py = np.array(nc_group["gamma4_py"]["pages"])
            g4_pz = np.array(nc_group["gamma4_pz"]["pages"])

            # p_mean_r and p_mean_z: mean over all gammas
            # Radial momentum per gamma: p_r = (px*x + py*y) / r
            # Average over active gammas (E > 0)
            n_events = len(evtid)
            
            # Stack gamma data: (n_events, 4)
            gamma_es = np.stack([g1_e, g2_e, g3_e, g4_e], axis=1)
            gamma_pxs = np.stack([g1_px, g2_px, g3_px, g4_px], axis=1)
            gamma_pys = np.stack([g1_py, g2_py, g3_py, g4_py], axis=1)
            gamma_pzs = np.stack([g1_pz, g2_pz, g3_pz, g4_pz], axis=1)
            
            # Radial projection: p_r = (px*x + py*y) / r
            r_safe = np.where(r_mm > 0, r_mm, 1.0)
            pr_per_gamma = (gamma_pxs * x_mm[:, None] + gamma_pys * y_mm[:, None]) / r_safe[:, None]
            pr_per_gamma = np.where(r_mm[:, None] > 0, pr_per_gamma, 0.0)
            
            # Mask active gammas (E > 0)
            active_mask = gamma_es > 0  # (n_events, 4)
            count = active_mask.sum(axis=1)  # (n_events,)
            count_safe = np.where(count > 0, count, 1.0)
            
            p_mean_r = np.where(count > 0, (pr_per_gamma * active_mask).sum(axis=1) / count_safe, 0.0)
            p_mean_z = np.where(count > 0, (gamma_pzs * active_mask).sum(axis=1) / count_safe, 0.0)

            # Vectorized deduplication
            keys = np.stack([evtid, track_id], axis=1)
            _, unique_indices = np.unique(keys, axis=0, return_index=True)
            mask = np.zeros(n_events, dtype=bool)
            mask[unique_indices] = True

            n_unique = np.sum(mask)
            if n_unique < n_events:
                print(f"  WARNING: {n_events - n_unique} duplicate NC events "
                      f"removed in {fpath}")

            # Append to accumulators
            fields["xNC_mm"].append(x_mm[mask])
            fields["yNC_mm"].append(y_mm[mask])
            fields["zNC_mm"].append(z_mm[mask])
            fields["r_NC_mm"].append(r_mm[mask])
            fields["phi_NC_rad"].append(phi_rad[mask])
            fields["matID"].append(mat_id[mask])
            fields["volID"].append(vol_id[mask])
            fields["#gamma"].append(n_gamma[mask])
            fields["E_gamma_tot_keV"].append(e_gamma_tot[mask])
            fields["dist_to_wall_mm"].append(dist_wall[mask])
            fields["dist_to_bot_mm"].append(dist_bot[mask])
            fields["dist_to_top_mm"].append(dist_top[mask])
            fields["gammaE1_keV"].append(g1_e[mask])
            fields["gammapx1"].append(g1_px[mask])
            fields["gammapy1"].append(g1_py[mask])
            fields["gammapz1"].append(g1_pz[mask])
            fields["gammaE2_keV"].append(g2_e[mask])
            fields["gammapx2"].append(g2_px[mask])
            fields["gammapy2"].append(g2_py[mask])
            fields["gammapz2"].append(g2_pz[mask])
            fields["gammaE3_keV"].append(g3_e[mask])
            fields["gammapx3"].append(g3_px[mask])
            fields["gammapy3"].append(g3_py[mask])
            fields["gammapz3"].append(g3_pz[mask])
            fields["gammaE4_keV"].append(g4_e[mask])
            fields["gammapx4"].append(g4_px[mask])
            fields["gammapy4"].append(g4_py[mask])
            fields["gammapz4"].append(g4_pz[mask])
            fields["p_mean_r"].append(p_mean_r[mask])
            fields["p_mean_z"].append(p_mean_z[mask])

            # Event IDs
            run_id_val = _parse_run_id(fpath)
            id_run.append(np.full(n_unique, run_id_val, dtype=np.int64))
            id_nc.append(track_id[mask].astype(np.int64))
            if is_muon_sim:
                id_muon.append(evtid[mask].astype(np.int64))

            total_ncs += n_unique

            # Early exit if we have enough events
            if max_events is not None and total_ncs >= max_events:
                print(f"  {fpath}: {n_unique} NC events extracted (reached {total_ncs:,} >= {max_events:,}, stopping)")
                break
            else:
                print(f"  {fpath}: {n_unique} NC events extracted")

    # Concatenate all files
    result = {k: np.concatenate(v).astype(np.float32) for k, v in fields.items()}

    run_id_arr  = np.concatenate(id_run)
    nc_id_arr   = np.concatenate(id_nc)
    if is_muon_sim:
        muon_id_arr = np.concatenate(id_muon)

    if max_events is not None and total_ncs > max_events:
        result      = {k: v[:max_events] for k, v in result.items()}
        run_id_arr  = run_id_arr[:max_events]
        nc_id_arr   = nc_id_arr[:max_events]
        if is_muon_sim:
            muon_id_arr = muon_id_arr[:max_events]
        print(f"\nTotal NC events: {total_ncs:,} → truncated to {max_events:,}")
    else:
        print(f"\nTotal NC events: {total_ncs:,}")

    if is_muon_sim:
        event_ids  = np.stack([run_id_arr, muon_id_arr, nc_id_arr], axis=1)
        id_columns = ["run_id", "muon_id", "nc_id"]
    else:
        event_ids  = np.stack([run_id_arr, nc_id_arr], axis=1)
        id_columns = ["run_id", "nc_id"]

    return result, event_ids, id_columns


# =============================================================================
# Step 3: Normalize phi for model input
# =============================================================================

def normalize_phi(phi_data: dict, config: dict) -> np.ndarray:
    """
    Normalize phi parameters for model inference.

    Mirrors data_loader._normalize_phi exactly.

    Args:
        phi_data: Dict of raw phi arrays
        config: Full config dict

    Returns:
        (n_events, phi_dim) normalized conditioning array
    """
    norm_cfg = config["normalization"]
    feat_cfg = config["features"]
    onehot_cfg = config["onehot"]
    mapping_cfg = config["mapping"]

    E_max = norm_cfg["E_max"]
    r_cyl = norm_cfg["r_cylinder"]
    z_min = norm_cfg["z_min"]
    z_max = norm_cfg["z_max"]
    h_cyl = z_max - z_min
    angle_max = norm_cfg["angle_max"]
    gamma_count_max = norm_cfg["gamma_count_max"]

    enable_mat_onehot = onehot_cfg["enable_material_onehot"]
    enable_vol_onehot = onehot_cfg["enable_volume_onehot"]

    # Determine active features (remove matID/volID if one-hot)
    active_phi_raw = feat_cfg["active_phi"]
    active_phi = [
        f for f in active_phi_raw
        if not ((f == "matID" and enable_mat_onehot) or
                (f == "volID" and enable_vol_onehot))
    ]

    n_events = len(phi_data["r_NC_mm"])
    normalized_features = []

    for feature_name in active_phi:
        raw = phi_data[feature_name]

        # Energy normalization
        if feature_name in ["E_gamma_tot_keV", "gammaE1_keV", "gammaE2_keV",
                            "gammaE3_keV", "gammaE4_keV"]:
            norm = raw / E_max
            if np.any(norm > 1.0):
                raise RuntimeError(
                    f"NORMALIZATION OVERFLOW: {feature_name}\n"
                    f"  Max: {np.max(raw):.2f}, E_max: {E_max:.2f}"
                )

        # Momentum (already normalized)
        elif feature_name in ["gammapx1", "gammapx2", "gammapx3", "gammapx4",
                              "gammapy1", "gammapy2", "gammapy3", "gammapy4",
                              "gammapz1", "gammapz2", "gammapz3", "gammapz4",
                              "p_mean_r", "p_mean_z"]:
            norm = raw

        # Radii
        elif feature_name in ["r_NC_mm", "dist_to_wall_mm"]:
            norm = raw / r_cyl

        # Z coordinates
        elif feature_name in ["zNC_mm", "dist_to_bot_mm", "dist_to_top_mm"]:
            norm = (raw - z_min) / h_cyl

        # X, Y
        elif feature_name in ["xNC_mm", "yNC_mm"]:
            norm = (raw + r_cyl) / (2 * r_cyl)

        # Angle
        elif feature_name == "phi_NC_rad":
            norm = np.mod(raw, angle_max) / angle_max

        # Gamma count
        elif feature_name == "#gamma":
            norm = raw / gamma_count_max

        # Categorical fallback
        elif feature_name in ["matID", "volID"]:
            norm = raw

        else:
            raise ValueError(f"Unknown feature: {feature_name}")

        normalized_features.append(norm)

    phi_base = np.stack(normalized_features, axis=1)

    # One-hot encoding
    onehot_parts = []

    if enable_mat_onehot and "matID" in active_phi_raw:
        with open(mapping_cfg["material_mapping_file"], "r") as f:
            mat_map_raw = json.load(f)

        # Build id → name mapping
        mat_id_to_name = {}
        for name, mid in mat_map_raw.items():
            if isinstance(mid, int):
                final_name = "noMaterial" if name == "" else name
                mat_id_to_name[mid] = final_name

        mat_names_sorted = sorted(set(mat_id_to_name.values()))
        mat_name_to_idx = {n: i for i, n in enumerate(mat_names_sorted)}
        n_mat = len(mat_names_sorted)

        mat_ids = phi_data["matID"].astype(int)
        mat_onehot = np.zeros((n_events, n_mat), dtype=np.float32)
        for i in range(n_events):
            mid = mat_ids[i]
            if mid not in mat_id_to_name:
                raise RuntimeError(f"Unknown matID: {mid}")
            mat_onehot[i, mat_name_to_idx[mat_id_to_name[mid]]] = 1.0

        onehot_parts.append(mat_onehot)
        print(f"  Material one-hot: {n_mat} categories")

    if enable_vol_onehot and "volID" in active_phi_raw:
        raise NotImplementedError(
            "Volume one-hot encoding in generate.py not yet implemented. "
            "Set enable_volume_onehot = false in config.toml"
        )

    if onehot_parts:
        phi_normalized = np.concatenate([phi_base] + onehot_parts, axis=1)
    else:
        phi_normalized = phi_base

    print(f"  Phi shape: {phi_normalized.shape} "
          f"(base={phi_base.shape[1]}, onehot={sum(p.shape[1] for p in onehot_parts)})")
    return phi_normalized.astype(np.float32)


# =============================================================================
# Step 4: Load voxel geometry
# =============================================================================

def load_voxel_geometry(voxel_json_path: str) -> Tuple[List[dict], List[str]]:
    """
    Load voxel definitions from JSON.

    Args:
        voxel_json_path: Path to currentDistZylVoxelsPMTSize.json

    Returns:
        voxels: List of voxel dicts with 'index', 'center', 'corners', 'layer'
        voxel_keys: Sorted list of voxel index strings
    """
    with open(voxel_json_path, "r") as f:
        voxels = json.load(f)

    voxel_keys = sorted([v["index"] for v in voxels])

    # Validate count
    region_counts = {"pit": 0, "bot": 0, "wall": 0, "top": 0}
    for v in voxels:
        region_counts[v["layer"]] += 1

    print(f"  Loaded {len(voxels)} voxels: "
          f"PIT={region_counts['pit']}, BOT={region_counts['bot']}, "
          f"WALL={region_counts['wall']}, TOP={region_counts['top']}")

    return voxels, voxel_keys


# =============================================================================
# Step 5: Build grid mappings (mirrors data_loader logic)
# =============================================================================

def build_grid_mappings(
    voxel_keys: List[str],
    config: dict,
) -> Tuple[dict, dict, dict]:
    """
    Build voxel-to-grid and grid-to-voxel mappings per region.

    Mirrors data_loader._infer_grid_shape_and_mapping logic.

    Args:
        voxel_keys: Sorted list of voxel index strings
        config: Full config dict

    Returns:
        grid_shapes: {region: (n_axis0, n_axis1, 1)}
        voxel_to_grid: {region: {voxel_key: (axis0, axis1)}}
        grid_to_voxel: {region: {(axis0, axis1): voxel_key}}
    """
    region_cfg = config["regions"]
    merged_regions = region_cfg.get("merged_regions", [])

    # Classify voxels by region
    region_voxels = {"PIT": [], "BOT": [], "WALL": [], "TOP": []}
    for key in voxel_keys:
        if key.startswith(region_cfg["pit_prefix"]):
            region_voxels["PIT"].append(key)
        elif key.startswith(region_cfg["bot_prefix"]):
            region_voxels["BOT"].append(key)
        elif key.startswith(region_cfg["top_prefix"]):
            region_voxels["TOP"].append(key)
        else:
            region_voxels["WALL"].append(key)

    # Determine which regions to process
    regions_to_process = []
    if merged_regions:
        merged_name = "".join(merged_regions)  # e.g. "PITBOT"
        merged_keys = []
        for r in merged_regions:
            merged_keys.extend(region_voxels[r])
        regions_to_process.append((merged_name, merged_regions, sorted(merged_keys)))

        for r in ["PIT", "BOT", "WALL", "TOP"]:
            if r not in merged_regions:
                regions_to_process.append((r, [r], sorted(region_voxels[r])))
    else:
        for r in ["PIT", "BOT", "WALL", "TOP"]:
            regions_to_process.append((r, [r], sorted(region_voxels[r])))

    grid_shapes = {}
    voxel_to_grid = {}
    grid_to_voxel = {}

    for region_name, source_regions, keys in regions_to_process:
        # Parse voxel indices
        parsed = {}
        for key in keys:
            # Determine parse region
            if key.startswith(region_cfg["pit_prefix"]):
                parse_region = "PIT"
            elif key.startswith(region_cfg["bot_prefix"]):
                parse_region = "BOT"
            elif key.startswith(region_cfg["top_prefix"]):
                parse_region = "TOP"
            else:
                parse_region = "WALL"

            coords = _parse_voxel_index(key, parse_region)
            parsed[key] = coords

        # Determine grid extents
        axis0_vals = [c[0] for c in parsed.values()]
        axis1_vals = [c[1] for c in parsed.values()]
        a0_min, a0_max = min(axis0_vals), max(axis0_vals)
        a1_min, a1_max = min(axis1_vals), max(axis1_vals)

        n_axis0 = a0_max - a0_min + 1
        n_axis1 = a1_max - a1_min + 1

        grid_shapes[region_name] = (n_axis0, n_axis1, 1)

        # Build mappings (normalize to 0-indexed)
        v2g = {}
        g2v = {}
        for key, (a0, a1) in parsed.items():
            norm_a0 = a0 - a0_min
            norm_a1 = a1 - a1_min
            v2g[key] = (norm_a0, norm_a1)
            g2v[(norm_a0, norm_a1)] = key

        voxel_to_grid[region_name] = v2g
        grid_to_voxel[region_name] = g2v

        print(f"  {region_name}: shape={grid_shapes[region_name]}, "
              f"voxels={len(keys)}, "
              f"grid_cells={n_axis0 * n_axis1}, "
              f"utilization={len(keys) / (n_axis0 * n_axis1) * 100:.1f}%")

    return grid_shapes, voxel_to_grid, grid_to_voxel


def _parse_voxel_index(index_str: str, region: str) -> Tuple[int, int]:
    """
    Parse voxel index string to grid coordinates.

    Mirrors data_loader._parse_voxel_index.
    """
    remainder = index_str[2:]

    if region == "TOP":
        assert len(remainder) == 4, f"TOP index '{index_str}' must be 6 digits"
        return (int(remainder[0:2]), int(remainder[2:4]))

    if region == "WALL":
        if len(remainder) == 4:
            return (int(remainder[:2]), int(remainder[2:4]))
        elif len(remainder) == 5:
            # LLZZPPP heuristic (phi usually has more bins)
            return (int(remainder[:2]), int(remainder[2:5]))
        else:
            raise ValueError(
                f"Cannot parse WALL index '{index_str}' "
                f"({len(remainder)} digits after prefix)"
            )

    if region in ("PIT", "BOT"):
        assert len(remainder) == 4, f"{region} index '{index_str}' must be 6 digits"
        return (int(remainder[0:2]), int(remainder[2:4]))

    raise ValueError(f"Unknown region: {region}")


# =============================================================================
# Step 6: Model loading & inference
# =============================================================================

def load_model(config: dict, checkpoint_dir: str):
    """
    Load NC-Score model with EMA weights.

    Args:
        config: Prepared config dict (with SHAPE_*, PAD, etc.)
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Loaded scoreDiffusion model with EMA weights
    """
    from src.models.scoreDiffusion import scoreDiffusion

    ckpt = Path(checkpoint_dir)

    # Verify EMA weights exist
    ema_area_path = ckpt / "ema_area_model_best.weights.h5"
    if not ema_area_path.exists():
        raise FileNotFoundError(
            f"EMA weights not found: {ema_area_path}\n"
            f"Re-train with EMA checkpoint saving enabled."
        )

    # Determine phi_dim from config
    feat_cfg = config["features"]
    onehot_cfg = config["onehot"]
    active_phi_raw = feat_cfg["active_phi"]
    active_phi = [
        f for f in active_phi_raw
        if not ((f == "matID" and onehot_cfg["enable_material_onehot"]) or
                (f == "volID" and onehot_cfg["enable_volume_onehot"]))
    ]
    phi_dim = len(active_phi)
    if onehot_cfg["enable_material_onehot"] and "matID" in active_phi_raw:
        with open(config["mapping"]["material_mapping_file"], "r") as f:
            mat_map = json.load(f)
        n_mat = len(set(v for v in mat_map.values() if isinstance(v, int)))
        phi_dim += n_mat
    if onehot_cfg["enable_volume_onehot"] and "volID" in active_phi_raw:
        raise NotImplementedError("Volume one-hot not implemented in generate.py")

    num_area = 4
    model = scoreDiffusion(num_area=num_area, num_cond=phi_dim, config=config)

    # Load EMA weights (used for generation)
    model.ema_area.load_weights(str(ckpt / "ema_area_model_best.weights.h5"))
    for area_name in model.active_areas:
        model.ema_voxels[area_name].load_weights(
            str(ckpt / f"ema_voxel_{area_name}_best.weights.h5")
        )

    print(f"  Model loaded with EMA weights from: {checkpoint_dir}")
    print(f"  Active areas: {model.active_areas}")
    print(f"  phi_dim: {phi_dim}")
    return model

def generate_area_hits_batched(model, phi_normalized, batch_size):
    """
    Stage 1: Generate area_hits for ALL events via ResNet.
    """
    n_events = phi_normalized.shape[0]
    n_batches = (n_events + batch_size - 1) // batch_size
    area_hits_list = []
    batch_times = []

    print(f"\n--- Stage 1: ResNet area_hits ({n_batches} batches) ---")

    t_start = time.perf_counter()
    for batch_idx in range(n_batches):
        t0 = time.perf_counter()
        start = batch_idx * batch_size
        end = min(start + batch_size, n_events)
        actual_size = end - start

        # Pad last batch to avoid re-tracing with different shape
        if actual_size < batch_size:
            padded = np.zeros((batch_size, phi_normalized.shape[1]), dtype=np.float32)
            padded[:actual_size] = phi_normalized[start:end]
            cond_batch = tf.constant(padded, dtype=tf.float32)
        else:
            cond_batch = tf.constant(phi_normalized[start:end], dtype=tf.float32)

        area_hits = model.DDPMSampler(
            cond_batch, model.ema_area,
            data_shape=[model.num_area], const_shape=[-1, 1]
        )
        # Trim padding before storing
        area_hits_list.append(area_hits.numpy()[:actual_size])

        dt = time.perf_counter() - t0
        batch_times.append(dt)

        if batch_idx < 3 or (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            elapsed = time.perf_counter() - t_start
            rate = end / elapsed if elapsed > 0 else 0
            label = " [WARMUP]" if batch_idx == 0 else ""
            print(f"    Batch {batch_idx:>4d}/{n_batches} dt={dt:.2f}s "
                  f"({end:,}/{n_events:,}) {rate:.0f} evt/s{label}")

    area_hits_all = np.concatenate(area_hits_list, axis=0)

    total_time = time.perf_counter() - t_start
    steady = np.array(batch_times[1:]) if len(batch_times) > 1 else np.array(batch_times)
    print(f"  ResNet total: {total_time:.1f}s "
          f"(batch 0={batch_times[0]:.2f}s, "
          f"steady mean={np.mean(steady):.3f}s, std={np.std(steady):.3f}s)")

    return area_hits_all, {"resnet_total": total_time, "resnet_batch_times": batch_times}


def generate_voxels_for_area(model, phi_normalized, area_hits_all,
                              area_name, batch_size):
    """
    Stage 2 (per area): Generate voxel grids, skipping events with 0 area hits.
    """
    n_events = phi_normalized.shape[0]
    data_shape = model.shapes[area_name]
    region_order = ["pit", "bot", "wall", "top"]

    # Determine which events have nonzero hits for this area
    if area_name == "PITBOT":
        area_total = area_hits_all[:, 0] + area_hits_all[:, 1]
    else:
        region_idx = region_order.index(area_name.lower())
        area_total = area_hits_all[:, region_idx]

    nonzero_mask = np.abs(area_total) > 1e-6
    active_indices = np.where(nonzero_mask)[0]
    n_active = len(active_indices)
    n_skipped = n_events - n_active
    skip_pct = n_skipped / n_events * 100 if n_events > 0 else 0

    print(f"\n--- Stage 2: U-Net {area_name} "
          f"(shape={data_shape}, {n_active:,} active, "
          f"{n_skipped:,} skipped [{skip_pct:.1f}%]) ---")

    # Pre-allocate full output with zeros
    voxel_grid_all = np.zeros(
        (n_events,) + tuple(data_shape), dtype=np.float32
    )

    if n_active == 0:
        print(f"  All events skipped for {area_name}!")
        return voxel_grid_all, {"total": 0.0, "n_active": 0, "n_skipped": n_skipped,
                                "batch_times": []}

    # Extract active subset
    phi_active = phi_normalized[active_indices]
    area_hits_active = area_hits_all[active_indices]
    phi_dim = phi_normalized.shape[1]

    n_batches = (n_active + batch_size - 1) // batch_size
    batch_times = []

    t_start = time.perf_counter()
    voxel_chunks = []

    for batch_idx in range(n_batches):
        t0 = time.perf_counter()
        start = batch_idx * batch_size
        end = min(start + batch_size, n_active)
        actual_size = end - start

        # Pad last batch to avoid re-tracing with different shape
        if actual_size < batch_size:
            padded_cond = np.zeros((batch_size, phi_dim), dtype=np.float32)
            padded_cond[:actual_size] = phi_active[start:end]
            padded_ah = np.zeros((batch_size, 4), dtype=np.float32)
            padded_ah[:actual_size] = area_hits_active[start:end]
            cond_batch = tf.constant(padded_cond, dtype=tf.float32)
            ah_batch = tf.constant(padded_ah, dtype=tf.float32)
        else:
            cond_batch = tf.constant(phi_active[start:end], dtype=tf.float32)
            ah_batch = tf.constant(area_hits_active[start:end], dtype=tf.float32)

        voxel_tensor = model.DDPMSampler(
            cond_batch, model.ema_voxels[area_name],
            data_shape=data_shape,
            const_shape=[-1] + [1] * len(data_shape),
            area_hits=ah_batch
        )
        # Trim padding before storing
        voxel_chunks.append(voxel_tensor.numpy()[:actual_size])

        dt = time.perf_counter() - t0
        batch_times.append(dt)

        if batch_idx < 3 or (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            elapsed = time.perf_counter() - t_start
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n_active - end) / rate if rate > 0 else 0
            label = " [WARMUP]" if batch_idx == 0 else ""
            print(f"    Batch {batch_idx:>4d}/{n_batches} dt={dt:.2f}s "
                  f"({end:,}/{n_active:,}) {rate:.0f} evt/s ETA={eta:.0f}s{label}")

    # Scatter active results back into full array
    voxel_active = np.concatenate(voxel_chunks, axis=0)
    voxel_grid_all[active_indices] = voxel_active

    total_time = time.perf_counter() - t_start
    steady = np.array(batch_times[1:]) if len(batch_times) > 1 else np.array(batch_times)
    print(f"  U-Net {area_name} total: {total_time:.1f}s "
          f"(batch 0={batch_times[0]:.2f}s, "
          f"steady mean={np.mean(steady):.3f}s, std={np.std(steady):.3f}s)")

    return voxel_grid_all, {
        "total": total_time,
        "n_active": n_active,
        "n_skipped": n_skipped,
        "batch_times": batch_times,
    }

def run_inference(
    model,
    phi_normalized: np.ndarray,
    batch_size: int,
    config: dict,
):
    """
    Two-stage inference pipeline with per-stage profiling and zero-skipping.
    
    Stage 1: Generate all area_hits via ResNet (batched)
    Stage 2: For each area, generate voxel grids (batched, zero-skip)
    """

    n_events = phi_normalized.shape[0]
    n_batches = (n_events + batch_size - 1) // batch_size
    voxel_hit_max = config["normalization"]["voxel_hit_max"]

    # Print inference config
    print(f"\n{'='*60}")
    print(f"INFERENCE CONFIG:")
    print(f"  n_events:      {n_events:,}")
    print(f"  batch_size:    {batch_size}")
    print(f"  n_batches:     {n_batches}")
    print(f"  num_steps:     {model.num_steps}")
    print(f"  active_areas:  {model.active_areas}")
    print(f"  shapes:        {model.shapes}")
    print(f"  num_cond:      {model.num_cond}")
    fwd_per_batch = model.num_steps
    fwd_total_resnet = fwd_per_batch * n_batches
    fwd_total_unet = fwd_per_batch * n_batches * len(model.active_areas)
    print(f"  ResNet forward passes:  {fwd_total_resnet:,} "
          f"({model.num_steps} steps × {n_batches} batches)")
    print(f"  U-Net forward passes (max): {fwd_total_unet:,} "
          f"({model.num_steps} steps × {n_batches} batches × "
          f"{len(model.active_areas)} areas)")
    print(f"  Total max forward passes: {fwd_total_resnet + fwd_total_unet:,}")
    print(f"{'='*60}")

    t_pipeline = time.perf_counter()

    # ===== STAGE 1: All ResNet calls =====
    area_hits_all, resnet_timings = generate_area_hits_batched(
        model, phi_normalized, batch_size
    )

    # Print area_hits statistics (useful for zero-skip analysis)
    print(f"\n  Area hits statistics (raw, before denorm):")
    for idx, name in enumerate(["pit", "bot", "wall", "top"]):
        col = area_hits_all[:, idx]
        n_zero = np.sum(np.abs(col) < 1e-6)
        print(f"    {name:>6s}: mean={np.mean(col):.4f} std={np.std(col):.4f} "
              f"min={np.min(col):.4f} max={np.max(col):.4f} "
              f"n_zero={n_zero} ({n_zero/n_events*100:.1f}%)")

    # ===== STAGE 2: U-Net calls per area =====
    voxel_grids = {}
    unet_timings = {}

    for area_name in model.active_areas:
        voxel_grid, timing = generate_voxels_for_area(
            model, phi_normalized, area_hits_all,
            area_name, batch_size
        )
        voxel_grids[area_name] = voxel_grid
        unet_timings[area_name] = timing

    # ===== SUMMARY =====
    total_pipeline = time.perf_counter() - t_pipeline

    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Stage 1 (ResNet):    {resnet_timings['resnet_total']:.1f}s")
    for area_name, t in unet_timings.items():
        skip_info = (f" (skipped {t['n_skipped']:,}/{n_events:,} = "
                     f"{t['n_skipped']/n_events*100:.1f}%)")
        print(f"  Stage 2 ({area_name:>6s}):  {t['total']:.1f}s{skip_info}")
    print(f"  Total pipeline:      {total_pipeline:.1f}s "
          f"({n_events / total_pipeline:.0f} events/s)")
    print(f"  Projected for 1M events: "
          f"{total_pipeline / n_events * 1e6 / 3600:.1f}h")

    # Memory check
    try:
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
        print(f"  GPU memory: current={gpu_mem['current']/1e9:.2f}GB "
              f"peak={gpu_mem['peak']/1e9:.2f}GB")
    except Exception:
        pass

    print(f"{'='*60}")

    # ===== Denormalize =====
    area_hits_raw = area_hits_all * voxel_hit_max
    area_hits_raw = np.clip(area_hits_raw, 0, None)

    region_order = ["pit", "bot", "wall", "top"]
    for area_name in model.active_areas:
        if area_name == "PITBOT":
            area_total = area_hits_raw[:, 0] + area_hits_raw[:, 1]
        else:
            region_idx = region_order.index(area_name.lower())
            area_total = area_hits_raw[:, region_idx]

        n_spatial = len(voxel_grids[area_name].shape) - 1
        area_broadcast = area_total.reshape((-1,) + (1,) * n_spatial)
        voxel_grids[area_name] = voxel_grids[area_name] * area_broadcast
        voxel_grids[area_name] = np.clip(voxel_grids[area_name], 0, None)

    return voxel_grids, area_hits_raw


# =============================================================================
# Step 7: Grid-to-voxel remapping & HDF5 output
# =============================================================================

def remap_grids_to_voxels(
    voxel_grids, grid_to_voxel, voxel_to_grid,
    voxel_keys, region_cfg, merged_regions,
):
    """Vectorized grid-to-voxel remapping."""
    
    def get_grid_region(voxel_key):
        if voxel_key.startswith(region_cfg["pit_prefix"]):
            base = "PIT"
        elif voxel_key.startswith(region_cfg["bot_prefix"]):
            base = "BOT"
        elif voxel_key.startswith(region_cfg["top_prefix"]):
            base = "TOP"
        else:
            base = "WALL"
        if merged_regions and base in merged_regions:
            return "".join(merged_regions)
        return base

    # Group voxels by grid region for batch extraction
    region_groups = {}  # {grid_region: [(voxel_key, a0, a1), ...]}
    for voxel_key in voxel_keys:
        grid_region = get_grid_region(voxel_key)
        mapping = voxel_to_grid[grid_region]
        a0, a1 = mapping[voxel_key]
        if grid_region not in region_groups:
            region_groups[grid_region] = []
        region_groups[grid_region].append((voxel_key, a0, a1))

    # Extract all voxels per region in one vectorized operation
    voxel_hits = {}
    for grid_region, entries in region_groups.items():
        grid = voxel_grids[grid_region]  # (n_events, h, w, 1)
        keys = [e[0] for e in entries]
        a0s = np.array([e[1] for e in entries])
        a1s = np.array([e[2] for e in entries])
        
        # Single fancy-index operation: (n_events, n_voxels_in_region)
        extracted = grid[:, a0s, a1s, 0]
        
        for i, key in enumerate(keys):
            voxel_hits[key] = extracted[:, i]

    # Compute region sums
    region_sums = {"pit": None, "bot": None, "wall": None, "top": None}
    for voxel_key, hits in voxel_hits.items():
        if voxel_key.startswith(region_cfg["pit_prefix"]):
            region = "pit"
        elif voxel_key.startswith(region_cfg["bot_prefix"]):
            region = "bot"
        elif voxel_key.startswith(region_cfg["top_prefix"]):
            region = "top"
        else:
            region = "wall"
        if region_sums[region] is None:
            region_sums[region] = np.zeros(hits.shape[0], dtype=np.float32)
        region_sums[region] += hits

    return voxel_hits, region_sums


def write_output_hdf5(
    output_path: str,
    phi_data: dict,
    voxel_hits: dict,
    region_hits_from_voxels: dict,
    area_hits_from_resnet: np.ndarray,
    voxels_json: List[dict],
    voxel_keys: List[str],
    config: dict,
    event_ids: np.ndarray,
    event_id_columns: List[str],
) -> None:
    """
    Write ML-format HDF5 file with 2D matrix layout.

    Structure matches reference format:
      /event_id_columns (n_id_cols,)    string labels
      /event_ids        (N, n_id_cols)  int64  — run_id/[muon_id]/nc_id
      /phi_columns      (n_phi,)        string labels
      /phi_matrix       (N, n_phi)      float32
      /target_columns   (n_voxels,)     string labels
      /target_matrix    (N, n_voxels)   int32
      /region_columns   (4,)            string labels
      /region_matrix    (N, 4)          int32  — summed from voxels
      /region_matrix_resnet (N, 4)      int32  — direct ResNet output
      /primaries        ()              int64
      /mat_map/...      scalar datasets
      /vol_map/...      scalar datasets
      /voxels/...       geometry groups
    """
    n_events = len(phi_data["xNC_mm"])

    # --- Phi matrix ---
    phi_order = [
        "xNC_mm", "yNC_mm", "zNC_mm", "matID", "volID", "#gamma",
        "E_gamma_tot_keV", "r_NC_mm", "phi_NC_rad",
        "dist_to_wall_mm", "dist_to_bot_mm", "dist_to_top_mm",
        "p_mean_r", "p_mean_z",
        "gammaE1_keV", "gammapx1", "gammapy1", "gammapz1",
        "gammaE2_keV", "gammapx2", "gammapy2", "gammapz2",
        "gammaE3_keV", "gammapx3", "gammapy3", "gammapz3",
        "gammaE4_keV", "gammapx4", "gammapy4", "gammapz4",
    ]

    phi_matrix = np.column_stack([phi_data[name] for name in phi_order])

    # --- Target matrix (N, n_voxels) int32 ---
    target_matrix = np.column_stack(
        [np.rint(voxel_hits[key]).astype(np.int32) for key in voxel_keys]
    )

    # --- Region matrices (N, 4) int32 ---
    region_names = ["pit", "bot", "wall", "top"]

    region_matrix_from_voxels = np.column_stack(
        [np.rint(region_hits_from_voxels[r]).astype(np.int32) for r in region_names]
    )

    region_matrix_from_resnet = np.column_stack(
        [np.rint(area_hits_from_resnet[:, idx]).astype(np.int32)
         for idx, r in enumerate(region_names)]
    )

    # --- Write ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:

        # Event IDs
        f.create_dataset("event_id_columns",
                         data=np.array(event_id_columns, dtype=object))
        f.create_dataset("event_ids", data=event_ids, dtype="int64")

        # Phi
        f.create_dataset("phi_columns", data=np.array(phi_order, dtype=object))
        f.create_dataset("phi_matrix", data=phi_matrix, dtype="float32")

        # Target
        f.create_dataset("target_columns", data=np.array(voxel_keys, dtype=object))
        f.create_dataset("target_matrix", data=target_matrix, dtype="int32")

        # Regions (cross-check pair)
        f.create_dataset("region_columns", data=np.array(region_names, dtype=object))
        f.create_dataset("region_matrix", data=region_matrix_from_voxels, dtype="int32")
        f.create_dataset("region_matrix_resnet",
                         data=region_matrix_from_resnet, dtype="int32")

        # Primaries
        f.create_dataset("primaries", data=n_events)

        # Material map
        with open(config["mapping"]["material_mapping_file"], "r") as mf:
            mat_map = json.load(mf)
        mat_grp = f.create_group("mat_map")
        for name, mid in mat_map.items():
            if isinstance(mid, int):
                mat_grp.create_dataset(name, data=mid)

        # Volume map
        with open(config["mapping"]["volume_mapping_file"], "r") as vf:
            vol_map = json.load(vf)
        vol_grp = f.create_group("vol_map")
        for name, vid in vol_map.items():
            if isinstance(vid, int):
                vol_grp.create_dataset(name, data=vid)

        # Voxel geometry
        voxels_grp = f.create_group("voxels")
        for voxel in voxels_json:
            idx = voxel["index"]
            v_grp = voxels_grp.create_group(idx)
            v_grp.create_dataset(
                "center", data=np.array(voxel["center"], dtype=np.float32))
            corners_grp = v_grp.create_group("corners")
            corners = np.array(voxel["corners"])
            corners_grp.create_dataset("x", data=corners[:, 0])
            corners_grp.create_dataset("y", data=corners[:, 1])
            corners_grp.create_dataset("z", data=corners[:, 2])
            v_grp.create_dataset("layer", data=voxel["layer"])

    file_size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n✓ Output written: {output_path}")
    print(f"  Events:        {n_events:,}")
    print(f"  phi_matrix:    {phi_matrix.shape} float32")
    print(f"  target_matrix: {target_matrix.shape} int32")
    print(f"  region_matrix: {region_matrix_from_voxels.shape} int32")
    print(f"  Voxels:        {len(voxel_keys)}")
    print(f"  Size:          {file_size_mb:.1f} MB")

    # Cross-check: region sums
    diff = np.abs(region_matrix_from_voxels - region_matrix_from_resnet)
    print(f"\n  Region cross-check (voxel_sum vs resnet):")
    for idx, name in enumerate(region_names):
        mean_diff = np.mean(diff[:, idx])
        max_diff = np.max(diff[:, idx])
        print(f"    {name:>5s}: mean_abs_diff={mean_diff:.2f}, "
              f"max_abs_diff={max_diff:.0f}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("NC-Score Generative Pipeline")
    print("=" * 70)

    # --- Config ---
    print("\n[1/7] Loading config...")
    config = toml.load(args.config)

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        print(f"  Random seed: {args.seed}")
    else:
        print(f"  Random seed: None (non-deterministic)")

    # --- Hardware ---
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        print("  Device: CPU")
    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Device: GPU ({len(gpus)} available)")
        else:
            print("  Device: CPU (no GPUs found)")

    # --- Discover files ---
    t_step = time.time()
    print("\n[2/7] Discovering simulation files...")
    sim_files = discover_sim_files(args.input_dir, args.nested)
    print(f"  Found {len(sim_files)} simulation file(s)")
    print(f"  ⏱ {time.time() - t_step:.1f}s")

    # --- Extract NC events ---
    t_step = time.time()
    print("\n[3/7] Extracting NC events...")
    phi_data, event_ids, event_id_columns = extract_nc_events(
        sim_files, config, max_events=args.max_events
    )
    print(f"  ⏱ {time.time() - t_step:.1f}s")

    # --- Normalize phi ---
    t_step = time.time()
    print("\n[4/7] Normalizing phi parameters...")
    phi_normalized = normalize_phi(phi_data, config)
    print(f"  ⏱ {time.time() - t_step:.1f}s")

    # --- Load voxel geometry & build mappings ---
    t_step = time.time()
    print("\n[5/7] Loading voxel geometry...")
    voxels_json, voxel_keys = load_voxel_geometry(args.voxel_json)
    grid_shapes, voxel_to_grid, grid_to_voxel = build_grid_mappings(
        voxel_keys, config
    )
    print(f"  ⏱ {time.time() - t_step:.1f}s")

    # --- Prepare config for model ---
    # Mirror train.py config preparation
    config["EMBED"] = config["model"]["embed_dim"]
    config["PAD"] = {}

    merged_regions = config["regions"].get("merged_regions", [])
    active_regions = config["training"].get("active_regions", {}).get(
        "enabled", ["PIT", "BOT", "WALL", "TOP"]
    )

    if set(merged_regions).issubset(set(active_regions)):
        layer_names = [r for r in active_regions if r not in merged_regions]
        layer_names.append("PITBOT")
        layer_names.sort()
    else:
        layer_names = active_regions

    config["LAYER_NAMES"] = layer_names
    config["num_steps"] = config["diffusion"]["num_steps"]
    config["ema_decay"] = config["training"]["ema_decay"]

    config["AREA_RATIOS"] = {}
    for region in layer_names:
        if region == "PITBOT":
            config["AREA_RATIOS"][region] = config["normalization"]["area_ratios"]["PITBOT"]
        else:
            config["AREA_RATIOS"][region] = config["normalization"]["area_ratios"][region]

    # Set shapes and auto-padding from inferred grid shapes
    for region_name, shape in grid_shapes.items():
        config[f"SHAPE_{region_name}"] = list(shape)

        # Auto-padding (mirrors data_loader)
        geo_source = region_name
        if region_name == "PITBOT":
            geo_source = merged_regions[0]  # Use PIT's geometry config

        geo_cfg = config["geometry"][geo_source]
        periodic = []
        if geo_source in ["TOP", "PIT", "BOT"]:
            if geo_cfg.get("periodic_y", False):
                periodic.append(0)
            if geo_cfg.get("periodic_x", False):
                periodic.append(1)
        else:
            if geo_cfg.get("periodic_phi", False):
                periodic.append(1)
            if geo_cfg.get("periodic_z", False):
                periodic.append(0)

        depth = config["model"]["unet_block_depth"]
        divisor = 2 ** depth
        n0, n1 = shape[0], shape[1]

        pad0 = (0, 0)
        if n0 % divisor != 0:
            target = ((n0 // divisor) + 1) * divisor
            total = target - n0
            pad0 = (total // 2, total - total // 2)

        pad1 = (0, 0)
        if n1 % divisor != 0:
            target = ((n1 // divisor) + 1) * divisor
            total = target - n1
            pad1 = (total // 2, total - total // 2)

        config["PAD"][region_name] = (pad0, pad1)

    # --- Load model & generate ---
    t_step = time.time()
    print("\n[6/7] Loading model and generating...")
    t_load = time.time()
    model = load_model(config, args.checkpoint_dir)
    print(f"  Model load: {time.time() - t_load:.1f}s")
    voxel_grids_raw, area_hits_raw = run_inference(
        model, phi_normalized, args.batch_size, config
    )
    print(f"  ⏱ Total step 6: {time.time() - t_step:.1f}s")

    # --- Remap & write output ---
    t_step = time.time()
    print("\n[7/7] Remapping grids to voxels and writing output...")
    t_remap = time.time()
    voxel_hits, region_hits_from_voxels = remap_grids_to_voxels(
        voxel_grids_raw,
        grid_to_voxel,
        voxel_to_grid,
        voxel_keys,
        config["regions"],
        merged_regions,
    )

    print(f"  Remap: {time.time() - t_remap:.1f}s")
    t_write = time.time()
    write_output_hdf5(
        args.output_file,
        phi_data,
        voxel_hits,
        region_hits_from_voxels,
        area_hits_raw,
        voxels_json,
        voxel_keys,
        config,
        event_ids=event_ids,
        event_id_columns=event_id_columns,
    )

    print(f"  HDF5 write: {time.time() - t_write:.1f}s")
    print(f"  ⏱ Total step 7: {time.time() - t_step:.1f}s")

    elapsed = time.time() - t_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"\nTotal runtime: {h:02d}:{m:02d}:{s:05.2f}")
    print("Done.")
    print(f"\n{'='*50}")


if __name__ == "__main__":
    main()