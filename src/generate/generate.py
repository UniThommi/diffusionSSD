#!/usr/bin/env python3
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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import toml


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

def extract_nc_events(sim_files: List[str], config: dict) -> dict:
    """
    Extract neutron capture events from simulation HDF5 files.

    Each NC event is identified by unique (evtid, nC_track_id) tuple.

    Args:
        sim_files: List of simulation HDF5 paths
        config: Full config dict

    Returns:
        Dict with keys matching ML-format phi fields, each value is np.ndarray
        of shape (n_nc_events,)
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

    for fpath in sim_files:
        with h5py.File(fpath, "r") as f:
            nc_group = f["hit"]["MyNeutronCaptureOutput"]

            # Read NC-level data (pages format)
            evtid = np.array(nc_group["evtid"]["pages"])
            track_id = np.array(nc_group["nC_track_id"]["pages"])

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
            mat_id = np.array(nc_group["nC_material_id"]["pages"]).astype(np.float32)
            vol_id = np.array(nc_group["nC_phys_vol_id"]["pages"]).astype(np.float32)
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
            p_mean_r = np.zeros(n_events, dtype=np.float64)
            p_mean_z = np.zeros(n_events, dtype=np.float64)

            for i in range(n_events):
                gamma_es = [g1_e[i], g2_e[i], g3_e[i], g4_e[i]]
                gamma_pxs = [g1_px[i], g2_px[i], g3_px[i], g4_px[i]]
                gamma_pys = [g1_py[i], g2_py[i], g3_py[i], g4_py[i]]
                gamma_pzs = [g1_pz[i], g2_pz[i], g3_pz[i], g4_pz[i]]

                r_val = r_mm[i]
                count = 0
                sum_pr = 0.0
                sum_pz = 0.0

                for j in range(4):
                    if gamma_es[j] > 0:
                        # Radial projection: p_r = (px*x + py*y) / r
                        if r_val > 0:
                            pr = (gamma_pxs[j] * x_mm[i] + gamma_pys[j] * y_mm[i]) / r_val
                        else:
                            pr = 0.0
                        sum_pr += pr
                        sum_pz += gamma_pzs[j]
                        count += 1

                if count > 0:
                    p_mean_r[i] = sum_pr / count
                    p_mean_z[i] = sum_pz / count

            # Deduplicate by (evtid, track_id) - should already be unique
            # but verify just in case
            nc_keys = set()
            mask = np.ones(n_events, dtype=bool)
            for i in range(n_events):
                key = (int(evtid[i]), int(track_id[i]))
                if key in nc_keys:
                    mask[i] = False
                else:
                    nc_keys.add(key)

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

            total_ncs += n_unique

        print(f"  {fpath}: {n_unique} NC events extracted")

    # Concatenate all files
    result = {k: np.concatenate(v).astype(np.float32) for k, v in fields.items()}
    print(f"\nTotal NC events: {total_ncs:,}")
    return result


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
                mat_id_to_name[mid] = name

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
    import tensorflow as tf
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


def run_inference(
    model,
    phi_normalized: np.ndarray,
    batch_size: int,
    config: dict,
) -> Tuple[dict, np.ndarray]:
    """
    Run batched model inference.

    Args:
        model: Loaded scoreDiffusion model
        phi_normalized: (n_events, phi_dim) normalized conditioning
        batch_size: Inference batch size
        config: Full config dict

    Returns:
        voxel_grids: {region: (n_events, n_axis0, n_axis1, 1)} generated grids
        area_hits_raw: (n_events, 4) denormalized area hit counts
    """
    import tensorflow as tf

    n_events = phi_normalized.shape[0]
    n_batches = (n_events + batch_size - 1) // batch_size
    voxel_hit_max = config["normalization"]["voxel_hit_max"]

    # Collect results
    area_hits_list = []
    voxel_grids = {area: [] for area in model.active_areas}

    print(f"\n=== Generating {n_events:,} events in {n_batches} batches ===")
    t_start = time.time()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_events)
        cond_batch = tf.constant(phi_normalized[start:end], dtype=tf.float32)

        # Generate via model (uses EMA weights internally)
        voxels_batch, area_hits_batch = model.generate(cond_batch)

        area_hits_list.append(area_hits_batch)
        for area_name in model.active_areas:
            voxel_grids[area_name].append(voxels_batch[area_name])

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t_start
            rate = (end) / elapsed
            eta = (n_events - end) / rate if rate > 0 else 0
            print(f"  Batch {batch_idx + 1}/{n_batches} "
                  f"({end}/{n_events}) - "
                  f"{rate:.0f} events/s, ETA: {eta:.0f}s")

    # Concatenate
    area_hits_norm = np.concatenate(area_hits_list, axis=0)  # (n_events, 4)
    for area_name in model.active_areas:
        voxel_grids[area_name] = np.concatenate(
            voxel_grids[area_name], axis=0
        )

    # Denormalize area hits: area_norm * voxel_hit_max
    area_hits_raw = area_hits_norm * voxel_hit_max
    area_hits_raw = np.clip(area_hits_raw, 0, None)

    # Denormalize voxels: voxel_fraction * area_hits_raw[region]
    region_order = ["pit", "bot", "wall", "top"]
    for area_name in model.active_areas:
        if area_name == "PITBOT":
            # Merged: pit + bot
            area_total = area_hits_raw[:, 0] + area_hits_raw[:, 1]
        else:
            region_idx = region_order.index(area_name.lower())
            area_total = area_hits_raw[:, region_idx]

        # Broadcast: (n,) → (n, 1, 1, 1)
        n_spatial = len(voxel_grids[area_name].shape) - 1
        area_broadcast = area_total.reshape((-1,) + (1,) * n_spatial)
        voxel_grids[area_name] = voxel_grids[area_name] * area_broadcast
        voxel_grids[area_name] = np.clip(voxel_grids[area_name], 0, None)

    elapsed = time.time() - t_start
    print(f"\nGeneration complete: {elapsed:.1f}s "
          f"({n_events / elapsed:.0f} events/s)")

    return voxel_grids, area_hits_raw


# =============================================================================
# Step 7: Grid-to-voxel remapping & HDF5 output
# =============================================================================

def remap_grids_to_voxels(
    voxel_grids: dict,
    grid_to_voxel: dict,
    voxel_to_grid: dict,
    voxel_keys: List[str],
    region_cfg: dict,
    merged_regions: List[str],
) -> Tuple[dict, dict]:
    """
    Remap generated grids back to individual voxel hit arrays.

    Only physical voxels (those in voxel_json) get values.
    Padding positions are ignored.

    Args:
        voxel_grids: {region: (n_events, n_axis0, n_axis1, 1)}
        grid_to_voxel: {region: {(a0, a1): voxel_key}}
        voxel_to_grid: {region: {voxel_key: (a0, a1)}}
        voxel_keys: Full sorted list of voxel keys
        region_cfg: config['regions']
        merged_regions: List of merged region names

    Returns:
        voxel_hits: {voxel_key: (n_events,) float32}
        region_hits_from_voxels: {'pit': (n_events,), ...} summed from voxels
    """
    n_events = None

    # Determine which region grid each voxel belongs to
    def get_grid_region(voxel_key: str) -> str:
        """Map voxel key to grid region name."""
        if voxel_key.startswith(region_cfg["pit_prefix"]):
            base = "PIT"
        elif voxel_key.startswith(region_cfg["bot_prefix"]):
            base = "BOT"
        elif voxel_key.startswith(region_cfg["top_prefix"]):
            base = "TOP"
        else:
            base = "WALL"

        # Check if this region is merged
        if merged_regions and base in merged_regions:
            return "".join(merged_regions)  # e.g. "PITBOT"
        return base

    voxel_hits = {}
    region_sums = {"pit": None, "bot": None, "wall": None, "top": None}

    for voxel_key in voxel_keys:
        grid_region = get_grid_region(voxel_key)
        grid = voxel_grids[grid_region]

        if n_events is None:
            n_events = grid.shape[0]

        # Look up grid position for this voxel
        mapping = voxel_to_grid[grid_region]
        if voxel_key not in mapping:
            raise RuntimeError(
                f"Voxel {voxel_key} not found in grid mapping for {grid_region}"
            )

        a0, a1 = mapping[voxel_key]
        voxel_hits[voxel_key] = grid[:, a0, a1, 0]  # (n_events,)

    # Compute region sums from voxels
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
            region_sums[region] = np.zeros(n_events, dtype=np.float32)
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
) -> None:
    """
    Write ML-format HDF5 file.

    Args:
        output_path: Output file path
        phi_data: Dict of raw phi arrays (all 30 features)
        voxel_hits: {voxel_key: (n_events,)}
        region_hits_from_voxels: {'pit': (n_events,), ...}
        area_hits_from_resnet: (n_events, 4) - ResNet-generated area hits
        voxels_json: Voxel geometry from JSON
        voxel_keys: Sorted voxel index list
        config: Full config dict
    """
    n_events = len(phi_data["xNC_mm"])

    # PHI_HDF5_ORDER (must match data_loader)
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:

        # === /phi/ ===
        phi_grp = f.create_group("phi")
        for name in phi_order:
            phi_grp.create_dataset(name, data=phi_data[name], dtype="float32")

        # === /target/ (generated voxel hits) ===
        target_grp = f.create_group("target")
        for key in voxel_keys:
            target_grp.create_dataset(
                key, data=voxel_hits[key], dtype="float32"
            )

        # === /target_regions/ (summed from voxels) ===
        region_grp = f.create_group("target_regions")
        for region_name in ["pit", "bot", "wall", "top"]:
            region_grp.create_dataset(
                region_name,
                data=region_hits_from_voxels[region_name],
                dtype="float32",
            )

        # === /target_regions_resnet/ (direct ResNet output for comparison) ===
        resnet_grp = f.create_group("target_regions_resnet")
        for idx, region_name in enumerate(["pit", "bot", "wall", "top"]):
            resnet_grp.create_dataset(
                region_name,
                data=area_hits_from_resnet[:, idx],
                dtype="float32",
            )

        # === /mat_map/ ===
        with open(config["mapping"]["material_mapping_file"], "r") as mf:
            mat_map = json.load(mf)
        mat_grp = f.create_group("mat_map")
        for name, mid in mat_map.items():
            if isinstance(mid, int):
                mat_grp.create_dataset(name, data=mid)

        # === /vol_map/ ===
        with open(config["mapping"]["volume_mapping_file"], "r") as vf:
            vol_map = json.load(vf)
        vol_grp = f.create_group("vol_map")
        for name, vid in vol_map.items():
            if isinstance(vid, int):
                vol_grp.create_dataset(name, data=vid)

        # === /voxels/ (geometry from JSON) ===
        voxels_grp = f.create_group("voxels")
        for voxel in voxels_json:
            idx = voxel["index"]
            v_grp = voxels_grp.create_group(idx)
            v_grp.create_dataset(
                "center", data=np.array(voxel["center"], dtype=np.float32)
            )
            corners_grp = v_grp.create_group("corners")
            corners = np.array(voxel["corners"])  # (8, 3)
            corners_grp.create_dataset("x", data=corners[:, 0])
            corners_grp.create_dataset("y", data=corners[:, 1])
            corners_grp.create_dataset("z", data=corners[:, 2])
            v_grp.create_dataset("layer", data=voxel["layer"])

        # === /primaries ===
        f.create_dataset("primaries", data=n_events)

    print(f"\n✓ Output written: {output_path}")
    print(f"  Events: {n_events:,}")
    print(f"  Voxels: {len(voxel_keys)}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")


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

    # --- Hardware ---
    import tensorflow as tf
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
    print("\n[2/7] Discovering simulation files...")
    sim_files = discover_sim_files(args.input_dir, args.nested)
    print(f"  Found {len(sim_files)} simulation file(s)")

    # --- Extract NC events ---
    print("\n[3/7] Extracting NC events...")
    phi_data = extract_nc_events(sim_files, config)

    # --- Normalize phi ---
    print("\n[4/7] Normalizing phi parameters...")
    phi_normalized = normalize_phi(phi_data, config)

    # --- Load voxel geometry & build mappings ---
    print("\n[5/7] Loading voxel geometry...")
    voxels_json, voxel_keys = load_voxel_geometry(args.voxel_json)
    grid_shapes, voxel_to_grid, grid_to_voxel = build_grid_mappings(
        voxel_keys, config
    )

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
    print("\n[6/7] Loading model and generating...")
    model = load_model(config, args.checkpoint_dir)
    voxel_grids_raw, area_hits_raw = run_inference(
        model, phi_normalized, args.batch_size, config
    )

    # --- Remap & write output ---
    print("\n[7/7] Remapping grids to voxels and writing output...")
    voxel_hits, region_hits_from_voxels = remap_grids_to_voxels(
        voxel_grids_raw,
        grid_to_voxel,
        voxel_to_grid,
        voxel_keys,
        config["regions"],
        merged_regions,
    )

    write_output_hdf5(
        args.output_file,
        phi_data,
        voxel_hits,
        region_hits_from_voxels,
        area_hits_raw,
        voxels_json,
        voxel_keys,
        config,
    )

    elapsed = time.time() - t_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"\nTotal runtime: {h:02d}:{m:02d}:{s:05.2f}")
    print("Done.")


if __name__ == "__main__":
    main()