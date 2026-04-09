#!/usr/bin/env python3
"""
NC-Score Dataset Loader (v2 — Consolidated HDF5 Format)

Reads directly from consolidated HDF5 matrices:
  - phi_matrix:    (n_events, 30)    float32
  - target_matrix: (n_events, 9583)  int32
  - region_matrix: (n_events, 4)     int32

No .npy preprocessing cache needed.

Data loading modes (auto-selected based on available RAM):
  - RAM mode: Full dataset loaded into memory → tf.data.Dataset.from_tensor_slices
  - Memmap mode: HDF5 chunk-streaming → tf.data.Dataset.from_generator

Author: Thomas Buerger (University of Tübingen)
"""

import h5py
import tensorflow as tf
import numpy as np
import json
import os
import time
import psutil
from pathlib import Path


class voxelDataset:
    """
    NC-Score Dataset Loader (Consolidated HDF5 Format)
    
    Loads neutron capture events from consolidated 2D HDF5 matrices.
    Handles normalization and One-Hot encoding based on config.toml.
    """
    
    def __init__(self, h5_path: str, config: dict, is_validation: bool = False):
        self.h5_path = h5_path
        self.config = config
        self.is_validation = is_validation
        
        self.feature_config = config['features']
        self.norm_config = config['normalization']
        self.mapping_config = config['mapping']
        self.onehot_config = config['onehot']
        self.region_config = config['regions']
        self.model_config = config['model']

        # Read structure from HDF5
        with h5py.File(h5_path, "r") as f:
            if "phi_matrix" not in f or "target_matrix" not in f or "region_matrix" not in f:
                raise ValueError(
                    "HDF5 must contain 'phi_matrix', 'target_matrix', and 'region_matrix'. "
                    "Regenerate data with updated post-processing script."
                )
            self.n_events_total = f["phi_matrix"].shape[0]
            self.n_voxels = f["target_matrix"].shape[1]
            
            # Read column metadata
            self.phi_columns = [c.decode() if isinstance(c, bytes) else c 
                               for c in f["phi_columns"][:]]
            self.voxel_keys = [c.decode() if isinstance(c, bytes) else c 
                              for c in f["target_columns"][:]]
            self.region_columns = [c.decode() if isinstance(c, bytes) else c 
                                  for c in f["region_columns"][:]]
        
        # === EVENT LIMITING ===
        max_events = config['training'].get('max_events', None)
        print(f"[DEBUG] max_events from config: {max_events}")
        print(f"[DEBUG] is_validation: {is_validation}")
        
        if max_events is not None:
            if is_validation:
                train_events = min(max_events, self.n_events_total)
                train_frac = config['training']['train_val_split']
                val_frac = 1.0 - train_frac
                self.n_events = int(train_events * val_frac / train_frac)
                self.n_events = min(self.n_events, self.n_events_total)
                print(f"\n[Event Limiting - Validation]")
                print(f"  Total available: {self.n_events_total:,}")
                print(f"  Train events: {train_events:,}")
                print(f"  Val events (proportional): {self.n_events:,}")
            else:
                self.n_events = min(max_events, self.n_events_total)
                print(f"\n[Event Limiting - Training]")
                print(f"  Total available: {self.n_events_total:,}")
                print(f"  Using: {self.n_events:,} ({100*self.n_events/self.n_events_total:.1f}%)")
        else:
            self.n_events = self.n_events_total
            print(f"\n[Event Limiting] DISABLED - Using all {self.n_events:,} events")
        
        # Active features
        self.active_phi_raw = self.feature_config['active_phi'].copy()
        
        # === ONE-HOT ENCODING SETUP ===
        self.enable_material_onehot = self.onehot_config['enable_material_onehot']
        self.enable_volume_onehot = self.onehot_config['enable_volume_onehot']
        
        # Material mapping
        print("\n[Material Mapping]")
        with open(self.mapping_config['material_mapping_file'], 'r') as f:
            material_mapping_raw = json.load(f)
        
        self.material_mapping = {}
        for mat_name, mat_id in material_mapping_raw.items():
            if isinstance(mat_id, int):
                self.material_mapping[mat_id] = mat_name
        self.n_materials = len(self.material_mapping)
        print(f"  Loaded {self.n_materials} materials")
        
        if self.enable_material_onehot:
            self.material_names = sorted(set(self.material_mapping.values()))
            self.material_to_idx = {name: idx for idx, name in enumerate(self.material_names)}
            self.n_material_categories = len(self.material_names)
            print(f"  One-Hot categories: {self.n_material_categories}")
            print(f"  Materials: {', '.join(self.material_names)}")
        else:
            self.n_material_categories = 0
        
        # Volume mapping
        if self.enable_volume_onehot:
            print("\n[Volume Mapping]")
            with open(self.mapping_config['volume_mapping_file'], 'r') as f:
                volume_mapping_raw = json.load(f)
            self.volume_mapping = {}
            for vol_name, vol_id in volume_mapping_raw.items():
                if isinstance(vol_id, int):
                    final_name = "noVolume" if vol_name == "" else vol_name
                    self.volume_mapping[vol_id] = final_name
            print(f"  Loaded {len(self.volume_mapping):,} volumes")
            
            from src.data.volume_groups import VolumeGrouper
            self.volume_grouper = VolumeGrouper()
            print(f"  Defined groups: {self.volume_grouper.n_groups}")
            
            self.volume_to_group, self.volume_group_contents = \
                self.volume_grouper.build_lookup_dict(self.volume_mapping)
            non_empty = sum(1 for v in self.volume_group_contents.values() if len(v) > 0)
            print(f"  ✓ All volumes matched successfully")
            print(f"  Active groups: {non_empty}/{self.volume_grouper.n_groups}")
            self.n_volume_categories = self.volume_grouper.n_groups
        else:
            self.n_volume_categories = 0
            print("\n[Volume Mapping] DISABLED - Skipped")
        
        # Adjust active_phi
        self.active_phi = [
            f for f in self.active_phi_raw
            if not ((f == 'matID' and self.enable_material_onehot) or
                   (f == 'volID' and self.enable_volume_onehot))
        ]
        if self.enable_material_onehot and 'matID' in self.active_phi_raw:
            print(f"  ✓ matID removed from active_phi (replaced by One-Hot)")
        if self.enable_volume_onehot and 'volID' in self.active_phi_raw:
            print(f"  ✓ volID removed from active_phi (replaced by One-Hot)")
        
        # Build phi column index mapping
        self._phi_col_indices = {name: idx for idx, name in enumerate(self.phi_columns)}
        
        # === VOXEL-TO-REGION MAPPING ===
        print("\n[Voxel-to-Region Mapping]")
        self.voxel_to_region_idx = np.zeros(len(self.voxel_keys), dtype=np.int32)
        self.merged_regions = config['regions'].get('merged_regions', [])
        self.region_voxel_indices = {'PIT': [], 'BOT': [], 'WALL': [], 'TOP': []}
        
        for voxel_idx, voxel_key in enumerate(self.voxel_keys):
            if voxel_key.startswith(self.region_config['pit_prefix']):
                region = 'PIT'
                region_idx = 0
            elif voxel_key.startswith(self.region_config['bot_prefix']):
                region = 'BOT'
                region_idx = 1
            elif voxel_key.startswith(self.region_config['top_prefix']):
                region = 'TOP'
                region_idx = 3
            else:
                region = 'WALL'
                region_idx = 2
            self.voxel_to_region_idx[voxel_idx] = region_idx
            self.region_voxel_indices[region].append(voxel_idx)
        
        for region in self.region_voxel_indices:
            self.region_voxel_indices[region] = np.array(
                self.region_voxel_indices[region], dtype=np.int32
            )
        
        if self.merged_regions:
            merged_name = ''.join(self.merged_regions)
            merged_indices = []
            for region in self.merged_regions:
                merged_indices.extend(self.region_voxel_indices[region].tolist())
            self.region_voxel_indices[merged_name] = np.array(merged_indices, dtype=np.int32)
            print(f"  {merged_name}: {len(merged_indices)} voxels (merged from {' + '.join(self.merged_regions)})")
        
        expected = self.region_config['expected_voxel_counts']
        for region, indices in self.region_voxel_indices.items():
            if region not in expected:
                continue
            actual = len(indices)
            exp = expected[region]
            if actual != exp:
                raise ValueError(f"Region {region}: Expected {exp} voxels, found {actual}")
            print(f"  {region.upper()}: {actual} voxels")
        
        # === GEOMETRY PROCESSING ===
        self.grid_shapes = {}
        self.periodic_axes = {}
        self.voxel_to_grid_mapping = {}
        
        regions_to_process = []
        for region in ['PIT', 'BOT', 'WALL', 'TOP']:
            if region in self.merged_regions and region != self.merged_regions[0]:
                continue
            if region in self.merged_regions and region == self.merged_regions[0]:
                merged_name = ''.join(self.merged_regions)
                regions_to_process.append((merged_name, self.merged_regions))
            else:
                regions_to_process.append((region, [region]))
        
        for region_name, source_regions in regions_to_process:
            geo_config = config['geometry'][source_regions[0]]
            
            periodic = []
            actual_region = source_regions[0]
            if actual_region in ['TOP', 'PIT', 'BOT']:
                if geo_config.get('periodic_y', False):
                    periodic.append(0)
                if geo_config.get('periodic_x', False):
                    periodic.append(1)
            else:
                if geo_config.get('periodic_phi', False):
                    periodic.append(1)
                if geo_config.get('periodic_r', False) or geo_config.get('periodic_z', False):
                    periodic.append(0)
            self.periodic_axes[region_name] = periodic
            
            if geo_config['use_auto_geometry']:
                print(f"\n[Auto-Geometry: {region_name}]")
                if len(source_regions) > 1:
                    print(f"  Merging: {' + '.join(source_regions)}")
                inferred_shape, voxel_mapping = self._infer_grid_shape_and_mapping(
                    region_name, source_regions
                )
                
                if len(source_regions) > 1:
                    expected_count = sum(
                        self.region_config['expected_voxel_counts'][r]
                        for r in source_regions
                    )
                else:
                    expected_count = self.region_config['expected_voxel_counts'][region_name]
                actual_count = len(self.region_voxel_indices[region_name])
                
                sparse_regions = ['TOP', 'PITBOT']
                if region_name not in sparse_regions:
                    if inferred_shape[0] * inferred_shape[1] != actual_count:
                        raise ValueError(
                            f"GEOMETRY MISMATCH: {region_name}\n"
                            f"  Inferred shape: {inferred_shape} → {inferred_shape[0]*inferred_shape[1]} voxels\n"
                            f"  Actual voxels: {actual_count}"
                        )
                else:
                    grid_size = inferred_shape[0] * inferred_shape[1]
                    utilization = actual_count / grid_size
                    print(f"  Grid utilization: {utilization*100:.1f}% ({actual_count}/{grid_size})")
                    if utilization < 0.3:
                        import warnings
                        warnings.warn(f"{region_name} grid utilization very low ({utilization*100:.1f}%).")
                
                if actual_count != expected_count:
                    raise ValueError(f"VOXEL COUNT MISMATCH: {region_name}")
                
                auto_pad = self._calculate_auto_padding(
                    inferred_shape,
                    depth=config['model']['unet_block_depth'],
                    periodic_axes=periodic
                )
                
                self.grid_shapes[region_name] = inferred_shape
                config['PAD'][region_name] = auto_pad
                self.voxel_to_grid_mapping[region_name] = voxel_mapping
                
                print(f"  Inferred shape: {inferred_shape}")
                print(f"  Auto-padding: {auto_pad}")
                print(f"  Periodic axes: {periodic}")
            else:
                self.grid_shapes[region_name] = tuple(geo_config['SHAPE'])
                pad_value = config['model']['padding'].get(region_name, 0)
                config['PAD'][region_name] = pad_value
                self.voxel_to_grid_mapping[region_name] = None
                print(f"\n[Manual Geometry: {region_name}]")
                print(f"  Shape: {self.grid_shapes[region_name]}")
                print(f"  Padding: {config['PAD'][region_name]}")
        
        # Calculate phi_dim
        base_dim = len(self.active_phi)
        self.phi_dim = base_dim + self.n_material_categories + self.n_volume_categories
        self.target_dim = len(self.voxel_keys)
        
        # === DATA LOADING MODE ===
        self._select_data_mode()
        
        print(f"\n✓ voxelDataset initialized")
        print(f"  Using events: {self.n_events:,} (Total available: {self.n_events_total:,})")
        print(f"  Base phi features: {base_dim}")
        if self.enable_material_onehot:
            print(f"  + Material One-Hot: {self.n_material_categories}")
        if self.enable_volume_onehot:
            print(f"  + Volume One-Hot: {self.n_volume_categories}")
        print(f"  = Total phi_dim: {self.phi_dim}")
        print(f"  Target voxels: {self.target_dim}")
        print(f"  Data mode: {self._data_mode}")

    def _select_data_mode(self):
        """Select RAM or HDF5-streaming mode based on config or available memory."""
        forced_mode = self.config['training'].get('data_mode', 'auto')
        
        bytes_per_event = (
            len(self.phi_columns) * 4 +
            self.n_voxels * 4 +
            4 * 4
        )
        estimated_ram = self.n_events * bytes_per_event
        available_ram = psutil.virtual_memory().available
        est_mb = estimated_ram / 1e6
        avail_mb = available_ram / 1e6
        
        if forced_mode == "hdf5":
            self._data_mode = "hdf5"
            print(f"\n[Data Mode] HDF5 streaming (forced, dataset={est_mb:.0f}MB)")
        elif forced_mode == "ram":
            self._data_mode = "ram"
            print(f"\n[Data Mode] RAM (forced, dataset={est_mb:.0f}MB, available={avail_mb:.0f}MB)")
            self._load_into_ram()
        else:
            threshold = available_ram * 0.8
            if estimated_ram < threshold:
                self._data_mode = "ram"
                print(f"\n[Data Mode] RAM (dataset={est_mb:.0f}MB, available={avail_mb:.0f}MB)")
                self._load_into_ram()
            else:
                self._data_mode = "hdf5"
                print(f"\n[Data Mode] HDF5 streaming (dataset={est_mb:.0f}MB > threshold={est_mb:.0f}MB)")
    
    def _load_into_ram(self):
        """Load data directly from HDF5 into RAM."""
        t_start = time.time()
        with h5py.File(self.h5_path, "r") as f:
            self._phi_raw = f["phi_matrix"][:self.n_events].copy()
            self._targets_raw = f["target_matrix"][:self.n_events].copy()
            self._regions_raw = f["region_matrix"][:self.n_events].copy()
        elapsed = time.time() - t_start
        print(f"  Loaded {self.n_events:,} events into RAM in {elapsed:.1f}s")

    # =========================================================================
    # Voxel Index Parsing & Geometry (unchanged)
    # =========================================================================

    def _parse_voxel_index(self, index_str: str, region: str):
        remainder = index_str[2:]
        if region == 'TOP':
            if len(remainder) != 4:
                raise ValueError(f"TOP voxel index '{index_str}' must have format 99YYXX (6 digits).")
            return (int(remainder[0:2]), int(remainder[2:4]))
        if region == 'WALL':
            if len(remainder) == 4:
                return (int(remainder[:2]), int(remainder[2:4]))
            elif len(remainder) == 5:
                return (int(remainder[:2]), int(remainder[2:5]))
            else:
                raise ValueError(f"Cannot parse WALL voxel index '{index_str}'.")
        if region in ['PIT', 'BOT']:
            if len(remainder) != 4:
                raise ValueError(f"{region} voxel index '{index_str}' must be 6 digits.")
            return (int(remainder[0:2]), int(remainder[2:4]))
        raise ValueError(f"Unknown region: {region}")
    
    def _infer_grid_shape_and_mapping(self, region: str, source_regions: list = None):
        if source_regions is None:
            source_regions = [region]
        
        region_indices = self.region_voxel_indices[region]
        region_keys = [self.voxel_keys[i] for i in region_indices]
        
        voxel_to_source = {}
        if len(source_regions) > 1:
            for source_region in source_regions:
                source_indices = self.region_voxel_indices[source_region]
                for idx in source_indices:
                    voxel_to_source[self.voxel_keys[idx]] = source_region
        
        region_keys_sorted = sorted(region_keys)
        parsed_coords = []
        for key in region_keys_sorted:
            parse_region = voxel_to_source.get(key, region)
            coords = self._parse_voxel_index(key, parse_region)
            parsed_coords.append(coords)
        
        axis0_coords = [c[0] for c in parsed_coords]
        axis1_coords = [c[1] for c in parsed_coords]
        axis0_min, axis0_max = min(axis0_coords), max(axis0_coords)
        axis1_min, axis1_max = min(axis1_coords), max(axis1_coords)
        n_axis0 = axis0_max - axis0_min + 1
        n_axis1 = axis1_max - axis1_min + 1
        
        print(f"  Axis 0: min={axis0_min}, max={axis0_max}, n={n_axis0}")
        print(f"  Axis 1: min={axis1_min}, max={axis1_max}, n={n_axis1}")
        
        mapping = {}
        for list_idx in region_indices:
            key = self.voxel_keys[list_idx]
            sorted_idx = region_keys_sorted.index(key)
            raw_coords = parsed_coords[sorted_idx]
            normalized_coords = (raw_coords[0] - axis0_min, raw_coords[1] - axis1_min)
            mapping[list_idx] = normalized_coords
        
        return (n_axis0, n_axis1, 1), mapping
    
    def _calculate_auto_padding(self, shape, depth=4, periodic_axes=[]):
        divisor = 2 ** depth
        n_axis0, n_axis1 = shape[0], shape[1]
        
        if n_axis0 % divisor != 0:
            target = ((n_axis0 // divisor) + 1) * divisor
            total_pad = target - n_axis0
            pad_axis0 = (total_pad // 2, total_pad - total_pad // 2)
        else:
            pad_axis0 = (0, 0)
        
        if n_axis1 % divisor != 0:
            target = ((n_axis1 // divisor) + 1) * divisor
            total_pad = target - n_axis1
            pad_axis1 = (total_pad // 2, total_pad - total_pad // 2)
        else:
            pad_axis1 = (0, 0)
        
        return (pad_axis0, pad_axis1)

    # =========================================================================
    # Voxel Splitting
    # =========================================================================
    
    def _split_voxels_to_regions(self, voxel_batch: np.ndarray) -> dict:
        batch_size = voxel_batch.shape[0]
        region_batches = {}
        
        for region_name in self.grid_shapes.keys():
            indices = self.region_voxel_indices[region_name]
            region_voxels = voxel_batch[:, indices]
            grid_shape = self.grid_shapes[region_name]
            
            if self.voxel_to_grid_mapping[region_name] is not None:
                mapping = self.voxel_to_grid_mapping[region_name]
                grid = np.zeros(
                    (batch_size, grid_shape[0], grid_shape[1], grid_shape[2]),
                    dtype=np.float32
                )
                for voxel_list_idx, (axis0, axis1) in mapping.items():
                    region_pos = np.where(indices == voxel_list_idx)[0]
                    if len(region_pos) > 0:
                        grid[:, axis0, axis1, 0] = region_voxels[:, region_pos[0]]
                region_batches[region_name] = grid
            else:
                n_voxels = len(indices)
                grid_size = np.prod(grid_shape[:-1])
                if n_voxels != grid_size:
                    if n_voxels < grid_size:
                        pad_width = ((0, 0), (0, grid_size - n_voxels))
                        region_voxels = np.pad(region_voxels, pad_width, mode='constant')
                    else:
                        region_voxels = region_voxels[:, :grid_size]
                target_shape = (batch_size,) + grid_shape
                region_batches[region_name] = region_voxels.reshape(target_shape)
        
        return region_batches

    # =========================================================================
    # Normalization
    # =========================================================================
    
    def _normalize_phi(self, phi_raw: np.ndarray) -> np.ndarray:
        import warnings
        batch_size = phi_raw.shape[0]
        
        # Build dict from column names
        phi_dict = {name: phi_raw[:, self._phi_col_indices[name]] 
                   for name in self.phi_columns}
        
        E_max = self.norm_config['E_max']
        r_cyl = self.norm_config['r_cylinder']
        z_min = self.norm_config['z_min']
        z_max = self.norm_config['z_max']
        h_cyl = z_max - z_min
        angle_max = self.norm_config['angle_max']
        gamma_count_max = self.norm_config['gamma_count_max']
        
        phi_normalized = []
        
        for feature_name in self.active_phi:
            if feature_name not in phi_dict:
                raise ValueError(f"Feature '{feature_name}' not in HDF5!")
            raw_values = phi_dict[feature_name]
            
            if feature_name in ['E_gamma_tot_keV', 'gammaE1_keV', 'gammaE2_keV',
                               'gammaE3_keV', 'gammaE4_keV']:
                normalized = raw_values / E_max
                if np.any(normalized > 1.0):
                    max_val = np.max(raw_values)
                    raise RuntimeError(
                        f"NORMALIZATION OVERFLOW: {feature_name}\n"
                        f"  Max value: {max_val:.2f} keV, Norm factor: {E_max:.2f} keV\n"
                        f"  → Update config.toml: normalization.E_max = {np.ceil(max_val)}"
                    )
            elif feature_name in ['gammapx1', 'gammapx2', 'gammapx3', 'gammapx4',
                                 'gammapy1', 'gammapy2', 'gammapy3', 'gammapy4',
                                 'gammapz1', 'gammapz2', 'gammapz3', 'gammapz4',
                                 'p_mean_r', 'p_mean_z']:
                normalized = raw_values
            elif feature_name in ['r_NC_mm', 'dist_to_wall_mm']:
                normalized = raw_values / r_cyl
            elif feature_name in ['zNC_mm', 'dist_to_bot_mm', 'dist_to_top_mm']:
                normalized = (raw_values - z_min) / h_cyl
            elif feature_name in ['xNC_mm', 'yNC_mm']:
                normalized = (raw_values + r_cyl) / (2 * r_cyl)
                if np.any((normalized < 0) | (normalized > 1)):
                    warnings.warn(f"WARNING: {feature_name} outside [-r_cyl, r_cyl]")
            elif feature_name == 'phi_NC_rad':
                normalized = np.mod(raw_values, angle_max) / angle_max
            elif feature_name == '#gamma':
                normalized = raw_values / gamma_count_max
            elif feature_name in ['matID', 'volID']:
                normalized = raw_values
            else:
                raise ValueError(f"Unknown feature: {feature_name}")
            
            phi_normalized.append(normalized)
        
        phi_base = np.stack(phi_normalized, axis=1)
        onehot_features = []
        
        if self.enable_material_onehot:
            matID_raw = phi_dict['matID'].astype(int)
            material_onehot = np.zeros((batch_size, self.n_material_categories), dtype=np.float32)
            for i in range(batch_size):
                mat_id = matID_raw[i]
                if mat_id in self.material_mapping:
                    mat_name = self.material_mapping[mat_id]
                    mat_idx = self.material_to_idx[mat_name]
                    material_onehot[i, mat_idx] = 1.0
                else:
                    raise RuntimeError(f"Unknown matID: {mat_id}")
            onehot_features.append(material_onehot)
        
        if self.enable_volume_onehot:
            volID_raw = phi_dict['volID'].astype(int)
            volume_onehot = np.zeros((batch_size, self.n_volume_categories), dtype=np.float32)
            for i in range(batch_size):
                vol_id = volID_raw[i]
                if vol_id in self.volume_to_group:
                    vol_group_idx = self.volume_to_group[vol_id]
                    volume_onehot[i, vol_group_idx] = 1.0
                else:
                    vol_name = self.volume_mapping.get(vol_id, f"UNKNOWN_{vol_id}")
                    raise RuntimeError(f"Unknown volID: {vol_id} ('{vol_name}')")
            onehot_features.append(volume_onehot)
        
        if onehot_features:
            return np.concatenate([phi_base] + onehot_features, axis=1).astype(np.float32)
        return phi_base.astype(np.float32)
    
    def _normalize_target(self, target_raw: np.ndarray, region_raw: np.ndarray) -> np.ndarray:
        region_hits_per_voxel = region_raw[:, self.voxel_to_region_idx]
        safe_divisor = np.where(region_hits_per_voxel > 0, region_hits_per_voxel, 1.0)
        target_normalized = np.where(
            region_hits_per_voxel > 0,
            target_raw / safe_divisor,
            0.0
        )
        return target_normalized.astype(np.float32)
    
    def _normalize_region_targets(self, region_raw: np.ndarray) -> np.ndarray:
        voxel_hit_max = self.norm_config['voxel_hit_max']
        normalized = region_raw / voxel_hit_max
        if np.any(normalized > 1.0):
            max_val = np.max(region_raw)
            raise RuntimeError(
                f"NORMALIZATION OVERFLOW: Voxel Hits\n"
                f"  Max value: {max_val:.1f}, Norm factor: {voxel_hit_max:.1f}\n"
                f"  → Update config.toml: normalization.voxel_hit_max = {np.ceil(max_val)}"
            )
        return normalized.astype(np.float32)

    # =========================================================================
    # Dataset Creation
    # =========================================================================

    def _process_chunk(self, start_idx: int, end_idx: int):
        phi_chunk = self._phi_raw[start_idx:end_idx]
        target_chunk = self._targets_raw[start_idx:end_idx].astype(np.float32)
        region_chunk = self._regions_raw[start_idx:end_idx].astype(np.float32)
        
        phi_normalized = self._normalize_phi(phi_chunk)
        target_normalized = self._normalize_target(target_chunk, region_chunk)
        region_normalized = self._normalize_region_targets(region_chunk)
        voxel_regions = self._split_voxels_to_regions(target_normalized)
        
        return voxel_regions, region_normalized, phi_normalized

    def get_dataset(self, shuffle=True, rank=0, size=1):
        if self._data_mode == "ram":
            return self._get_dataset_ram(shuffle, rank, size)
        else:
            return self._get_dataset_hdf5(shuffle, rank, size)
    
    def _get_dataset_ram(self, shuffle=True, rank=0, size=1):
        print(f"  [RAM Dataset] Processing {self.n_events:,} events...")
        t_start = time.time()
        
        voxel_regions, region_normalized, phi_normalized = self._process_chunk(0, self.n_events)
        
        elapsed = time.time() - t_start
        print(f"  [RAM Dataset] Normalized in {elapsed:.1f}s")
        
        data_dict = {}
        for region_name, grid in voxel_regions.items():
            data_dict[f"voxel_{region_name}"] = grid.astype(np.float32)
        data_dict["region_hits"] = region_normalized.astype(np.float32)
        data_dict["phi"] = phi_normalized.astype(np.float32)
        
        ds = tf.data.Dataset.from_tensor_slices(data_dict)
        
        if size > 1:
            ds = ds.shard(num_shards=size, index=rank)
        
        region_keys = list(voxel_regions.keys())
        def reformat(sample):
            voxel_dict = {region: sample[f"voxel_{region}"] for region in region_keys}
            return (voxel_dict, sample["region_hits"], sample["phi"])
        
        ds = ds.map(reformat, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            ds = ds.shuffle(buffer_size=self.n_events)
        
        return ds
    
    def _get_dataset_hdf5(self, shuffle=True, rank=0, size=1):
        """HDF5 streaming mode — reads chunks directly from file."""
        chunk_size = 5000
        
        def generator():
            with h5py.File(self.h5_path, "r") as f:
                phi_ds = f["phi_matrix"]
                target_ds = f["target_matrix"]
                region_ds = f["region_matrix"]
                
                indices = np.arange(self.n_events)
                if shuffle:
                    np.random.shuffle(indices)
                
                if size > 1:
                    indices = indices[rank::size]
                
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_indices = indices[chunk_start:chunk_start + chunk_size]
                    chunk_indices_sorted = np.sort(chunk_indices)
                    
                    phi_chunk = phi_ds[chunk_indices_sorted]
                    target_chunk = target_ds[chunk_indices_sorted].astype(np.float32)
                    region_chunk = region_ds[chunk_indices_sorted].astype(np.float32)
                    
                    phi_normalized = self._normalize_phi(phi_chunk)
                    target_normalized = self._normalize_target(target_chunk, region_chunk)
                    region_normalized = self._normalize_region_targets(region_chunk)
                    voxel_regions = self._split_voxels_to_regions(target_normalized)
                    
                    for i in range(len(chunk_indices_sorted)):
                        voxel_dict = {
                            region: voxel_regions[region][i]
                            for region in voxel_regions.keys()
                        }
                        yield (voxel_dict, region_normalized[i], phi_normalized[i])
        
        voxel_spec = {
            region: tf.TensorSpec(shape=shape, dtype=tf.float32)
            for region, shape in self.grid_shapes.items()
        }
        output_signature = (
            voxel_spec,
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(self.phi_dim,), dtype=tf.float32)
        )
        
        ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return ds

    # =========================================================================
    # Denormalization
    # =========================================================================
    
    def denormalize_predictions(self, voxels_normalized, areas_normalized):
        areas_raw = areas_normalized * self.norm_config['voxel_hit_max']
        areas_raw = np.clip(areas_raw, 0, None)
        
        region_order = ['pit', 'bot', 'wall', 'top']
        voxels_raw = {}
        for region_name, grid in voxels_normalized.items():
            if region_name == 'PITBOT':
                area_total = areas_raw[:, 0] + areas_raw[:, 1]
            else:
                region_idx = region_order.index(region_name.lower())
                area_total = areas_raw[:, region_idx]
            n_spatial_dims = len(grid.shape) - 1
            area_broadcast = area_total.reshape((-1,) + (1,) * n_spatial_dims)
            voxel_raw = grid * area_broadcast
            voxels_raw[region_name] = np.clip(voxel_raw, 0, None)
        
        return voxels_raw, areas_raw