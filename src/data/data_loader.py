# data_loader.py
import h5py
import tensorflow as tf
import numpy as np
import toml
import json
from pathlib import Path

class voxelDataset:
    """
    NC-Score Dataset Loader
    
    Loads neutron capture events from HDF5 with:
    - Phi parameters (physics conditioning)
    - Region targets (4 areas: pit, bot, wall, top)
    - Voxel targets (9583 individual PMT responses)
    
    Handles normalization and One-Hot encoding based on data_config.toml
    """
    
    # HDF5 Spaltenreihenfolge (vollständig)
    PHI_HDF5_ORDER = [
        "xNC_mm", "yNC_mm", "zNC_mm", "matID", "volID", "#gamma", "E_gamma_tot_keV",
        "r_NC_mm", "phi_NC_rad", "dist_to_wall_mm", "dist_to_bot_mm", "dist_to_top_mm",
        "p_mean_r", "p_mean_z",
        "gammaE1_keV", "gammapx1", "gammapy1", "gammapz1",
        "gammaE2_keV", "gammapx2", "gammapy2", "gammapz2",
        "gammaE3_keV", "gammapx3", "gammapy3", "gammapz3",
        "gammaE4_keV", "gammapx4", "gammapy4", "gammapz4"
    ]
    
    def __init__(self, h5_path: str, config: dict, is_validation: bool = False):
        """
        Initialize dataset for CaloScore-compatible output
        
        Args:
            h5_path: Path to HDF5 file
            config: Full config dict from config.toml
            is_validation: If True, apply proportional event limiting
        """
        self.h5_path = h5_path
        self.config = config
        self.is_validation = is_validation
        
        # Extract sub-configs
        self.feature_config = config['features']
        self.norm_config = config['normalization']
        self.mapping_config = config['mapping']
        self.onehot_config = config['onehot']
        self.region_config = config['regions']
        self.model_config = config['model']

        # Read shapes from HDF5
        with h5py.File(h5_path, "r") as f:
            if "phi" not in f or "target" not in f:
                raise ValueError("HDF5 must contain 'target' and 'phi' groups")
            self.n_events_total = f["phi"]["#gamma"].shape[0]
        
        # === EVENT LIMITING ===
        max_events = config['training'].get('max_events', None)

        print(f"[DEBUG] max_events from config: {max_events}")  # ← Debug-Print!
        print(f"[DEBUG] is_validation: {is_validation}")
        
        if max_events is not None:
            if is_validation:
                # Proportional limiting for validation
                train_events = min(max_events, self.n_events_total)
                # Berechne Verhältnis aus train.py's train_val_split
                train_frac = config['training']['train_val_split']
                val_frac = 1.0 - train_frac
                
                # Proportionale Val-Events
                self.n_events = int(train_events * val_frac / train_frac)
                self.n_events = min(self.n_events, self.n_events_total)
                
                print(f"\n[Event Limiting - Validation]")
                print(f"  Total available: {self.n_events_total:,}")
                print(f"  Train events: {train_events:,}")
                print(f"  Val events (proportional): {self.n_events:,}")
            else:
                # Direct limiting for training
                self.n_events = min(max_events, self.n_events_total)
                
                print(f"\n[Event Limiting - Training]")
                print(f"  Total available: {self.n_events_total:,}")
                print(f"  Using: {self.n_events:,} ({100*self.n_events/self.n_events_total:.1f}%)")
        else:
            # No limiting - use all events
            self.n_events = self.n_events_total
            print(f"\n[Event Limiting] DISABLED - Using all {self.n_events:,} events")
        
        # Active features
        self.active_phi_raw = self.feature_config['active_phi'].copy()
        
        # === ONE-HOT ENCODING SETUP ===
        self.enable_material_onehot = self.onehot_config['enable_material_onehot']
        self.enable_volume_onehot = self.onehot_config['enable_volume_onehot']
        
        # Load material mapping
        print("\n[Material Mapping]")
        with open(self.mapping_config['material_mapping_file'], 'r') as f:
            material_mapping_raw = json.load(f)
        
        # Invert: {name: id} → {id: name}
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
        
        # Load volume mapping (only if enabled)
        if self.enable_volume_onehot:
            print("\n[Volume Mapping]")
            with open(self.mapping_config['volume_mapping_file'], 'r') as f:
                volume_mapping_raw = json.load(f)
            
            # Invert: {name: id} → {id: name}
            self.volume_mapping = {}
            for vol_name, vol_id in volume_mapping_raw.items():
                if isinstance(vol_id, int):
                    final_name = "noVolume" if vol_name == "" else vol_name
                    self.volume_mapping[vol_id] = final_name
            print(f"  Loaded {len(self.volume_mapping):,} volumes")
            
            # Initialize volume grouper
            from src.data.volume_groups import VolumeGrouper
            self.volume_grouper = VolumeGrouper()
            print(f"  Defined groups: {self.volume_grouper.n_groups}")
            
            # Build lookup
            print("  Building volume group lookup...")
            try:
                self.volume_to_group, self.volume_group_contents = \
                    self.volume_grouper.build_lookup_dict(self.volume_mapping)
                
                non_empty = sum(1 for v in self.volume_group_contents.values() if len(v) > 0)
                print(f"  ✓ All volumes matched successfully")
                print(f"  Active groups: {non_empty}/{self.volume_grouper.n_groups}")
                
                self.n_volume_categories = self.volume_grouper.n_groups
                
                # Save mapping
                output_path = Path("./data/volume_group_mapping.json")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                output_data = {
                    'metadata': {
                        'total_volumes': len(self.volume_mapping),
                        'total_groups': self.volume_grouper.n_groups,
                        'non_empty_groups': non_empty
                    },
                    'group_names': self.volume_grouper.group_names,
                    'group_contents': self.volume_group_contents
                }
                
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                print(f"  Volume group mapping saved: {output_path}")
                
            except ValueError as e:
                raise RuntimeError(
                    f"VOLUME GROUPING FAILED\n{e}\n"
                    f"Training aborted before any computation."
                )
        else:
            self.n_volume_categories = 0
            print("\n[Volume Mapping] DISABLED - Skipped")
        
        # Adjust active_phi: Remove matID/volID if One-Hot enabled
        self.active_phi = [
            f for f in self.active_phi_raw 
            if not ((f == 'matID' and self.enable_material_onehot) or 
                   (f == 'volID' and self.enable_volume_onehot))
        ]
        
        if self.enable_material_onehot and 'matID' in self.active_phi_raw:
            print(f"  ✓ matID removed from active_phi (replaced by One-Hot)")
        if self.enable_volume_onehot and 'volID' in self.active_phi_raw:
            print(f"  ✓ volID removed from active_phi (replaced by One-Hot)")
        
        # Read shapes from HDF5
        with h5py.File(h5_path, "r") as f:
            self.voxel_keys = list(f["target"].keys())
            
            # Check if region targets exist
            self.has_region_targets = "target_regions" in f
            if self.has_region_targets:
                self.region_keys = list(f["target_regions"].keys())
            else:
                raise ValueError("HDF5 must contain 'target_regions' group for NC-Score training!")
            
            # Check if voxel_metadata exists
            self.has_voxel_metadata = "voxel_metadata" in f
        
        # Build voxel-to-region mapping (once, for efficiency)
        print("\n[Voxel-to-Region Mapping]")
        self.voxel_to_region_idx = np.zeros(len(self.voxel_keys), dtype=np.int32)
        
        # Check for merged regions
        self.merged_regions = config['regions'].get('merged_regions', [])
        
        # Initialize region indices (original regions)
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
        
        # Convert to numpy arrays
        for region in self.region_voxel_indices:
            self.region_voxel_indices[region] = np.array(
                self.region_voxel_indices[region], dtype=np.int32
            )

        # Create merged region indices if needed
        if self.merged_regions:
            merged_name = ''.join(self.merged_regions)  # e.g., "PITBOT"
            merged_indices = []
            for region in self.merged_regions:
                merged_indices.extend(self.region_voxel_indices[region].tolist())
            self.region_voxel_indices[merged_name] = np.array(merged_indices, dtype=np.int32)
            print(f"  {merged_name}: {len(merged_indices)} voxels (merged from {' + '.join(self.merged_regions)})")
        
        # Validate voxel counts (skip merged regions, validate later)
        expected = self.region_config['expected_voxel_counts']
        for region, indices in self.region_voxel_indices.items():
            # Skip validation for merged region names (not in original config)
            if region not in expected:
                continue
            
            actual = len(indices)
            exp = expected[region]
            if actual != exp:
                raise ValueError(
                    f"Region {region}: Expected {exp} voxels, found {actual}"
                )
            print(f"  {region.upper()}: {actual} voxels")

        # === GEOMETRY PROCESSING ===
        self.grid_shapes = {}
        self.periodic_axes = {}  # Maps region → list of periodic axes
        self.voxel_to_grid_mapping = {}  # Maps region → {voxel_idx: (axis0, axis1)}
        
        # Determine which regions to process for geometry
        regions_to_process = []
        for region in ['PIT', 'BOT', 'WALL', 'TOP']:
            # Skip regions that are merged (process merged version instead)
            if region in self.merged_regions and region != self.merged_regions[0]:
                continue
            
            # For first merged region, process as merged
            if region in self.merged_regions and region == self.merged_regions[0]:
                merged_name = ''.join(self.merged_regions)
                regions_to_process.append((merged_name, self.merged_regions))
            else:
                regions_to_process.append((region, [region]))
        
        for region_name, source_regions in regions_to_process:
            # Use first source region's config for merged regions
            geo_config = config['geometry'][source_regions[0]]
            
            # Determine periodic axes (region-specific naming)
            periodic = []
            
            # Check actual region name (not merged name)
            actual_region = source_regions[0]
            
            if actual_region in ['TOP', 'PIT', 'BOT']:
                # TOP uses cartesian coordinates (y, x)
                if geo_config.get('periodic_y', False):
                    periodic.append(0)  # Y is axis 0
                if geo_config.get('periodic_x', False):
                    periodic.append(1)  # X is axis 1
            else:
                # Other regions use cylindrical/radial naming
                if geo_config.get('periodic_phi', False):
                    periodic.append(1)  # Phi is axis 1
                if geo_config.get('periodic_r', False) or geo_config.get('periodic_z', False):
                    periodic.append(0)  # R/Z is axis 0
            
            self.periodic_axes[region] = periodic
            
            if geo_config['use_auto_geometry']:
                # Auto-infer shape from voxel indices
                print(f"\n[Auto-Geometry: {region_name}]")
                if len(source_regions) > 1:
                    print(f"  Merging: {' + '.join(source_regions)}")
                inferred_shape, voxel_mapping = self._infer_grid_shape_and_mapping(region_name, source_regions)
                
                # Validate against expected count
                if len(source_regions) > 1:
                    # Merged region: sum expected counts
                    expected_count = sum(
                        self.region_config['expected_voxel_counts'][r] 
                        for r in source_regions
                    )
                    actual_count = len(self.region_voxel_indices[region_name])
                else:
                    expected_count = self.region_config['expected_voxel_counts'][region_name]
                    actual_count = len(self.region_voxel_indices[region_name])
                
                # Check if inferred shape matches actual count
                # Exception: Sparse grids (TOP, PITBOT) allow grid_size > actual_voxels
                sparse_regions = ['TOP', 'PITBOT']
                if region_name not in sparse_regions:
                    if inferred_shape[0] * inferred_shape[1] != actual_count:
                        raise ValueError(
                            f"GEOMETRY MISMATCH: {region}\n"
                            f"  Inferred shape: {inferred_shape} → {inferred_shape[0] * inferred_shape[1]} voxels\n"
                            f"  Actual voxels: {actual_count}\n"
                            f"  Expected: {expected_count}"
                        )
                else:
                    # TOP: Sparse grid (donut), verify it's reasonable
                    grid_size = inferred_shape[0] * inferred_shape[1]
                    utilization = actual_count / grid_size
                    
                    print(f"  Grid utilization: {utilization*100:.1f}% ({actual_count}/{grid_size})")
                    
                    if utilization < 0.3:
                        import warnings
                        warnings.warn(
                            f"TOP grid utilization very low ({utilization*100:.1f}%). "
                            f"Grid might be inefficient. Consider polar coordinates."
                        )
                
                if actual_count != expected_count:
                    raise ValueError(
                        f"VOXEL COUNT MISMATCH: {region}\n"
                        f"  Expected: {expected_count}\n"
                        f"  Found: {actual_count}"
                    )
                
                # Calculate auto-padding
                auto_pad = self._calculate_auto_padding(
                    inferred_shape, 
                    depth=config['model']['unet_block_depth'],
                    periodic_axes=periodic
                )
                
                self.grid_shapes[region_name] = inferred_shape
                config['PAD'][region_name] = auto_pad
                
                # Store mapping for later use
                self.voxel_to_grid_mapping[region_name] = voxel_mapping
                
                print(f"  Inferred shape: {inferred_shape}")
                print(f"  Auto-padding: {auto_pad}")
                print(f"  Periodic axes: {periodic}")
                
            else:
                # Manual shape from config
                self.grid_shapes[region_name] = tuple(geo_config['SHAPE'])
                
                # Use legacy padding
                pad_value = config['model']['padding'].get(region, 0)
                config['PAD'][region] = pad_value
                
                # No voxel mapping needed (uses old reshape logic)
                self.voxel_to_grid_mapping[region] = None
                
                print(f"\n[Manual Geometry: {region_name}]")
                print(f"  Shape: {self.grid_shapes[region_name]}")
                print(f"  Padding: {config['PAD'][region_name]}")
        
        # Calculate phi_dim
        base_dim = len(self.active_phi)
        self.phi_dim = base_dim + self.n_material_categories + self.n_volume_categories
        
        self.target_dim = len(self.voxel_keys)
        
        print(f"\n✓ voxelDataset initialized")
        print(f"  Using events: {self.n_events:,} (Total available: {self.n_events_total:,})")
        print(f"  Base phi features: {base_dim}")
        if self.enable_material_onehot:
            print(f"  + Material One-Hot: {self.n_material_categories}")
        if self.enable_volume_onehot:
            print(f"  + Volume One-Hot: {self.n_volume_categories}")
        print(f"  = Total phi_dim: {self.phi_dim}")
        print(f"  Target voxels: {self.target_dim}")
        print(f"  Region targets: {'✓' if self.has_region_targets else '✗'}")
        print(f"  Voxel metadata: {'✓' if self.has_voxel_metadata else '✗'}")

    def _parse_voxel_index(self, index_str: str, region: str):
        """
        Parse voxel index string to extract grid coordinates
        
        Format depends on region:
        
        WALL (cylindrical):
          Format: LLZZPP or LLZZZPP or LLZZPPP (auto-detected)
          - LL: Layer prefix (30)
          - ZZ(Z): Z index
          - PP(P): Phi index
        
        TOP (cartesian):
          Format: LLYYXX (always 6 digits)
          - LL: Layer prefix (99)
          - YY: Y index (2 digits)
          - XX: X index (2 digits)
        
        PIT, BOT (cartesian, legacy):
          Format: LLYYXX (6 digits)
          - LL: Layer prefix (00 or 01)
          - YY: Y index
          - XX: X index
        
        Args:
            index_str: e.g. "300512" or "990312" or "001523"
            region: 'PIT', 'BOT', 'WALL', 'TOP'
        
        Returns:
            (axis0_idx, axis1_idx): Grid coordinates
        """
        layer_prefix = index_str[:2]
        remainder = index_str[2:]
        
        # TOP: Fixed format ZZYYXX (99YYXX)
        if region == 'TOP':
            if len(remainder) != 4:
                raise ValueError(
                    f"TOP voxel index '{index_str}' must have format 99YYXX (6 digits total). "
                    f"Got {len(index_str)} digits."
                )
            axis0_idx = int(remainder[0:2])  # YY
            axis1_idx = int(remainder[2:4])  # XX
            return (axis0_idx, axis1_idx)
        
        # WALL: Dynamic parsing for variable phi/z dimensions
        if region == 'WALL':
            if len(index_str) > 6 and self.config['training'].get('verbose', False):
                import warnings
                warnings.warn(
                    f"Voxel index '{index_str}' is {len(index_str)} digits (expected 6). "
                    f"Using dynamic parsing."
                )
            
            # Dynamic parsing: Try to infer format
            if len(remainder) == 4:
                # Standard ZZPP format
                axis0_idx = int(remainder[:2])
                axis1_idx = int(remainder[2:4])
                format_used = "LLZZPP"
            elif len(remainder) == 5:
                # Either ZZZPP or ZZPPP
                # Heuristic: Phi usually has more voxels (cylindrical)
                # Try ZZPPP first
                try:
                    axis0_idx = int(remainder[:2])
                    axis1_idx = int(remainder[2:5])
                    format_used = "LLZZPPP"
                except:
                    # Fallback to ZZZPP
                    axis0_idx = int(remainder[:3])
                    axis1_idx = int(remainder[3:5])
                    format_used = "LLZZZPP"
            else:
                raise ValueError(
                    f"Cannot parse WALL voxel index '{index_str}' with {len(remainder)} "
                    f"digits after layer prefix. Expected 4 or 5."
                )
            
            if len(index_str) > 6:
                import warnings
                if self.config['training'].get('verbose', False):
                    warnings.warn(f"  → Parsed as {format_used}: axis0={axis0_idx}, axis1={axis1_idx}")
            return (axis0_idx, axis1_idx)
        
        # PIT, BOT: Standard LLYYXX format
        if region in ['PIT', 'BOT']:
            if len(remainder) != 4:
                raise ValueError(
                    f"{region} voxel index '{index_str}' must have format LLYYXX (6 digits total). "
                    f"Got {len(index_str)} digits."
                )
            axis0_idx = int(remainder[0:2])  # YY
            axis1_idx = int(remainder[2:4])  # XX
            return (axis0_idx, axis1_idx)
        
        # Fallback for unknown regions
        raise ValueError(
            f"Cannot parse voxel index '{index_str}' for region '{region}'. "
            f"Unknown region type."
        )
    
    def _infer_grid_shape_and_mapping(self, region: str, source_regions: list = None):
        """
        Infer grid shape from voxel indices and create mapping
        
        Handles both 0-indexed grids (WALL) and offset grids (TOP, PITBOT).
        For merged regions, combines voxel indices from multiple source regions.
        
        Args:
            region: Target region name ('PIT', 'BOT', 'WALL', 'TOP', 'PITBOT')
            source_regions: List of source regions to merge (e.g., ['PIT', 'BOT'])
                           If None, uses [region]
        
        Returns:
            shape: (n_axis0, n_axis1, 1)
            mapping: dict {voxel_list_idx: (axis0, axis1)}
        """
        # Handle merged regions
        if source_regions is None:
            source_regions = [region]
        
        # Get all voxel keys for this region (or merged regions)
        region_indices = self.region_voxel_indices[region]
        region_keys = [self.voxel_keys[i] for i in region_indices]
        
        # For merged regions, we need to parse using correct region prefix
        # Build a mapping: voxel_key → source_region
        voxel_to_source = {}
        if len(source_regions) > 1:
            for source_region in source_regions:
                source_indices = self.region_voxel_indices[source_region]
                for idx in source_indices:
                    voxel_to_source[self.voxel_keys[idx]] = source_region
        
        # Sort keys to ensure consistent ordering
        region_keys_sorted = sorted(region_keys)
        
        # Parse all indices
        parsed_coords = []
        for key in region_keys_sorted:
            # For merged regions, use source region for parsing
            parse_region = voxel_to_source.get(key, region)
            coords = self._parse_voxel_index(key, parse_region)
            parsed_coords.append(coords)
        
        # Determine grid dimensions and offsets
        axis0_coords = [c[0] for c in parsed_coords]
        axis1_coords = [c[1] for c in parsed_coords]
        
        axis0_min = min(axis0_coords)
        axis0_max = max(axis0_coords)
        axis1_min = min(axis1_coords)
        axis1_max = max(axis1_coords)
        
        # Grid dimensions (inclusive range)
        n_axis0 = axis0_max - axis0_min + 1
        n_axis1 = axis1_max - axis1_min + 1
        
        print(f"  Axis 0: min={axis0_min}, max={axis0_max}, n={n_axis0}")
        print(f"  Axis 1: min={axis1_min}, max={axis1_max}, n={n_axis1}")
        
        # Create mapping: original voxel list index → grid coordinates (normalized to 0)
        # Need to map from unsorted to sorted order
        mapping = {}
        for list_idx in region_indices:
            key = self.voxel_keys[list_idx]
            sorted_idx = region_keys_sorted.index(key)
            raw_coords = parsed_coords[sorted_idx]
            
            # Normalize coordinates to start at 0
            normalized_coords = (
                raw_coords[0] - axis0_min,
                raw_coords[1] - axis1_min
            )
            mapping[list_idx] = normalized_coords
        
        return (n_axis0, n_axis1, 1), mapping
    
    def _calculate_auto_padding(self, shape, depth=4, periodic_axes=[]):
        """
        Calculate padding to make dimensions divisible by 2^depth
        
        IMPORTANT: Padding amount is ALWAYS calculated, regardless of periodic_axes.
        The periodic_axes parameter only determines the TYPE of padding (zero vs circular),
        not WHETHER to pad.
        
        Args:
            shape: (n_axis0, n_axis1, channels)
            depth: U-Net pooling depth
            periodic_axes: List of axes that use circular padding [0, 1]
        
        Returns:
            ((pad_axis0_top, pad_axis0_bot), (pad_axis1_left, pad_axis1_right))
        """
        divisor = 2 ** depth
        n_axis0, n_axis1 = shape[0], shape[1]
        
        # Axis 0 padding (ALWAYS calculate, type determined by periodic_axes)
        if n_axis0 % divisor != 0:
            target = ((n_axis0 // divisor) + 1) * divisor
            total_pad = target - n_axis0
            pad_axis0 = (total_pad // 2, total_pad - total_pad // 2)
        else:
            pad_axis0 = (0, 0)
        
        # Axis 1 padding (ALWAYS calculate, type determined by periodic_axes)
        if n_axis1 % divisor != 0:
            target = ((n_axis1 // divisor) + 1) * divisor
            total_pad = target - n_axis1
            pad_axis1 = (total_pad // 2, total_pad - total_pad // 2)
        else:
            pad_axis1 = (0, 0)
        
        return (pad_axis0, pad_axis1)
    
    def _split_voxels_to_regions(self, voxel_batch: np.ndarray) -> dict:
        """
        Split voxel array into region grids
        
        Handles both individual and merged regions (e.g., PITBOT).
        
        Args:
            voxel_batch: (batch, n_voxels) - All voxels
        
        Returns:
            Dict of region tensors: {'WALL': (batch, ...), 'TOP': ..., 'PITBOT': ...}
        """
        batch_size = voxel_batch.shape[0]
        region_batches = {}
        
        # Process regions based on grid_shapes (includes merged regions)
        for region_name in self.grid_shapes.keys():
            indices = self.region_voxel_indices[region_name]
            region_voxels = voxel_batch[:, indices]  # (batch, n_voxels_region)
            grid_shape = self.grid_shapes[region_name]
            
            if self.voxel_to_grid_mapping[region_name] is not None:
                # Use proper voxel-to-grid mapping (auto-geometry)
                mapping = self.voxel_to_grid_mapping[region_name]
                
                # Initialize grid with zeros
                grid = np.zeros((batch_size, grid_shape[0], grid_shape[1], grid_shape[2]), 
                               dtype=np.float32)
                
                # Fill grid using mapping
                for voxel_list_idx, (axis0, axis1) in mapping.items():
                    # Find position in region_voxels array
                    region_pos = np.where(indices == voxel_list_idx)[0]
                    if len(region_pos) > 0:
                        grid[:, axis0, axis1, 0] = region_voxels[:, region_pos[0]]
                
                region_batches[region_name] = grid
                
            else:
                # Legacy: Simple reshape (manual geometry)
                n_voxels = len(indices)
                grid_size = np.prod(grid_shape[:-1])
                
                if n_voxels != grid_size:
                    if n_voxels < grid_size:
                        pad_width = ((0, 0), (0, grid_size - n_voxels))
                        region_voxels = np.pad(region_voxels, pad_width, mode='constant')
                    else:
                        region_voxels = region_voxels[:, :grid_size]
                
                target_shape = (batch_size,) + grid_shape
                region_grid = region_voxels.reshape(target_shape)
                region_batches[region_name] = region_grid
        
        return region_batches

    def _generator_chunked(self, chunk_size, rank=0, size=1):
        """Memory-efficient generator with CaloScore-compatible output"""
        import warnings
        
        with h5py.File(self.h5_path, "r") as f:
            phi_group = f["phi"]
            target_group = f["target"]
            target_regions_group = f["target_regions"]
            
            # Process in chunks
            for start_idx in range(rank, self.n_events, chunk_size * size):
                end_idx = min(start_idx + chunk_size, self.n_events)
               
                # Load phi chunk
                phi_data_raw = []
                for name in self.PHI_HDF5_ORDER:
                    phi_data_raw.append(phi_group[name][start_idx:end_idx])
                phi_chunk_raw = np.array(phi_data_raw, dtype=np.float32).T
                
                # Normalize phi
                phi_chunk_normalized = self._normalize_phi(phi_chunk_raw)
                
                # Load target chunk
                target_data = []
                for key in self.voxel_keys:
                    target_data.append(target_group[key][start_idx:end_idx])
                target_chunk_raw = np.array(target_data, dtype=np.float32).T
                
                # Load region targets FIRST (needed for voxel normalization)
                region_data = []
                for key in self.region_keys:
                    region_data.append(target_regions_group[key][start_idx:end_idx])
                region_chunk_raw = np.array(region_data, dtype=np.float32).T
                
                # Normalize target (requires region_chunk_raw)
                target_chunk_normalized = self._normalize_target(target_chunk_raw, region_chunk_raw)
                region_chunk_normalized = self._normalize_region_targets(region_chunk_raw)
                
                # Split voxels into 4 regions
                voxel_regions = self._split_voxels_to_regions(target_chunk_normalized)
                
                # Yield single samples
                for i in range(phi_chunk_normalized.shape[0]):
                    # Build voxel dict (dynamic regions)
                    voxel_dict = {
                        region: voxel_regions[region][i]
                        for region in voxel_regions.keys()
                    }
                    
                    yield (
                        voxel_dict,
                        region_chunk_normalized[i],    # area_hits
                        phi_chunk_normalized[i]        # cond
                    )
    
    def _normalize_phi(self, phi_raw: np.ndarray) -> np.ndarray:
        """
        Normalize phi parameters according to config
        
        Args:
            phi_raw: (batch, 30) - all HDF5 columns in PHI_HDF5_ORDER
        
        Returns:
            phi_normalized: (batch, phi_dim) - only active features, normalized
        """
        import warnings
        
        batch_size = phi_raw.shape[0]
        
        # Mapping: HDF5 column index -> name
        phi_dict = {name: phi_raw[:, idx] for idx, name in enumerate(self.PHI_HDF5_ORDER)}
        
        # Normalization parameters
        E_max = self.norm_config['E_max']
        r_cyl = self.norm_config['r_cylinder']
        z_min = self.norm_config['z_min']
        z_max = self.norm_config['z_max']
        h_cyl = z_max - z_min
        angle_max = self.norm_config['angle_max']
        gamma_count_max = self.norm_config['gamma_count_max']
        
        # Normalized features
        phi_normalized = []
        
        for feature_name in self.active_phi:
            if feature_name not in phi_dict:
                raise ValueError(f"Feature '{feature_name}' not in HDF5!")
            
            raw_values = phi_dict[feature_name]
            
            # === ENERGY NORMALIZATION ===
            if feature_name in ['E_gamma_tot_keV', 'gammaE1_keV', 'gammaE2_keV', 
                               'gammaE3_keV', 'gammaE4_keV']:
                normalized = raw_values / E_max
                
                if np.any(normalized > 1.0):
                    max_val = np.max(raw_values)
                    raise RuntimeError(
                        f"NORMALIZATION OVERFLOW: {feature_name}\n"
                        f"  Max value: {max_val:.2f} keV\n"
                        f"  Norm factor: {E_max:.2f} keV\n"
                        f"  → Update data_config.toml: normalization.E_max = {np.ceil(max_val)}"
                    )
            
            # === MOMENTUM (already normalized) ===
            elif feature_name in ['gammapx1', 'gammapx2', 'gammapx3', 'gammapx4',
                                 'gammapy1', 'gammapy2', 'gammapy3', 'gammapy4',
                                 'gammapz1', 'gammapz2', 'gammapz3', 'gammapz4',
                                 'p_mean_r', 'p_mean_z']:
                normalized = raw_values
            
            # === RADII ===
            elif feature_name in ['r_NC_mm', 'dist_to_wall_mm']:
                normalized = raw_values / r_cyl
            
            # === Z-COORDINATES ===
            elif feature_name in ['zNC_mm', 'dist_to_bot_mm', 'dist_to_top_mm']:
                normalized = (raw_values - z_min) / h_cyl
            
            # === X, Y (if activated) ===
            elif feature_name in ['xNC_mm', 'yNC_mm']:
                normalized = (raw_values + r_cyl) / (2 * r_cyl)
                
                if np.any((normalized < 0) | (normalized > 1)):
                    warnings.warn(
                        f"WARNING: {feature_name} outside [-r_cyl, r_cyl]\n"
                        f"  Range: [{np.min(raw_values):.1f}, {np.max(raw_values):.1f}]"
                    )
            
            # === ANGLE ===
            elif feature_name == 'phi_NC_rad':
                normalized = np.mod(raw_values, angle_max) / angle_max
            
            # === GAMMA COUNT ===
            elif feature_name == '#gamma':
                normalized = raw_values / gamma_count_max
            
            # === CATEGORICAL (fallback if no One-Hot) ===
            elif feature_name in ['matID', 'volID']:
                normalized = raw_values
            
            else:
                raise ValueError(f"Unknown feature: {feature_name}")
            
            phi_normalized.append(normalized)
        
        # Stack base features
        phi_base = np.stack(phi_normalized, axis=1)
        
        # === ONE-HOT ENCODING ===
        onehot_features = []
        
        # Material One-Hot
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
        
        # Volume One-Hot
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
        
        # Concatenate
        if onehot_features:
            phi_normalized = np.concatenate([phi_base] + onehot_features, axis=1)
        else:
            phi_normalized = phi_base
        
        return phi_normalized.astype(np.float32)
    
    def _normalize_target(self, target_raw: np.ndarray, region_raw: np.ndarray) -> np.ndarray:
        """
        Normalize voxel hit counts to corresponding area hits (like CaloScore)
        
        Args:
            target_raw: (batch, n_voxels) - Raw voxel hits
            region_raw: (batch, 4) - Raw region hits [pit, bot, wall, top]
        
        Returns:
            target_normalized: (batch, n_voxels) - Normalized as fraction of area
        """
        batch_size = target_raw.shape[0]
        n_voxels = target_raw.shape[1]
        
        # Get voxel keys to determine which region each voxel belongs to
        # Voxel keys format: "000013", "010045", "990123", "123456"
        # Prefix: 00=pit, 01=bot, 99=top, other=wall
        
        # Build region assignment for each voxel (done once in init would be better, but ok here)
        voxel_to_region = np.zeros(n_voxels, dtype=np.int32)  # 0=pit, 1=bot, 2=wall, 3=top
        
        for voxel_idx, voxel_key in enumerate(self.voxel_keys):
            if voxel_key.startswith(self.region_config['pit_prefix']):
                voxel_to_region[voxel_idx] = 0  # pit
            elif voxel_key.startswith(self.region_config['bot_prefix']):
                voxel_to_region[voxel_idx] = 1  # bot
            elif voxel_key.startswith(self.region_config['top_prefix']):
                voxel_to_region[voxel_idx] = 3  # top
            else:
                voxel_to_region[voxel_idx] = 2  # wall
        
        # Normalize each voxel by its area's total hits
        target_normalized = np.zeros_like(target_raw, dtype=np.float32)
        
        for batch_idx in range(batch_size):
            for voxel_idx in range(n_voxels):
                region_idx = voxel_to_region[voxel_idx]
                area_hits = region_raw[batch_idx, region_idx]
                
                if area_hits > 0:
                    target_normalized[batch_idx, voxel_idx] = target_raw[batch_idx, voxel_idx] / area_hits
                else:
                    target_normalized[batch_idx, voxel_idx] = 0.0
        
        return target_normalized
    
    def _normalize_region_targets(self, region_raw: np.ndarray) -> np.ndarray:
        """
        Normalize region hit counts with area correction (Strategy C)
        FIXED normalization to max_hit_global (like CaloScore layer energies)
        
        Args:
            region_raw: (batch, 4) - [pit, bot, wall, top]
        
        Returns:
            region_normalized: (batch, 4)
        """
        voxel_hit_max = self.norm_config['voxel_hit_max']
        
        normalized = region_raw / voxel_hit_max
        
        if np.any(normalized > 1.0):
            max_val = np.max(region_raw)
            raise RuntimeError(
                f"NORMALIZATION OVERFLOW: Voxel Hits\n"
                f"  Max value: {max_val:.1f}\n"
                f"  Norm factor: {voxel_hit_max:.1f}\n"
                f"  → Update data_config.toml: normalization.voxel_hit_max = {np.ceil(max_val)}"
            )
        
        return normalized.astype(np.float32)

    def get_dataset(self, shuffle=True, rank=0, size=1):
        """
        Create clean dataset for CaloScore training (no noise injection)
        
        Args:
            shuffle: Shuffle data
        
        Returns:
            tf.data.Dataset yielding (voxel_areas_list, area_hits, cond)
        """
        # Define output signature (dynamic based on grid_shapes)
        voxel_spec = {
            region: tf.TensorSpec(shape=shape, dtype=tf.float32)
            for region, shape in self.grid_shapes.items()
        }
        
        output_signature = (
            voxel_spec,
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(self.phi_dim,), dtype=tf.float32)
        )
        
        def generator():
            for sample in self._generator_chunked(chunk_size=1000, rank=rank, size=size):
                yield sample
        
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if shuffle:
            # Dynamischer Buffer: Min(20% der Events, aber maximal 10k)
            buffer_size = min(
                max(int(0.2 * self.n_events), 100),  # Mindestens 100
                10000  # Maximal 10k für Memory
            )
            print(f"  Shuffle buffer: {buffer_size:,} events")
            ds = ds.shuffle(buffer_size=buffer_size)
        
        return ds
    
    def denormalize_predictions(self, voxels_normalized, areas_normalized):
        """
        Reverse normalization after model inference
        
        Args:
            voxels_normalized: List of 4 region grids (normalized)
            areas_normalized: (batch, 4) - normalized area hits
        
        Returns:
            voxels_raw, areas_raw - in original hit counts
        """
        # 1. Denormalize areas
        area_ratios = np.array([
            self.norm_config['area_ratios']['pit'],
            self.norm_config['area_ratios']['bot'],
            self.norm_config['area_ratios']['wall'],
            self.norm_config['area_ratios']['top']
        ])
        
        areas_raw = areas_normalized * self.norm_config['region_hit_max_global'] * area_ratios
        
        # 2. Denormalize voxels (multiply by area hits)
        voxels_raw = []
        for region_idx, region_name in enumerate(['pit', 'bot', 'wall', 'top']):
            voxel_grid = voxels_normalized[region_idx]  # (batch, *grid_shape)
            area_hits = areas_raw[:, region_idx:region_idx+1, None, None, None]  # Broadcasting shape
            voxel_raw = voxel_grid * area_hits
            voxels_raw.append(voxel_raw)
        
        return voxels_raw, areas_raw