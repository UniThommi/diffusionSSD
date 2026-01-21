import h5py
import tensorflow as tf
import numpy as np

class voxelDataset:
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
    
    def __init__(self, h5_path: str, config_path: str = None):
        """
        Initialize dataset with normalization and One-Hot encoding from config
        
        Args:
            h5_path: Path to HDF5 file
            config_path: Path to config.toml (if None, auto-detect)
        """
        from config.config_loader import ConfigLoader
        from data.volume_groups import VolumeGrouper
        import json
        
        self.h5_path = h5_path
        
        # Load config
        cfg_loader = ConfigLoader(config_path)
        self.feature_config = cfg_loader.get_feature_config()
        self.norm_config = cfg_loader.get_normalization_config()
        self.mapping_config = cfg_loader.get_mapping_config()
        self.onehot_config = cfg_loader.get_onehot_config()
        
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
            # Build material index
            self.material_names = sorted(set(self.material_mapping.values()))
            self.material_to_idx = {name: idx for idx, name in enumerate(self.material_names)}
            self.n_material_categories = len(self.material_names)
            print(f"  One-Hot categories: {self.n_material_categories}")
            print(f"  Materials: {', '.join(self.material_names)}")
        else:
            self.n_material_categories = 0
        
        # Load volume mapping
        print("\n[Volume Mapping]")
        with open(self.mapping_config['volume_mapping_file'], 'r') as f:
            volume_mapping_raw = json.load(f)
        
        # Invert: {name: id} → {id: name}
        self.volume_mapping = {}
        for vol_name, vol_id in volume_mapping_raw.items():
            if isinstance(vol_id, int):
                # Handle empty string as "noVolume"
                final_name = "noVolume" if vol_name == "" else vol_name
                self.volume_mapping[vol_id] = final_name
        print(f"  Loaded {len(self.volume_mapping):,} volumes")
        
        if self.enable_volume_onehot:
            # Initialize volume grouper
            self.volume_grouper = VolumeGrouper()
            print(f"  Defined groups: {self.volume_grouper.n_groups}")
            
            # Build lookup (validiert alle Volumes)
            print("  Building volume group lookup...")
            try:
                self.volume_to_group, self.volume_group_contents = \
                    self.volume_grouper.build_lookup_dict(self.volume_mapping)
                
                # Count non-empty groups
                non_empty = sum(1 for v in self.volume_group_contents.values() if len(v) > 0)
                print(f"  ✓ All volumes matched successfully")
                print(f"  Active groups: {non_empty}/{self.volume_grouper.n_groups}")
                
                self.n_volume_categories = self.volume_grouper.n_groups
                
                # Save group mapping to JSON (überschreibt bei jedem Run)
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
                    f"VOLUME GROUPING FAILED\n"
                    f"{e}\n"
                    f"Training aborted before any computation."
                )
        else:
            self.n_volume_categories = 0
        
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
            if "phi" not in f or "target" not in f:
                raise ValueError("HDF5 must contain 'target' and 'phi' groups")
            self.n_events = f["phi"]["#gamma"].shape[0]
            self.voxel_keys = list(f["target"].keys())
            
            # Check if region targets exist
            self.has_region_targets = "target_regions" in f
            if self.has_region_targets:
                self.region_keys = list(f["target_regions"].keys())
            
            # Check if voxel_metadata exists
            self.has_voxel_metadata = "voxel_metadata" in f
        
        # Calculate phi_dim
        base_dim = len(self.active_phi)
        self.phi_dim = base_dim + self.n_material_categories + self.n_volume_categories
        
        self.target_dim = len(self.voxel_keys)
        
        print(f"\n✓ voxelDataset initialized")
        print(f"  Events: {self.n_events:,}")
        print(f"  Base phi features: {base_dim}")
        if self.enable_material_onehot:
            print(f"  + Material One-Hot: {self.n_material_categories}")
        if self.enable_volume_onehot:
            print(f"  + Volume One-Hot: {self.n_volume_categories}")
        print(f"  = Total phi_dim: {self.phi_dim}")
        print(f"  Target voxels: {self.target_dim}")
        print(f"  Region targets: {self.has_region_targets}")
        print(f"  Voxel metadata: {self.has_voxel_metadata}")

    def _generator_chunked(self, chunk_size):
        """Memory-efficient generator with normalization"""
        import warnings
        
        with h5py.File(self.h5_path, "r") as f:
            phi_group = f["phi"]
            target_group = f["target"]
            
            if self.has_region_targets:
                target_regions_group = f["target_regions"]
            
            if self.has_voxel_metadata:
                voxel_metadata_group = f["voxel_metadata"]
            
            # Process in chunks
            for start_idx in range(0, self.n_events, chunk_size):
                end_idx = min(start_idx + chunk_size, self.n_events)
                
                # Load phi chunk (alle HDF5 Spalten)
                phi_data_raw = []
                for name in self.PHI_HDF5_ORDER:
                    phi_data_raw.append(phi_group[name][start_idx:end_idx])
                phi_chunk_raw = np.array(phi_data_raw, dtype=np.float32).T  # (chunk_size, 30)
                
                # Normalize phi
                phi_chunk_normalized = self._normalize_phi(phi_chunk_raw)
                
                # Load target chunk
                target_data = []
                for key in self.voxel_keys:
                    target_data.append(target_group[key][start_idx:end_idx])
                target_chunk_raw = np.array(target_data, dtype=np.float32).T
                
                # Normalize target
                target_chunk_normalized = self._normalize_target(target_chunk_raw)
                
                # Load region targets (optional)
                if self.has_region_targets:
                    region_data = []
                    for key in self.region_keys:
                        region_data.append(target_regions_group[key][start_idx:end_idx])
                    region_chunk_raw = np.array(region_data, dtype=np.float32).T
                    region_chunk_normalized = self._normalize_region_targets(region_chunk_raw)
                else:
                    region_chunk_normalized = None
                
                # Load voxel metadata (optional)
                if self.has_voxel_metadata:
                    # TODO: Implement voxel_metadata normalization
                    # One-Hot für regions + normierte r, phi, z
                    pass
                
                # Yield einzelne Samples
                for i in range(phi_chunk_normalized.shape[0]):
                    sample = {
                        'phi': phi_chunk_normalized[i],
                        'target': target_chunk_normalized[i]
                    }
                    
                    if region_chunk_normalized is not None:
                        sample['target_regions'] = region_chunk_normalized[i]
                    
                    yield sample
    
    def _normalize_phi(self, phi_raw: np.ndarray) -> np.ndarray:
        """
        Normalize phi parameters according to config
        
        Args:
            phi_raw: (batch, 30) - alle HDF5 Spalten in PHI_HDF5_ORDER
        
        Returns:
            phi_normalized: (batch, phi_dim) - nur active features, normiert
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
        
        # Normierte Features (in Reihenfolge von active_phi)
        phi_normalized = []
        
        for feature_name in self.active_phi:
            if feature_name not in phi_dict:
                raise ValueError(f"Feature '{feature_name}' nicht in HDF5 gefunden!")
            
            raw_values = phi_dict[feature_name]
            
            # === ENERGIE-NORMIERUNG (alle auf E_max) ===
            if feature_name in ['E_gamma_tot_keV', 'gammaE1_keV', 'gammaE2_keV', 
                               'gammaE3_keV', 'gammaE4_keV']:
                normalized = raw_values / E_max
                
                if np.any(normalized > 1.0):
                    max_val = np.max(raw_values)
                    raise RuntimeError(
                        f"NORMALIZATION OVERFLOW: {feature_name}\n"
                        f"  Max value: {max_val:.2f} keV\n"
                        f"  Norm factor: {E_max:.2f} keV\n"
                        f"  → Update config.toml: normalization.E_max = {np.ceil(max_val)}"
                    )
            
            # === IMPULSE (bereits normiert, keine Änderung) ===
            elif feature_name in ['gammapx1', 'gammapx2', 'gammapx3', 'gammapx4',
                                 'gammapy1', 'gammapy2', 'gammapy3', 'gammapy4',
                                 'gammapz1', 'gammapz2', 'gammapz3', 'gammapz4',
                                 'p_mean_r', 'p_mean_z']:
                normalized = raw_values  # Bereits normiert
            
            # === RADIEN (auf r_cylinder) ===
            elif feature_name in ['r_NC_mm', 'dist_to_wall_mm']:
                normalized = raw_values / r_cyl
                # Kann negativ sein (NC außerhalb) - erlaubt, keine Warnung
            
            # === Z-KOORDINATEN (auf Zylinderhöhe) ===
            elif feature_name in ['zNC_mm', 'dist_to_bot_mm', 'dist_to_top_mm']:
                normalized = (raw_values - z_min) / h_cyl
                # Kann außerhalb [0,1] sein - erlaubt für NC, keine Warnung
            
            # === X, Y (falls aktiviert - standardmäßig nicht) ===
            elif feature_name in ['xNC_mm', 'yNC_mm']:
                normalized = (raw_values + r_cyl) / (2 * r_cyl)  # [-r, r] -> [0, 1]
                
                if np.any((normalized < 0) | (normalized > 1)):
                    warnings.warn(
                        f"WARNING: {feature_name} outside [-r_cyl, r_cyl]\n"
                        f"  Range: [{np.min(raw_values):.1f}, {np.max(raw_values):.1f}]"
                    )
            
            # === WINKEL (auf 2π) ===
            elif feature_name == 'phi_NC_rad':
                # Bereits in [0, 2π] oder [-π, π]?
                # Normiere auf [0, 1]
                normalized = np.mod(raw_values, angle_max) / angle_max
            
            # === GAMMA-ANZAHL ===
            elif feature_name == '#gamma':
                # Diskret {0, 1, 2, 3, 4} -> normiere auf [0, 1]
                normalized = raw_values / 4.0
            
            # KATEGORIALE (matID, volID, FALLBACK falls kein one hot encoding)
            elif feature_name in ['matID', 'volID']:
                normalized = raw_values 
            
            else:
                raise ValueError(f"Unbekanntes Feature: {feature_name}")
            
            phi_normalized.append(normalized)        

        # Stack base features
        phi_base = np.stack(phi_normalized, axis=1)  # (batch, base_dim)
        
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
                    raise RuntimeError(
                        f"Unknown matID: {mat_id}\n"
                        f"Available materials: {list(self.material_mapping.keys())}"
                    )
            
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
                    # Should never happen (validated in __init__)
                    vol_name = self.volume_mapping.get(vol_id, f"UNKNOWN_{vol_id}")
                    raise RuntimeError(
                        f"Unknown volID: {vol_id} ('{vol_name}')\n"
                        f"This should have been caught during initialization!"
                    )
            
            onehot_features.append(volume_onehot)
        
        # Concatenate all features
        if onehot_features:
            phi_normalized = np.concatenate([phi_base] + onehot_features, axis=1)
        else:
            phi_normalized = phi_base
        
        return phi_normalized.astype(np.float32)
    
    def _normalize_target(self, target_raw: np.ndarray) -> np.ndarray:
        """
        Normalize voxel hit counts
        
        Args:
            target_raw: (batch, n_voxels)
        
        Returns:
            target_normalized: (batch, n_voxels)
        """
        voxel_hit_max = self.norm_config['voxel_hit_max']
        
        normalized = target_raw / voxel_hit_max
        
        if np.any(normalized > 1.0):
            max_val = np.max(target_raw)
            raise RuntimeError(
                f"NORMALIZATION OVERFLOW: Voxel Hits\n"
                f"  Max value: {max_val:.1f}\n"
                f"  Norm factor: {voxel_hit_max:.1f}\n"
                f"  → Update config.toml: normalization.voxel_hit_max = {np.ceil(max_val)}"
            )
        
        return normalized.astype(np.float32)
    
    def _normalize_region_targets(self, region_raw: np.ndarray) -> np.ndarray:
        """
        Normalize region hit counts with area correction (Strategy C)
        
        Args:
            region_raw: (batch, 4) - [pit, bot, wall, top]
        
        Returns:
            region_normalized: (batch, 4)
        """
        area_ratios = self.norm_config['area_ratios']
        max_hit_global = self.norm_config['region_hit_max_global']
        
        # Area ratios in order [pit, bot, wall, top]
        ratios = np.array([
            area_ratios['pit'],
            area_ratios['bot'],
            area_ratios['wall'],
            area_ratios['top']
        ], dtype=np.float32)
        
        # Area-corrected densities
        densities = region_raw / ratios  # (batch, 4) / (4,) -> (batch, 4)
        
        # Normalize to global max
        normalized = densities / max_hit_global
        
        if np.any(normalized > 1.0):
            max_density = np.max(densities)
            max_region_idx = np.argmax(np.max(densities, axis=0))
            region_names = ['pit', 'bot', 'wall', 'top']
            
            raise RuntimeError(
                f"NORMALIZATION OVERFLOW: Region Hits ({region_names[max_region_idx]})\n"
                f"  Max density (area-corrected): {max_density:.1f}\n"
                f"  Norm factor: {max_hit_global:.1f}\n"
                f"  → Update config.toml: normalization.region_hits.max_hit_global = {np.ceil(max_density)}"
            )
        
        return normalized.astype(np.float32)

    def get_base_dataset(self, shuffle: bool=True):
        """Basis Dataset mit Normierung"""
        # Output signature mit dict
        output_signature = {
            'phi': tf.TensorSpec(shape=(self.phi_dim,), dtype=tf.float32),
            'target': tf.TensorSpec(shape=(self.target_dim,), dtype=tf.float32)
        }
        
        # Add region targets if available
        if self.has_region_targets:
            output_signature['target_regions'] = tf.TensorSpec(shape=(4,), dtype=tf.float32)
        
        def generator():
            for sample in self._generator_chunked(chunk_size=1000):
                if self.has_region_targets:
                    yield (sample['phi'], sample['target'], sample['target_regions'])
                else:
                    yield (sample['phi'], sample['target'])
        
        # Create dataset
        if self.has_region_targets:
            ds = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    output_signature['phi'],
                    output_signature['target'],
                    output_signature['target_regions']
                )
            )
        else:
            ds = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    output_signature['phi'],
                    output_signature['target']
                )
            )
        
        if shuffle:
            ds = ds.shuffle(buffer_size=min(5000, self.n_events))
        
        return ds    
    
    def get_noisy_dataset(self, batch_size: int = 32, diffusion_schedule: dict = None, 
                         shuffle: bool = True):
        """
        Erstelle noisy dataset für Diffusion Training
        
        Args:
            batch_size: Training batch size
            diffusion_schedule: Dict mit 'T' und 'alphas_cumprod' aus config
            shuffle: Shuffle data
        """
        if diffusion_schedule is None:
            raise ValueError(
                "Must provide diffusion_schedule from config!\n"
                "Usage: config = ConfigLoader(); schedule = config.diffusion_schedule.to_dict()"
            )
        
        T = diffusion_schedule['T']
        alphas_cumprod = diffusion_schedule['alphas_cumprod']
        
        base = self.get_base_dataset(shuffle=shuffle)

        if self.has_region_targets:
            def _add_noise(phi, target, target_regions):
                t = tf.random.uniform([], minval=0, maxval=T, dtype=tf.int32)
                alpha_t = tf.gather(alphas_cumprod, t)
                sqrt_alpha = tf.sqrt(alpha_t)
                sqrt_one_minus = tf.sqrt(1.0 - alpha_t)
                noise = tf.random.normal(shape=tf.shape(target), dtype=tf.float32)
                x_noisy = sqrt_alpha * target + sqrt_one_minus * noise
                return phi, x_noisy, noise, t, target_regions
            
            return (base
                    .map(_add_noise, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
        else:
            def _add_noise(phi, target):
                t = tf.random.uniform([], minval=0, maxval=T, dtype=tf.int32)
                alpha_t = tf.gather(alphas_cumprod, t)
                sqrt_alpha = tf.sqrt(alpha_t)
                sqrt_one_minus = tf.sqrt(1.0 - alpha_t)
                noise = tf.random.normal(shape=tf.shape(target), dtype=tf.float32)
                x_noisy = sqrt_alpha * target + sqrt_one_minus * noise
                return phi, x_noisy, noise, t

            return (base
                    .map(_add_noise, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    
    # Für Debugging
    def get_small_test_dataset(self, batch_size: int=8, num_samples: int=100): 
        """Kleines Test-Dataset für Memory-Tests"""
        base = self.get_base_dataset(shuffle=False, use_chunking=True)
        
        def _add_noise(phi, target):
            t = tf.random.uniform([], minval=0, maxval=self.T, dtype=tf.int32)
            alpha_t = tf.gather(self.alphas_cumprod, t)
            sqrt_alpha = tf.sqrt(alpha_t)
            sqrt_one_minus = tf.sqrt(1.0 - alpha_t)
            noise = tf.random.normal(shape=tf.shape(target), dtype=tf.float32)
            x_noisy = sqrt_alpha * target + sqrt_one_minus * noise
            return phi, x_noisy, noise, t
        
        return (base
                .take(num_samples)
                .map(_add_noise)
                .batch(batch_size)
                .prefetch(1))

    # Testing 
    def check_memory_usage(self):
        """Memory-Usage Check"""
        print(f"Dataset Info:")
        print(f"  Events: {self.n_events:,}")
        print(f"  Phi dimension: {self.phi_dim}")
        print(f"  Target dimension: {self.target_dim:,}")
        
        # Geschätzte Memory-Nutzung pro Sample
        phi_memory = self.phi_dim * 4  # float32 = 4 bytes
        target_memory = self.target_dim * 4
        total_per_sample = phi_memory + target_memory
        
        print(f"  Memory per sample: {total_per_sample/1024:.2f} KB")
        print(f"  Total dataset memory: {(total_per_sample * self.n_events)/(1024**2):.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("h5file", help="Path to your HDF5 file")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--test-memory", action="store_true", help="Test memory efficiency")
    args = parser.parse_args()

    loader = voxelDataset(args.h5file)
    print("=== Meta ===")
    print("n_events:", loader.n_events)
    print("phi_dim:", loader.phi_dim, " (phi keys in this order):")
    print("target_dim (num voxels):", loader.target_dim)
    print("Diffusion T:", loader.T)
    
    # Memory Check
    loader.check_memory_usage()

    if args.test_memory:
        print("\n=== Memory Efficiency Test ===")
        
        # Test kleines Dataset
        print("Testing small dataset...")
        small_ds = loader.get_small_test_dataset(batch_size=args.batch, num_samples=50)
        for i, (phi_b, x_noisy_b, noise_b, t_b) in enumerate(small_ds.take(3)):
            print(f"Batch {i}: phi {phi_b.shape}, x_noisy {x_noisy_b.shape}")
        
        # Test memory-efficient dataset
        print("Testing memory-efficient dataset...")
        efficient_ds = loader.get_noisy_dataset(
            batch_size=args.batch, 
            buffer_size=500
        )
        for i, (phi_b, x_noisy_b, noise_b, t_b) in enumerate(efficient_ds.take(3)):
            print(f"Batch {i}: phi {phi_b.shape}, x_noisy {x_noisy_b.shape}")
    else:
        print("\n=== Erstes Basissample (ungebatcht) ===")
        ds = loader.get_base_dataset(shuffle=False)
        for phi, target in ds.take(1):
            print("phi shape:", phi.shape)
            print("phi sample (first 8):", phi[:8].numpy())
            print("target shape:", target.shape)
            print("target sample (first 20):", target[:20].numpy())

        print("\n=== Memory-efficient Noisy-Batch ===")
        noisy_ds = loader.get_noisy_dataset(
            batch_size=args.batch, 
            shuffle=True, 
            buffer_size=1000
        )
        for phi_b, x_noisy_b, noise_b, t_b in noisy_ds.take(1):
            print("phi batch shape:", phi_b.shape)
            print("x_noisy batch shape:", x_noisy_b.shape)
            print("noise batch shape:", noise_b.shape)
            print("t batch shape (per-sample):", t_b.shape, "values:", t_b.numpy())