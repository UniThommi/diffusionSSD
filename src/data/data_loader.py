import h5py
import tensorflow as tf
import numpy as np

class voxelDataset:
    PHI = [
    "#gamma", "E_gamma_tot_keV",
    "gammaE1_keV", "gammaE2_keV", "gammaE3_keV", "gammaE4_keV",
    "gammapx1", "gammapx2", "gammapx3", "gammapx4",
    "gammapy1", "gammapy2", "gammapy3", "gammapy4",
    "gammapz1", "gammapz2", "gammapz3", "gammapz4",
    "matID", "xNC_mm", "yNC_mm", "zNC_mm"
    ]
    def __init__(self, h5_path: str, T: int=1000, beta_start: float=1e-4, beta_end: float=0.02):
        self.h5_path = h5_path
        # Read out shapes
        with h5py.File(h5_path, "r")as f:
            if "phi" not in f or "target" not in f:
                raise ValueError("HDF5 must contain 'targe' and 'phi' groups")
            self.n_events =f["phi"]["#gamma"].shape[0]
            self.voxel_keys = list(f["target"].keys())

        self.phi_dim = len(self.PHI) # 22
        self.target_dim = len(self.voxel_keys) # 7789

        # Diffusion parameter
        betas = tf.linspace(beta_start, beta_end, T)    # noise schedule
        alphas = 1.0 - betas                            # signal = total - noise
        self.alphas_cumprod = tf.math.cumprod(alphas)   # cumulative product of signal after n steps (hom much signal is left)
        self.T = T                                      # time steps

    def _generator(self):
        """Original generator"""
        with h5py.File(self.h5_path, "r") as f:
            phi_group = f["phi"]
            target_group = f["target"]

            phi_dsets = [phi_group[name] for name in self.PHI]
            voxel_dsets = [target_group[key] for key in self.voxel_keys]

            for i in range(self.n_events):
                phi = np.array([ds[i] for ds in phi_dsets], dtype=np.float32)
                target = np.array([ds[i] for ds in voxel_dsets], dtype=np.float32)
                yield phi, target

    def _generator_chunked(self, chunk_size=1000):
        """Memory-effizienterer Generator mit Chunking"""
        with h5py.File(self.h5_path, "r") as f:
            phi_group = f["phi"]
            target_group = f["target"]
            
            # Verarbeite in Chunks
            for start_idx in range(0, self.n_events, chunk_size):
                end_idx = min(start_idx + chunk_size, self.n_events)
                
                # Lade Chunk
                phi_data = []
                for name in self.PHI:
                    phi_data.append(phi_group[name][start_idx:end_idx])
                phi_chunk = np.array(phi_data, dtype=np.float32).T  # (chunk_size, phi_dim)
                
                target_data = []
                for key in self.voxel_keys:
                    target_data.append(target_group[key][start_idx:end_idx])
                target_chunk = np.array(target_data, dtype=np.float32).T  # (chunk_size, target_dim)
                
                # Yield einzelne Samples aus dem Chunk
                for i in range(phi_chunk.shape[0]):
                    yield phi_chunk[i], target_chunk[i]

    def get_base_dataset(self, shuffle: bool=True, use_chunking: bool=True):
        """Basis Dataset mit optionalem Chunking"""
        output_signature = (
            tf.TensorSpec(shape=(self.phi_dim,), dtype=tf.float32),
            tf.TensorSpec(shape=(self.target_dim,), dtype=tf.float32),
        )
        
        if use_chunking:
            ds = tf.data.Dataset.from_generator(
                lambda: self._generator_chunked(chunk_size=1000), 
                output_signature=output_signature
            )
        else:
            ds = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        
        if shuffle:
            # Kleinerer Buffer für Memory-Effizienz
            ds = ds.shuffle(buffer_size=min(5000, self.n_events))
        return ds
    
    
    def get_noisy_dataset(self, batch_size: int=32, shuffle: bool=True, buffer_size: int=1000):
        # Memory-effiziente mit kontrollierten Buffern
        base = self.get_base_dataset(shuffle=False, use_chunking=True)  # Chunking verwenden
        
        if shuffle:
            # Kontrollierter Shuffle-Buffer
            base = base.shuffle(buffer_size=buffer_size)

        def _add_noise(phi, target):
            t = tf.random.uniform([], minval=0, maxval=self.T, dtype=tf.int32)
            alpha_t = tf.gather(self.alphas_cumprod, t)
            sqrt_alpha = tf.sqrt(alpha_t)
            sqrt_one_minus = tf.sqrt(1.0 - alpha_t)
            noise = tf.random.normal(shape=tf.shape(target), dtype=tf.float32)
            x_noisy = sqrt_alpha * target + sqrt_one_minus * noise
            return phi, x_noisy, noise, t

        return (base
                .map(_add_noise, num_parallel_calls=tf.data.AUTOTUNE)  # ← PARALLEL!
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))  # ← PREFETCH!
    
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
        efficient_ds = loader.get_noisy_dataset_memory_efficient(
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
        noisy_ds = loader.get_noisy_dataset_memory_efficient(
            batch_size=args.batch, 
            shuffle=True, 
            buffer_size=1000
        )
        for phi_b, x_noisy_b, noise_b, t_b in noisy_ds.take(1):
            print("phi batch shape:", phi_b.shape)
            print("x_noisy batch shape:", x_noisy_b.shape)
            print("noise batch shape:", noise_b.shape)
            print("t batch shape (per-sample):", t_b.shape, "values:", t_b.numpy())