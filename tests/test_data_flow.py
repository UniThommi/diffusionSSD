from config import Config
from data_loader import voxelDataset
from diffusion_model import build_diffusion_model_memory_efficient, monitor_memory

def test_data_flow():
    monitor_memory()
    config = Config()
    
    # Data Loader
    loader = voxelDataset(config.DATA_PATH)
    print(f"Dataset loaded: {loader.n_events} events")
    
    # Test verschiedene Batch-Sizes
    for batch_size in [4, 8, 16]:
        print(f"\nTesting batch_size={batch_size}")
        try:
            dataset = loader.get_noisy_dataset(batch_size=batch_size, shuffle=False)
            for phi_b, x_noisy_b, noise_b, t_b in dataset.take(1):
                print(f"✓ Batch shapes: phi={phi_b.shape}, x_noisy={x_noisy_b.shape}")
                break
        except Exception as e:
            print(f"✗ batch_size={batch_size} failed: {e}")

if __name__ == "__main__":
    test_data_flow()