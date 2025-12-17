
# gpu_minimal.py - GPU mit extremen Workarounds
import tensorflow as tf
import os
from data_loader import voxelDataset
from diffusion_model_optimized import build_diffusion_model_memory_efficient

def setup_minimal_gpu():
    # Libdevice workarounds
    os.environ['TF_DISABLE_JIT'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
    
    # Alle GPU Optimierungen deaktivieren
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    
    # GPU Memory Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Float32 only
    tf.keras.mixed_precision.set_global_policy('float32')

def run_gpu_inference_only():
    print("=== GPU INFERENCE ONLY (No Training) ===")
    
    setup_minimal_gpu()
    
    try:
        # Mini Model
        model = build_diffusion_model_memory_efficient(
            phi_dim=22, target_dim=7789, T=1000,
            hidden_dim=32,  # SEHR klein
            n_layers=1,
            t_emb_dim=8
        )
        
        print(f"✓ Minimal GPU Model: {model.count_params():,} Parameter")
        
        loader = voxelDataset("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5")
        dataset = loader.get_noisy_dataset(batch_size=1, shuffle=False)
        
        # NUR INFERENCE - kein Training
        print("GPU Inference Test (10 Forward Passes)...")
        for i, (phi_b, x_noisy_b, noise_b, t_b) in enumerate(dataset.take(10)):
            pred = model([x_noisy_b, phi_b, t_b], training=False)  # training=False!
            print(f"Forward {i+1}: {pred.shape}")
        
        print("✓ GPU Inference funktioniert!")
        print("HINWEIS: Nur Forward Pass möglich - Training scheitert an libdevice")
        return True
        
    except Exception as e:
        print(f"✗ Auch GPU Inference fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    success = run_gpu_inference_only()
    if not success:
        print("\nGPU ist komplett unbrauchbar auf diesem System")
