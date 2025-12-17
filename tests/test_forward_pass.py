# test_forward_pass.py - Nächster Test für Ihr Setup

import tensorflow as tf
import numpy as np
from data_loader import voxelDataset
from diffusion_model_optimized import build_diffusion_model_memory_efficient, monitor_memory

def setup_gpu_for_limited_memory():
    """GPU Setup für begrenzte Memory-Situation"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory Growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # WICHTIG: Memory-Limit setzen für Stabilität
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=80)]  # 80MB limit
            )
            print("✓ GPU Memory auf 80MB begrenzt")
            
        except RuntimeError as e:
            print(f"GPU Konfiguration: {e}")
    
    # Mixed Precision für Memory-Effizienz  
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✓ Mixed Precision (float16) aktiviert")

def test_data_loading():
    """Test ob Daten richtig laden"""
    print("\n=== SCHRITT 1: Daten-Test ===")
    
    try:
        loader = voxelDataset("/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5")
        print(f"✓ Dataset geladen: {loader.n_events} events")
        
        # Test sehr kleine Batch-Size
        for batch_size in [1, 2]:
            print(f"Testing batch_size={batch_size}")
            dataset = loader.get_noisy_dataset(batch_size=batch_size, shuffle=False)
            
            for phi_b, x_noisy_b, noise_b, t_b in dataset.take(1):
                def tensor_nbytes(t):
                    return np.prod(t.shape) * t.dtype.size

                total_bytes = (
                    tensor_nbytes(phi_b) +
                    tensor_nbytes(x_noisy_b) +
                    tensor_nbytes(noise_b)
                )
                print(f"✓ Batch {batch_size}: phi={phi_b.shape}, x_noisy={x_noisy_b.shape}, noise={noise_b.shape}")
                print(f"  Batch Memory: ~{total_bytes/1024:.2f} KB")
        
        return loader
        
    except Exception as e:
        print(f"✗ Daten-Loading fehlgeschlagen: {e}")
        return None

def test_model_creation():
    """Test Modell-Erstellung mit optimalen Einstellungen"""
    print("\n=== SCHRITT 2: Modell-Test ===")
    
    # Basierend auf Ihrem Memory-Test: hidden_dim=256 funktioniert
    model = build_diffusion_model_memory_efficient(
        phi_dim=22,
        target_dim=7789,
        T=1000,
        hidden_dim=256,  # Funktioniert laut Ihrem Test
        n_layers=3,
        t_emb_dim=32
    )
    
    print(f"✓ Modell erstellt: {model.count_params():,} Parameter")
    print(f"✓ Memory Estimate: ~{model.count_params() * 2 / (1024**2):.1f}MB (mixed precision)")
    
    return model

def test_forward_pass(model, loader):
    """Test Forward und Backward Pass"""
    print("\n=== SCHRITT 3: Forward/Backward Pass Test ===")
    
    # Sehr kleine Batch-Size
    dataset = loader.get_noisy_dataset(batch_size=1, shuffle=False)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)  # Für Mixed Precision
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    try:
        for phi_b, x_noisy_b, noise_b, t_b in dataset.take(1):
            print(f"Input shapes: phi={phi_b.shape}, x_noisy={x_noisy_b.shape}, t={t_b.shape}")
            
            # Forward Pass
            print("Testing forward pass...")
            pred = model([x_noisy_b, phi_b, t_b], training=False)
            print(f"✓ Forward Pass: Output shape {pred.shape}")
            
            # Training Forward + Backward Pass
            print("Testing training pass...")
            with tf.GradientTape() as tape:
                pred_train = model([x_noisy_b, phi_b, t_b], training=True)
                loss = loss_fn(noise_b, pred_train)
                scaled_loss = optimizer.get_scaled_loss(loss)
            
            print(f"✓ Loss berechnet: {loss.numpy():.6f}")
            
            # Gradients
            print("Testing gradients...")
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            
            # Check für None-Gradients
            valid_grads = [g for g in gradients if g is not None]
            print(f"✓ Gradients: {len(valid_grads)}/{len(gradients)} valid")
            
            # Apply gradients
            gradients_clipped = [tf.clip_by_norm(g, 1.0) for g in valid_grads]
            optimizer.apply_gradients(zip(gradients_clipped, [v for v, g in zip(model.trainable_variables, gradients) if g is not None]))
            
            print("✓ Backward Pass erfolgreich!")
            
            return True
            
    except tf.errors.ResourceExhaustedError as e:
        print(f"✗ MEMORY ERROR: {e}")
        print("LÖSUNG: Verwenden Sie batch_size=1 und hidden_dim=128")
        return False
    except Exception as e:
        print(f"✗ ANDERER FEHLER: {e}")
        return False

def main():
    """Haupttest-Funktion"""
    print("=== FORWARD PASS TEST - Basierend auf Memory Test Ergebnissen ===")
    
    # GPU Setup
    setup_gpu_for_limited_memory()
    
    # Schritt 1: Daten
    loader = test_data_loading()
    if loader is None:
        return
    
    # Schritt 2: Modell
    try:
        model = test_model_creation()
    except Exception as e:
        print(f"✗ Modell-Erstellung fehlgeschlagen: {e}")
        print("FALLBACK: Verwende hidden_dim=128...")
        
        model = build_diffusion_model_memory_efficient(
            phi_dim=22, target_dim=7789, T=1000,
            hidden_dim=128,  # Fallback
            n_layers=2,      # Weniger Layer
            t_emb_dim=16     # Kleinere Embeddings
        )
        print(f"✓ Fallback-Modell: {model.count_params():,} Parameter")
    
    # Schritt 3: Forward/Backward Pass
    success = test_forward_pass(model, loader)
    
    if success:
        print("\n🎉 ALLE TESTS ERFOLGREICH!")
        print("Nächster Schritt: Vollständiges Training starten")
        
        # Empfohlene finale Konfiguration
        print("\n=== EMPFOHLENE KONFIGURATION ===")
        print("- hidden_dim: 256 (oder 128 als Backup)")
        print("- batch_size: 1 (maximal 2)")
        print("- mixed_precision: True")
        print("- memory_limit: 80MB")
        print("- n_layers: 3 (oder 2 als Backup)")
    else:
        print("\n❌ TEST FEHLGESCHLAGEN")
        print("Empfehlung: Noch kleinere Konfiguration verwenden")

if __name__ == "__main__":
    main()