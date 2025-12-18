# train_diffusion_cpu_fixed.py - Memory-optimierte Version

import os
# ─────────────────────────────────────────────
# ABSOLUT ZUERST – bevor TensorFlow importiert wird
# ─────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Threading VOR JEGLICHER TF-NUTZUNG
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

import time
import numpy as np
from datetime import datetime
from data.data_loader import voxelDataset
from models.diffusion_model import build_diffusion_model


class DiffusionTrainer:
    def __init__(self, config_path="/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/outdated/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5"):
        self.data_path = config_path
        
        self.config = {
            'phi_dim': 22,
            'target_dim': 7789,
            'T': 1000,
            
            # Model Architecture (VERBESSERT)
            'hidden_dim': 512,         # War ok
            'n_layers': 6,             # Erhöht von 4 → Mehr Kapazität
            't_emb_dim': 64,           # Erhöht von 32 → Bessere Zeit-Konditionierung
            
            # Training Parameters (ANGEPASST)
            'batch_size': 32,
            'learning_rate': 5e-5,     # Reduziert von 1e-4 → Stabileres Training
            'epochs': 30,              # Erhöht von 5 → Genug für Konvergenz
            'steps_per_epoch': 200,    # Erhöht von 50 → Mehr pro Epoch
            
            # Optimization
            'gradient_clip_norm': 1.0,
            'use_ema': False,
            'ema_decay': 0.999,
            
            # Checkpointing
            'checkpoint_dir': './checkpoints_cpu',
            'log_dir': './logs_cpu',
            'save_every_n_epochs': 2,  # Öfter speichern
            'eval_every_n_epochs': 1,  # Jede Epoch evaluieren
        }

        # Diffusion schedule (für Physics Loss)
        betas = tf.linspace(1e-4, 0.02, self.config['T'])
        self.alphas = 1.0 - betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        
        # Setup directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Setup TensorFlow optimizations
        self.setup_cpu_optimizations()
        
    def setup_cpu_optimizations(self):
        tf.keras.mixed_precision.set_global_policy('float32')
        tf.config.experimental.enable_tensor_float_32_execution(False)

    
    def build_model(self):
        model = build_diffusion_model(
            phi_dim=self.config['phi_dim'],
            target_dim=self.config['target_dim'],
            T=self.config['T'],
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            t_emb_dim=self.config['t_emb_dim']
        )
        
        print(f"Model built: {model.count_params():,} parameters")
        return model
    
    def setup_training(self, model):
        # Training setup 
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Loss function
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Kein EMA für Memory-Effizienz
        ema_model = None
        
        # Metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        
        return optimizer, loss_fn, ema_model, train_loss
    
    def compute_physics_loss(self, x_pred, x_true, phi, weight=0.1):
        """
        Physics-Informed Loss
        
        Prüft:
        1. Energieerhaltung: Summe(Signal) ∝ E_gamma
        2. Multiplizität: Mindestens N aktive Voxel
        3. Räumliche Lokalität: Signal um NC-Position konzentriert
        
        Args:
            x_pred: Predicted signal (batch, target_dim)
            x_true: True signal (batch, target_dim)
            phi: NC parameters (batch, phi_dim)
            weight: Gewichtung des Physics-Loss
        
        Returns:
            physics_loss: Scalar Loss
        """
        batch_size = tf.shape(x_pred)[0]
        
        # 1. Energieerhaltung
        # E_gamma_tot_keV ist phi[:, 1]
        energy_true = phi[:, 1]  # (batch,)
        
        # Totales Signal sollte mit Energie korrelieren
        signal_sum_pred = tf.reduce_sum(x_pred, axis=1)  # (batch,)
        signal_sum_true = tf.reduce_sum(x_true, axis=1)  # (batch,)
        
        # Relative Abweichung
        energy_loss = tf.reduce_mean(
            tf.abs(signal_sum_pred - signal_sum_true) / (signal_sum_true + 1e-6)
        )
        
        # 2. Multiplizitäts-Loss (Soft Constraint)
        # Wir wollen dass mindestens N=6 Voxel > threshold
        threshold = 0.5
        active_pred = tf.cast(x_pred > threshold, tf.float32)
        active_true = tf.cast(x_true > threshold, tf.float32)
        
        multiplicity_pred = tf.reduce_sum(active_pred, axis=1)  # (batch,)
        multiplicity_true = tf.reduce_sum(active_true, axis=1)  # (batch,)
        
        # Loss: Abweichung der Multiplizität
        multiplicity_loss = tf.reduce_mean(
            tf.abs(multiplicity_pred - multiplicity_true)
        )
        
        # 3. Räumliche Lokalität (Center of Mass sollte ähnlich sein)
        voxel_indices = tf.range(tf.shape(x_pred)[1], dtype=tf.float32)  # (target_dim,)
        
        # Center of Mass (gewichteter Durchschnitt der Voxel-Indices)
        com_pred = tf.reduce_sum(x_pred * voxel_indices, axis=1) / (signal_sum_pred + 1e-6)
        com_true = tf.reduce_sum(x_true * voxel_indices, axis=1) / (signal_sum_true + 1e-6)
        
        locality_loss = tf.reduce_mean(tf.abs(com_pred - com_true))
        
        # Kombiniere Physics-Losses
        total_physics_loss = (
            0.4 * energy_loss +
            0.3 * multiplicity_loss +
            0.3 * locality_loss
        )
        
        return weight * total_physics_loss
    
    @tf.function
    def train_step(self, model, optimizer, loss_fn, batch):
        """Training Step mit Physics Loss"""
        phi_b, x_noisy_b, noise_b, t_b = batch
        
        with tf.GradientTape() as tape:
            # Noise prediction
            predictions = model([x_noisy_b, phi_b, t_b], training=True)
            
            # Standard Diffusion Loss (MSE)
            diffusion_loss = loss_fn(noise_b, predictions)
            
            # Physics Loss (auf denoised signal)
            # Approximate x_0 von x_noisy und predicted noise
            alpha_t = tf.gather(self.alphas_cumprod, t_b)
            sqrt_alpha = tf.sqrt(alpha_t)
            sqrt_one_minus = tf.sqrt(1.0 - alpha_t)
            
            # x_0 ≈ (x_noisy - sqrt(1-α_t)*ε) / sqrt(α_t)
            x_pred = (x_noisy_b - sqrt_one_minus[:, None] * predictions) / sqrt_alpha[:, None]
            
            # Ground truth x_0 (von noisy rekonstruiert)
            x_true = (x_noisy_b - sqrt_one_minus[:, None] * noise_b) / sqrt_alpha[:, None]
            
            # Physics Loss
            physics_loss = self.compute_physics_loss(x_pred, x_true, phi_b, weight=0.1)
            
            # Total Loss
            total_loss = diffusion_loss + physics_loss
        
        # Gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Gradient clipping
        gradients = [tf.clip_by_norm(g, self.config['gradient_clip_norm']) 
                    for g in gradients if g is not None]
        
        # Apply gradients
        valid_gradients = [(g, v) for g, v in zip(gradients, model.trainable_variables) 
                        if g is not None]
        optimizer.apply_gradients(valid_gradients)
        
        return total_loss, diffusion_loss, physics_loss
    
    def train_epoch(self, model, optimizer, loss_fn, ema_model, train_loss, dataset, epoch):
        """Training Epoch mit Physics Loss Tracking"""
        print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
        start_time = time.time()
        
        # Reset metrics
        train_loss.reset_state()
        diffusion_losses = []
        physics_losses = []
        
        step_count = 0
        try:
            for batch in dataset.take(self.config['steps_per_epoch']):
                # Training step
                total_loss, diff_loss, phys_loss = self.train_step(
                    model, optimizer, loss_fn, batch
                )
                
                train_loss.update_state(total_loss)
                diffusion_losses.append(diff_loss.numpy())
                physics_losses.append(phys_loss.numpy())
                
                step_count += 1
                
                # Progress updates
                if step_count % 10 == 0:
                    print(f"  Step {step_count}: Total={total_loss.numpy():.6f}, "
                        f"Diffusion={diff_loss.numpy():.6f}, Physics={phys_loss.numpy():.6f}")
                    
                # Memory cleanup
                if step_count % 50 == 0:
                    tf.keras.backend.clear_session()
                    
        except Exception as e:
            print(f"Error in training step {step_count}: {e}")
            return train_loss.result()
        
        epoch_time = time.time() - start_time
        print(f"  Epoch completed in {epoch_time:.1f}s")
        print(f"  Average Total Loss:     {train_loss.result():.6f}")
        print(f"  Average Diffusion Loss: {np.mean(diffusion_losses):.6f}")
        print(f"  Average Physics Loss:   {np.mean(physics_losses):.6f}")
        
        return train_loss.result()
    
    def train(self):
        # Training Loop
        print(f"Daten: {self.data_path}")
        print(f"Config: {self.config}")
        
        # Data loader
        print("\nLade Daten...")
        loader = voxelDataset(self.data_path)
        
        # Split: 90% Training, 10% Validation
        n_total_batches = loader.n_events // self.config['batch_size']
        n_train_batches = int(0.9 * n_total_batches)
        n_val_batches = n_total_batches - n_train_batches
        
        print(f"Total Events: {loader.n_events:,}")
        print(f"  Training Batches: {n_train_batches:,}")
        print(f"  Validation Batches: {n_val_batches:,}")
        
        # Full dataset (Training + Validation)
        full_dataset = loader.get_noisy_dataset(
            batch_size=self.config['batch_size'], 
            shuffle=True,
            buffer_size=1000
        )
        
        # Split in Train und Val
        train_dataset = full_dataset.take(n_train_batches)
        val_dataset = full_dataset.skip(n_train_batches).take(n_val_batches)
        
        # Model
        print("\nErstelle Model...")
        model = self.build_model()
        
        # Training setup
        print("\nSetup Training...")
        optimizer, loss_fn, ema_model, train_loss = self.setup_training(model)
        
        # Best model tracking
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Early stopping nach 5 Epochen ohne Verbesserung
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        # Training loop
        print(f"\nStarte Training für {self.config['epochs']} Epochen...")
        print(f"Steps per Epoch: {self.config['steps_per_epoch']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Early Stopping: Patience = {patience}")
        
        training_start = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training
            epoch_loss = self.train_epoch(
                model, optimizer, loss_fn, ema_model, 
                train_loss, train_dataset, epoch
            )
            
            # Validation
            val_loss = self.validate(model, loss_fn, val_dataset, epoch)
            
            # Track history
            history['train_loss'].append(float(epoch_loss))
            history['val_loss'].append(float(val_loss))
            history['epochs'].append(epoch + 1)
            
            # Check if best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ★ New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
            
            # Checkpointing
            if (epoch + 1) % self.config['save_every_n_epochs'] == 0 or is_best:
                self.save_checkpoint(model, ema_model, optimizer, epoch, epoch_loss, is_best=is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered (no improvement for {patience} epochs)")
                print(f"  Best validation loss was: {best_val_loss:.6f}")
                break
            
            # Memory cleanup nach jeder Epoch
            tf.keras.backend.clear_session()
        
        # Final save
        final_time = time.time() - training_start
        print(f"\n✓ Training abgeschlossen in {final_time/60:.2f} Minuten")
        
        # Save final checkpoint
        self.save_checkpoint(model, ema_model, optimizer, 
                        epoch, epoch_loss, is_best=False)
        
        # Save training history
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        import json
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved: {history_path}")
        
        # Plot training curves
        self.plot_training_curves(history)
        
        print(f"\n{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Final Training Loss:  {epoch_loss:.6f}")
        print(f"Total Epochs:         {epoch + 1}")
        print(f"Best Model:           checkpoints_cpu/best_model.weights.h5")
        print(f"{'='*70}")
        
        return model, ema_model
    
    def validate(self, model, loss_fn, val_dataset, epoch):
        """Validation Step"""
        print(f"  Validating...")
        val_losses = []
        
        for batch in val_dataset:
            phi_b, x_noisy_b, noise_b, t_b = batch
            predictions = model([x_noisy_b, phi_b, t_b], training=False)
            loss = loss_fn(noise_b, predictions)
            val_losses.append(loss.numpy())
        
        val_loss = np.mean(val_losses)
        print(f"  Validation Loss: {val_loss:.6f}")
        
        return val_loss

    def plot_training_curves(self, history):
        """Plot Training und Validation Loss"""
        import matplotlib
        matplotlib.use('Agg')  # Für Headless-Server
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['epochs'], history['train_loss'], 'b-o', 
                label='Training Loss', linewidth=2, markersize=6)
        plt.plot(history['epochs'], history['val_loss'], 'r-s', 
                label='Validation Loss', linewidth=2, markersize=6)
        
        # Markiere bestes Epoch
        best_epoch_idx = np.argmin(history['val_loss'])
        best_epoch = history['epochs'][best_epoch_idx]
        best_val_loss = history['val_loss'][best_epoch_idx]
        
        plt.axvline(best_epoch, color='green', linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'Best (Epoch {best_epoch})')
        plt.scatter([best_epoch], [best_val_loss], 
                    color='gold', s=200, zorder=5, edgecolor='black', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.config['log_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved: {plot_path}")
        plt.close()
        
    def save_checkpoint(self, model, ema_model, optimizer, epoch, loss, is_best=False):
        """
        Speichere Checkpoint
        
        Args:
            is_best: True wenn dies das beste Modell bisher ist
        """
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}')
        
        # Main model
        model.save_weights(f"{checkpoint_path}_model.weights.h5")
        
        # Training state
        checkpoint_info = {
            'epoch': epoch + 1,
            'loss': float(loss),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(f"{checkpoint_path}_info.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Best model tracking - NEU!
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model')
            model.save_weights(f"{best_path}.weights.h5")
            
            with open(f"{best_path}_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            print(f"  ★ Best model saved: {best_path}")

def main():
    """Hauptfunktion"""
    print("Start Training")
    
    # Trainer erstellen
    trainer = DiffusionTrainer()
    
    # Training starten
    try:
        model, ema_model = trainer.train()
        print("\nTraining erfolgreich abgeschlossen!")
    except Exception as e:
        print(f"\nTraining fehler: {e}")

if __name__ == "__main__":
    main()