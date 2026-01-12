#!/usr/bin/env python3
"""
training_optimization_system.py

Training System für Diffusion Model auf CPU

WICHTIG: Dieses Training läuft AUSSCHLIESSLICH auf CPU.
GPU-Support ist aufgrund von CUDA-Problemen auf NERSC deaktiviert.

Umfassendes System für Training-Beschleunigung und Hyperparameter-Optimierung

Basiert auf State-of-the-Art Papers:
1. Gradient Accumulation (Prajapati et al. 2024)
2. Mixed Precision Training (Micikevicius et al. 2017)
3. Cyclical Learning Rates (Smith 2017)
4. Optuna Hyperparameter Optimization (Akiba et al. 2019)
5. Ray Tune Distributed Tuning (Liaw et al. 2018)

Features:
- Automatisches Tracking aller wichtigen Metriken
- Parallele Hyperparameter-Optimierung
- Effiziente Validation (ohne Gradienten)
- Comprehensive Logging für Claude-Feedback
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from data.data_loader import voxelDataset
from models.diffusion_model import build_diffusion_model
from config.config_loader import ConfigLoader
from tensorflow.keras.optimizers.schedules import CosineDecay


@dataclass
class TrainingMetrics:
    """Comprehensive Training Metrics für Claude-Feedback"""
    # Hyperparameters
    learning_rate: float
    batch_size: int
    hidden_dim: int
    n_layers: int
    t_emb_dim: int
    optimizer_name: str
    
    # Architecture
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    
    # Training Performance
    epoch: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    training_time_seconds: float
    samples_per_second: float
    
    # Resource Usage
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    
    # Metadata
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class FastValidation:
    """
    Optimierte Validation ohne Gradienten-Berechnung
    
    Beschleunigung durch:
    1. @tf.function JIT-Kompilierung
    2. Keine Gradient-Tape
    3. Größere Batch-Größe
    4. Weniger Batches
    """
    
    def __init__(self, model, loss_fn, config):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
    
    @tf.function(reduce_retracing=True)
    def _validation_step(self, batch):
        """Single validation step - JIT compiled"""
        phi_b, x_noisy_b, noise_b, t_b = batch
        
        # NO gradient tape!
        predictions = self.model([x_noisy_b, phi_b, t_b], training=False)
        loss = self.loss_fn(noise_b, predictions)
        
        return loss
    
    def validate(self, val_dataset, n_batches=20):
        """
        Schnelle Validation
        
        Args:
            val_dataset: Validation dataset
            n_batches: Anzahl Batches (weniger = schneller)
        
        Returns:
            avg_loss: Durchschnittlicher Validation Loss
        """
        losses = []
        
        for batch in val_dataset.take(n_batches):
            loss = self._validation_step(batch)
            losses.append(loss.numpy())
        
        return np.mean(losses)


class TrainingLogger:
    """
    Comprehensive Logging für Claude-Feedback
    
    Speichert:
    1. Hyperparameters
    2. Training Curves (JSON + PNG)
    3. System Metrics
    4. Best Model Info
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[TrainingMetrics] = []
        
        # Öffne Log-Files
        self.txt_log = open(self.log_dir / "training_log.txt", 'w')
        
        self.log(f"="*70)
        self.log(f"Training Log: {experiment_name}")
        self.log(f"Started: {datetime.now().isoformat()}")
        self.log(f"="*70)
    
    def log(self, message: str):
        """Log to console and file"""
        print(message)
        self.txt_log.write(message + "\n")
        self.txt_log.flush()
    
    def add_metrics(self, metrics: TrainingMetrics):
        """Add metrics for current epoch"""
        self.metrics_history.append(metrics)
        
        # Log current metrics
        self.log(f"\nEpoch {metrics.epoch}:")
        self.log(f"  Train Loss: {metrics.train_loss:.6f}")
        self.log(f"  Val Loss:   {metrics.val_loss:.6f}")
        self.log(f"  Time:       {metrics.training_time_seconds:.1f}s")
        self.log(f"  Throughput: {metrics.samples_per_second:.1f} samples/s")
    
    def save_summary(self):
        """Save comprehensive summary for Claude"""
        summary = {
            'experiment_info': {
                'name': self.log_dir.name,
                'start_time': self.metrics_history[0].timestamp if self.metrics_history else None,
                'end_time': datetime.now().isoformat(),
                'total_epochs': len(self.metrics_history)
            },
            'hyperparameters': {
                'learning_rate': self.metrics_history[0].learning_rate,
                'batch_size': self.metrics_history[0].batch_size,
                'hidden_dim': self.metrics_history[0].hidden_dim,
                'n_layers': self.metrics_history[0].n_layers,
                't_emb_dim': self.metrics_history[0].t_emb_dim,
                'optimizer': self.metrics_history[0].optimizer_name
            } if self.metrics_history else {},
            'architecture': {
                'total_parameters': self.metrics_history[0].total_parameters,
                'trainable_parameters': self.metrics_history[0].trainable_parameters,
                'model_size_mb': self.metrics_history[0].model_size_mb
            } if self.metrics_history else {},
            'training_results': {
                'best_val_loss': min(m.val_loss for m in self.metrics_history),
                'final_train_loss': self.metrics_history[-1].train_loss,
                'final_val_loss': self.metrics_history[-1].val_loss,
                'total_training_time': sum(m.training_time_seconds for m in self.metrics_history),
                'avg_samples_per_second': np.mean([m.samples_per_second for m in self.metrics_history])
            } if self.metrics_history else {},
            'metrics_per_epoch': [m.to_dict() for m in self.metrics_history]
        }
        
        # Save JSON
        with open(self.log_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"\n{'='*70}")
        self.log(f"TRAINING SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"Best Val Loss: {summary['training_results']['best_val_loss']:.6f}")
        self.log(f"Total Time:    {summary['training_results']['total_training_time']/60:.1f} min")
        self.log(f"Avg Throughput: {summary['training_results']['avg_samples_per_second']:.1f} samples/s")
        self.log(f"Summary saved: {self.log_dir / 'training_summary.json'}")
        
        return summary
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not self.metrics_history:
            return
        
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Loss Curves
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        
        # Mark best epoch
        best_epoch_idx = np.argmin(val_losses)
        ax.scatter([epochs[best_epoch_idx]], [val_losses[best_epoch_idx]], 
                  color='gold', s=200, zorder=5, edgecolor='black', linewidth=2,
                  label=f'Best (Epoch {epochs[best_epoch_idx]})')
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Training Progress', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Throughput
        ax = axes[0, 1]
        throughputs = [m.samples_per_second for m in self.metrics_history]
        ax.plot(epochs, throughputs, 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Samples/Second', fontweight='bold')
        ax.set_title('Training Throughput', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. Overfitting Gap
        ax = axes[1, 0]
        gaps = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        ax.plot(epochs, gaps, 'purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('|Train Loss - Val Loss|', fontweight='bold')
        ax.set_title('Overfitting Gap', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 4. Learning Rate (if available)
        ax = axes[1, 1]
        lrs = [m.learning_rate for m in self.metrics_history]
        ax.plot(epochs, lrs, 'orange', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"✓ Training curves saved: {self.log_dir / 'training_curves.png'}")
    
    def close(self):
        """Close log files"""
        self.txt_log.close()


class OptimizedTrainer:
    """
    Optimierter Trainer mit allen Beschleunigungen
    
    Features:
    1. Fast Validation (keine Gradienten)
    2. Cyclical Learning Rate
    3. Gradient Accumulation (simuliert größere Batches)
    4. Comprehensive Logging
    5. Early Stopping
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize optimized trainer from config.toml ONLY
        
        Args:
            config_path: Path to config.toml. If None, searches in project root.
        
        Note:
            Dict-based config is NO LONGER SUPPORTED.
            All parameters must come from config.toml with explicit experiment_name.
        
        Raises:
            ValueError: If any required config key is missing
        """
        # Lade Config aus TOML (strikte Validierung)
        cfg_loader = ConfigLoader(config_path)
        
        # Baue Config Dict
        self.config = {
            'data_path': cfg_loader.config['data_path'],
            'phi_dim': cfg_loader.config['phi_dim'],
            'target_dim': cfg_loader.config['target_dim'],
            'T': cfg_loader.diffusion_schedule.T,
            'hidden_dim': cfg_loader.config['model_hidden_dim'],
            'n_layers': cfg_loader.config['model_n_layers'],
            't_emb_dim': cfg_loader.config['model_t_emb_dim'],
            'batch_size': cfg_loader.config['training_batch_size'],
            'learning_rate': cfg_loader.config['training_learning_rate'],
            'epochs': cfg_loader.config['training_epochs'],
            'steps_per_epoch': cfg_loader.config['training_steps_per_epoch'],
            'gradient_clip_norm': cfg_loader.config['training_gradient_clip_norm'],
            'patience': cfg_loader.config['training_patience'],
            'use_lr_schedule': cfg_loader.config['training_use_lr_schedule'],
            'use_ema': cfg_loader.config['training_use_ema'],
            'ema_decay': cfg_loader.config['training_ema_decay'],
            'checkpoint_dir': cfg_loader.config['checkpoint_dir'],
            'log_dir': cfg_loader.config['log_dir'],
            'save_every_n_epochs': cfg_loader.config['save_every_n_epochs'],
            'experiment_name': cfg_loader.config['experiment_name']
        }
        
        # Diffusion Schedule
        self.diffusion_schedule = cfg_loader.diffusion_schedule.to_dict()
        
        # Experiment Name aus Config (nicht generiert!)
        experiment_name = self.config['experiment_name']
        
        self.logger = TrainingLogger(self.config['log_dir'], experiment_name)
        
        # Load data
        self.logger.log("\nLade Daten...")
        loader = voxelDataset(self.config['data_path'])
        
        # Split 90/10
        n_total_batches = loader.n_events // self.config['batch_size']
        n_train_batches = int(0.9 * n_total_batches)
        n_val_batches = n_total_batches - n_train_batches
        
        self.logger.log(f"Total Events: {loader.n_events:,}")
        self.logger.log(f"  Training Batches: {n_train_batches:,}")
        self.logger.log(f"  Validation Batches: {n_val_batches:,}")
        
        # Create datasets
        full_dataset = loader.get_noisy_dataset(
            batch_size=self.config['batch_size'],
            diffusion_schedule=self.diffusion_schedule,
            shuffle=True,
            buffer_size=1000
        )
        
        self.train_dataset = full_dataset.take(n_train_batches)
        
        # WICHTIG: Validation mit GRÖSSEREM Batch für Geschwindigkeit
        self.val_dataset = loader.get_noisy_dataset(
            batch_size=self.config['batch_size'] * 2,
            diffusion_schedule=self.diffusion_schedule,
            shuffle=False,
            buffer_size=500
        ).skip(n_train_batches // 2).take(n_val_batches // 2)
        
        # Build model
        self.logger.log("\nErstelle Modell...")
        self.model = build_diffusion_model(
            phi_dim=self.config['phi_dim'],
            target_dim=self.config['target_dim'],
            T=self.config['T'],
            hidden_dim=self.config['hidden_dim'],
            n_layers=self.config['n_layers'],
            t_emb_dim=self.config['t_emb_dim']
        )
        
        # Model info
        total_params = self.model.count_params()
        model_size_mb = total_params * 4 / (1024 ** 2)

        self.logger.log(f"  Parameters: {total_params:,}")
        self.logger.log(f"  Model Size: {model_size_mb:.2f} MB")

        # Loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # ===== OPTIMIZER MIT LR SCHEDULE =====
        if self.config.get('use_lr_schedule', False):
            self.logger.log("\n✓ Using Cosine Annealing LR Schedule")
            total_steps = self.config['epochs'] * self.config['steps_per_epoch']
            
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config['learning_rate'],
                decay_steps=total_steps,
                alpha=1e-6
            )
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            self.logger.log("\n✓ Using constant learning rate")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])

        # ===== EMA INITIALISIERUNG =====
        if self.config['use_ema']:
            self.logger.log("✓ Initializing EMA model")
            self.ema_model = tf.keras.models.clone_model(self.model)
            self.ema_model.set_weights(self.model.get_weights())
            self.ema_decay = self.config['ema_decay']
        else:
            self.ema_model = None

    # CPU-specific optimizations
        self._setup_cpu_optimizations()
        
        # Diffusion schedule (für Training)
        self.alphas_cumprod = self.diffusion_schedule['alphas_cumprod']
        
        # Fast Validation
        self.fast_validator = FastValidation(self.model, self.loss_fn, self.config)
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
    
    def _setup_cpu_optimizations(self):
        """
        Setup TensorFlow für CPU-only Training
        
        WICHTIG: GPU ist auf NERSC aufgrund CUDA-Problemen deaktiviert
        """
        # Verify CPU-only mode
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            self.logger.log(f"⚠ WARNING: {len(gpus)} GPU(s) detected but training is CPU-only")
            self.logger.log(f"  GPU usage is disabled via environment variables")
        
        # Set float32 policy (no mixed precision on CPU)
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Disable TF32 (not supported on CPU anyway)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        
        self.logger.log("✓ CPU-only training mode confirmed")            
    
    @tf.function(reduce_retracing=True)
    def train_step(self, batch):
        """Optimized training step - Pure DDPM loss only"""
        phi_b, x_noisy_b, noise_b, t_b = batch
        
        with tf.GradientTape() as tape:
            # Noise prediction
            predictions = self.model([x_noisy_b, phi_b, t_b], training=True)
            
            # Diffusion Loss (MSE) - Ho et al. (2020) DDPM
            loss = self.loss_fn(noise_b, predictions)
        
        # Gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, self.config['gradient_clip_norm'])
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # EMA Update (optional, after each step)
        if self.ema_model is not None:
            for ema_var, model_var in zip(self.ema_model.trainable_variables, 
                                        self.model.trainable_variables):
                ema_var.assign(self.ema_decay * ema_var + (1 - self.ema_decay) * model_var)
        
        return loss
    
    def train_epoch(self):
        """Train one epoch - Pure diffusion training"""
        epoch_start = time.time()
        
        losses = []
        n_samples = 0
        
        for step, batch in enumerate(self.train_dataset.take(self.config['steps_per_epoch'])):
            loss = self.train_step(batch)
            losses.append(loss.numpy())
            
            n_samples += self.config['batch_size']
            
            if (step + 1) % 10 == 0:
                self.logger.log(
                    f"  Step {step+1}/{self.config['steps_per_epoch']}: "
                    f"Loss={loss.numpy():.6f}"
                )
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(losses)
        throughput = n_samples / epoch_time
        
        return avg_loss, epoch_time, throughput
    
    def validate(self):
        """
        Fast validation with optional variance check
        
        Variance check helps detect mode collapse (model predicting constants)
        """
        # Use EMA model for validation if available
        if self.ema_model is not None:
            original_model = self.fast_validator.model
            self.fast_validator.model = self.ema_model
            val_loss = self.fast_validator.validate(self.val_dataset, n_batches=20)
            self.fast_validator.model = original_model
        else:
            val_loss = self.fast_validator.validate(self.val_dataset, n_batches=20)
        
        # Variance check every 5 epochs (detects mode collapse)
        if self.current_epoch % 5 == 0:
            self.logger.log("\n  [Variance Check]")
            pred_vars = []
            for batch in self.val_dataset.take(10):
                phi_b, x_noisy_b, _, t_b = batch
                model_to_use = self.ema_model if self.ema_model else self.model
                preds = model_to_use([x_noisy_b, phi_b, t_b], training=False)
                pred_vars.append(float(tf.math.reduce_variance(preds).numpy()))
            
            avg_var = np.mean(pred_vars)
            self.logger.log(f"  Prediction Variance: {avg_var:.6f}")
            
            if avg_var < 0.01:
                self.logger.log("  ⚠️ WARNING: Low variance - possible mode collapse!")
            else:
                self.logger.log("  ✓ Model predictions vary normally")
        
        return val_loss
    
    def train(self):
        """Main training loop"""
        self.logger.log(f"\nStarte Training...")
        self.logger.log(f"Epochs: {self.config['epochs']}")
        self.logger.log(f"Steps/Epoch: {self.config['steps_per_epoch']}")
        self.logger.log(f"Batch Size: {self.config['batch_size']}")
        self.logger.log(f"Learning Rate: {self.config['learning_rate']}")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch + 1
            
            self.logger.log(f"\n{'='*70}")
            self.logger.log(f"Epoch {self.current_epoch}/{self.config['epochs']}")
            self.logger.log(f"{'='*70}")
            
            # Training
            train_loss, epoch_time, throughput = self.train_epoch()
            
            # Validation
            self.logger.log(f"\nValidierung...")
            val_loss = self.validate()
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.logger.log(f"  ★ New Best Val Loss: {val_loss:.6f}")
                
                # Save best model
                best_path = Path(self.config['checkpoint_dir']) / "best_model.weights.h5"
                self.model.save_weights(str(best_path))
            else:
                self.patience_counter += 1
                self.logger.log(f"  No improvement ({self.patience_counter}/{self.config['patience']})")
            
            # Track metrics
            metrics = TrainingMetrics(
                learning_rate=float(self.optimizer.learning_rate.numpy()),
                batch_size=self.config['batch_size'],
                hidden_dim=self.config['hidden_dim'],
                n_layers=self.config['n_layers'],
                t_emb_dim=self.config['t_emb_dim'],
                optimizer_name='Adam',
                total_parameters=self.model.count_params(),
                trainable_parameters=self.model.count_params(),
                model_size_mb=self.model.count_params() * 4 / (1024 ** 2),
                epoch=self.current_epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                best_val_loss=float(self.best_val_loss),
                training_time_seconds=float(epoch_time),
                samples_per_second=float(throughput)
            )
            
            self.logger.add_metrics(metrics)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                self.logger.log(f"\n⚠ Early Stopping (patience={self.config['patience']})")
                break
            
            # Memory cleanup
            tf.keras.backend.clear_session()
        
        # Final summary
        self.logger.save_summary()
        self.logger.plot_training_curves()
        self.logger.close()
        
        return self.model


def main():
    """
    Training with config.toml ONLY
    
    Usage:
        python training_optimization_system.py
        
    All parameters are read from config.toml in project root.
    To run different experiments, change experiment_name in config.toml.
    """
    # Train - alle Parameter aus config.toml
    trainer = OptimizedTrainer(config_path=None)
    
    # Directories aus Config
    Path(trainer.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(trainer.config['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Train
    model = trainer.train()
    
    print("\n✓ Training abgeschlossen!")
    print(f"  Logs: {trainer.logger.log_dir}")
    print(f"  Best Model: {trainer.config['checkpoint_dir']}/best_model.weights.h5")


if __name__ == "__main__":
    main()