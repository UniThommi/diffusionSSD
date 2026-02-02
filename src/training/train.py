#!/usr/bin/env python3
"""
scoreDiffusion Training Script
Adapted from CaloScore train.py for neutron capture optical response prediction
"""
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import toml
from pathlib import Path
import gc
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for NERSC
import matplotlib.pyplot as plt

print(f"Eager execution: {tf.executing_eagerly()}")

# Reproducibility
tf.random.set_seed(1234)

if __name__ == '__main__':
    # === Config Loading ===
    config = toml.load('config.toml')

    # Transformiere config für CaloScore
    config['EMBED'] = config['model']['embed_dim']
    
    # Initialize PAD dict (will be filled by data_loader for auto-geometry regions)
    config['PAD'] = {}

    # Filter active regions
    active_regions = config['training'].get('active_regions', {}).get('enabled', ['PIT', 'BOT', 'WALL', 'TOP'])
    
    # Check for merged regions
    merged_regions = config['regions'].get('merged_regions', [])
    
    # Build LAYER_NAMES (U-Net names)
    if set(merged_regions).issubset(set(active_regions)):
        # Replace merged regions with single combined name
        layer_names = [r for r in active_regions if r not in merged_regions]
        layer_names.append('PITBOT')  # Add merged name
        layer_names.sort()  # Keep consistent order
    else:
        layer_names = active_regions
    
    config['LAYER_NAMES'] = layer_names
    config['ALL_REGIONS'] = ['PIT', 'BOT', 'WALL', 'TOP']  # Original regions for data_loader
    
    print(f"[Model Architecture] U-Nets: {config['LAYER_NAMES']}")
    if merged_regions:
        print(f"[Merged Regions] {' + '.join(merged_regions)} → PITBOT")

    # Grid shapes (will be set by data_loader based on geometry config)
    config['SHAPE_PIT'] = None
    config['SHAPE_BOT'] = None
    config['SHAPE_WALL'] = None
    config['SHAPE_TOP'] = None
    config['SHAPE_PITBOT'] = None  # For merged PIT+BOT grid

    config['num_steps'] = config['diffusion']['num_steps']
    config['ema_decay'] = config['training']['ema_decay']

    # Area ratios (für loss weighting)
    config['AREA_RATIOS'] = {}
    for region in config['LAYER_NAMES']:
        if region == 'PITBOT':
            # Merged region: sum individual ratios
            config['AREA_RATIOS'][region] = config['normalization']['area_ratios']['PITBOT']
        else:
            config['AREA_RATIOS'][region] = config['normalization']['area_ratios'][region]
    
    # === Hardware Setup ===
    if config['training']['enable_horovod']:
        import horovod.tensorflow.keras as hvd
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
        rank = hvd.rank()
        size = hvd.size()
    else:
        hvd = None
        rank = 0
        size = 1
        
        if config['training']['device'] == 'cpu':
            # Force CPU
            tf.config.set_visible_devices([], 'GPU')
            print("[CPU Mode] GPUs hidden")
        else:
            # GPU without Horovod
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[GPU Mode] Found {len(gpus)} GPU(s)")
    
    # === Data Loading ===
    from src.data.data_loader import voxelDataset

    data_dir = Path(config['training']['data_dir'])
    train_file = data_dir / config['training']['train_file']
    val_file = data_dir / config['training']['val_file']

    print("\n=== Loading Data ===")
    print(f"Train: {train_file}")
    print(f"Val: {val_file}")

    # Create datasets (mit is_validation Flag)
    train_dataset = voxelDataset(str(train_file), config, is_validation=False)
    val_dataset = voxelDataset(str(val_file), config, is_validation=True)

    # Update config with inferred shapes from data_loader
    for region in config['LAYER_NAMES']:
        shape_key = f'SHAPE_{region}'
        config[shape_key] = list(train_dataset.grid_shapes[region])
    
    # Get clean datasets (CaloScore style - no noise injection)
    train_data_clean = train_dataset.get_dataset(
        rank=rank, size=size
    ).repeat()
    val_data_clean = val_dataset.get_dataset(
        rank=rank, size=size
    ).repeat()

    # === Hyperparameters ===
    BATCH_SIZE = config['training']['batch_size']
    LR = config['training']['learning_rate']
    NUM_EPOCHS = config['training']['num_epochs']
    EARLY_STOP = config['training']['early_stop_patience']

    print("\n=== Dataset Shape Verification ===")
    sample_batch = next(iter(train_data_clean.batch(BATCH_SIZE)))
    voxels_batch, area_hits_batch, cond_batch = sample_batch

    print(f"Voxel shapes after batching:")
    for region_name in config['LAYER_NAMES']:
        shape = voxels_batch[region_name].shape
        print(f"  {region_name}: {shape}")
        expected = (BATCH_SIZE,) + tuple(config['SHAPE_' + region_name])
        if shape != expected:
            raise ValueError(
                f"Shape mismatch for {region_name}!\n"
                f"  Expected: {expected}\n"
                f"  Got: {shape}"
            )

    print(f"Area hits shape: {area_hits_batch.shape}")
    print(f"Condition shape: {cond_batch.shape}")
    print(f"✓ All shapes correct\n")
        
    # Train/Val split
    train_frac = config['training']['train_val_split']
    num_cond = train_dataset.phi_dim
    num_area = 4
    
    # Training nutzt bereits die limitierte Zahl OHNE nochmalige Multiplikation
    train_size = train_dataset.n_events
    val_size = val_dataset.n_events

    print(f"\n=== DEBUG ===")
    print(f"train_dataset.n_events = {train_dataset.n_events}")
    print(f"train_dataset.n_events_total = {train_dataset.n_events_total}")
    print(f"train_size = {train_size}")
    print(f"val_size = {val_size}")

    print(f"\n=== Training Configuration ===")
    print(f"Training events: {train_size:,}")
    print(f"Validation events: {val_size:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch (train): {int(train_size / BATCH_SIZE)}")
    print(f"Validation steps: {int(val_size / BATCH_SIZE)}")
    print(f"Conditioning dims: {num_cond}")
    print(f"Number of areas: {num_area}")
    
    # === Callbacks ===
    callbacks = []
    
    if hvd is not None:
        callbacks.extend([
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback()
        ])
    
    callbacks.append(
        EarlyStopping(patience=EARLY_STOP, restore_best_weights=True)
    )
    
    # Checkpoint folder (include run_name)
    checkpoint_folder = Path(config['training']['checkpoint_dir']) / config['training']['run_name']
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    # === Model Initialization ===
    from src.models.scoreDiffusion import scoreDiffusion

    num_layer = config['model'].get('num_resnet_layers', 3) ##FIX wird nie benutzt
    
    model = scoreDiffusion(
        num_area=num_area, 
        num_cond=num_cond,
        config=config
        )
    
    # === Distillation (NOT IMPLEMENTED - Structure only) ===
    if config['training']['enable_distillation']:
        raise NotImplementedError(
            "Distillation not yet implemented for NC-Score.\n"
            "Set enable_distillation = false in config.toml"
        )
    
    # === Optimizer ===
    if config['training']['use_cosine_decay']:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LR * size,
            decay_steps=NUM_EPOCHS * int(train_size / BATCH_SIZE)  # ← Korrigiert
        )
    else:
        lr_schedule = LR * size
    
    if config['training']['optimizer'] == 'Adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    if hvd is not None:
        opt = hvd.DistributedOptimizer(opt, average_aggregated_gradients=True)
    
    # === Compile Model ===
    model.compile(
        optimizer=opt,
        weighted_metrics=[]
    )
    
    # Custom Checkpoint Callback für Multi-Model Architektur
    class MultiModelCheckpoint(Callback):
        """Speichert Area- und Voxel-Modelle separat"""
        def __init__(self, checkpoint_dir, monitor='val_loss', save_best_only=True):
            super().__init__()
            self.checkpoint_dir = checkpoint_dir
            self.monitor = monitor
            self.save_best_only = save_best_only
            self.best_loss = float('inf')
        
        def on_epoch_end(self, epoch, logs=None):
            current_loss = logs.get(self.monitor)
            
            if current_loss is None:
                return
            
            if not self.save_best_only or current_loss < self.best_loss:
                self.best_loss = current_loss
                
                # Save Area Model
                area_path = os.path.join(self.checkpoint_dir, 'area_model_best.weights.h5')
                self.model.model_area.save_weights(area_path)
                
                # Save Voxel Models
                for area_name in self.model.active_areas:
                    voxel_path = os.path.join(
                        self.checkpoint_dir, 
                        f'voxel_{area_name}_best.weights.h5'
                    )
                    self.model.model_voxels[area_name].save_weights(voxel_path)
                
                print(f"\n✓ Checkpoint saved (epoch {epoch+1}, {self.monitor}={current_loss:.4f})")

    if rank == 0:
        checkpoint = MultiModelCheckpoint(
            checkpoint_dir=str(checkpoint_folder),
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)

    class MemoryCleanup(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    callbacks.append(MemoryCleanup())

    # === Incremental Plotting ===
    if rank == 0 and config['training'].get('save_training_plots', False):
        class IncrementalPlotter(Callback):
            def __init__(self, checkpoint_dir, plot_every=10, run_name=""):
                super().__init__()
                self.checkpoint_dir = Path(checkpoint_dir)
                self.plot_every = plot_every
                self.run_name = run_name
                self.epoch_history = {
                    'loss': [], 'val_loss': [],
                    'loss_area': [], 'val_loss_layer': [],
                    'loss_WALL': [], 'val_loss_WALL': []
                }
            
            def on_epoch_end(self, epoch, logs=None):
                # Append current metrics
                for key in self.epoch_history.keys():
                    if key in logs:
                        self.epoch_history[key].append(float(logs[key]))
                
                # Plot every N epochs or on last epoch
                if self.plot_every > 0:
                    if (epoch + 1) % self.plot_every == 0 or epoch == 0:
                        self._plot_and_save()
                        gc.collect()
            
            def on_train_end(self, logs=None):
                # Final plot at end of training
                self._plot_and_save(final=True)
                gc.collect()
            
            def _plot_and_save(self, final=False):
                if len(self.epoch_history['loss']) == 0:
                    return
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                suffix = "_final" if final else "_progress"
                title_suffix = " (Final)" if final else f" (Epoch {len(self.epoch_history['loss'])})"
                
                fig.suptitle(f"Training History - {self.run_name}{title_suffix}", 
                            fontsize=14, fontweight='bold')
                
                epochs = range(1, len(self.epoch_history['loss']) + 1)
                
                # Total Loss
                axes[0, 0].plot(epochs, self.epoch_history['loss'], 
                            label='Train', linewidth=2, marker='o', markersize=3)
                axes[0, 0].plot(epochs, self.epoch_history['val_loss'], 
                            label='Val', linewidth=2, marker='s', markersize=3)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Total Loss')
                axes[0, 0].set_title('Total Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Area Loss
                axes[0, 1].plot(epochs, self.epoch_history['loss_area'], 
                            label='Train', linewidth=2, marker='o', markersize=3)
                axes[0, 1].plot(epochs, self.epoch_history['val_loss_layer'], 
                            label='Val', linewidth=2, marker='s', markersize=3)
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Area Loss')
                axes[0, 1].set_title('Area Hits Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Voxel Loss (WALL)
                axes[1, 0].plot(epochs, self.epoch_history['loss_WALL'], 
                            label='Train', linewidth=2, marker='o', markersize=3)
                axes[1, 0].plot(epochs, self.epoch_history['val_loss_WALL'], 
                            label='Val', linewidth=2, marker='s', markersize=3)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Voxel Loss')
                axes[1, 0].set_title('WALL Voxel Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Best Epoch Info
                best_epoch = int(np.argmin(self.epoch_history['val_loss']))
                best_val_loss = self.epoch_history['val_loss'][best_epoch]
                
                axes[1, 1].axis('off')
                info_text = f"""
                Best Epoch: {best_epoch + 1}
                Best Val Loss: {best_val_loss:.4f}
                
                Current Metrics:
                Train Loss: {self.epoch_history['loss'][-1]:.4f}
                Val Loss: {self.epoch_history['val_loss'][-1]:.4f}
                
                Total Epochs: {len(self.epoch_history['loss'])}
                """
                axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, 
                            verticalalignment='center', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                plot_path = self.checkpoint_dir / f'training_history{suffix}.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)  # CRITICAL: Free memory
                
                if not final:
                    print(f"  → Plot updated: {plot_path}")
        
        plot_freq = config['training'].get('plot_update_frequency', 10)
        incremental_plotter = IncrementalPlotter(
            checkpoint_dir=str(checkpoint_folder),
            plot_every=plot_freq,
            run_name=config['training']['run_name']
        )
        callbacks.append(incremental_plotter)
        print(f"✓ Incremental plotting enabled (every {plot_freq} epochs)")

    # === TensorBoard Logging ===
    if rank == 0 and config['training'].get('enable_tensorboard', False):
        log_dir = Path('./logs') / config['training']['run_name']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard = TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,  # Keine Weight-Histogramme (langsam)
            write_graph=False,  # Kein Graph (komplex bei Multi-Model)
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        print(f"✓ TensorBoard logging enabled: {log_dir}")
    
    # === Training ===
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    history = model.fit(
        train_data_clean.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(train_size / BATCH_SIZE),  # ← Korrigiert: Keine train_frac mehr!
        validation_data=val_data_clean.batch(BATCH_SIZE),
        validation_steps=int(val_size / BATCH_SIZE),  # ← Korrigiert
        verbose=1 if rank == 0 else 0,
        callbacks=callbacks
    )   
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # === Save Training History ===
    if rank == 0:
        # Save history as JSON (optional)
        if config['training'].get('save_history_json', False):
            history_path = checkpoint_folder / 'history.json'
            with open(history_path, 'w') as f:
                history_dict = {k: [float(v) for v in vals] 
                               for k, vals in history.history.items()}
                json.dump(history_dict, f, indent=2)
            print(f"✓ Training history saved: {history_path}")
        
        # Save best epoch info
        best_epoch = np.argmin(history.history['val_loss'])
        best_val_loss = history.history['val_loss'][best_epoch]
        
        best_epoch_path = checkpoint_folder / 'best_epoch.txt'
        with open(best_epoch_path, 'w') as f:
            f.write(f"Best Epoch: {best_epoch + 1}\n")
            f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
            f.write(f"\nMetrics at best epoch:\n")
            for key in history.history.keys():
                if 'val_' in key:
                    f.write(f"  {key}: {history.history[key][best_epoch]:.4f}\n")
        print(f"✓ Best epoch info saved: {best_epoch_path}")
        
        print("\n✓ All outputs saved successfully")