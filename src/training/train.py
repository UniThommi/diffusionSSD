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
import time

print(f"Eager execution: {tf.executing_eagerly()}")

# Reproducibility
tf.random.set_seed(1234)

# Runtime Report
def report_time():
    elapsed = time.perf_counter() - start_time

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60

    print("\n" + "=" * 50)
    print("⏱️  Runtime Report")
    print("-" * 50)
    print(f"   Total runtime : {h:02d}:{m:02d}:{s:05.2f}")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    # Time Tracking 
    start_time = time.perf_counter()
    # === Config Loading ===
    config = toml.load('config.toml')

    # Suppress XLA warnings if verbose=false
    if not config['training'].get('verbose', False):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Mixed Precision (Optional)
    if config['training'].get('use_mixed_precision', False):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("[Mixed Precision] Enabled (float16)")

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

    report_time()
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
    # Add Prefetch (NERSC Dokumentation)
    train_data_clean = train_data_clean.prefetch(tf.data.AUTOTUNE)
    val_data_clean = val_data_clean.prefetch(tf.data.AUTOTUNE)

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

    report_time()
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
    
    # Multi-GPU Strategy (for NERSC Perlmutter)
    if config['training'].get('use_multi_gpu', False):
        strategy = tf.distribute.MirroredStrategy()
        print(f'[Multi-GPU] Using {strategy.num_replicas_in_sync} GPUs')
    else:
        strategy = tf.distribute.get_strategy()  # Default (single device)

    with strategy.scope():
        model = scoreDiffusion(num_area=num_area, num_cond=num_cond, config=config)

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
                
                # Save EMA weights (used for generation)
                ema_area_path = os.path.join(self.checkpoint_dir, 'ema_area_model_best.weights.h5')
                self.model.ema_area.save_weights(ema_area_path)
                
                for area_name in self.model.active_areas:
                    ema_voxel_path = os.path.join(
                        self.checkpoint_dir,
                        f'ema_voxel_{area_name}_best.weights.h5'
                    )
                    self.model.ema_voxels[area_name].save_weights(ema_voxel_path)
                
                print(f"\n✓ Checkpoint saved (epoch {epoch+1}, {self.monitor}={current_loss:.4f}) [EMA weights]")
                report_time()

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
            def __init__(self, checkpoint_dir, plot_every=10, run_name="", model_config=None):
                super().__init__()
                self.checkpoint_dir = Path(checkpoint_dir)
                self.plot_every = plot_every
                self.run_name = run_name
                self.model_config = model_config
                
                # Initialize history for all metrics
                self.epoch_history = {'loss': [], 'val_loss': [], 'loss_area': [], 'val_loss_area': []}
                
                # Add voxel model histories dynamically
                for area_name in model_config['LAYER_NAMES']:
                    self.epoch_history[f'loss_{area_name}'] = []
                    self.epoch_history[f'val_loss_{area_name}'] = []
            
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
                
                suffix = "_final" if final else "_progress"
                title_suffix = " (Final)" if final else f" (Epoch {len(self.epoch_history['loss'])})"
                epochs = range(1, len(self.epoch_history['loss']) + 1)
                
                # === 1. Overview Plot (Total + ResNet) ===
                fig_overview, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig_overview.suptitle(f"Training Overview - {self.run_name}{title_suffix}", 
                                     fontsize=14, fontweight='bold')
                
                # Total Loss (top left)
                axes[0, 0].plot(epochs, self.epoch_history['loss'], 
                            label='Train', linewidth=2, marker='o', markersize=2, 
                            color='#1f77b4')  # Blue
                axes[0, 0].plot(epochs, self.epoch_history['val_loss'], 
                            label='Val', linewidth=2, marker='s', markersize=2,
                            color='#ff7f0e')  # Orange
                axes[0, 0].set_yscale('log')  # Total Loss logarithmic
                axes[0, 0].set_xlabel('Epoch', fontsize=11)
                axes[0, 0].set_ylabel('Total Loss', fontsize=11)
                axes[0, 0].set_title('Total Loss (ResNet + U-Nets weighted)', fontweight='bold')
                axes[0, 0].legend(fontsize=10)
                axes[0, 0].grid(True, alpha=0.3)
                
                # ResNet Area Loss (top right)
                axes[0, 1].plot(epochs, self.epoch_history['loss_area'], 
                            label='Train', linewidth=2, marker='o', markersize=2,
                            color='#2ca02c')  # Green
                axes[0, 1].plot(epochs, self.epoch_history['val_loss_area'], 
                            label='Val', linewidth=2, marker='s', markersize=2,
                            color='#d62728')  # Red
                axes[0, 1].set_yscale('log')  # ResNet Area Loss logarithmic
                axes[0, 1].set_xlabel('Epoch', fontsize=11)
                axes[0, 1].set_ylabel('Area Loss', fontsize=11)
                axes[0, 1].set_title('ResNet: Area Hits (4D vector)', fontweight='bold')
                axes[0, 1].legend(fontsize=10)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Summary Statistics (bottom left)
                axes[1, 0].axis('off')
                best_epoch = int(np.argmin(self.epoch_history['val_loss']))
                best_val_loss = self.epoch_history['val_loss'][best_epoch]
                
                summary_text = f"""Loss Composition:
                Total = ResNet Loss + Σ(U-Net Losses × weights)

                Area Weights:"""
                for area_name in self.model_config['LAYER_NAMES']:
                    weight = self.model_config['AREA_RATIOS'][area_name]
                    summary_text += f"\n  • {area_name}: {weight:.3f}"
                
                summary_text += f"""

                Best Epoch: {best_epoch + 1} / {len(self.epoch_history['loss'])}
                Best Val Loss: {best_val_loss:.4f}

                Current Metrics:
                Train Loss: {self.epoch_history['loss'][-1]:.4f}
                Val Loss:   {self.epoch_history['val_loss'][-1]:.4f}
                """
                
                axes[1, 0].text(0.05, 0.5, summary_text, fontsize=11, 
                              verticalalignment='center', family='monospace',
                              bbox=dict(boxstyle='round', facecolor='#fffacd', 
                                       edgecolor='gray', linewidth=1.5, alpha=0.9))
                
                # Loss Components Breakdown (bottom right)
                axes[1, 1].axis('off')
                current_epoch = len(self.epoch_history['loss'])
                
                breakdown_text = f"""Current Loss Breakdown (Epoch {current_epoch}):

                ResNet (Area):
                Train: {self.epoch_history['loss_area'][-1]:.4f}
                Val:   {self.epoch_history['val_loss_area'][-1]:.4f}

                U-Nets (Voxels):"""
                
                for area_name in self.model_config['LAYER_NAMES']:
                    train_loss = self.epoch_history[f'loss_{area_name}'][-1]
                    val_loss = self.epoch_history[f'val_loss_{area_name}'][-1]
                    breakdown_text += f"\n  {area_name}:"
                    breakdown_text += f"\n    Train: {train_loss:.4f}"
                    breakdown_text += f"\n    Val:   {val_loss:.4f}"
                
                axes[1, 1].text(0.05, 0.5, breakdown_text, fontsize=10,
                              verticalalignment='center', family='monospace',
                              bbox=dict(boxstyle='round', facecolor='#e6f2ff',
                                       edgecolor='gray', linewidth=1.5, alpha=0.9))
                
                plt.tight_layout()
                overview_path = self.checkpoint_dir / f'overview{suffix}.png'
                plt.savefig(overview_path, dpi=150, bbox_inches='tight')
                plt.close(fig_overview)
                
                # === 2. U-Nets Detailed Plot ===
                n_unets = len(self.model_config['LAYER_NAMES'])
                n_cols = 2
                n_rows = (n_unets + 1) // 2  # Ceiling division
                
                fig_unets, axes_unets = plt.subplots(n_rows, n_cols, 
                                                     figsize=(14, 5 * n_rows))
                fig_unets.suptitle(f"U-Net Losses - {self.run_name}{title_suffix}",
                                  fontsize=14, fontweight='bold')
                
                # Flatten axes for easier indexing
                if n_unets == 1:
                    axes_unets = np.array([axes_unets])
                axes_flat = axes_unets.flatten()
                
                # Color pairs (train, val) for each U-Net
                color_pairs = [
                    ('#1f77b4', '#ff7f0e'),  # Blue / Orange
                    ('#2ca02c', '#d62728'),  # Green / Red
                    ('#9467bd', '#e377c2'),  # Purple / Pink
                    ('#8c564b', '#bcbd22'),  # Brown / Yellow-green
                ]
                
                for idx, area_name in enumerate(self.model_config['LAYER_NAMES']):
                    ax = axes_flat[idx]
                    train_color, val_color = color_pairs[idx % len(color_pairs)]
                    
                    ax.plot(epochs, self.epoch_history[f'loss_{area_name}'],
                           label='Train', linewidth=2, marker='o', markersize=2,
                           color=train_color)
                    ax.plot(epochs, self.epoch_history[f'val_loss_{area_name}'],
                           label='Val', linewidth=2, marker='s', markersize=2,
                           color=val_color)
                    ax.set_yscale('log')                    
                    ax.set_xlabel('Epoch', fontsize=11)
                    ax.set_ylabel('Loss (MSE)', fontsize=11)
                    
                    ax.set_title(f'{area_name} (unweighted MSE)', fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(n_unets, len(axes_flat)):
                    axes_flat[idx].axis('off')
                
                plt.tight_layout()
                unets_path = self.checkpoint_dir / f'unets{suffix}.png'
                plt.savefig(unets_path, dpi=150, bbox_inches='tight')
                plt.close(fig_unets)
                
                # === 3. Individual Model Plots (unchanged) ===
                # ResNet plot
                fig_resnet, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.plot(epochs, self.epoch_history['loss_area'], 
                       label='Train', linewidth=2, marker='o', markersize=3,
                       color='#2ca02c')
                ax.plot(epochs, self.epoch_history['val_loss_area'],
                       label='Val', linewidth=2, marker='s', markersize=3,
                       color='#d62728')
                ax.set_yscale('log')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Loss', fontsize=12)
                ax.set_title(f'ResNet: Area Hits - {self.run_name}{title_suffix}', 
                           fontsize=13, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                resnet_path = self.checkpoint_dir / f'resnet{suffix}.png'
                plt.savefig(resnet_path, dpi=150, bbox_inches='tight')
                plt.close(fig_resnet)
                
                # U-Net individual plots
                for idx, area_name in enumerate(self.model_config['LAYER_NAMES']):
                    train_color, val_color = color_pairs[idx % len(color_pairs)]
                    
                    fig_unet, ax = plt.subplots(1, 1, figsize=(8, 6))
                    ax.plot(epochs, self.epoch_history[f'loss_{area_name}'], 
                           label='Train', linewidth=2, marker='o', markersize=3,
                           color=train_color)
                    ax.plot(epochs, self.epoch_history[f'val_loss_{area_name}'], 
                           label='Val', linewidth=2, marker='s', markersize=3,
                           color=val_color)
                    ax.set_yscale('log')
                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Loss (MSE)', fontsize=12)
                    
                    ax.set_title(f'{area_name} Voxel Loss (unweighted MSE) - {self.run_name}{title_suffix}',
                               fontsize=13, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    unet_path = self.checkpoint_dir / f'unet_{area_name}{suffix}.png'
                    plt.savefig(unet_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_unet)
                
                if not final:
                    n_plots = 2 + 1 + len(self.model_config['LAYER_NAMES'])  # overview + unets + resnet + individual unets
                    print(f"  → Saved {n_plots} plots (overview, unets, resnet, {len(self.model_config['LAYER_NAMES'])} individual)")
        
        plot_freq = config['training'].get('plot_update_frequency', 10)
        incremental_plotter = IncrementalPlotter(
            checkpoint_dir=str(checkpoint_folder),
            plot_every=plot_freq,
            run_name=config['training']['run_name'],
            model_config=config
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
    report_time()
    
    # Verbose: 2=one line per epoch (minimal), 1=progress bar (detailed)
    verbose_level = 1 if config['training'].get('verbose', False) else 2
    if rank != 0:
        verbose_level = 0  # Multi-GPU: nur rank 0 printet
    
    history = model.fit(
        train_data_clean.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(train_size / BATCH_SIZE),
        validation_data=val_data_clean.batch(BATCH_SIZE),
        validation_steps=int(val_size / BATCH_SIZE),
        verbose=verbose_level,
        callbacks=callbacks
    )   
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    report_time()

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
        report_time()