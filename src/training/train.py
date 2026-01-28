#!/usr/bin/env python3
"""
scoreDiffusion Training Script
Adapted from CaloScore train.py for neutron capture optical response prediction
"""
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import toml
from pathlib import Path
import gc

print(f"Eager execution: {tf.executing_eagerly()}")

# Reproducibility
tf.random.set_seed(1234)

if __name__ == '__main__':
    # === Config Loading ===
    config = toml.load('config.toml')

    # Transformiere config für CaloScore
    config['EMBED'] = config['model']['embed_dim']
    # Filter active regions
    active_regions = config['training'].get('active_regions', {}).get('enabled', ['PIT', 'BOT', 'WALL', 'TOP'])
    config['LAYER_NAMES'] = active_regions
    config['ALL_REGIONS'] = ['PIT', 'BOT', 'WALL', 'TOP']  # Für data_loader
    config['SHAPE_PIT'] = config['model']['SHAPE_PIT']
    config['SHAPE_BOT'] = config['model']['SHAPE_BOT']
    config['SHAPE_WALL'] = config['model']['SHAPE_WALL']
    config['SHAPE_TOP'] = config['model']['SHAPE_TOP']
    config['num_steps'] = config['diffusion']['num_steps']
    config['ema_decay'] = config['training']['ema_decay']

    # Padding pro Region
    config['PAD'] = {}
    for region in config['LAYER_NAMES']:
        pad_value = config['model']['padding'].get(region, 0)
        config['PAD'][region] = pad_value

    # Area ratios (für loss weighting)
    config['AREA_RATIOS'] = {
        region: config['normalization']['area_ratios'][region]
        for region in config['LAYER_NAMES']
    }
    
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
    
    # Create datasets (pass full config)
    train_dataset = voxelDataset(str(train_file), config)
    val_dataset = voxelDataset(str(val_file), config)
    
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
    data_size = train_dataset.n_events
    num_cond = train_dataset.phi_dim
    num_area = 4
    
    print(f"\nData size: {data_size}")
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
            decay_steps=NUM_EPOCHS * int(data_size * train_frac / BATCH_SIZE)
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
    
    # === Checkpoint (master only) ===
    if rank == 0:
        checkpoint = ModelCheckpoint(
            str(checkpoint_folder / 'checkpoint.weights.h5'),
            save_best_only=True,
            mode='auto',
            save_weights_only=True
        )
        callbacks.append(checkpoint)
    
    # === Training ===
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    history = model.fit(
        train_data_clean.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS,
        steps_per_epoch=int(data_size * train_frac / BATCH_SIZE),
        validation_data=val_data_clean.batch(BATCH_SIZE),
        validation_steps=int(val_dataset.n_events / BATCH_SIZE),
        verbose=1 if rank == 0 else 0,
        callbacks=callbacks
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)