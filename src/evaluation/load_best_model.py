#!/usr/bin/env python3
"""
load_best_model.py

Utility zum Laden des besten trainierten Modells für Verifikation/Inferenz

Usage:
    from load_best_model import load_best_model
    
    model, info = load_best_model()
    # oder
    model, info = load_best_model(checkpoint_dir='./custom_checkpoints')
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
from diffusion_model import build_diffusion_model


def load_best_model(checkpoint_dir: str = "./checkpoints_cpu") -> Tuple[tf.keras.Model, Dict]:
    """
    Lade das beste trainierte Modell
    
    Sucht in dieser Reihenfolge:
    1. best_model.weights.h5 (von Training mit Validation)
    2. Letztes Checkpoint (highest epoch number)
    
    Args:
        checkpoint_dir: Verzeichnis mit Checkpoints
        
    Returns:
        model: Geladenes TensorFlow Model
        info: Dict mit Checkpoint-Informationen
        
    Raises:
        FileNotFoundError: Wenn keine Checkpoints gefunden
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory nicht gefunden: {checkpoint_dir}")
    
    # 1. Versuche best_model zu laden
    best_model_path = checkpoint_dir / "best_model.weights.h5"
    best_info_path = checkpoint_dir / "best_model_info.json"
    
    if best_model_path.exists() and best_info_path.exists():
        print("✓ Lade Best Model...")
        
        with open(best_info_path, 'r') as f:
            info = json.load(f)
        
        model = build_diffusion_model(
            phi_dim=info['config']['phi_dim'],
            target_dim=info['config']['target_dim'],
            T=info['config']['T'],
            hidden_dim=info['config']['hidden_dim'],
            n_layers=info['config']['n_layers'],
            t_emb_dim=info['config']['t_emb_dim']
        )
        
        model.load_weights(str(best_model_path))
        
        print(f"  Epoch: {info['epoch']}")
        print(f"  Loss: {info['loss']:.6f}")
        print(f"  Trained: {info['timestamp']}")
        print(f"  Parameters: {model.count_params():,}")
        
        return model, info
    
    # 2. Fallback: Lade letztes Checkpoint
    print("⚠ Best model nicht gefunden, lade letztes Checkpoint...")
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*_model.weights.h5"))
    
    if not checkpoint_files:
        raise FileNotFoundError(
            f"Keine Checkpoints gefunden in {checkpoint_dir}\n"
            "Bitte erst Training ausführen: python train_diffusion_cpu.py"
        )
    
    # Sortiere nach Epoch-Nummer
    def extract_epoch(path):
        # checkpoint_epoch_5_model.weights.h5 -> 5
        name = path.stem  # checkpoint_epoch_5_model.weights
        parts = name.split('_')
        return int(parts[2])  # epoch number
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    epoch_num = extract_epoch(latest_checkpoint)
    
    info_path = latest_checkpoint.parent / f"checkpoint_epoch_{epoch_num}_info.json"
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    model = build_diffusion_model(
        phi_dim=info['config']['phi_dim'],
        target_dim=info['config']['target_dim'],
        T=info['config']['T'],
        hidden_dim=info['config']['hidden_dim'],
        n_layers=info['config']['n_layers'],
        t_emb_dim=info['config']['t_emb_dim']
    )
    
    model.load_weights(str(latest_checkpoint))
    
    print(f"✓ Geladen: {latest_checkpoint.name}")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Loss: {info['loss']:.6f}")
    print(f"  Trained: {info['timestamp']}")
    print(f"  Parameters: {model.count_params():,}")
    
    return model, info


def list_available_checkpoints(checkpoint_dir: str = "./checkpoints_cpu"):
    """
    Liste alle verfügbaren Checkpoints
    
    Args:
        checkpoint_dir: Verzeichnis mit Checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"⚠ Directory nicht gefunden: {checkpoint_dir}")
        return
    
    print("=" * 70)
    print("VERFÜGBARE CHECKPOINTS")
    print("=" * 70)
    
    # Best model
    best_model_path = checkpoint_dir / "best_model.weights.h5"
    if best_model_path.exists():
        info_path = checkpoint_dir / "best_model_info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"\n★ BEST MODEL:")
        print(f"  Path: {best_model_path}")
        print(f"  Epoch: {info['epoch']}")
        print(f"  Loss: {info['loss']:.6f}")
        print(f"  Date: {info['timestamp']}")
    else:
        print(f"\n⚠ Best model nicht gefunden")
    
    # Regular checkpoints
    checkpoint_files = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*_info.json"),
        key=lambda p: int(p.stem.split('_')[2])
    )
    
    if checkpoint_files:
        print(f"\nREGULAR CHECKPOINTS ({len(checkpoint_files)}):")
        for info_path in checkpoint_files:
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            epoch = info['epoch']
            loss = info['loss']
            timestamp = info['timestamp'].split('T')[0]  # Nur Datum
            
            print(f"  Epoch {epoch:2d}: Loss={loss:.6f}  ({timestamp})")
    else:
        print("\n⚠ Keine regulären Checkpoints gefunden")
    
    print("=" * 70)


def compare_checkpoints(checkpoint_dir: str = "./checkpoints_cpu"):
    """
    Vergleiche alle Checkpoints und empfehle bestes
    
    Args:
        checkpoint_dir: Verzeichnis mit Checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"⚠ Directory nicht gefunden: {checkpoint_dir}")
        return
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*_info.json"))
    
    if not checkpoint_files:
        print("⚠ Keine Checkpoints gefunden")
        return
    
    print("=" * 70)
    print("CHECKPOINT COMPARISON")
    print("=" * 70)
    
    checkpoints_data = []
    
    for info_path in checkpoint_files:
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        checkpoints_data.append({
            'epoch': info['epoch'],
            'loss': info['loss'],
            'path': str(info_path.with_suffix('.weights.h5').name.replace('_info', '_model'))
        })
    
    # Sortiere nach Loss
    checkpoints_data.sort(key=lambda x: x['loss'])
    
    print(f"\n{'Rank':<6} {'Epoch':<8} {'Loss':<12} {'File':<40}")
    print("-" * 70)
    
    for i, cp in enumerate(checkpoints_data[:10], 1):  # Top 10
        marker = "★" if i == 1 else " "
        print(f"{marker} {i:<4} {cp['epoch']:<8} {cp['loss']:<12.6f} {cp['path']:<40}")
    
    best = checkpoints_data[0]
    print("\n" + "=" * 70)
    print(f"EMPFEHLUNG: Nutze Epoch {best['epoch']} (Loss: {best['loss']:.6f})")
    print("=" * 70)


def main():
    """CLI Interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verwalte trainierte Diffusion Model Checkpoints'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Liste alle verfügbaren Checkpoints'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Vergleiche Checkpoints und empfehle bestes'
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Lade und teste best model'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='./checkpoints_cpu',
        help='Checkpoint directory (default: ./checkpoints_cpu)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_checkpoints(args.checkpoint_dir)
    elif args.compare:
        compare_checkpoints(args.checkpoint_dir)
    elif args.load:
        try:
            model, info = load_best_model(args.checkpoint_dir)
            print("\n✓ Model erfolgreich geladen und bereit für Inferenz")
            print(f"  Model input shape: {model.input_shape}")
            print(f"  Model output shape: {model.output_shape}")
        except Exception as e:
            print(f"\n✗ Fehler beim Laden: {e}")
    else:
        # Default: List
        list_available_checkpoints(args.checkpoint_dir)


if __name__ == "__main__":
    main()