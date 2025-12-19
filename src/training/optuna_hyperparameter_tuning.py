#!/usr/bin/env python3
"""
optuna_hyperparameter_tuning.py

Parallele Hyperparameter-Optimierung mit Optuna

Basiert auf:
- Optuna Framework (Akiba et al. 2019)
- TPE Sampler (Bergstra et al. 2011)
- Median Pruner für Early Stopping

Usage:
    # Single-Node (4 parallel trials)
    python optuna_hyperparameter_tuning.py --n-trials 50 --n-jobs 4
    
    # Dashboard (in separatem Terminal)
    optuna-dashboard sqlite:///optuna_study.db
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
from training.training_optimization_system import OptimizedTrainer


def objective(trial: optuna.Trial) -> float:
    """
    Objective function für Optuna
    
    Definiert Search Space und führt Training aus
    
    Args:
        trial: Optuna Trial object
        
    Returns:
        best_val_loss: Zu minimierender Wert
    """
    # Define search space
    config = {
        # Data (fixed)
        'data_path': '/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/outdated/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5',
        'phi_dim': 22,
        'target_dim': 7789,
        'T': 1000,
        
        # Model Architecture (TUNABLE)
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 768]),
        'n_layers': trial.suggest_int('n_layers', 4, 8),
        't_emb_dim': trial.suggest_categorical('t_emb_dim', [32, 64, 128]),
        
        # Training Parameters (TUNABLE)
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        
        # Fixed training params (für Geschwindigkeit)
        'epochs': 10,  # Weniger für schnelles Tuning
        'steps_per_epoch': 100,  # Weniger Steps
        'gradient_clip_norm': 1.0,
        'patience': 3,  # Früh stoppen
        
        # Paths
        'checkpoint_dir': f'./optuna_checkpoints/trial_{trial.number}',
        'log_dir': './optuna_logs'
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(exist_ok=True)
    
    try:
        # Train model
        trainer = OptimizedTrainer(
            config, 
            experiment_name=f"trial_{trial.number}"
        )
        
        # Überprüfe Pruning nach jeder Epoch
        # (Optuna kann unpromising trials früh beenden)
        # Dies müsste in OptimizedTrainer integriert werden
        
        model = trainer.train()
        
        # Return best validation loss
        best_val_loss = trainer.best_val_loss
        
        # Log trial info
        trial.set_user_attr("training_time", sum(
            m.training_time_seconds for m in trainer.logger.metrics_history
        ))
        trial.set_user_attr("final_train_loss", trainer.logger.metrics_history[-1].train_loss)
        
        return best_val_loss
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')  # Worst possible value


def run_optuna_study(
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = "diffusion_hyperopt",
    storage: str = "sqlite:///optuna_study.db"
):
    """
    Führe Optuna Hyperparameter-Suche aus
    
    Args:
        n_trials: Anzahl Trials insgesamt
        n_jobs: Anzahl parallele Jobs (CPU cores)
        study_name: Name der Study
        storage: Database für Persistence
    """
    print("="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Study: {study_name}")
    print(f"Trials: {n_trials}")
    print(f"Parallel Jobs: {n_jobs}")
    print(f"Storage: {storage}")
    print("="*70)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,  # Resume wenn vorhanden
        direction="minimize",  # Minimize validation loss
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(
            n_startup_trials=5,  # Warte 5 trials bevor pruning
            n_warmup_steps=3     # Warte 3 epochs bevor pruning
        )
    )
    
    # Run optimization
    print("\nStarting optimization...")
    print("Dashboard: optuna-dashboard " + storage)
    print()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,  # Parallele Trials
        show_progress_bar=True
    )
    
    # Best trial
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    best_trial = study.best_trial
    
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"  Best Val Loss: {best_trial.value:.6f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print(f"\nTraining Time: {best_trial.user_attrs.get('training_time', 'N/A')} seconds")
    print(f"Final Train Loss: {best_trial.user_attrs.get('final_train_loss', 'N/A'):.6f}")
    
    # Save results
    results_dir = Path("./optuna_results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'study_name': study_name,
        'timestamp': datetime.now().isoformat(),
        'n_trials': len(study.trials),
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params,
            'user_attrs': best_trial.user_attrs
        },
        'all_trials': [
            {
                'number': t.number,
                'value': t.value if t.value is not None else float('inf'),
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }
    
    results_file = results_dir / f"{study_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_file}")
    
    # Visualizations
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(results_dir / f"{study_name}_history.png", dpi=300)
        plt.close()
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(results_dir / f"{study_name}_importance.png", dpi=300)
        plt.close()
        
        print(f"✓ Visualizations saved: {results_dir}")
    except Exception as e:
        print(f"⚠ Could not create visualizations: {e}")
    
    print("="*70)
    
    return study


def analyze_optuna_results(storage: str = "sqlite:///optuna_study.db"):
    """
    Analysiere Optuna Results für Claude-Feedback
    
    Erstellt detaillierte Zusammenfassung für Hyperparameter-Empfehlungen
    """
    # Load study
    study = optuna.load_study(
        study_name="diffusion_hyperopt",
        storage=storage
    )
    
    # Analyze
    print("="*70)
    print("OPTUNA ANALYSIS FOR CLAUDE")
    print("="*70)
    
    # 1. Best Configurations
    print("\n[1] TOP-5 CONFIGURATIONS:")
    trials = sorted([t for t in study.trials if t.value is not None], 
                   key=lambda t: t.value)[:5]
    
    for i, trial in enumerate(trials, 1):
        print(f"\n  Rank {i}: Trial #{trial.number}")
        print(f"    Val Loss: {trial.value:.6f}")
        print(f"    Params:")
        for k, v in trial.params.items():
            print(f"      {k}: {v}")
    
    # 2. Hyperparameter Importance
    print("\n[2] HYPERPARAMETER IMPORTANCE:")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in sorted(importances.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"    {param}: {importance:.3f}")
    except:
        print("    (Not enough trials for importance calculation)")
    
    # 3. Recommendations
    print("\n[3] RECOMMENDATIONS:")
    best = study.best_trial
    
    print(f"\n  Optimal Configuration:")
    print(f"    hidden_dim: {best.params['hidden_dim']}")
    print(f"    n_layers: {best.params['n_layers']}")
    print(f"    t_emb_dim: {best.params['t_emb_dim']}")
    print(f"    batch_size: {best.params['batch_size']}")
    print(f"    learning_rate: {best.params['learning_rate']:.2e}")
    
    print(f"\n  Expected Performance:")
    print(f"    Val Loss: {best.value:.6f}")
    
    # 4. Failed trials analysis
    failed_trials = [t for t in study.trials if t.value is None or t.value == float('inf')]
    if failed_trials:
        print(f"\n  ⚠ Warning: {len(failed_trials)} trials failed")
        print(f"    Common failure patterns:")
        # Analyze common params in failed trials
        failed_params = {}
        for t in failed_trials:
            for k, v in t.params.items():
                if k not in failed_params:
                    failed_params[k] = []
                failed_params[k].append(v)
        
        for k, values in failed_params.items():
            if len(set(values)) < len(values):
                # Repeated value in failures
                from collections import Counter
                counter = Counter(values)
                most_common = counter.most_common(1)[0]
                print(f"      {k}={most_common[0]} appears in {most_common[1]} failures")
    
    print("\n" + "="*70)


def main():
    """CLI Interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization mit Optuna'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Anzahl Optimization Trials (default: 20)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Anzahl parallele Jobs (default: 1, empfohlen: 4)'
    )
    parser.add_argument(
        '--study-name',
        default='diffusion_hyperopt',
        help='Study name'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Nur Analyse existierender Results'
    )
    
    args = parser.parse_args()
    
    storage = "sqlite:///optuna_study.db"
    
    if args.analyze:
        analyze_optuna_results(storage)
    else:
        study = run_optuna_study(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            study_name=args.study_name,
            storage=storage
        )
        
        # Auto-analyze
        print("\n")
        analyze_optuna_results(storage)


if __name__ == "__main__":
    main()