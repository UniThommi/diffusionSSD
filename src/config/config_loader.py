"""
Config Loader for Diffusion Training
Loads flat TOML configuration from project root.
For Python ≥3.11, uses stdlib tomllib.
"""

import tomllib
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DiffusionSchedule:
    """Computed diffusion schedule from config parameters"""
    T: int
    betas: tf.Tensor
    alphas: tf.Tensor
    alphas_cumprod: tf.Tensor
    
    def to_dict(self) -> Dict:
        return {
            'T': self.T,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod
        }


class ConfigLoader:
    """Load flat TOML configuration from project root"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Auto-detect in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.toml"
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create config.toml in project root."
            )
        
        with open(self.config_path, 'rb') as f:
            self.config = tomllib.load(f)
        
        self._validate_config()
        self.diffusion_schedule = self._create_diffusion_schedule()
    
    def _validate_config(self):
        """
        Strikte Validierung - ALLE benötigten Keys müssen vorhanden sein
        Keine Defaults, keine Fallbacks
        """
        required_keys = [
            # Data
            'data_path', 'phi_dim', 'target_dim',
            # Diffusion
            'diffusion_T', 'diffusion_beta_start', 'diffusion_beta_end', 'diffusion_schedule_type',
            # Model
            'model_hidden_dim', 'model_n_layers', 'model_t_emb_dim',
            # Training
            'training_batch_size', 'training_learning_rate',
            'training_epochs', 'training_steps_per_epoch',
            'training_gradient_clip_norm', 'training_patience',
            # Validation
            'validation_split_ratio', 'validation_n_batches', 'validation_batch_size_multiplier',
            # Paths
            'checkpoint_dir', 'log_dir', 'save_every_n_epochs',
            # Experiment
            'experiment_name'
        ]
        
        missing = [key for key in required_keys if key not in self.config]
        if missing:
            raise ValueError(
                f"Missing required config keys: {missing}\n"
                f"Config file: {self.config_path}\n"
                f"All keys must be explicitly set, no defaults allowed!"
            )
        
        if self.config['diffusion_beta_start'] >= self.config['diffusion_beta_end']:
            raise ValueError("diffusion_beta_start must be < diffusion_beta_end")
    
    def _create_diffusion_schedule(self) -> DiffusionSchedule:
        """
        Create diffusion noise schedule
        
        Reference: 
        - Ho et al. (2020) - DDPM: Linear schedule
        - Nichol & Dhariwal (2021) - Improved DDPM: Cosine schedule (RECOMMENDED)
        
        Cosine schedule empirically shown to produce better sample quality
        """
        T = self.config['diffusion_T']
        beta_start = self.config['diffusion_beta_start']
        beta_end = self.config['diffusion_beta_end']
        schedule_type = self.config['diffusion_schedule_type']  # NO DEFAULT - must be explicit
        
        if schedule_type == 'linear':
            betas = tf.linspace(beta_start, beta_end, T)
        elif schedule_type == 'cosine':
            steps = tf.range(T, dtype=tf.float32)
            alpha_bar = tf.cos(((steps / T) + 0.008) / 1.008 * (np.pi / 2)) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = tf.concat([[beta_start], betas], axis=0)
            betas = tf.clip_by_value(betas, 0, 0.999)
        elif schedule_type == 'quadratic':
            betas = tf.linspace(beta_start**0.5, beta_end**0.5, T) ** 2
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
        
        alphas = 1.0 - betas
        alphas_cumprod = tf.math.cumprod(alphas)
        
        return DiffusionSchedule(T=T, betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            'hidden_dim': self.config['model_hidden_dim'],
            'n_layers': self.config['model_n_layers'],
            't_emb_dim': self.config['model_t_emb_dim']
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training parameters - NO DEFAULTS, all must be in config"""
        return {
            'batch_size': self.config['training_batch_size'],
            'learning_rate': self.config['training_learning_rate'],
            'epochs': self.config['training_epochs'],
            'steps_per_epoch': self.config['training_steps_per_epoch'],
            'gradient_clip_norm': self.config['training_gradient_clip_norm'],
            'patience': self.config['training_patience'],
            'use_lr_schedule': self.config['training_use_lr_schedule'],
            'use_ema': self.config['training_use_ema'],
            'ema_decay': self.config['training_ema_decay']
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation parameters - NO DEFAULTS"""
        return {
            'split_ratio': self.config['validation_split_ratio'],
            'n_batches': self.config['validation_n_batches'],
            'batch_size_multiplier': self.config['validation_batch_size_multiplier']
        }
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration - NO DEFAULTS"""
        return {
            'checkpoint_dir': self.config['checkpoint_dir'],
            'log_dir': self.config['log_dir'],
            'save_every_n_epochs': self.config['save_every_n_epochs'],
            'experiment_name': self.config['experiment_name']
        }