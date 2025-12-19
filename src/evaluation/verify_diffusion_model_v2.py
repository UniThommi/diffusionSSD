import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from evaluation.load_best_model import load_best_model
from models.diffusion_model import build_diffusion_model


@dataclass
class VerificationConfig:
    """Configuration for physics-driven verification"""
    checkpoint_dir: str = "./checkpoints_cpu"
    data_path: str = "/path/to/resum_output_0.hdf5"
    n_test_events: int = 500
    n_sampling_steps: int = 50
    output_dir: str = "./verification_results"


class PhysicsVerificationPipeline:
    """
    Complete physics verification pipeline
    
    Verification Hierarchy:
    1. Reconstruction Fidelity (ML metric)
    2. Physics Constraints (domain metric)
    3. Spatial Locality (detector metric)
    4. Conditional Consistency (generalization metric)
    """
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        Path(config.output_dir).mkdir(exist_ok=True)
        
        # Load best validated model
        print("="*70)
        print("PHYSICS-DRIVEN DIFFUSION MODEL VERIFICATION")
        print("="*70)
        
        self.model, self.model_info = load_best_model(config.checkpoint_dir)
        print(f"\n✓ Loaded best model:")
        print(f"  Epoch: {self.model_info['epoch']}")
        print(f"  Validation Loss: {self.model_info['loss']:.6f}")
        print(f"  Parameters: {self.model.count_params():,}")
        
        # Load test data (held-out)
        print(f"\n✓ Loading test data from: {config.data_path}")
        self.test_data = self._load_test_data()
        print(f"  Test events: {len(self.test_data)}")
        
        # Load geometry
        self.voxel_positions = self._load_voxel_positions()
        print(f"  Voxel positions: {self.voxel_positions.shape}")
        
        # Initialize DDIM sampler
        from verify_diffusion_model import DDIMSampler
        self.sampler = DDIMSampler(self.model, T=1000)
        
        # Generate predictions for all test events
        print(f"\n✓ Generating predictions (DDIM, {config.n_sampling_steps} steps)...")
        self.predictions = self._generate_predictions()
        print(f"  Predictions shape: {self.predictions.shape}")
        
    def run_full_verification(self):
        """Execute complete verification pipeline"""
        
        results = {}
        
        # Phase 1: Reconstruction Fidelity
        print("\n" + "="*70)
        print("[1/4] RECONSTRUCTION FIDELITY")
        print("="*70)
        results['fidelity'] = self._verify_reconstruction_fidelity()
        
        # Phase 2: Physics Constraints (Material-aware energy)
        print("\n" + "="*70)
        print("[2/4] PHYSICS CONSTRAINTS (Material-Aware Energy)")
        print("="*70)
        results['physics'] = self._verify_physics_constraints()
        
        # Phase 3: Spatial Locality
        print("\n" + "="*70)
        print("[3/4] SPATIAL LOCALITY")
        print("="*70)
        results['spatial'] = self._verify_spatial_locality()
        
        # Phase 4: Summary
        print("\n" + "="*70)
        print("[4/4] SUMMARY REPORT")
        print("="*70)
        self._generate_summary_report(results)
        
        return results
    
    def _verify_reconstruction_fidelity(self):
        """
        Verify model can reconstruct voxel activation patterns
        
        Metrics:
        - MSE, R², Correlation
        - Residual analysis
        - Per-voxel error profile
        """
        
        # Extract ground truth
        x_true = np.array([event['signal'] for event in self.test_data])
        x_pred = self.predictions
        
        # Flatten for analysis
        x_true_flat = x_true.flatten()
        x_pred_flat = x_pred.flatten()
        
        # Compute metrics
        mse = np.mean((x_true_flat - x_pred_flat)**2)
        
        # R² score
        ss_res = np.sum((x_true_flat - x_pred_flat)**2)
        ss_tot = np.sum((x_true_flat - np.mean(x_true_flat))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Correlation
        corr = np.corrcoef(x_true_flat, x_pred_flat)[0, 1]
        
        results = {
            'mse': float(mse),
            'r2': float(r2),
            'correlation': float(corr)
        }
        
        print(f"  MSE: {mse:.6f}")
        print(f"  R²: {r2:.3f}")
        print(f"  Correlation: {corr:.3f}")
        
        # Visualize (using code from section 2)
        self._plot_reconstruction_quality(x_true_flat, x_pred_flat)
        
        return results
    
    def _verify_physics_constraints(self):
        """
        Material-aware energy-photon correlation
        
        Uses code from section 3
        """
        
        # Extract physics quantities
        energies = np.array([event['phi'][1] for event in self.test_data])  # E_gamma_tot_keV
        materials = self._identify_capture_materials()
        
        # Compute photon yields
        photon_yields = np.sum(self.predictions, axis=1)
        
        # Material-dependent analysis
        results = {}
        
        unique_materials = np.unique(materials)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, material in enumerate(unique_materials[:4]):  # Max 4 materials
            ax = axes[idx]
            
            # Filter by material
            mask = materials == material
            E_mat = energies[mask]
            N_mat = photon_yields[mask]
            
            # Scatter
            ax.scatter(E_mat, N_mat, alpha=0.4, s=20, label=material)
            
            # Linear fit (RANSAC for robustness)
            from sklearn.linear_model import RANSACRegressor
            
            if len(E_mat) > 10:
                ransac = RANSACRegressor()
                ransac.fit(E_mat.reshape(-1, 1), N_mat)
                
                E_range = np.linspace(E_mat.min(), E_mat.max(), 100)
                N_fit = ransac.predict(E_range.reshape(-1, 1))
                
                ax.plot(E_range, N_fit, 'r-', linewidth=2.5,
                       label=f'Fit: {ransac.estimator_.coef_[0]:.2f} photons/keV')
                
                results[material] = {
                    'efficiency': float(ransac.estimator_.coef_[0]),
                    'r2': float(ransac.score(E_mat.reshape(-1, 1), N_mat))
                }
            
            ax.set_xlabel('Capture Energy [keV]', fontweight='bold')
            ax.set_ylabel('Detected Photons', fontweight='bold')
            ax.set_title(f'{material} (N={np.sum(mask)})', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Material-Dependent Energy-Photon Correlation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / 'energy_conservation_by_material.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
        
        return results
    
    def _verify_spatial_locality(self):
        """
        Spatial locality analysis
        
        Uses code from section 4
        """
        
        # NC positions
        nc_positions = np.array([event['phi'][19:22] for event in self.test_data])
        
        # Center-of-light
        col_positions = self._compute_center_of_light(self.predictions)
        
        # Position errors
        position_errors = np.linalg.norm(col_positions - nc_positions, axis=1)
        
        # Spatial spreads
        spatial_spreads = self._compute_spatial_spread(self.predictions, nc_positions)
        
        results = {
            'mean_position_error': float(np.mean(position_errors)),
            'median_position_error': float(np.median(position_errors)),
            'mean_spatial_spread': float(np.mean(spatial_spreads))
        }
        
        print(f"  Mean Position Error: {results['mean_position_error']:.1f} mm")
        print(f"  Median Position Error: {results['median_position_error']:.1f} mm")
        print(f"  Mean Spatial Spread: {results['mean_spatial_spread']:.1f} mm")
        
        # Visualize (using code from section 4)
        self._plot_spatial_locality(position_errors, spatial_spreads, 
                                    nc_positions, col_positions)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate comprehensive summary report with plots"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Add summary plots here...
        
        plt.suptitle('Verification Summary Report',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / 'verification_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
    
    # Helper methods (load data, compute COL, etc.)
    # ... (implementation details)


def main():
    config = VerificationConfig(
        checkpoint_dir="./checkpoints_cpu",
        data_path="/path/to/resum_output_0.hdf5",
        n_test_events=500,
        output_dir="./verification_results_v2"
    )
    
    pipeline = PhysicsVerificationPipeline(config)
    results = pipeline.run_full_verification()
    
    print("\n" + "="*70)
    print("✓ VERIFICATION COMPLETE")
    print("="*70)
    print(f"Results saved in: {config.output_dir}")


if __name__ == "__main__":
    main()