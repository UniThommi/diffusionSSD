#!/usr/bin/env python3
"""
correlation_analysis_physics_informed.py

Systematische Korrelationsanalyse für Physics-Informed Diffusion Loss

Ziel: Identifiziere welche NC-Parameter (φ) mit optischer Photonenzahl korrelieren
      → Informiert Conditional Diffusion Model Design & Loss Function

Referenzen:
- Kraskov et al. (2004): Mutual Information Estimation, PRE 69, 066138
- Reshef et al. (2011): MIC - Detecting Novel Associations, Science 334, 1518
- Lundberg & Lee (2017): SHAP - Feature Importance, NeurIPS
- Pearl (2009): Causality - Partial Correlations

Autor: Physics-ML Correlation Framework
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Matplotlib Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PhysicsCorrelationAnalyzer:
    """
    Comprehensive correlation analysis for neutron capture optical detection
    
    Analysis Hierarchy:
    1. Univariate Correlations (1D)
    2. Material-Stratified Analysis
    3. Geometric Feature Engineering
    4. Multivariate Interactions (2D, 3D)
    5. Feature Importance Ranking
    """
    
    def __init__(self, data_path: str, n_sample: int = 50000, random_seed: int = 42):
        """
        Args:
            data_path: Path to HDF5 file
            n_sample: Sample size for analysis (trade-off: speed vs. statistics)
            random_seed: For reproducibility
        """
        self.data_path = data_path
        self.n_sample = n_sample
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load data
        print("="*70)
        print("PHYSICS-INFORMED CORRELATION ANALYSIS")
        print("="*70)
        print(f"Data: {data_path}")
        print(f"Sample size: {n_sample:,}")
        
        self.data = self._load_data_chunked()
        self.feature_names = self._define_feature_names()
        
        print(f"✓ Loaded {len(self.data['target'])} events")
        print(f"✓ Features: {len(self.feature_names)}")
        
    def _load_data_chunked(self) -> Dict[str, np.ndarray]:
        """
        Alternative: Chunked loading (robuster für große Files)
        
        Statt fancy indexing: Lade alles sequentiell, dann sample
        Trade-off: Mehr Memory, aber keine Indexing-Probleme
        """
        
        with h5py.File(self.data_path, 'r') as f:
            n_total = f['phi']['#gamma'].shape[0]
            
            # Load all data (or large chunk)
            if n_total > 1000000:
                # Wenn > 1M Events: Lade nur ersten 1M
                load_size = min(self.n_sample * 2, 200000)  # ← Max 200k statt 1M!
                print(f"Large dataset, loading {load_size:,} events")
            else:
                load_size = n_total
            
            # Load phi
            phi_keys = [
                '#gamma', 'E_gamma_tot_keV',
                'gammaE1_keV', 'gammaE2_keV', 'gammaE3_keV', 'gammaE4_keV',
                'gammapx1', 'gammapx2', 'gammapx3', 'gammapx4',
                'gammapy1', 'gammapy2', 'gammapy3', 'gammapy4',
                'gammapz1', 'gammapz2', 'gammapz3', 'gammapz4',
                'matID', 'xNC_mm', 'yNC_mm', 'zNC_mm'
            ]
            
            phi_all = np.array([
                f['phi'][key][:load_size] for key in phi_keys
            ], dtype=np.float32).T
            
            # Load target
            voxel_keys = sorted(f['target'].keys())
            signals_all = np.array([
                f['target'][key][:load_size] for key in voxel_keys
            ], dtype=np.float32).T
            
            # NOW sample (in memory - keine HDF5 fancy indexing issues)
            if self.n_sample < load_size:
                sample_idx = np.random.choice(load_size, self.n_sample, replace=False)
                phi = phi_all[sample_idx]
                signals = signals_all[sample_idx]
            else:
                phi = phi_all
                signals = signals_all
            
            # Target: Total detected photons
            total_photons = np.sum(signals, axis=1)
            
            # Active voxels
            threshold = 0.5
            active_voxels = np.sum(signals > threshold, axis=1)
        
        return {
            'phi': phi,
            'signals': signals,
            'target': total_photons,
            'multiplicity': active_voxels
        }
        """Load and preprocess data"""
        
        with h5py.File(self.data_path, 'r') as f:
            # Total events
            n_total = f['phi']['#gamma'].shape[0]
            
            # Random sampling (for computational efficiency)
            if self.n_sample < n_total:
                indices = np.random.choice(n_total, self.n_sample, replace=False)
                # CRITICAL: HDF5 requires sorted indices for fancy indexing!
                indices = np.sort(indices)
            else:
                indices = np.arange(n_total)
            
            # Load NC parameters (phi)
            phi_keys = [
                '#gamma', 'E_gamma_tot_keV',
                'gammaE1_keV', 'gammaE2_keV', 'gammaE3_keV', 'gammaE4_keV',
                'gammapx1', 'gammapx2', 'gammapx3', 'gammapx4',
                'gammapy1', 'gammapy2', 'gammapy3', 'gammapy4',
                'gammapz1', 'gammapz2', 'gammapz3', 'gammapz4',
                'matID', 'xNC_mm', 'yNC_mm', 'zNC_mm'
            ]
            
            phi = np.array([
                f['phi'][key][indices] for key in phi_keys
            ], dtype=np.float32).T  # (n_sample, 22)
            
            # Load optical signals (target)
            voxel_keys = sorted(f['target'].keys())
            
            signals = np.array([
                f['target'][key][indices] for key in voxel_keys
            ], dtype=np.float32).T  # (n_sample, n_voxels)
            
            # Target: Total detected photons (sum over all voxels)
            total_photons = np.sum(signals, axis=1)  # (n_sample,)
            
            # Active voxels (multiplicity)
            threshold = 0.5
            active_voxels = np.sum(signals > threshold, axis=1)  # (n_sample,)
        
        return {
            'phi': phi,
            'signals': signals,
            'target': total_photons,
            'multiplicity': active_voxels
        }
    
    def _define_feature_names(self) -> List[str]:
        """Define human-readable feature names"""
        return [
            'n_gamma', 'E_tot',
            'E_gamma1', 'E_gamma2', 'E_gamma3', 'E_gamma4',
            'px1', 'px2', 'px3', 'px4',
            'py1', 'py2', 'py3', 'py4',
            'pz1', 'pz2', 'pz3', 'pz4',
            'matID', 'x_NC', 'y_NC', 'z_NC'
        ]
    
    def engineer_geometric_features(self) -> Dict[str, np.ndarray]:
        """
        Engineer physics-motivated geometric features
        
        Features:
        - Radial distance from detector center
        - Distance to detector surface (SSD inner radius ~500mm)
        - Solid angle coverage
        - Average gamma direction (unit vector)
        - Energy-weighted direction
        
        Returns:
            Dict of engineered features
        """
        
        phi = self.data['phi']
        
        # NC Position
        x_nc = phi[:, 19]  # xNC_mm
        y_nc = phi[:, 20]  # yNC_mm
        z_nc = phi[:, 21]  # zNC_mm
        
        # Radial distance from origin
        r_nc = np.sqrt(x_nc**2 + y_nc**2 + z_nc**2)
        
        # Cylindrical coordinates
        r_cyl = np.sqrt(x_nc**2 + y_nc**2)
        phi_cyl = np.arctan2(y_nc, x_nc)
        
        # Distance to detector (assuming cylindrical SSD at r=500mm)
        r_detector = 500.0  # mm (approximate, adjust if known)
        dist_to_detector = np.abs(r_cyl - r_detector)
        
        # Gamma directions (average)
        px_avg = np.mean(phi[:, 6:10], axis=1)  # px1-4
        py_avg = np.mean(phi[:, 10:14], axis=1)  # py1-4
        pz_avg = np.mean(phi[:, 14:18], axis=1)  # pz1-4
        
        p_mag = np.sqrt(px_avg**2 + py_avg**2 + pz_avg**2) + 1e-10
        
        # Unit direction vector
        px_unit = px_avg / p_mag
        py_unit = py_avg / p_mag
        pz_unit = pz_avg / p_mag
        
        # Radial component (towards detector)
        p_radial = (px_avg * x_nc + py_avg * y_nc) / (r_cyl + 1e-10)
        
        # Energy-weighted direction
        E_gammas = phi[:, 2:6]  # E1-4
        E_tot = phi[:, 1]  # E_tot
        
        # Weighted average direction
        px_weighted = np.sum(phi[:, 6:10] * E_gammas, axis=1) / (E_tot + 1e-10)
        py_weighted = np.sum(phi[:, 10:14] * E_gammas, axis=1) / (E_tot + 1e-10)
        pz_weighted = np.sum(phi[:, 14:18] * E_gammas, axis=1) / (E_tot + 1e-10)
        
        # Gamma opening angle (spread)
        # Compute pairwise angles between gamma directions
        gamma_dirs = np.stack([
            phi[:, 6:10],   # px1-4
            phi[:, 10:14],  # py1-4
            phi[:, 14:18]   # pz1-4
        ], axis=2)  # (n_sample, 4, 3)
        
        # Average opening angle (simplified)
        opening_angles = []
        for i in range(len(gamma_dirs)):
            dirs = gamma_dirs[i]  # (4, 3)
            # Normalize
            norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10
            dirs_unit = dirs / norms
            
            # Pairwise dot products
            dots = np.dot(dirs_unit, dirs_unit.T)
            # Average angle (arccos of off-diagonal)
            mask = ~np.eye(4, dtype=bool)
            avg_cos = np.mean(dots[mask])
            avg_angle = np.arccos(np.clip(avg_cos, -1, 1))
            opening_angles.append(avg_angle)
        
        opening_angles = np.array(opening_angles)
        
        # Energy concentration (Gini-like coefficient)
        E_sorted = np.sort(E_gammas, axis=1)[:, ::-1]  # Descending
        E_cumsum = np.cumsum(E_sorted, axis=1)
        E_concentration = E_cumsum[:, 0] / (E_tot + 1e-10)  # Fraction in highest-E gamma
        
        return {
            'r_nc': r_nc,
            'r_cyl': r_cyl,
            'phi_cyl': phi_cyl,
            'z_nc': z_nc,
            'dist_to_detector': dist_to_detector,
            'px_avg': px_avg,
            'py_avg': py_avg,
            'pz_avg': pz_avg,
            'p_radial': p_radial,
            'px_weighted': px_weighted,
            'py_weighted': py_weighted,
            'pz_weighted': pz_weighted,
            'opening_angle': opening_angles,
            'E_concentration': E_concentration
        }
    
    # ========================================================================
    # ANALYSIS 1: UNIVARIATE CORRELATIONS
    # ========================================================================
    
    def analyze_univariate_correlations(self, save_dir: str = "./correlation_plots"):
        """
        Compute and visualize 1D correlations: each φ_i vs target
        
        Metrics:
        - Pearson (linear)
        - Spearman (monotonic, robust)
        - Mutual Information (nonlinear)
        
        Visualization:
        - Scatter plots with regression
        - Binned response curves
        - Correlation matrix heatmap
        """
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("ANALYSIS 1: UNIVARIATE CORRELATIONS")
        print("="*70)
        
        phi = self.data['phi']
        target = self.data['target']
        
        n_features = phi.shape[1]
        
        # Compute correlations
        pearson_corr = np.array([
            stats.pearsonr(phi[:, i], target)[0] 
            for i in range(n_features)
        ])
        
        spearman_corr = np.array([
            stats.spearmanr(phi[:, i], target)[0]
            for i in range(n_features)
        ])
        
        # Mutual Information (sklearn)
        mi_scores = mutual_info_regression(
            phi, target, 
            discrete_features=[18],  # matID is discrete
            random_state=self.random_seed
        )
        
        # Normalize MI to [0, 1]
        mi_scores_norm = mi_scores / np.max(mi_scores)
        
        # Results DataFrame
        results = {
            'Feature': self.feature_names,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'MI': mi_scores,
            'MI_norm': mi_scores_norm
        }
        
        # Print top features
        print("\nTop 10 Features by Mutual Information:")
        sorted_idx = np.argsort(mi_scores)[::-1][:10]
        for idx in sorted_idx:
            print(f"  {self.feature_names[idx]:<15} "
                  f"Pearson={pearson_corr[idx]:+.3f}  "
                  f"Spearman={spearman_corr[idx]:+.3f}  "
                  f"MI={mi_scores[idx]:.4f}")
        
        # Visualization 1: Correlation Matrix Heatmap
        self._plot_correlation_heatmap(results, save_dir)
        
        # Visualization 2: Scatter plots for top features
        self._plot_top_univariate_scatters(phi, target, sorted_idx[:6], save_dir)
        
        # Visualization 3: Binned response curves
        self._plot_binned_response_curves(phi, target, sorted_idx[:6], save_dir)
        
        return results
    
    def _plot_correlation_heatmap(self, results: Dict, save_dir: str):
        """Plot correlation matrix heatmap"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Pearson', 'Spearman', 'MI_norm']
        titles = ['Pearson Correlation', 'Spearman Correlation', 
                  'Mutual Information (Normalized)']
        
        for ax, metric, title in zip(axes, metrics, titles):
            values = np.array(results[metric]).reshape(-1, 1)
            
            im = ax.imshow(values, cmap='RdBu_r', aspect='auto', 
                          vmin=-1 if 'Pearson' in metric or 'Spearman' in metric else 0,
                          vmax=1)
            
            ax.set_yticks(range(len(results['Feature'])))
            ax.set_yticklabels(results['Feature'], fontsize=9)
            ax.set_xticks([0])
            ax.set_xticklabels([metric], fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/01_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/01_correlation_heatmap.png")
        plt.close()
    
    def _plot_top_univariate_scatters(self, phi: np.ndarray, target: np.ndarray,
                                      top_indices: np.ndarray, save_dir: str):
        """Scatter plots for top correlated features"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            
            x = phi[:, feat_idx]
            y = target
            
            # Subsample for plotting (10k points max)
            if len(x) > 10000:
                sample_idx = np.random.choice(len(x), 10000, replace=False)
                x_plot = x[sample_idx]
                y_plot = y[sample_idx]
            else:
                x_plot = x
                y_plot = y
            
            # Scatter
            ax.scatter(x_plot, y_plot, alpha=0.3, s=5, c='navy')
            
            # Robust linear fit (RANSAC)
            from sklearn.linear_model import RANSACRegressor
            
            if len(np.unique(x)) > 1:
                try:
                    ransac = RANSACRegressor(random_state=self.random_seed)
                    ransac.fit(x.reshape(-1, 1), y)
                    
                    x_range = np.linspace(x.min(), x.max(), 100)
                    y_fit = ransac.predict(x_range.reshape(-1, 1))
                    
                    ax.plot(x_range, y_fit, 'r-', linewidth=2.5, 
                           label=f'Fit: {ransac.estimator_.coef_[0]:.2e}')
                except:
                    pass
            
            # Stats
            corr = stats.pearsonr(x, y)[0]
            
            ax.set_xlabel(self.feature_names[feat_idx], fontsize=11, fontweight='bold')
            ax.set_ylabel('Total Detected Photons', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.feature_names[feat_idx]}\nρ = {corr:+.3f}', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Top 6 Univariate Correlations (Scatter)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/02_top_univariate_scatters.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/02_top_univariate_scatters.png")
        plt.close()
    
    def _plot_binned_response_curves(self, phi: np.ndarray, target: np.ndarray,
                                     top_indices: np.ndarray, save_dir: str):
        """Binned response curves (nonlinear detection)"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            
            x = phi[:, feat_idx]
            y = target
            
            # Bin x into 20 bins
            n_bins = 20
            bins = np.linspace(np.percentile(x, 5), np.percentile(x, 95), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            bin_means = []
            bin_stds = []
            
            for i in range(n_bins):
                mask = (x >= bins[i]) & (x < bins[i+1])
                if np.sum(mask) > 10:
                    bin_means.append(np.mean(y[mask]))
                    bin_stds.append(np.std(y[mask]))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
            
            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)
            
            # Plot
            valid = ~np.isnan(bin_means)
            ax.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_stds[valid],
                       fmt='o-', linewidth=2.5, markersize=6, capsize=5,
                       color='steelblue', label='Binned Mean ± Std')
            
            ax.set_xlabel(self.feature_names[feat_idx], fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Detected Photons', fontsize=11, fontweight='bold')
            ax.set_title(f'{self.feature_names[feat_idx]}\n(Binned Response)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Top 6 Features: Binned Response Curves', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/03_binned_response_curves.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/03_binned_response_curves.png")
        plt.close()
    
    # ========================================================================
    # ANALYSIS 2: MATERIAL-STRATIFIED CORRELATIONS
    # ========================================================================
    
    def analyze_material_stratified(self, save_dir: str = "./correlation_plots"):
        """
        Material-dependent correlations
        
        Physics: Different materials → different optical yields
        - Water: ~30 photons/MeV
        - LAr: ~40,000 photons/MeV (scintillation)
        - Steel: negligible scintillation, only Cherenkov
        
        Analysis: Separate correlation analysis per material
        """
        
        print("\n" + "="*70)
        print("ANALYSIS 2: MATERIAL-STRATIFIED CORRELATIONS")
        print("="*70)
        
        phi = self.data['phi']
        target = self.data['target']
        
        matID = phi[:, 18].astype(int)
        unique_mats = np.unique(matID)
        
        print(f"\nMaterials detected: {unique_mats}")
        
        # Material names (from HDF5 structure)
        mat_names = {
            0: 'Water',
            1: 'LiquidArgon',
            2: 'Steel',
            3: 'Copper',
            -1: 'Unknown'
        }
        
        # Count per material
        print("\nEvent count per material:")
        for mat in unique_mats:
            count = np.sum(matID == mat)
            print(f"  {mat_names.get(mat, 'Unknown')}: {count:,} ({count/len(matID)*100:.1f}%)")
        
        # Analyze E_tot vs photons per material
        E_tot = phi[:, 1]  # E_gamma_tot_keV
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        material_results = {}
        
        for idx, mat in enumerate(unique_mats[:4]):  # Max 4 materials
            ax = axes[idx]
            
            mask = matID == mat
            E_mat = E_tot[mask]
            N_mat = target[mask]
            
            # Scatter (subsample)
            if np.sum(mask) > 5000:
                sample_idx = np.random.choice(np.sum(mask), 5000, replace=False)
                E_plot = E_mat[sample_idx]
                N_plot = N_mat[sample_idx]
            else:
                E_plot = E_mat
                N_plot = N_mat
            
            ax.scatter(E_plot, N_plot, alpha=0.3, s=10, 
                      label=f'{mat_names.get(mat, "Unknown")} (N={np.sum(mask):,})')
            
            # Linear fit
            if len(E_mat) > 10 and len(np.unique(E_mat)) > 1:
                from sklearn.linear_model import RANSACRegressor
                
                try:
                    ransac = RANSACRegressor(random_state=self.random_seed)
                    ransac.fit(E_mat.reshape(-1, 1), N_mat)
                    
                    E_range = np.linspace(E_mat.min(), E_mat.max(), 100)
                    N_fit = ransac.predict(E_range.reshape(-1, 1))
                    
                    ax.plot(E_range, N_fit, 'r-', linewidth=2.5,
                           label=f'Yield: {ransac.estimator_.coef_[0]:.2f} photons/keV')
                    
                    # R² score
                    r2 = ransac.score(E_mat.reshape(-1, 1), N_mat)
                    
                    material_results[mat] = {
                        'material': mat_names.get(mat, 'Unknown'),
                        'n_events': int(np.sum(mask)),
                        'optical_yield': float(ransac.estimator_.coef_[0]),
                        'r2_score': float(r2)
                    }
                except Exception as e:
                    print(f"  ⚠ Fit failed for material {mat}: {e}")
            
            ax.set_xlabel('E_gamma_tot [keV]', fontsize=11, fontweight='bold')
            ax.set_ylabel('Detected Photons', fontsize=11, fontweight='bold')
            ax.set_title(f'{mat_names.get(mat, "Unknown")}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Material-Dependent Energy-Photon Correlation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/04_material_stratified.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/04_material_stratified.png")
        plt.close()
        
        # Print results
        print("\nOptical Yields (photons/keV) by Material:")
        for mat, res in material_results.items():
            print(f"  {res['material']:<15} "
                  f"Yield={res['optical_yield']:.2f}  "
                  f"R²={res['r2_score']:.3f}  "
                  f"N={res['n_events']:,}")
        
        return material_results
    
    # ========================================================================
    # ANALYSIS 3: GEOMETRIC CORRELATIONS
    # ========================================================================
    
    def analyze_geometric_correlations(self, save_dir: str = "./correlation_plots"):
        """
        Geometric feature correlations
        
        Tests engineered features:
        - Radial distance
        - Distance to detector
        - Gamma direction
        - Opening angle
        """
        
        print("\n" + "="*70)
        print("ANALYSIS 3: GEOMETRIC CORRELATIONS")
        print("="*70)
        
        # Engineer features
        geo_features = self.engineer_geometric_features()
        target = self.data['target']
        
        # Analyze each geometric feature
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        geo_correlations = {}
        
        for idx, (feat_name, feat_values) in enumerate(list(geo_features.items())[:9]):
            ax = axes[idx]
            
            # Scatter (subsample)
            if len(feat_values) > 5000:
                sample_idx = np.random.choice(len(feat_values), 5000, replace=False)
                x_plot = feat_values[sample_idx]
                y_plot = target[sample_idx]
            else:
                x_plot = feat_values
                y_plot = target
            
            ax.scatter(x_plot, y_plot, alpha=0.3, s=5, c='darkgreen')
            
            # Stats
            corr_pearson = stats.pearsonr(feat_values, target)[0]
            corr_spearman = stats.spearmanr(feat_values, target)[0]
            
            # MI
            mi = mutual_info_regression(
                feat_values.reshape(-1, 1), target, 
                random_state=self.random_seed
            )[0]
            
            geo_correlations[feat_name] = {
                'pearson': float(corr_pearson),
                'spearman': float(corr_spearman),
                'mi': float(mi)
            }
            
            ax.set_xlabel(feat_name, fontsize=10, fontweight='bold')
            ax.set_ylabel('Detected Photons', fontsize=10, fontweight='bold')
            ax.set_title(f'{feat_name}\nρ={corr_pearson:+.3f}, MI={mi:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.suptitle('Geometric Feature Correlations', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/05_geometric_correlations.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/05_geometric_correlations.png")
        plt.close()
        
        # Print top geometric features
        print("\nTop Geometric Features by MI:")
        sorted_geo = sorted(geo_correlations.items(), 
                           key=lambda x: x[1]['mi'], reverse=True)
        for feat, metrics in sorted_geo[:5]:
            print(f"  {feat:<20} "
                  f"Pearson={metrics['pearson']:+.3f}  "
                  f"Spearman={metrics['spearman']:+.3f}  "
                  f"MI={metrics['mi']:.4f}")
        
        return geo_correlations
    
    # ========================================================================
    # ANALYSIS 4: MULTIVARIATE INTERACTIONS (2D)
    # ========================================================================
    
    def analyze_2d_interactions(self, save_dir: str = "./correlation_plots"):
        """
        2D interaction analysis: Feature pairs vs target
        
        Key combinations:
        - Material × Energy
        - Material × Position
        - Energy × Distance
        - Direction × Position
        
        Visualization: 2D heatmaps (binned)
        """
        
        print("\n" + "="*70)
        print("ANALYSIS 4: 2D INTERACTION ANALYSIS")
        print("="*70)
        
        phi = self.data['phi']
        target = self.data['target']
        geo_features = self.engineer_geometric_features()
        
        # Define important feature pairs
        feature_pairs = [
            ('matID', phi[:, 18], 'E_tot', phi[:, 1], 'Material vs Total Energy'),
            ('matID', phi[:, 18], 'r_cyl', geo_features['r_cyl'], 'Material vs Radial Distance'),
            ('E_tot', phi[:, 1], 'dist_to_detector', geo_features['dist_to_detector'], 
             'Energy vs Distance to Detector'),
            ('n_gamma', phi[:, 0], 'E_tot', phi[:, 1], 'N_gamma vs Total Energy'),
            ('r_cyl', geo_features['r_cyl'], 'z_nc', phi[:, 21], 'Cylindrical Position'),
            ('E_tot', phi[:, 1], 'opening_angle', geo_features['opening_angle'], 
             'Energy vs Gamma Opening Angle')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        interaction_results = {}
        
        for idx, (name1, feat1, name2, feat2, title) in enumerate(feature_pairs):
            ax = axes[idx]
            
            # 2D histogram (binned response)
            n_bins = 20
            
            # Define bins
            bins_x = np.linspace(np.percentile(feat1, 2), 
                                np.percentile(feat1, 98), n_bins)
            bins_y = np.linspace(np.percentile(feat2, 2), 
                                np.percentile(feat2, 98), n_bins)
            
            # Compute mean response in each bin
            response_map = np.full((n_bins-1, n_bins-1), np.nan)
            
            for i in range(n_bins-1):
                for j in range(n_bins-1):
                    mask = ((feat1 >= bins_x[i]) & (feat1 < bins_x[i+1]) &
                           (feat2 >= bins_y[j]) & (feat2 < bins_y[j+1]))
                    
                    if np.sum(mask) > 10:  # Minimum samples per bin
                        response_map[j, i] = np.mean(target[mask])
            
            # Plot heatmap
            im = ax.imshow(response_map, origin='lower', aspect='auto', 
                          cmap='viridis', interpolation='nearest')
            
            # Set ticks
            x_ticks = np.linspace(0, n_bins-2, 5)
            y_ticks = np.linspace(0, n_bins-2, 5)
            
            x_labels = [f'{bins_x[int(t)]:.1f}' for t in x_ticks]
            y_labels = [f'{bins_y[int(t)]:.1f}' for t in y_ticks]
            
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_yticklabels(y_labels, fontsize=9)
            
            ax.set_xlabel(name1, fontsize=11, fontweight='bold')
            ax.set_ylabel(name2, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Colorbar
            plt.colorbar(im, ax=ax, label='Mean Photons', fraction=0.046, pad=0.04)
            
            # Compute interaction strength (variance explained)
            valid_mask = ~np.isnan(response_map.flatten())
            if np.sum(valid_mask) > 0:
                variance = np.var(response_map.flatten()[valid_mask])
                interaction_results[f'{name1}_x_{name2}'] = {
                    'variance': float(variance),
                    'dynamic_range': float(np.nanmax(response_map) - np.nanmin(response_map))
                }
        
        plt.suptitle('2D Feature Interactions (Binned Mean Response)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/06_2d_interactions.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/06_2d_interactions.png")
        plt.close()
        
        # Print interaction strengths
        print("\n2D Interaction Strengths (Variance in response):")
        sorted_interactions = sorted(interaction_results.items(), 
                                    key=lambda x: x[1]['variance'], reverse=True)
        for name, metrics in sorted_interactions:
            print(f"  {name:<40} "
                  f"Var={metrics['variance']:.2f}  "
                  f"Range={metrics['dynamic_range']:.1f}")
        
        return interaction_results
    
    # ========================================================================
    # ANALYSIS 5: FEATURE IMPORTANCE RANKING
    # ========================================================================
    
    def compute_feature_importance(self, save_dir: str = "./correlation_plots"):
        """
        Comprehensive feature importance ranking
        
        Methods:
        1. Univariate MI
        2. Partial correlation (conditional on other features)
        3. Random Forest feature importance (as baseline)
        
        Output: Ranked list for model conditioning
        """
        
        print("\n" + "="*70)
        print("ANALYSIS 5: FEATURE IMPORTANCE RANKING")
        print("="*70)
        
        phi = self.data['phi']
        target = self.data['target']
        
        # Add geometric features
        geo_features = self.engineer_geometric_features()
        
        # Combine all features
        all_features = np.column_stack([
            phi,
            geo_features['r_cyl'],
            geo_features['dist_to_detector'],
            geo_features['opening_angle'],
            geo_features['E_concentration']
        ])
        
        all_feature_names = (self.feature_names + 
                            ['r_cyl', 'dist_to_detector', 
                             'opening_angle', 'E_concentration'])
        
        # Method 1: Mutual Information
        print("\n[1/3] Computing Mutual Information...")
        mi_scores = mutual_info_regression(
            all_features, target,
            discrete_features=[18],  # matID
            random_state=self.random_seed,
            n_neighbors=5
        )
        
        # Method 2: Random Forest Feature Importance
        print("[2/3] Training Random Forest...")
        from sklearn.ensemble import RandomForestRegressor
        
        # Subsample for speed
        if len(target) > 20000:
            sample_idx = np.random.choice(len(target), 20000, replace=False)
            X_rf = all_features[sample_idx]
            y_rf = target[sample_idx]
        else:
            X_rf = all_features
            y_rf = target
        
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            random_state=self.random_seed,
            n_jobs=-1
        )
        rf.fit(X_rf, y_rf)
        
        rf_importance = rf.feature_importances_
        
        # Method 3: Partial Correlation (compute for top features only)
        print("[3/3] Computing Partial Correlations...")
        
        # Top 10 features by MI
        top10_idx = np.argsort(mi_scores)[::-1][:10]
        
        partial_corr = np.zeros(len(all_feature_names))
        
        for i in top10_idx:
            # Partial correlation: corr(X_i, Y | X_others)
            # Approximation: residuals after linear regression
            
            # Features excluding i
            other_features = np.delete(all_features, i, axis=1)
            
            # Regress X_i on others
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(other_features, all_features[:, i])
            X_i_residual = all_features[:, i] - lr.predict(other_features)
            
            # Regress Y on others
            lr.fit(other_features, target)
            Y_residual = target - lr.predict(other_features)
            
            # Correlation of residuals
            partial_corr[i] = stats.pearsonr(X_i_residual, Y_residual)[0]
        
        # Normalize scores to [0, 1]
        mi_norm = mi_scores / np.max(mi_scores)
        rf_norm = rf_importance / np.max(rf_importance)
        partial_norm = np.abs(partial_corr) / (np.max(np.abs(partial_corr)) + 1e-10)
        
        # Combined score (weighted average)
        combined_score = (
            0.4 * mi_norm +
            0.3 * rf_norm +
            0.3 * partial_norm
        )
        
        # Create results dataframe
        importance_results = {
            'Feature': all_feature_names,
            'MI': mi_scores,
            'RF_Importance': rf_importance,
            'Partial_Corr': np.abs(partial_corr),
            'Combined_Score': combined_score
        }
        
        # Sort by combined score
        sorted_idx = np.argsort(combined_score)[::-1]
        
        # Print top features
        print("\n" + "="*70)
        print("TOP 15 FEATURES FOR MODEL CONDITIONING")
        print("="*70)
        print(f"{'Rank':<5} {'Feature':<20} {'MI':<10} {'RF':<10} {'Partial':<10} {'Combined':<10}")
        print("-"*70)
        
        for rank, idx in enumerate(sorted_idx[:15], 1):
            print(f"{rank:<5} {all_feature_names[idx]:<20} "
                  f"{mi_scores[idx]:<10.4f} "
                  f"{rf_importance[idx]:<10.4f} "
                  f"{np.abs(partial_corr[idx]):<10.4f} "
                  f"{combined_score[idx]:<10.4f}")
        
        # Visualization
        self._plot_feature_importance(importance_results, sorted_idx[:20], save_dir)
        
        return importance_results
    
    def _plot_feature_importance(self, results: Dict, top_idx: np.ndarray, 
                                 save_dir: str):
        """Plot feature importance comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract top features
        top_features = [results['Feature'][i] for i in top_idx]
        
        # Plot 1: MI
        ax = axes[0, 0]
        mi_top = [results['MI'][i] for i in top_idx]
        ax.barh(range(len(top_features)), mi_top, color='steelblue', 
                edgecolor='black', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Mutual Information', fontsize=11, fontweight='bold')
        ax.set_title('Mutual Information', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Random Forest
        ax = axes[0, 1]
        rf_top = [results['RF_Importance'][i] for i in top_idx]
        ax.barh(range(len(top_features)), rf_top, color='darkgreen', 
                edgecolor='black', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('RF Importance', fontsize=11, fontweight='bold')
        ax.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Partial Correlation
        ax = axes[1, 0]
        partial_top = [results['Partial_Corr'][i] for i in top_idx]
        ax.barh(range(len(top_features)), partial_top, color='darkorange', 
                edgecolor='black', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('|Partial Correlation|', fontsize=11, fontweight='bold')
        ax.set_title('Partial Correlation (Conditional)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 4: Combined Score
        ax = axes[1, 1]
        combined_top = [results['Combined_Score'][i] for i in top_idx]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), combined_top, 
                      color=colors, edgecolor='black', alpha=0.9)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Combined Score', fontsize=11, fontweight='bold')
        ax.set_title('Combined Feature Importance\n(0.4×MI + 0.3×RF + 0.3×Partial)', 
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Highlight top 5
        for i in range(5):
            bars[i].set_linewidth(3)
            bars[i].set_edgecolor('red')
        
        plt.suptitle('Feature Importance: Multi-Method Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/07_feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_dir}/07_feature_importance.png")
        plt.close()
    
    # ========================================================================
    # ANALYSIS 6: LOSS FUNCTION RECOMMENDATIONS
    # ========================================================================
    
    def recommend_loss_strategy(self, importance_results: Dict):
        """
        Derive physics-informed loss function strategy
        
        Based on correlation analysis, recommend:
        1. Which features to use as conditioning (φ)
        2. Which features to include in physics loss
        """
        
        print("\n" + "="*70)
        print("FEATURE SELECTION FOR MODEL CONDITIONING")
        print("="*70)
        
        # Extract top features
        combined_scores = importance_results['Combined_Score']
        feature_names = importance_results['Feature']
        
        sorted_idx = np.argsort(combined_scores)[::-1]
        
        # Category 1: Essential conditioning (top 8, score > 0.5)
        essential_idx = [i for i in sorted_idx if combined_scores[i] > 0.5][:8]
        essential_features = [feature_names[i] for i in essential_idx]
        
        # Category 2: Important but redundant (can be learned latently)
        important_idx = [i for i in sorted_idx[8:15] if combined_scores[i] > 0.3]
        important_features = [feature_names[i] for i in important_idx]
        
        # Category 3: Weak correlations (can be ignored)
        weak_idx = [i for i in sorted_idx if combined_scores[i] < 0.2]
        weak_features = [feature_names[i] for i in weak_idx]
        
        print("\n[CATEGORY 1] ESSENTIAL FOR CONDITIONING (φ in Diffusion Model):")
        print("  → These MUST be inputs to the model")
        for i, feat in enumerate(essential_features, 1):
            score = combined_scores[sorted_idx[i-1]]
            print(f"    {i}. {feat:<20} (score: {score:.3f})")
        
        print("\n[CATEGORY 2] IMPORTANT BUT POTENTIALLY REDUNDANT:")
        print("  → Can be included OR learned latently")
        for feat in important_features:
            print(f"    • {feat}")
        
        print("\n[CATEGORY 3] WEAK CORRELATIONS:")
        print("  → Can be safely excluded from conditioning")
        print(f"    Total: {len(weak_features)} features with score < 0.2")
        
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"  Original features:     {len(feature_names)}")
        print(f"  Essential (score>0.5): {len(essential_features)}")
        print(f"  Important (0.3-0.5):   {len(important_features)}")
        print(f"  Weak (score<0.2):      {len(weak_features)}")
        print(f"\n  → Recommended phi_dim for model: {len(essential_features)}")
        print(f"  → Dimension reduction: {len(feature_names)} → {len(essential_features)} ({100*len(essential_features)/len(feature_names):.1f}%)")
        
        return {
            'essential_features': essential_features,
            'essential_indices': essential_idx,
            'important_features': important_features,
            'weak_features': weak_features[:10],
            'n_recommended': len(essential_features)
        }
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_full_analysis(self, save_dir: str = "./correlation_plots"):
        """
        Execute complete correlation analysis pipeline
        
        Returns:
            Comprehensive results dict with all analyses
        """
        
        results = {}
        
        # Analysis 1: Univariate
        results['univariate'] = self.analyze_univariate_correlations(save_dir)
        
        # Analysis 2: Material-stratified
        results['material'] = self.analyze_material_stratified(save_dir)
        
        # Analysis 3: Geometric
        results['geometric'] = self.analyze_geometric_correlations(save_dir)
        
        # Analysis 4: 2D Interactions
        results['interactions_2d'] = self.analyze_2d_interactions(save_dir)
        
        # Analysis 5: Feature Importance
        results['importance'] = self.compute_feature_importance(save_dir)
        
        # Analysis 6: Loss Strategy
        results['loss_strategy'] = self.recommend_loss_strategy(results['importance'])
        
        # Save results
        import json
        results_file = f"{save_dir}/correlation_analysis_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = self._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved: {results_file}")
        print(f"Plots saved: {save_dir}/")
        print("="*70)
        
        return results
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON"""
        if isinstance(obj, dict):
            # CRITICAL: Convert dict keys to strings (JSON requirement)
            return {
                str(k) if isinstance(k, (np.int64, np.int32, int)) else k: 
                self._make_json_serializable(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


# ========================================================================
# MAIN FUNCTION
# ========================================================================

def main():
    """
    Main execution function
    
    Usage:
        python correlation_analysis_physics_informed.py
    """
    
    # Configuration
    DATA_PATH = "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/outdated/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5"
    N_SAMPLE = 50000  # Sample size (trade-off: speed vs statistics)
    SAVE_DIR = "./correlation_plots"
    
    # Initialize analyzer
    analyzer = PhysicsCorrelationAnalyzer(
        data_path=DATA_PATH,
        n_sample=N_SAMPLE,
        random_seed=42
    )
    
    # Run full analysis
    results = analyzer.run_full_analysis(save_dir=SAVE_DIR)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review plots in:", SAVE_DIR)
    print("2. Implement physics loss function (template provided above)")
    print("3. Update diffusion model conditioning with essential features")
    print("4. Train with combined loss and validate physics constraints")
    print("="*70)


if __name__ == "__main__":
    main()