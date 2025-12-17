#!/usr/bin/env python3
"""
verify_diffusion_model.py

Umfassende Verifikation eines trainierten Conditional Diffusion Models
für Neutron Capture → Optical Signal Konversion

Basiert auf State-of-the-Art Diffusion Papers:
1. Ho et al. (2020) - DDPM: Reconstruction Fidelity
2. Rombach et al. (2022) - Stable Diffusion: Conditional Consistency
3. Song et al. (2021) - Score-Based: Interpolation Smoothness
4. Karras et al. (2022) - EDM: Physics-Informed Constraints
5. Ho & Salimans (2022) - Classifier-Free Guidance

Autor: Physics-ML Verification Framework
"""

import os
# CRITICAL: Disable GPU/XLA to avoid CUDA library issues on NERSC
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import h5py
from tqdm import tqdm
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.signal import correlate
from diffusion_model import build_diffusion_model


@dataclass
class VerificationConfig:
    """Konfiguration für Modellverifikation"""
    # Model
    checkpoint_path: str = "./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5"
    phi_dim: int = 22
    target_dim: int = 7789
    T: int = 1000
    hidden_dim: int = 512
    n_layers: int = 4
    t_emb_dim: int = 32
    
    # Test data
    data_path: str = "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5"
    n_test_events: int = 100
    
    # Sampling parameters
    n_sampling_steps: int = 50  # DDIM acceleration
    n_conditional_samples: int = 50  # Für Consistency-Test
    n_interpolation_steps: int = 10
    
    # Physics constraints
    multiplicity_threshold: int = 6
    hit_threshold: float = 0.5
    energy_tolerance: float = 0.2  # ±20% Energieabweichung toleriert
    
    # Output
    output_dir: str = "./verification_results"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class DDIMSampler:
    """
    DDIM Sampler für schnellere Inferenz
    
    Referenz: Song et al. (2021) "Denoising Diffusion Implicit Models"
              https://arxiv.org/abs/2010.02502
    
    Vorteil: Deterministische Sampling mit weniger Steps (50 statt 1000)
    """
    
    def __init__(self, model: tf.keras.Model, T: int, eta: float = 0.0):
        self.model = model
        self.T = T
        self.eta = eta  # 0 = deterministisch, 1 = DDPM
        
        # Diffusion schedule
        self.betas = tf.linspace(1e-4, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
    
    def sample(self, 
               phi: tf.Tensor, 
               n_steps: int = 50,
               return_trajectory: bool = False) -> tf.Tensor:
        """
        Generiere Sample via DDIM
        
        Args:
            phi: Conditioning (batch, phi_dim)
            n_steps: Anzahl Sampling-Steps
            return_trajectory: Ob gesamte Denoising-Trajektorie zurückgeben
            
        Returns:
            x_0: Generiertes Signal (batch, target_dim)
            [trajectory]: Optional Liste von (x_t, t) Paaren
        """
        batch_size = phi.shape[0]
        target_dim = self.model.input_shape[0][1]
        
        # Start von Rauschen
        x_t = tf.random.normal((batch_size, target_dim), dtype=tf.float32)
        
        # Timestep Schedule
        timesteps = np.linspace(self.T - 1, 0, n_steps, dtype=np.int32)
        
        trajectory = [(x_t.numpy(), self.T)] if return_trajectory else None
        
        for i, t in enumerate(timesteps):
            t_tensor = tf.constant([t] * batch_size, dtype=tf.int32)
            
            # Predict noise
            epsilon_pred = self.model([x_t, phi, t_tensor], training=False)
            
            # DDIM update
            if t > 0:
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else 1.0
                
                # Predicted x_0
                x_0_pred = (x_t - tf.sqrt(1 - alpha_t) * epsilon_pred) / tf.sqrt(alpha_t)
                
                # Direction pointing to x_t
                dir_xt = tf.sqrt(1 - alpha_t_prev) * epsilon_pred
                
                # DDIM step
                x_t = tf.sqrt(alpha_t_prev) * x_0_pred + dir_xt
            else:
                # Final step
                alpha_t = self.alphas_cumprod[t]
                x_t = (x_t - tf.sqrt(1 - alpha_t) * epsilon_pred) / tf.sqrt(alpha_t)
            
            if return_trajectory:
                trajectory.append((x_t.numpy(), t))
        
        if return_trajectory:
            return x_t, trajectory
        return x_t


class DiffusionVerifier:
    """Hauptklasse für Modellverifikation"""
    
    def __init__(self, model: tf.keras.Model, config: VerificationConfig):
        self.model = model
        self.config = config
        self.sampler = DDIMSampler(model, config.T)
        
        # Lade Test-Daten
        self.test_data = self._load_test_data()
        
        # Ergebnisse
        self.results = {
            'reconstruction_fidelity': {},
            'conditional_consistency': {},
            'interpolation_smoothness': {},
            'physics_constraints': {},
            'overall_score': 0.0
        }
    
    def _load_test_data(self) -> List[Dict]:
        """Lade Test-Events aus HDF5"""
        print(f"Lade Test-Daten: {self.config.data_path}")
        
        test_data = []
        
        with h5py.File(self.config.data_path, 'r') as f:
            phi_group = f['phi']
            target_group = f['target']
            
            phi_keys = sorted(phi_group.keys())
            voxel_keys = sorted(target_group.keys())
            
            n_total = phi_group[phi_keys[0]].shape[0]
            event_indices = np.random.choice(
                n_total, 
                min(self.config.n_test_events, n_total), 
                replace=False
            )
            
            for event_idx in tqdm(event_indices, desc="Lade Events"):
                phi = np.array([
                    phi_group[key][event_idx] 
                    for key in phi_keys
                ], dtype=np.float32)
                
                signal = np.array([
                    target_group[key][event_idx] 
                    for key in voxel_keys
                ], dtype=np.float32)
                
                test_data.append({
                    'signal': signal,
                    'phi': phi
                })
        
        print(f"  ✓ {len(test_data)} Events geladen")
        return test_data
    
    # ========================================================================
    # METHODE 1: RECONSTRUCTION FIDELITY
    # ========================================================================
    
    def verify_reconstruction_fidelity(self) -> Dict:
        """
        Test 1: Reconstruction Fidelity (Ho et al. 2020, DDPM)
        
        Idee: Generiere aus NC-Parametern und vergleiche mit Ground Truth
        
        Metriken:
        - MSE (Mean Squared Error)
        - Correlation Coefficient
        - Active Voxel Precision/Recall
        """
        print("\n" + "="*70)
        print("TEST 1: RECONSTRUCTION FIDELITY")
        print("="*70)
        
        mse_scores = []
        corr_scores = []
        precision_scores = []
        recall_scores = []
        
        # Sample für jedes Test-Event
        for event in tqdm(self.test_data, desc="Rekonstruktion"):
            phi = tf.constant([event['phi']], dtype=tf.float32)
            x_true = event['signal']
            
            # Generiere Rekonstruktion
            x_recon = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
            x_recon = x_recon.numpy()[0]
            
            # MSE
            mse = np.mean((x_recon - x_true) ** 2)
            mse_scores.append(mse)
            
            # Correlation
            corr = np.corrcoef(x_recon, x_true)[0, 1]
            corr_scores.append(corr if not np.isnan(corr) else 0.0)
            
            # Active Voxel Metrics
            active_true = x_true > self.config.hit_threshold
            active_recon = x_recon > self.config.hit_threshold
            
            true_positives = np.sum(active_true & active_recon)
            false_positives = np.sum(~active_true & active_recon)
            false_negatives = np.sum(active_true & ~active_recon)
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        # Aggregiere Ergebnisse
        results = {
            'mse_mean': float(np.mean(mse_scores)),
            'mse_std': float(np.std(mse_scores)),
            'correlation_mean': float(np.mean(corr_scores)),
            'correlation_std': float(np.std(corr_scores)),
            'precision_mean': float(np.mean(precision_scores)),
            'recall_mean': float(np.mean(recall_scores)),
            'f1_score': float(2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                            (np.mean(precision_scores) + np.mean(recall_scores) + 1e-8))
        }
        
        # Score: Gewichtete Kombination (0-1)
        results['fidelity_score'] = (
            0.3 * (1 - np.tanh(results['mse_mean'])) +  # MSE (niedriger = besser)
            0.4 * results['correlation_mean'] +          # Correlation (höher = besser)
            0.3 * results['f1_score']                    # F1 (höher = besser)
        )
        
        print(f"\n✓ Reconstruction Fidelity Results:")
        print(f"  MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
        print(f"  Correlation: {results['correlation_mean']:.3f} ± {results['correlation_std']:.3f}")
        print(f"  Precision: {results['precision_mean']:.3f}")
        print(f"  Recall: {results['recall_mean']:.3f}")
        print(f"  F1-Score: {results['f1_score']:.3f}")
        print(f"  → Overall Fidelity Score: {results['fidelity_score']:.3f}/1.0")
        
        self.results['reconstruction_fidelity'] = results
        return results
    
    # ========================================================================
    # METHODE 2: CONDITIONAL CONSISTENCY
    # ========================================================================
    
    def verify_conditional_consistency(self) -> Dict:
        """
        Test 2: Conditional Consistency (Rombach et al. 2022)
        
        Idee: Multiple Samples für gleiche Kondition sollten konsistent sein
        
        Metriken:
        - Mean-Consistency: Wie nah ist Durchschnitt an Ground Truth?
        - Variance-Plausibility: Ist Varianz physikalisch sinnvoll?
        """
        print("\n" + "="*70)
        print("TEST 2: CONDITIONAL CONSISTENCY")
        print("="*70)
        
        consistency_scores = []
        variance_scores = []
        
        # Teste auf Subset (zu teuer für alle)
        n_test = min(20, len(self.test_data))
        
        for event in tqdm(self.test_data[:n_test], desc="Conditional Sampling"):
            phi = tf.constant([event['phi']], dtype=tf.float32)
            x_true = event['signal']
            
            # Generiere n_conditional_samples Samples
            samples = []
            for _ in range(self.config.n_conditional_samples):
                x_sample = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
                samples.append(x_sample.numpy()[0])
            
            samples = np.array(samples)  # (n_samples, target_dim)
            
            # Mean Consistency
            x_mean = np.mean(samples, axis=0)
            consistency = np.corrcoef(x_mean, x_true)[0, 1]
            consistency_scores.append(consistency if not np.isnan(consistency) else 0.0)
            
            # Variance Plausibility
            x_var = np.var(samples, axis=0)
            
            # Physikalisch plausibel: Varianz sollte niedrig sein für inaktive Voxel,
            # höher für aktive Voxel (aber nicht zu hoch)
            active_true = x_true > self.config.hit_threshold
            var_inactive = np.mean(x_var[~active_true])
            var_active = np.mean(x_var[active_true])
            
            # Score: Aktive sollten mehr Varianz haben, aber nicht kollabieren
            var_ratio = var_active / (var_inactive + 1e-8)
            variance_plausibility = np.clip(var_ratio / 10.0, 0, 1)  # Ideal: ~10x höher
            variance_scores.append(variance_plausibility)
        
        results = {
            'mean_consistency': float(np.mean(consistency_scores)),
            'mean_consistency_std': float(np.std(consistency_scores)),
            'variance_plausibility': float(np.mean(variance_scores)),
            'variance_plausibility_std': float(np.std(variance_scores))
        }
        
        # Overall Score
        results['consistency_score'] = (
            0.7 * results['mean_consistency'] +
            0.3 * results['variance_plausibility']
        )
        
        print(f"\n✓ Conditional Consistency Results:")
        print(f"  Mean Consistency: {results['mean_consistency']:.3f} ± {results['mean_consistency_std']:.3f}")
        print(f"  Variance Plausibility: {results['variance_plausibility']:.3f} ± {results['variance_plausibility_std']:.3f}")
        print(f"  → Overall Consistency Score: {results['consistency_score']:.3f}/1.0")
        
        self.results['conditional_consistency'] = results
        return results
    
    # ========================================================================
    # METHODE 3: INTERPOLATION SMOOTHNESS
    # ========================================================================
    
    def verify_interpolation_smoothness(self) -> Dict:
        """
        Test 3: Interpolation Smoothness (Song et al. 2021)
        
        Idee: Interpoliere zwischen zwei NC-Positionen → Signal sollte smooth wandern
        
        Metrik: Perceptual Path Length (PPL)
        """
        print("\n" + "="*70)
        print("TEST 3: INTERPOLATION SMOOTHNESS")
        print("="*70)
        
        path_lengths = []
        
        # Teste auf Paaren
        n_pairs = min(20, len(self.test_data) // 2)
        
        for i in tqdm(range(n_pairs), desc="Interpolation"):
            event_a = self.test_data[2*i]
            event_b = self.test_data[2*i + 1]
            
            phi_a = event_a['phi']
            phi_b = event_b['phi']
            
            # Interpoliere zwischen phi_a und phi_b
            alphas = np.linspace(0, 1, self.config.n_interpolation_steps)
            
            signals_interp = []
            for alpha in alphas:
                phi_interp = (1 - alpha) * phi_a + alpha * phi_b
                phi_interp_batch = tf.constant([phi_interp], dtype=tf.float32)
                
                x_interp = self.sampler.sample(phi_interp_batch, 
                                              n_steps=self.config.n_sampling_steps)
                signals_interp.append(x_interp.numpy()[0])
            
            signals_interp = np.array(signals_interp)  # (n_steps, target_dim)
            
            # Berechne Path Length (Summe der Distanzen zwischen Steps)
            path_length = 0.0
            for j in range(len(signals_interp) - 1):
                dist = np.linalg.norm(signals_interp[j+1] - signals_interp[j])
                path_length += dist
            
            # Normalisiere durch direkte Distanz (Idealfall: Gerade Linie)
            direct_dist = np.linalg.norm(signals_interp[-1] - signals_interp[0])
            normalized_path_length = path_length / (direct_dist + 1e-8)
            
            path_lengths.append(normalized_path_length)
        
        results = {
            'path_length_mean': float(np.mean(path_lengths)),
            'path_length_std': float(np.std(path_lengths))
        }
        
        # Score: Ideal ist 1.0 (gerade Linie), höher = weniger smooth
        results['smoothness_score'] = np.clip(2.0 / results['path_length_mean'], 0, 1)
        
        print(f"\n✓ Interpolation Smoothness Results:")
        print(f"  Normalized Path Length: {results['path_length_mean']:.3f} ± {results['path_length_std']:.3f}")
        print(f"  (Ideal: 1.0 = perfekt smooth)")
        print(f"  → Overall Smoothness Score: {results['smoothness_score']:.3f}/1.0")
        
        self.results['interpolation_smoothness'] = results
        return results
    
    # ========================================================================
    # METHODE 4: PHYSICS CONSTRAINTS
    # ========================================================================
    
    def verify_physics_constraints(self) -> Dict:
        """
        Test 4: Physics-Informed Constraints (Karras et al. 2022)
        
        Prüfe physikalische Plausibilität:
        1. Energieerhaltung
        2. Räumliche Lokalität
        3. Multiplizitätsbedingung
        """
        print("\n" + "="*70)
        print("TEST 4: PHYSICS CONSTRAINTS")
        print("="*70)
        
        energy_violations = []
        multiplicity_success = []
        spatial_locality_scores = []
        
        for event in tqdm(self.test_data, desc="Physics Checks"):
            phi = tf.constant([event['phi']], dtype=tf.float32)
            x_true = event['signal']
            
            # Generiere Sample
            x_gen = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
            x_gen = x_gen.numpy()[0]
            
            # 1. Energieerhaltung (approximativ)
            # Total Signal sollte mit E_gamma_tot_keV korrelieren
            energy_true = event['phi'][1]  # E_gamma_tot_keV
            signal_sum_true = np.sum(x_true)
            signal_sum_gen = np.sum(x_gen)
            
            # Relative Abweichung
            if signal_sum_true > 0:
                energy_violation = abs(signal_sum_gen - signal_sum_true) / signal_sum_true
            else:
                energy_violation = 1.0
            
            energy_violations.append(energy_violation)
            
            # 2. Multiplizitätsbedingung
            active_voxels = np.sum(x_gen > self.config.hit_threshold)
            multiplicity_ok = active_voxels >= self.config.multiplicity_threshold
            multiplicity_success.append(1.0 if multiplicity_ok else 0.0)
            
            # 3. Räumliche Lokalität (vereinfacht)
            # Signal sollte um NC-Position konzentriert sein
            # (Hier: Nutze Variance als Proxy - niedrig = lokalisiert)
            signal_center_of_mass = np.sum(x_gen * np.arange(len(x_gen))) / (np.sum(x_gen) + 1e-8)
            signal_spread = np.sqrt(np.sum(x_gen * (np.arange(len(x_gen)) - signal_center_of_mass)**2) / (np.sum(x_gen) + 1e-8))
            
            # Normalisiere Spread (niedriger = besser lokalisiert)
            locality_score = np.exp(-signal_spread / 1000.0)  # Heuristisch
            spatial_locality_scores.append(locality_score)
        
        results = {
            'energy_violation_mean': float(np.mean(energy_violations)),
            'energy_violation_std': float(np.std(energy_violations)),
            'multiplicity_success_rate': float(np.mean(multiplicity_success)),
            'spatial_locality_mean': float(np.mean(spatial_locality_scores)),
            'spatial_locality_std': float(np.std(spatial_locality_scores))
        }
        
        # Overall Score
        results['physics_score'] = (
            0.4 * (1 - np.tanh(results['energy_violation_mean'] / self.config.energy_tolerance)) +
            0.4 * results['multiplicity_success_rate'] +
            0.2 * results['spatial_locality_mean']
        )
        
        print(f"\n✓ Physics Constraints Results:")
        print(f"  Energy Violation: {results['energy_violation_mean']:.3f} ± {results['energy_violation_std']:.3f}")
        print(f"  Multiplicity Success: {results['multiplicity_success_rate']:.3f}")
        print(f"  Spatial Locality: {results['spatial_locality_mean']:.3f} ± {results['spatial_locality_std']:.3f}")
        print(f"  → Overall Physics Score: {results['physics_score']:.3f}/1.0")
        
        self.results['physics_constraints'] = results
        return results
    
    # ========================================================================
    # AGGREGATION & VISUALIZATION
    # ========================================================================
    
    def compute_overall_score(self) -> float:
        """Berechne gewichteten Gesamtscore"""
        scores = {
            'fidelity': self.results['reconstruction_fidelity'].get('fidelity_score', 0),
            'consistency': self.results['conditional_consistency'].get('consistency_score', 0),
            'smoothness': self.results['interpolation_smoothness'].get('smoothness_score', 0),
            'physics': self.results['physics_constraints'].get('physics_score', 0)
        }
        
        # Gewichte (Physik ist am wichtigsten!)
        weights = {
            'fidelity': 0.3,
            'consistency': 0.2,
            'smoothness': 0.1,
            'physics': 0.4
        }
        
        overall = sum(scores[k] * weights[k] for k in scores.keys())
        self.results['overall_score'] = float(overall)
        
        return overall
    
    def run_all_tests(self):
        """Führe alle Verifikationstests aus"""
        print("\n" + "="*70)
        print("DIFFUSION MODEL VERIFICATION SUITE")
        print("="*70)
        print(f"Model: {self.config.checkpoint_path}")
        print(f"Test Events: {len(self.test_data)}")
        print("="*70)
        
        # Führe Tests aus
        self.verify_reconstruction_fidelity()
        self.verify_conditional_consistency()
        self.verify_interpolation_smoothness()
        self.verify_physics_constraints()
        
        # Gesamtscore
        overall = self.compute_overall_score()
        
        print("\n" + "="*70)
        print("OVERALL VERIFICATION SCORE")
        print("="*70)
        print(f"  Reconstruction Fidelity:    {self.results['reconstruction_fidelity']['fidelity_score']:.3f}")
        print(f"  Conditional Consistency:    {self.results['conditional_consistency']['consistency_score']:.3f}")
        print(f"  Interpolation Smoothness:   {self.results['interpolation_smoothness']['smoothness_score']:.3f}")
        print(f"  Physics Constraints:        {self.results['physics_constraints']['physics_score']:.3f}")
        print(f"\n  → OVERALL SCORE: {overall:.3f}/1.0")
        
        # Interpretation
        if overall >= 0.8:
            verdict = "✓ EXCELLENT - Model learned NC→Signal mapping well"
        elif overall >= 0.6:
            verdict = "⚠ GOOD - Model works but has some issues"
        elif overall >= 0.4:
            verdict = "⚠ FAIR - Model needs more training"
        else:
            verdict = "✗ POOR - Model did not learn the task properly"
        
        print(f"\n  {verdict}")
        print("="*70)
        
        # Speichere Ergebnisse
        self._save_results()
    
    def _save_results(self):
        """Speichere Verifikationsergebnisse"""
        output_path = Path(self.config.output_dir) / "verification_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Ergebnisse gespeichert: {output_path}")
    
    def visualize_results(self):
        """Erstelle umfassende Visualisierung"""
        print("\nGeneriere Visualisierungen...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Reconstruction Examples (3 Samples)
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            
            if i < len(self.test_data):
                event = self.test_data[i]
                phi = tf.constant([event['phi']], dtype=tf.float32)
                x_true = event['signal']
                
                x_recon = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
                x_recon = x_recon.numpy()[0]
                
                # Plot nur erste 500 Voxel (übersichtlicher)
                voxels = np.arange(min(500, len(x_true)))
                
                ax.plot(voxels, x_true[:len(voxels)], 'b-', alpha=0.7, linewidth=1.5, label='Ground Truth')
                ax.plot(voxels, x_recon[:len(voxels)], 'r--', alpha=0.7, linewidth=1.5, label='Reconstructed')
                ax.axhline(self.config.hit_threshold, color='gray', linestyle=':', 
                          linewidth=1, alpha=0.5, label='Hit Threshold')
                
                corr = np.corrcoef(x_recon, x_true)[0, 1]
                ax.set_title(f'Sample {i+1}\nCorr: {corr:.3f}', fontweight='bold')
                ax.set_xlabel('Voxel Index')
                ax.set_ylabel('Signal')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        
        # 2. Score Summary (Radar Chart)
        ax_radar = fig.add_subplot(gs[1, 0], projection='polar')
        
        categories = ['Fidelity', 'Consistency', 'Smoothness', 'Physics']
        scores = [
            self.results['reconstruction_fidelity'].get('fidelity_score', 0),
            self.results['conditional_consistency'].get('consistency_score', 0),
            self.results['interpolation_smoothness'].get('smoothness_score', 0),
            self.results['physics_constraints'].get('physics_score', 0)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]
        angles_plot = angles + [angles[0]]
        
        ax_radar.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='#3498db')
        ax_radar.fill(angles_plot, scores_plot, alpha=0.25, color='#3498db')
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(categories, fontweight='bold')
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Verification Scores', fontweight='bold', pad=20)
        ax_radar.grid(True)
        
        # 3. Conditional Consistency (Sample Variability)
        ax_consistency = fig.add_subplot(gs[1, 1])
        
        if len(self.test_data) > 0:
            event = self.test_data[0]
            phi = tf.constant([event['phi']], dtype=tf.float32)
            
            # Generiere 10 Samples
            samples = []
            for _ in range(10):
                x_sample = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
                samples.append(x_sample.numpy()[0][:500])  # Erste 500 Voxel
            
            samples = np.array(samples)
            voxels = np.arange(samples.shape[1])
            
            mean_signal = np.mean(samples, axis=0)
            std_signal = np.std(samples, axis=0)
            
            ax_consistency.plot(voxels, mean_signal, 'k-', linewidth=2, label='Mean')
            ax_consistency.fill_between(voxels, 
                                       mean_signal - std_signal, 
                                       mean_signal + std_signal,
                                       alpha=0.3, color='blue', label='±1 Std')
            ax_consistency.plot(voxels, event['signal'][:500], 'r--', 
                              linewidth=1.5, alpha=0.7, label='Ground Truth')
            
            ax_consistency.set_xlabel('Voxel Index')
            ax_consistency.set_ylabel('Signal')
            ax_consistency.set_title('Conditional Consistency\n(10 Samples)', fontweight='bold')
            ax_consistency.legend()
            ax_consistency.grid(alpha=0.3)
        
        # 4. Interpolation Path
        ax_interp = fig.add_subplot(gs[1, 2])
        
        if len(self.test_data) >= 2:
            event_a = self.test_data[0]
            event_b = self.test_data[1]
            
            phi_a = event_a['phi']
            phi_b = event_b['phi']
            
            alphas = np.linspace(0, 1, 5)
            
            for idx, alpha in enumerate(alphas):
                phi_interp = (1 - alpha) * phi_a + alpha * phi_b
                phi_interp_batch = tf.constant([phi_interp], dtype=tf.float32)
                
                x_interp = self.sampler.sample(phi_interp_batch, 
                                              n_steps=self.config.n_sampling_steps)
                x_interp = x_interp.numpy()[0][:500]
                
                ax_interp.plot(x_interp, alpha=0.6, 
                             label=f'α={alpha:.2f}',
                             linewidth=1.5)
            
            ax_interp.set_xlabel('Voxel Index')
            ax_interp.set_ylabel('Signal')
            ax_interp.set_title('Interpolation Smoothness\nφ_A → φ_B', fontweight='bold')
            ax_interp.legend(fontsize=8)
            ax_interp.grid(alpha=0.3)
        
        # 5. Physics: Energy Conservation
        ax_energy = fig.add_subplot(gs[2, 0])
        
        energy_true_list = []
        energy_gen_list = []
        
        for event in self.test_data[:50]:  # Sample
            phi = tf.constant([event['phi']], dtype=tf.float32)
            x_true = event['signal']
            
            x_gen = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
            x_gen = x_gen.numpy()[0]
            
            energy_true_list.append(np.sum(x_true))
            energy_gen_list.append(np.sum(x_gen))
        
        ax_energy.scatter(energy_true_list, energy_gen_list, alpha=0.6, s=50)
        
        # Perfekte Linie
        min_e = min(min(energy_true_list), min(energy_gen_list))
        max_e = max(max(energy_true_list), max(energy_gen_list))
        ax_energy.plot([min_e, max_e], [min_e, max_e], 'r--', 
                      linewidth=2, label='Perfect Conservation')
        
        ax_energy.set_xlabel('True Total Signal', fontweight='bold')
        ax_energy.set_ylabel('Generated Total Signal', fontweight='bold')
        ax_energy.set_title('Energy Conservation Check', fontweight='bold')
        ax_energy.legend()
        ax_energy.grid(alpha=0.3)
        
        # 6. Physics: Multiplicity Distribution
        ax_mult = fig.add_subplot(gs[2, 1])
        
        multiplicities_true = []
        multiplicities_gen = []
        
        for event in self.test_data[:50]:
            phi = tf.constant([event['phi']], dtype=tf.float32)
            x_true = event['signal']
            
            x_gen = self.sampler.sample(phi, n_steps=self.config.n_sampling_steps)
            x_gen = x_gen.numpy()[0]
            
            mult_true = np.sum(x_true > self.config.hit_threshold)
            mult_gen = np.sum(x_gen > self.config.hit_threshold)
            
            multiplicities_true.append(mult_true)
            multiplicities_gen.append(mult_gen)
        
        bins = np.arange(0, max(max(multiplicities_true), max(multiplicities_gen)) + 2)
        
        ax_mult.hist(multiplicities_true, bins=bins, alpha=0.6, 
                    color='blue', label='Ground Truth', edgecolor='black')
        ax_mult.hist(multiplicities_gen, bins=bins, alpha=0.6, 
                    color='red', label='Generated', edgecolor='black')
        ax_mult.axvline(self.config.multiplicity_threshold, color='green', 
                       linestyle='--', linewidth=2, label='N≥6 Threshold')
        
        ax_mult.set_xlabel('Multiplicity (# Active Voxels)', fontweight='bold')
        ax_mult.set_ylabel('Frequency', fontweight='bold')
        ax_mult.set_title('Multiplicity Distribution', fontweight='bold')
        ax_mult.legend()
        ax_mult.grid(alpha=0.3)
        
        # 7. Overall Score Bar
        ax_overall = fig.add_subplot(gs[2, 2])
        
        score_names = ['Fidelity', 'Consistency', 'Smoothness', 'Physics', 'OVERALL']
        score_values = [
            self.results['reconstruction_fidelity'].get('fidelity_score', 0),
            self.results['conditional_consistency'].get('consistency_score', 0),
            self.results['interpolation_smoothness'].get('smoothness_score', 0),
            self.results['physics_constraints'].get('physics_score', 0),
            self.results['overall_score']
        ]
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
        bars = ax_overall.barh(score_names, score_values, color=colors, 
                              alpha=0.8, edgecolor='black')
        
        # Annotate
        for bar, score in zip(bars, score_values):
            width = bar.get_width()
            ax_overall.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                          f'{score:.3f}',
                          ha='left', va='center', fontweight='bold', fontsize=11)
        
        ax_overall.set_xlim(0, 1.1)
        ax_overall.set_xlabel('Score', fontweight='bold')
        ax_overall.set_title('Verification Summary', fontweight='bold')
        ax_overall.axvline(0.6, color='orange', linestyle=':', linewidth=2, alpha=0.5)
        ax_overall.axvline(0.8, color='green', linestyle=':', linewidth=2, alpha=0.5)
        ax_overall.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Diffusion Model Verification Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Speichern
        output_path = Path(self.config.output_dir) / "verification_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualisierung gespeichert: {output_path}")
        
        plt.show()
    
    def visualize_denoising_trajectory(self):
        """Visualisiere komplette Denoising-Trajektorie"""
        print("\nGeneriere Denoising-Trajektorie Visualisierung...")
        
        if len(self.test_data) == 0:
            print("⚠ Keine Test-Daten verfügbar")
            return
        
        event = self.test_data[0]
        phi = tf.constant([event['phi']], dtype=tf.float32)
        x_true = event['signal']
        
        # Sample mit Trajektorie
        x_final, trajectory = self.sampler.sample(
            phi, 
            n_steps=20,  # Weniger Steps für Visualisierung
            return_trajectory=True
        )
        
        # Plotte Trajektorie
        fig, axes = plt.subplots(5, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, (x_t, t) in enumerate(trajectory):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Plotte erste 500 Voxel
            voxels = np.arange(min(500, len(x_t[0])))
            ax.plot(voxels, x_t[0][:len(voxels)], 'b-', linewidth=1, alpha=0.8)
            
            if idx == len(trajectory) - 1:
                # Final: Vergleiche mit Ground Truth
                ax.plot(voxels, x_true[:len(voxels)], 'r--', 
                       linewidth=1.5, alpha=0.7, label='Ground Truth')
                ax.legend(fontsize=8)
            
            ax.axhline(self.config.hit_threshold, color='gray', 
                      linestyle=':', linewidth=1, alpha=0.5)
            ax.set_title(f't = {t}', fontsize=10, fontweight='bold')
            ax.set_ylim(-2, 10)
            ax.grid(alpha=0.3)
            
            if idx >= 16:  # Nur unterste Reihe
                ax.set_xlabel('Voxel Index', fontsize=9)
        
        plt.suptitle('Denoising Trajectory: Noise → Signal', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(self.config.output_dir) / "denoising_trajectory.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trajektorie gespeichert: {output_path}")
        
        plt.show()


def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verifiziere trainiertes Diffusionsmodell'
    )
    parser.add_argument(
        '--checkpoint',
        default='./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5',
        help='Pfad zum Modell-Checkpoint'
    )
    parser.add_argument(
        '--data-path',
        default='/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/'
                'resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5',
        help='Pfad zur HDF5-Datendatei'
    )
    parser.add_argument(
        '--n-test-events',
        type=int,
        default=100,
        help='Anzahl Test-Events'
    )
    parser.add_argument(
        '--output-dir',
        default='./verification_results',
        help='Output-Verzeichnis'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generiere Visualisierungen'
    )
    parser.add_argument(
        '--show-trajectory',
        action='store_true',
        help='Zeige Denoising-Trajektorie'
    )
    
    args = parser.parse_args()
    
    # Konfiguration
    config = VerificationConfig(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        n_test_events=args.n_test_events,
        output_dir=args.output_dir
    )
    
    print("="*70)
    print("DIFFUSION MODEL VERIFICATION")
    print("="*70)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Data: {config.data_path}")
    print(f"Test Events: {config.n_test_events}")
    print("="*70)
    
    # Lade Modell
    print("\nLade Modell...")
    
    # Versuche best model zu laden
    try:
        from load_best_model import load_best_model
        checkpoint_dir = str(Path(config.checkpoint_path).parent)
        model, model_info = load_best_model(checkpoint_dir)
        print(f"✓ Best Model geladen aus Epoch {model_info['epoch']}")
    except Exception as e:
        print(f"⚠ Konnte Best Model nicht laden: {e}")
        print(f"  Fallback: Lade spezifisches Checkpoint...")
        
        model = build_diffusion_model(
            phi_dim=config.phi_dim,
            target_dim=config.target_dim,
            T=config.T,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            t_emb_dim=config.t_emb_dim
        )
        model.load_weights(config.checkpoint_path)
        print(f"✓ Modell geladen: {model.count_params():,} Parameter")
    
    # Initialisiere Verifier
    verifier = DiffusionVerifier(model, config)
    
    # Führe Tests aus
    verifier.run_all_tests()
    
    # Visualisierungen
    if args.visualize:
        verifier.visualize_results()
    
    if args.show_trajectory:
        verifier.visualize_denoising_trajectory()
    
    print("\n" + "="*70)
    print("✓ Verifikation abgeschlossen")
    print(f"Ergebnisse: {config.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()