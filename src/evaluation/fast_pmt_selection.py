# fast_pmt_selection_fixed.py
"""
Schnelle PMT-Selektion mit reduziertem Computational Budget
Alternative zu vollständiger Shapley-Berechnung

Nutzt drei schnellere Ansätze:
1. Gradient-basierte Importance (Backprop durch Diffusionsmodell)
2. Reconstruction-basierte Importance (Ablation)
3. Empirische Hit-Rate Analyse (datengetrieben)

Laufzeit: ~1-2h statt mehrere Tage

Referenzen:
- Gradient: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
- Ablation: Zeiler & Fergus (2014) "Visualizing and Understanding CNNs"
- Empirisch: Datenanalyse-Standard
"""

import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
from models.diffusion_model import build_diffusion_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@dataclass
class FastSelectionConfig:
    """Konfiguration für schnelle PMT-Selektion"""
    # Model
    checkpoint_path: str = "./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5"
    phi_dim: int = 22
    target_dim: int = 7789
    T: int = 1000
    hidden_dim: int = 512
    n_layers: int = 4
    t_emb_dim: int = 32
    
    # Fast selection parameters
    n_test_events: int = 500           # Mehr Events als Shapley (schneller!)
    n_gradient_samples: int = 10       # Timesteps für Gradient-Analyse
    n_ablation_voxels: int = 100       # Sample für Ablation (nicht alle!)
    
    # Selection
    n_pmts_to_select: int = 300
    multiplicity_threshold: int = 6
    hit_threshold: float = 0.5
    
    # Methods to use
    use_gradient_importance: bool = True
    use_reconstruction_importance: bool = True
    use_empirical_hitrate: bool = True
    
    # Output
    output_dir: str = "./fast_pmt_results"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class GradientImportanceCalculator:
    """
    Methode 1: Gradient-basierte Feature Importance
    
    Idee: Berechne ||∂L/∂x_i|| für jeden Voxel i
    Voxel mit hohen Gradienten sind wichtig für die Rekonstruktion
    
    Referenz: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
              https://arxiv.org/abs/1703.01365
    """
    
    def __init__(self, model: tf.keras.Model, config: FastSelectionConfig):
        self.model = model
        self.config = config
    
    @tf.function
    def compute_gradient_norm(self, 
                             x: tf.Tensor, 
                             phi: tf.Tensor, 
                             t: tf.Tensor) -> tf.Tensor:
        """
        Berechne ||∂noise_pred/∂x_i|| für jeden Voxel i
        
        Args:
            x: Input signal (batch, target_dim)
            phi: Conditioning (batch, phi_dim)
            t: Timestep (batch,)
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            noise_pred = self.model([x, phi, t], training=False)
            
            # Loss: Summe über Output (als Proxy für Wichtigkeit)
            loss = tf.reduce_sum(noise_pred ** 2)
        
        # Gradienten bezüglich Input x
        gradients = tape.gradient(loss, x)
        
        # L2-Norm der Gradienten pro Voxel
        gradient_norms = tf.abs(gradients)  # (batch, target_dim)
        
        return gradient_norms
    
    def compute_importance_scores(self, test_data: List[Dict]) -> np.ndarray:
        """
        Berechne Gradient-basierte Importance für alle Voxel
        
        Mittelt über:
        - Verschiedene Test-Events
        - Verschiedene Timesteps (früh, mittel, spät im Diffusion-Prozess)
        
        Returns:
            importance_scores: (target_dim,) - höher = wichtiger
        """
        n_voxels = self.config.target_dim
        importance_scores = np.zeros(n_voxels)
        
        # Timesteps zum Samplen (über den ganzen Diffusion-Prozess)
        timesteps = np.linspace(
            0, self.config.T - 1, self.config.n_gradient_samples
        ).astype(np.int32)
        
        print(f"Berechne Gradient-Importance über {len(test_data)} Events...")
        
        for event_idx, event in enumerate(tqdm(test_data, desc="Events")):
            x = tf.constant(event['signal'], dtype=tf.float32)
            phi = tf.constant(event['phi'], dtype=tf.float32)
            
            # Erweitere Dimensionen für Batch
            x = tf.expand_dims(x, 0)  # (1, target_dim)
            phi = tf.expand_dims(phi, 0)  # (1, phi_dim)
            
            event_importance = np.zeros(n_voxels)
            
            # Sample verschiedene Timesteps
            for t in timesteps:
                t_tensor = tf.constant([t], dtype=tf.int32)
                
                # Berechne Gradienten
                gradient_norms = self.compute_gradient_norm(x, phi, t_tensor)
                event_importance += gradient_norms.numpy()[0]
            
            # Normalisiere über Timesteps
            event_importance /= len(timesteps)
            importance_scores += event_importance
        
        # Durchschnitt über Events
        importance_scores /= len(test_data)
        
        return importance_scores


class ReconstructionImportanceCalculator:
    """
    Methode 2: Ablation-basierte Importance
    
    Idee: Für jeden Voxel i:
          - Maskiere Voxel i (setze auf 0)
          - Rekonstruiere Signal
          - Messe Rekonstruktionsfehler
          
    Hoher Fehler = Voxel ist wichtig
    
    Referenz: Zeiler & Fergus (2014) "Visualizing and Understanding CNNs"
              https://arxiv.org/abs/1311.2901
    """
    
    def __init__(self, model: tf.keras.Model, config: FastSelectionConfig):
        self.model = model
        self.config = config
        
        # Diffusion schedule
        self.betas = tf.linspace(1e-4, 0.02, config.T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
    
    def add_noise(self, x: tf.Tensor, t: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Füge Noise gemäß Diffusion-Schedule hinzu"""
        noise = tf.random.normal(shape=x.shape)
        alpha_t = self.alphas_cumprod[t]
        
        x_noisy = tf.sqrt(alpha_t) * x + tf.sqrt(1 - alpha_t) * noise
        return x_noisy, noise
    
    @tf.function
    def reconstruct_with_mask(self,
                             x: tf.Tensor,
                             phi: tf.Tensor,
                             t: tf.Tensor,
                             mask: tf.Tensor) -> tf.Tensor:
        """
        Rekonstruiere Noise mit maskiertem Input
        
        Args:
            x: Noisy input (batch, target_dim)
            phi: Conditioning (batch, phi_dim)
            t: Timestep (batch,)
            mask: (batch, target_dim) Boolean - False = Voxel ist maskiert
        """
        # Maskiere Input
        x_masked = x * tf.cast(mask, tf.float32)
        
        # Noise-Prädiktion
        noise_pred = self.model([x_masked, phi, t], training=False)
        
        return noise_pred
    
    def compute_ablation_importance(self, 
                                   test_data: List[Dict],
                                   sample_size: int = 100) -> np.ndarray:
        """
        Berechne Importance durch systematisches Maskieren
        
        WICHTIG: Sample nur sample_size Voxel (nicht alle 7789!)
        
        Returns:
            importance_scores: (target_dim,) - höher = wichtiger
        """
        n_voxels = self.config.target_dim
        importance_scores = np.zeros(n_voxels)
        voxel_sample_counts = np.zeros(n_voxels)
        
        # Sample zufällige Voxel-Indices für Ablation
        voxel_indices_to_test = np.random.choice(
            n_voxels, 
            size=min(sample_size, n_voxels), 
            replace=False
        )
        
        print(f"Berechne Ablation-Importance für {len(voxel_indices_to_test)} Voxel...")
        
        # Mittlerer Timestep für Rekonstruktion
        t_mid = self.config.T // 2
        
        for voxel_idx in tqdm(voxel_indices_to_test, desc="Voxel"):
            voxel_errors = []
            
            # Reduziere Events für Geschwindigkeit
            events_to_use = min(100, len(test_data))
            
            for event in test_data[:events_to_use]:
                x_true = tf.constant(event['signal'], dtype=tf.float32)
                phi = tf.constant(event['phi'], dtype=tf.float32)
                
                # Füge Noise hinzu
                x_noisy, true_noise = self.add_noise(tf.expand_dims(x_true, 0), t_mid)
                phi = tf.expand_dims(phi, 0)
                
                # Baseline: Rekonstruktion OHNE Maskierung
                t_tensor = tf.constant([t_mid], dtype=tf.int32)
                noise_pred_full = self.model([x_noisy, phi, t_tensor], training=False)
                error_full = tf.reduce_mean((noise_pred_full - true_noise) ** 2)
                
                # Rekonstruktion MIT Maskierung von Voxel i
                mask = tf.ones((1, n_voxels), dtype=tf.bool)
                mask = tf.tensor_scatter_nd_update(
                    mask, [[0, voxel_idx]], [False]
                )
                
                noise_pred_masked = self.reconstruct_with_mask(
                    x_noisy, phi, t_tensor, mask
                )
                error_masked = tf.reduce_mean((noise_pred_masked - true_noise) ** 2)
                
                # Importance = Wie viel schlechter wird Rekonstruktion?
                importance = (error_masked - error_full).numpy()
                voxel_errors.append(importance)
            
            # Durchschnittliche Importance für diesen Voxel
            importance_scores[voxel_idx] = np.mean(voxel_errors)
            voxel_sample_counts[voxel_idx] = len(voxel_errors)
        
        # Für nicht-getestete Voxel: Interpoliere
        untested = voxel_sample_counts == 0
        if np.any(untested):
            mean_importance = np.mean(importance_scores[~untested])
            importance_scores[untested] = mean_importance * 0.5  # Konservativ
        
        return importance_scores


class EmpiricalHitRateAnalyzer:
    """
    Methode 3: Daten-getriebene Hit-Rate Analyse
    
    Idee: Analysiere empirisch welche Voxel am häufigsten aktiv sind
          bei erfolgreichen NC-Detektionen
    
    Vorteil: Sehr schnell, keine Model-Inference nötig
    """
    
    def __init__(self, config: FastSelectionConfig):
        self.config = config
    
    def compute_hitrate_scores(self, test_data: List[Dict]) -> np.ndarray:
        """
        Berechne mehrere empirische Scores:
        1. Hit-Frequenz: Wie oft ist Voxel > threshold?
        2. Signal-Stärke: Durchschnittliches Signal wenn aktiv
        3. Multiplizitäts-Beitrag: Wie oft trägt Voxel zur N=6 bei?
        """
        n_voxels = self.config.target_dim
        
        # Statistiken sammeln
        hit_counts = np.zeros(n_voxels)
        signal_sums = np.zeros(n_voxels)
        multiplicity_contributions = np.zeros(n_voxels)
        
        print(f"Analysiere empirische Hit-Rates über {len(test_data)} Events...")
        
        for event in tqdm(test_data, desc="Events"):
            signal = event['signal']
            
            # Hit-Detection
            hits = signal > self.config.hit_threshold
            hit_counts[hits] += 1
            
            # Signal-Stärke
            signal_sums += signal
            
            # Multiplizitäts-Beitrag
            n_hits = np.sum(hits)
            if n_hits >= self.config.multiplicity_threshold:
                # Event ist detektierbar
                multiplicity_contributions[hits] += 1
        
        n_events = len(test_data)
        
        # Score 1: Hit-Frequenz (normalisiert)
        hit_frequency = hit_counts / n_events
        
        # Score 2: Durchschnittliche Signal-Stärke
        avg_signal = signal_sums / n_events
        
        # Score 3: Multiplizitäts-Relevanz
        multiplicity_relevance = multiplicity_contributions / n_events
        
        # Kombiniere Scores (gewichtet)
        combined_score = (
            0.3 * hit_frequency +
            0.2 * (avg_signal / (np.max(avg_signal) + 1e-8)) +
            0.5 * multiplicity_relevance
        )
        
        return combined_score
    
    def compute_correlation_scores(self, test_data: List[Dict]) -> np.ndarray:
        """
        Berechne Korrelation zwischen Voxel-Signal und NC-Parametern
        
        Idee: Voxel die stark mit NC-Position korrelieren sind wichtig
        """
        n_voxels = self.config.target_dim
        n_params = self.config.phi_dim
        
        # Sammle Daten
        signals = np.array([event['signal'] for event in test_data])  # (n_events, n_voxels)
        params = np.array([event['phi'] for event in test_data])  # (n_events, n_params)
        
        # Berechne Korrelation für jeden Voxel mit allen Parametern
        correlations = np.zeros((n_voxels, n_params))
        
        print("Berechne Voxel-Parameter Korrelationen...")
        for i in tqdm(range(n_voxels), desc="Voxel"):
            for j in range(n_params):
                # Pearson Korrelation
                corr = np.corrcoef(signals[:, i], params[:, j])[0, 1]
                correlations[i, j] = np.abs(corr) if not np.isnan(corr) else 0.0
        
        # Maximale Korrelation über alle Parameter
        max_correlation = np.max(correlations, axis=1)
        
        return max_correlation


class PMTSelector:
    """
    Attention-based PMT selection with multiplicity constraint
    
    Reference: Vishwasrao et al. (2025) - Diff-SPORT
    """
    
    def select_pmts_with_multiplicity_constraint(self, 
                                                  model, 
                                                  test_data,
                                                  n_pmts=300,
                                                  multiplicity_threshold=6,
                                                  hit_threshold=0.5):
        """
        Greedy PMT selection ensuring multiplicity constraint
        
        Strategy:
        1. Compute importance scores (attention/Shapley)
        2. Select PMTs greedily
        3. At each step, verify multiplicity condition
        4. Stop when both criteria met:
           - N = 300 PMTs selected
           - Detection rate > 90% (multiplicity ≥ 6)
        """
        
        # Compute attention scores (importance)
        importance_scores = self._compute_attention_scores(model, test_data)
        
        # Sort PMTs by importance
        sorted_pmts = np.argsort(importance_scores)[::-1]
        
        selected_pmts = []
        detection_rates = []
        
        for pmt_idx in sorted_pmts:
            selected_pmts.append(pmt_idx)
            
            # Test detection rate with current PMT set
            n_detected = 0
            for event in test_data:
                # Simulate detection: count active PMTs
                signal = event['signal']
                active_pmts = np.sum(signal[selected_pmts] > hit_threshold)
                
                if active_pmts >= multiplicity_threshold:
                    n_detected += 1
            
            detection_rate = n_detected / len(test_data)
            detection_rates.append(detection_rate)
            
            # Check stopping criteria
            if len(selected_pmts) >= n_pmts and detection_rate > 0.9:
                print(f"✓ Found optimal set: {len(selected_pmts)} PMTs, "
                      f"Detection rate: {detection_rate:.3f}")
                break
        
        return np.array(selected_pmts), detection_rates


def load_test_data(data_path: str, n_events: int = 500) -> List[Dict]:
    """
    Lade Test-Events KORREKT aus HDF5
    
    WICHTIG: Ein Event = Signal über ALLE Voxel
    """
    print(f"Lade Test-Daten: {data_path}")
    
    test_data = []
    
    with h5py.File(data_path, 'r') as f:
        phi_group = f['phi']
        target_group = f['target']
        
        # Keys sortieren
        phi_keys = sorted(phi_group.keys())
        voxel_keys = sorted(target_group.keys())
        
        # Anzahl verfügbare Events
        n_total = phi_group[phi_keys[0]].shape[0]
        print(f"  Verfügbare Events: {n_total:,}")
        
        # Sample zufällige Event-Indices
        event_indices = np.random.choice(
            n_total, 
            min(n_events, n_total), 
            replace=False
        )
        
        print(f"  Lade {len(event_indices)} Events...")
        
        for event_idx in tqdm(event_indices, desc="Events laden"):
            # Phi: Sammle alle 22 Parameter für dieses Event
            phi = np.array([
                phi_group[key][event_idx] 
                for key in phi_keys
            ], dtype=np.float32)
            
            # Signal: Sammle über alle Voxel für dieses Event
            signal = np.array([
                target_group[key][event_idx] 
                for key in voxel_keys
            ], dtype=np.float32)
            
            test_data.append({
                'signal': signal,  # (7789,)
                'phi': phi         # (22,)
            })
    
    print(f"  ✓ {len(test_data)} Events geladen")
    print(f"  Signal shape: {test_data[0]['signal'].shape}")
    print(f"  Phi shape: {test_data[0]['phi'].shape}")
    
    return test_data


def main():
    """Hauptfunktion: Schnelle PMT-Selektion"""
    
    # Konfiguration
    config = FastSelectionConfig(
        checkpoint_path="./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5",
        phi_dim=22,
        target_dim=7789,
        n_test_events=500,           # Viele Events (schnelle Methoden!)
        n_gradient_samples=10,
        n_ablation_voxels=100,       # Sample für Ablation
        n_pmts_to_select=300,
        multiplicity_threshold=6,
        # Alle Methoden aktivieren
        use_gradient_importance=True,
        use_reconstruction_importance=True,
        use_empirical_hitrate=True,
        output_dir="./fast_pmt_results"
    )
    
    print("=" * 70)
    print("Schnelle PMT-Selektion (Budget-optimiert) - KORRIGIERT")
    print("=" * 70)
    print(f"Methoden:")
    print(f"  [✓] Gradient-basiert: {config.use_gradient_importance}")
    print(f"  [✓] Reconstruction-basiert: {config.use_reconstruction_importance}")
    print(f"  [✓] Empirische Hit-Rate: {config.use_empirical_hitrate}")
    print(f"Test Events: {config.n_test_events}")
    print(f"Ablation Voxel-Sample: {config.n_ablation_voxels}")
    print(f"Geschätzte Laufzeit: ~1-2 Stunden")
    print("=" * 70)
    
    # 1. Lade Modell
    print("\n[1/4] Lade Modell...")
    model = build_diffusion_model(
        phi_dim=config.phi_dim,
        target_dim=config.target_dim,
        T=config.T,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        t_emb_dim=config.t_emb_dim
    )
    model.load_weights(config.checkpoint_path)
    print(f"  ✓ Geladen: {model.count_params():,} Parameter")
    
    # 2. Lade Daten KORREKT
    print("\n[2/4] Lade Test-Daten...")
    test_data = load_test_data(
        "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/"
        "resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5",
        n_events=config.n_test_events
    )
    
    # 3. Berechne Importance-Scores
    print("\n[3/4] Berechne Importance-Scores...")
    selector = FastPMTSelector(model, config)
    all_scores = selector.compute_combined_importance(test_data)
    
    # Speichere alle Scores
    for score_type, scores in all_scores.items():
        score_path = os.path.join(config.output_dir, f"importance_{score_type}.npy")
        np.save(score_path, scores)
        print(f"  ✓ Gespeichert: {score_path}")
    
    # 4. Selektiere PMTs
    print("\n[4/4] Selektiere optimale PMTs...")
    selected_pmts, stats = selector.select_pmts_with_multiplicity(
        all_scores['combined'], test_data
    )
    
    # Speichere Ergebnisse
    results = {
        'selected_pmt_indices': selected_pmts.tolist(),
        'importance_scores': {k: v.tolist() for k, v in all_scores.items()},
        'statistics': stats,
        'config': {
            'n_voxels': config.target_dim,
            'n_selected': config.n_pmts_to_select,
            'multiplicity': config.multiplicity_threshold,
            'methods_used': {
                'gradient': config.use_gradient_importance,
                'reconstruction': config.use_reconstruction_importance,
                'empirical': config.use_empirical_hitrate
            }
        }
    }
    
    results_path = os.path.join(config.output_dir, "fast_pmt_selection_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ERGEBNISSE")
    print("=" * 70)
    print(f"Selektierte PMTs: {len(selected_pmts)}")
    print(f"Detection Rate: {stats['final_detection_rate']:.3f}")
    print(f"Combined Importance (Mean): {np.mean(all_scores['combined']):.6f}")
    print(f"\nTop-10 wichtigste PMTs (Indices):")
    top10 = selected_pmts[:10]
    for i, idx in enumerate(top10):
        print(f"  {i+1}. Voxel {idx}: Importance = {all_scores['combined'][idx]:.6f}")
    print(f"\nErgebnisse: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()