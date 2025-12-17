#!/usr/bin/env python3
"""
visualize_pmt_results.py

Umfassende Visualisierung der PMT-Selektionsergebnisse
- Vergleich Fast vs. Shapley Methods
- Detection-Rate Kurven
- Importance Score Distributionen
- Voxel-Position Heatmaps
- Korrelationsanalysen

Autor: Basierend auf Diff-SPORT (Vishwasrao et al. 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
import h5py
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Matplotlib Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PMTResultsVisualizer:
    """
    Visualisiert Ergebnisse der PMT-Selektion
    """
    
    def __init__(self, 
                 fast_results_dir: str = "./fast_pmt_results",
                 shapley_results_dir: str = "./shapley_results",
                 data_path: str = None,
                 output_dir: str = "./figures"):
        
        self.fast_dir = Path(fast_results_dir)
        self.shapley_dir = Path(shapley_results_dir)
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Lade Ergebnisse
        self.fast_results = self._load_fast_results()
        self.shapley_results = self._load_shapley_results()
        
        # Lade Voxel-Geometrie (falls verfügbar)
        self.voxel_positions = self._load_voxel_positions()
        
        print(f"✓ Visualizer initialisiert")
        print(f"  Fast Results: {self.fast_results is not None}")
        print(f"  Shapley Results: {self.shapley_results is not None}")
        print(f"  Voxel Positions: {self.voxel_positions is not None}")
    
    def _load_fast_results(self) -> Optional[Dict]:
        """Lade Fast-Selection Ergebnisse"""
        results_file = self.fast_dir / "fast_pmt_selection_results.json"
        
        if not results_file.exists():
            print(f"⚠ Fast-Results nicht gefunden: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Lade zusätzliche Score-Arrays
        for score_type in ['gradient', 'reconstruction', 'hitrate', 'correlation', 'combined']:
            score_file = self.fast_dir / f"importance_{score_type}.npy"
            if score_file.exists():
                results[f'{score_type}_scores'] = np.load(score_file)
        
        return results
    
    def _load_shapley_results(self) -> Optional[Dict]:
        """Lade Shapley-Selection Ergebnisse"""
        results_file = self.shapley_dir / "pmt_selection_results.json"
        
        if not results_file.exists():
            print(f"⚠ Shapley-Results nicht gefunden: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Lade Shapley-Values
        shapley_file = self.shapley_dir / "shapley_values.npy"
        if shapley_file.exists():
            results['shapley_scores'] = np.load(shapley_file)
        
        return results
    
    def _load_voxel_positions(self) -> Optional[np.ndarray]:
        """Lade Voxel-Positionen aus HDF5 (falls verfügbar)"""
        if self.data_path is None:
            return None
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                if 'voxels' not in f:
                    return None
                
                voxel_group = f['voxels']
                voxel_keys = sorted(voxel_group.keys())
                
                positions = []
                for key in voxel_keys:
                    if 'center' in voxel_group[key]:
                        center = voxel_group[key]['center'][()]
                        positions.append(center)
                
                if len(positions) > 0:
                    return np.array(positions)  # (n_voxels, 3)
        except Exception as e:
            print(f"⚠ Konnte Voxel-Positionen nicht laden: {e}")
        
        return None
    
    # ========================================================================
    # HAUPTVISUALISIERUNGEN
    # ========================================================================
    
    def plot_detection_rate_comparison(self, save: bool = True):
        """
        Plot 1: Detection-Rate vs. Anzahl PMTs
        Vergleich Fast vs. Shapley Methods
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Fast Methods
        if self.fast_results and 'statistics' in self.fast_results:
            stats = self.fast_results['statistics']
            if 'detection_rate_curve' in stats:
                curve = stats['detection_rate_curve']
                n_pmts = np.arange(1, len(curve) + 1)
                ax.plot(n_pmts, curve, 'o-', linewidth=2.5, 
                       label='Fast Methods (Gradient + Ablation + Empirical)',
                       color='#2ecc71', markersize=4, alpha=0.8)
        
        # Shapley Methods
        if self.shapley_results and 'statistics' in self.shapley_results:
            stats = self.shapley_results['statistics']
            if 'detection_rate_curve' in stats:
                curve = stats['detection_rate_curve']
                n_pmts = np.arange(1, len(curve) + 1)
                ax.plot(n_pmts, curve, 's-', linewidth=2.5,
                       label='Shapley Value Method',
                       color='#3498db', markersize=4, alpha=0.8)
        
        # Target line: N=300 PMTs
        if self.fast_results:
            n_target = self.fast_results['config']['n_selected']
            ax.axvline(n_target, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Target: {n_target} PMTs')
        
        # Multiplicity threshold line
        if self.fast_results:
            mult_thresh = self.fast_results['config']['multiplicity']
            # Detection-Rate sollte idealerweise > 0.9 sein
            ax.axhline(0.9, color='gray', linestyle=':', 
                      linewidth=2, alpha=0.5, label='90% Detection Goal')
        
        ax.set_xlabel('Number of Selected PMTs', fontsize=13, fontweight='bold')
        ax.set_ylabel('Detection Rate', fontsize=13, fontweight='bold')
        ax.set_title('NC Detection Rate vs. PMT Count\n(Multiplicity N≥6)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(n_pmts) * 1.05)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "detection_rate_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_importance_distributions(self, save: bool = True):
        """
        Plot 2: Importance Score Distributionen
        Vergleich verschiedener Methoden
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        score_types = [
            ('gradient_scores', 'Gradient-Based Importance', '#e74c3c'),
            ('hitrate_scores', 'Empirical Hit-Rate', '#f39c12'),
            ('correlation_scores', 'NC-Parameter Correlation', '#9b59b6'),
            ('combined_scores', 'Combined Score (Ensemble)', '#2c3e50')
        ]
        
        for idx, (score_key, title, color) in enumerate(score_types):
            ax = axes[idx]
            
            if self.fast_results and score_key in self.fast_results:
                scores = np.array(self.fast_results[score_key])
                
                # Histogram
                ax.hist(scores, bins=50, alpha=0.7, color=color, 
                       edgecolor='black', linewidth=0.5)
                
                # Statistiken
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                std_score = np.std(scores)
                
                # Vertikale Linien
                ax.axvline(mean_score, color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {mean_score:.4f}')
                ax.axvline(median_score, color='blue', linestyle=':', 
                          linewidth=2, label=f'Median: {median_score:.4f}')
                
                # Top-300 Threshold
                if len(scores) >= 300:
                    threshold = np.sort(scores)[-300]
                    ax.axvline(threshold, color='green', linestyle='-', 
                              linewidth=2, alpha=0.8, 
                              label=f'Top-300 Threshold: {threshold:.4f}')
                
                ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{title}\nData not available', 
                       ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('Importance Score Distributions by Method', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "importance_distributions.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_method_correlation(self, save: bool = True):
        """
        Plot 3: Korrelation zwischen verschiedenen Methoden
        Scatter-Matrix
        """
        if not self.fast_results:
            print("⚠ Fast-Results nicht verfügbar")
            return
        
        # Sammle verfügbare Scores
        score_data = {}
        score_names = {
            'gradient_scores': 'Gradient',
            'hitrate_scores': 'Hit-Rate',
            'correlation_scores': 'Correlation',
            'combined_scores': 'Combined'
        }
        
        for key, name in score_names.items():
            if key in self.fast_results:
                score_data[name] = np.array(self.fast_results[key])
        
        if len(score_data) < 2:
            print("⚠ Nicht genug Scores für Korrelationsanalyse")
            return
        
        # Erstelle Scatter-Matrix
        n_methods = len(score_data)
        fig, axes = plt.subplots(n_methods, n_methods, figsize=(14, 14))
        
        method_names = list(score_data.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: Histogramm
                    ax.hist(score_data[method1], bins=30, alpha=0.7, 
                           color='steelblue', edgecolor='black')
                    ax.set_ylabel('Frequency', fontsize=9)
                    if i == n_methods - 1:
                        ax.set_xlabel(method1, fontsize=10, fontweight='bold')
                else:
                    # Off-diagonal: Scatter
                    scores1 = score_data[method1]
                    scores2 = score_data[method2]
                    
                    # Sample für Performance (zu viele Punkte)
                    if len(scores1) > 5000:
                        idx = np.random.choice(len(scores1), 5000, replace=False)
                        scores1 = scores1[idx]
                        scores2 = scores2[idx]
                    
                    ax.scatter(scores1, scores2, alpha=0.3, s=1, color='navy')
                    
                    # Pearson Korrelation
                    corr = np.corrcoef(score_data[method1], score_data[method2])[0, 1]
                    ax.text(0.05, 0.95, f'ρ = {corr:.3f}', 
                           transform=ax.transAxes, 
                           fontsize=10, fontweight='bold',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Labels
                if j == 0:
                    ax.set_ylabel(method1, fontsize=10, fontweight='bold')
                if i == n_methods - 1:
                    ax.set_xlabel(method2, fontsize=10, fontweight='bold')
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Correlation Matrix: Different Importance Methods', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "method_correlation.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_spatial_distribution(self, save: bool = True):
        """
        Plot 4: Räumliche Verteilung der selektierten PMTs
        3D-Scatter (falls Voxel-Positionen verfügbar)
        """
        if self.voxel_positions is None:
            print("⚠ Voxel-Positionen nicht verfügbar - überspringe 3D-Plot")
            return
        
        if not self.fast_results:
            print("⚠ Fast-Results nicht verfügbar")
            return
        
        fig = plt.figure(figsize=(16, 6))
        
        # Subplot 1: All Voxels mit Importance-Score
        ax1 = fig.add_subplot(131, projection='3d')
        
        if 'combined_scores' in self.fast_results:
            scores = np.array(self.fast_results['combined_scores'])
            
            # Normalisiere für Farbskala
            scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
            
            scatter = ax1.scatter(
                self.voxel_positions[:, 0],
                self.voxel_positions[:, 1],
                self.voxel_positions[:, 2],
                c=scores_norm,
                cmap='viridis',
                s=5,
                alpha=0.6
            )
            
            plt.colorbar(scatter, ax=ax1, label='Normalized Importance', shrink=0.7)
        
        ax1.set_xlabel('X [mm]', fontweight='bold')
        ax1.set_ylabel('Y [mm]', fontweight='bold')
        ax1.set_zlabel('Z [mm]', fontweight='bold')
        ax1.set_title('All Voxels\n(Color = Importance)', fontweight='bold')
        
        # Subplot 2: Selektierte PMTs (Fast)
        ax2 = fig.add_subplot(132, projection='3d')
        
        selected_indices = self.fast_results['selected_pmt_indices']
        selected_positions = self.voxel_positions[selected_indices]
        
        ax2.scatter(
            self.voxel_positions[:, 0],
            self.voxel_positions[:, 1],
            self.voxel_positions[:, 2],
            c='lightgray',
            s=2,
            alpha=0.2,
            label='Non-selected'
        )
        
        ax2.scatter(
            selected_positions[:, 0],
            selected_positions[:, 1],
            selected_positions[:, 2],
            c='red',
            s=20,
            alpha=0.8,
            label='Selected (Fast)'
        )
        
        ax2.set_xlabel('X [mm]', fontweight='bold')
        ax2.set_ylabel('Y [mm]', fontweight='bold')
        ax2.set_zlabel('Z [mm]', fontweight='bold')
        ax2.set_title(f'Selected PMTs: Fast Methods\n(N={len(selected_indices)})', 
                     fontweight='bold')
        ax2.legend()
        
        # Subplot 3: Vergleich Fast vs. Shapley
        ax3 = fig.add_subplot(133, projection='3d')
        
        ax3.scatter(
            self.voxel_positions[:, 0],
            self.voxel_positions[:, 1],
            self.voxel_positions[:, 2],
            c='lightgray',
            s=2,
            alpha=0.2
        )
        
        # Fast (rot)
        ax3.scatter(
            selected_positions[:, 0],
            selected_positions[:, 1],
            selected_positions[:, 2],
            c='red',
            s=20,
            alpha=0.6,
            label='Fast',
            marker='o'
        )
        
        # Shapley (blau) - falls verfügbar
        if self.shapley_results and 'selected_pmt_indices' in self.shapley_results:
            shapley_indices = self.shapley_results['selected_pmt_indices']
            shapley_positions = self.voxel_positions[shapley_indices]
            
            ax3.scatter(
                shapley_positions[:, 0],
                shapley_positions[:, 1],
                shapley_positions[:, 2],
                c='blue',
                s=20,
                alpha=0.6,
                label='Shapley',
                marker='^'
            )
            
            # Berechne Overlap
            overlap = len(set(selected_indices) & set(shapley_indices))
            overlap_pct = overlap / len(selected_indices) * 100
            
            ax3.set_title(f'Fast vs. Shapley Comparison\nOverlap: {overlap}/{len(selected_indices)} ({overlap_pct:.1f}%)', 
                         fontweight='bold')
        else:
            ax3.set_title('Selected PMTs: Fast Methods', fontweight='bold')
        
        ax3.set_xlabel('X [mm]', fontweight='bold')
        ax3.set_ylabel('Y [mm]', fontweight='bold')
        ax3.set_zlabel('Z [mm]', fontweight='bold')
        ax3.legend()
        
        plt.suptitle('Spatial Distribution of PMT Selection', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "spatial_distribution_3d.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_top_pmts_comparison(self, top_n: int = 50, save: bool = True):
        """
        Plot 5: Vergleich der Top-N PMTs zwischen Methoden
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sammle Top-N Indices verschiedener Methoden
        methods_top_pmts = {}
        
        if self.fast_results:
            if 'gradient_scores' in self.fast_results:
                scores = np.array(self.fast_results['gradient_scores'])
                methods_top_pmts['Gradient'] = np.argsort(scores)[-top_n:][::-1]
            
            if 'hitrate_scores' in self.fast_results:
                scores = np.array(self.fast_results['hitrate_scores'])
                methods_top_pmts['Hit-Rate'] = np.argsort(scores)[-top_n:][::-1]
            
            if 'combined_scores' in self.fast_results:
                scores = np.array(self.fast_results['combined_scores'])
                methods_top_pmts['Combined'] = np.argsort(scores)[-top_n:][::-1]
        
        if self.shapley_results and 'shapley_scores' in self.shapley_results:
            scores = np.array(self.shapley_results['shapley_scores'])
            methods_top_pmts['Shapley'] = np.argsort(scores)[-top_n:][::-1]
        
        if len(methods_top_pmts) < 2:
            print("⚠ Nicht genug Methoden für Vergleich")
            return
        
        # Berechne Pairwise Overlaps
        method_names = list(methods_top_pmts.keys())
        n_methods = len(method_names)
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                set1 = set(methods_top_pmts[method1])
                set2 = set(methods_top_pmts[method2])
                overlap = len(set1 & set2)
                overlap_matrix[i, j] = overlap / top_n * 100
        
        # Heatmap
        im = ax.imshow(overlap_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Overlap [%]', fontsize=12, fontweight='bold')
        
        # Annotationen
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax.text(j, i, f'{overlap_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", 
                             fontsize=11, fontweight='bold')
        
        # Achsen
        ax.set_xticks(np.arange(n_methods))
        ax.set_yticks(np.arange(n_methods))
        ax.set_xticklabels(method_names, fontsize=11, fontweight='bold')
        ax.set_yticklabels(method_names, fontsize=11, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_title(f'Overlap of Top-{top_n} PMTs Between Methods', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"top{top_n}_overlap_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_summary_statistics(self, save: bool = True):
        """
        Plot 6: Summary-Statistiken aller Methoden
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Detection Rates
        ax1 = fig.add_subplot(gs[0, :])
        methods = []
        detection_rates = []
        colors = []
        
        if self.fast_results:
            methods.append('Fast Methods')
            detection_rates.append(
                self.fast_results['statistics']['final_detection_rate']
            )
            colors.append('#2ecc71')
        
        if self.shapley_results:
            methods.append('Shapley Values')
            detection_rates.append(
                self.shapley_results['statistics']['final_detection_rate']
            )
            colors.append('#3498db')
        
        if methods:
            bars = ax1.bar(methods, detection_rates, color=colors, alpha=0.8, edgecolor='black')
            ax1.axhline(0.9, color='red', linestyle='--', linewidth=2, label='90% Goal')
            ax1.set_ylabel('Detection Rate', fontsize=12, fontweight='bold')
            ax1.set_title('Final Detection Rate (300 PMTs)', fontsize=13, fontweight='bold')
            ax1.set_ylim(0, 1.1)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Annotate bars
            for bar, rate in zip(bars, detection_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.3f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. Score Statistics (Fast Methods)
        if self.fast_results:
            score_types = ['gradient_scores', 'hitrate_scores', 'correlation_scores', 'combined_scores']
            score_labels = ['Gradient', 'Hit-Rate', 'Correlation', 'Combined']
            
            available_scores = []
            available_labels = []
            
            for score_type, label in zip(score_types, score_labels):
                if score_type in self.fast_results:
                    available_scores.append(np.array(self.fast_results[score_type]))
                    available_labels.append(label)
            
            if available_scores:
                # Box plot
                ax2 = fig.add_subplot(gs[1, :2])
                bp = ax2.boxplot(available_scores, labels=available_labels, 
                                patch_artist=True, showmeans=True)
                
                # Farben
                box_colors = ['#e74c3c', '#f39c12', '#9b59b6', '#2c3e50']
                for patch, color in zip(bp['boxes'], box_colors[:len(available_scores)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
                ax2.set_title('Score Distributions by Method', fontsize=13, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
        
        # 3. Top-10 PMTs
        ax3 = fig.add_subplot(gs[1, 2])
        if self.fast_results and 'combined_scores' in self.fast_results:
            scores = np.array(self.fast_results['combined_scores'])
            top10_indices = np.argsort(scores)[-10:][::-1]
            top10_scores = scores[top10_indices]
            
            y_pos = np.arange(len(top10_indices))
            ax3.barh(y_pos, top10_scores, color='steelblue', alpha=0.8, edgecolor='black')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f'Voxel {idx}' for idx in top10_indices], fontsize=9)
            ax3.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax3.set_title('Top-10 PMTs\n(Combined Score)', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. Method Comparison Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        
        if self.fast_results:
            stats = self.fast_results['statistics']
            table_data.append([
                'Fast Methods',
                f"{stats['n_selected']}",
                f"{stats['final_detection_rate']:.3f}",
                f"{stats.get('mean_importance', 0):.4f}",
                '~1-2h'
            ])
        
        if self.shapley_results:
            stats = self.shapley_results['statistics']
            table_data.append([
                'Shapley Values',
                f"{stats['n_selected']}",
                f"{stats['final_detection_rate']:.3f}",
                f"{stats['mean_shapley']:.4f}",
                '~8-24h'
            ])
        
        if table_data:
            table = ax4.table(
                cellText=table_data,
                colLabels=['Method', 'N PMTs', 'Detection Rate', 'Mean Score', 'Runtime'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.15, 0.2, 0.2, 0.2]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)

            # Header styling
            for i in range(5):
                table[(0, i)].set_facecolor('#34495e')
                table[(0, i)].set_text_props(weight='bold', color='white')

        plt.suptitle(
            'PMT Selection: Summary Statistics',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        if save:
            output_path = self.output_dir / "summary_statistics.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")

        plt.show()

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

    def generate_all_plots(self):
        """Generate all visualizations at once."""
        print("\n" + "="*70)
        print("Generating all visualizations...")
        print("="*70 + "\n")

        plots = [
            ("Detection Rate Comparison", self.plot_detection_rate_comparison),
            ("Importance Distributions", self.plot_importance_distributions),
            ("Method Correlation", self.plot_method_correlation),
            ("Spatial Distribution", self.plot_spatial_distribution),
            ("Top PMTs Comparison", self.plot_top_pmts_comparison),
            ("Summary Statistics", self.plot_summary_statistics)
        ]

        for plot_name, plot_func in plots:
            print(f"\n[{plot_name}]")
            try:
                plot_func(save=True)
            except Exception as e:
                print(f"⚠ Error in {plot_name}: {e}")

        print("\n" + "="*70)
        print(f"✓ All plots saved in: {self.output_dir}")
        print("="*70)

    def print_summary(self):
        """Print summary of the results."""
        print("\n" + "="*70)
        print("PMT SELECTION RESULTS SUMMARY")
        print("="*70)

        if self.fast_results:
            print("\n[FAST METHODS]")
            print(f"  Selected PMTs: {self.fast_results['statistics']['n_selected']}")
            print(f"  Detection Rate: {self.fast_results['statistics']['final_detection_rate']:.3f}")
            print(f"  Mean Importance: {self.fast_results['statistics'].get('mean_importance', 0):.4f}")

            if 'combined_scores' in self.fast_results:
                scores = np.array(self.fast_results['combined_scores'])
                print(f"  Score Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

        if self.shapley_results:
            print("\n[SHAPLEY VALUES]")
            print(f"  Selected PMTs: {self.shapley_results['statistics']['n_selected']}")
            print(f"  Detection Rate: {self.shapley_results['statistics']['final_detection_rate']:.3f}")
            print(f"  Mean Shapley: {self.shapley_results['statistics']['mean_shapley']:.4f}")

            if 'shapley_scores' in self.shapley_results:
                scores = np.array(self.shapley_results['shapley_scores'])
                print(f"  Score Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

        if self.fast_results and self.shapley_results:
            print("\n[COMPARISON]")
            fast_pmts = set(self.fast_results['selected_pmt_indices'])
            shapley_pmts = set(self.shapley_results['selected_pmt_indices'])
            overlap = len(fast_pmts & shapley_pmts)
            overlap_pct = overlap / len(fast_pmts) * 100

            print(f"  PMT Overlap: {overlap}/{len(fast_pmts)} ({overlap_pct:.1f}%)")
            print(
                f"  Detection Rate Diff: "
                f"{abs(self.fast_results['statistics']['final_detection_rate'] - self.shapley_results['statistics']['final_detection_rate']):.3f}"
            )

        print("\n" + "="*70)


# ========================================================================
# MAIN FUNCTION
# ========================================================================

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize PMT selection results'
    )
    parser.add_argument(
        '--fast-dir',
        default='./fast_pmt_results',
        help='Directory with Fast-Selection results'
    )
    parser.add_argument(
        '--shapley-dir',
        default='./shapley_results',
        help='Directory with Shapley-Selection results'
    )
    parser.add_argument(
        '--data-path',
        default=(
            '/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/'
            'resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5'
        ),
        help='Path to HDF5 data file (for voxel positions)'
    )
    parser.add_argument(
        '--output-dir',
        default='./figures',
        help='Directory for saved plots'
    )
    parser.add_argument(
        '--plot',
        choices=['all', 'detection', 'distribution', 'correlation',
                 'spatial', 'comparison', 'summary'],
        default='all',
        help='Which plots to generate'
    )

    args = parser.parse_args()

    # Initialize Visualizer
    viz = PMTResultsVisualizer(
        fast_results_dir=args.fast_dir,
        shapley_results_dir=args.shapley_dir,
        data_path=args.data_path,
        output_dir=args.output_dir
    )

    # Print summary
    viz.print_summary()

    # Generate selected plots
    if args.plot == 'all':
        viz.generate_all_plots()
    elif args.plot == 'detection':
        viz.plot_detection_rate_comparison()
    elif args.plot == 'distribution':
        viz.plot_importance_distributions()
    elif args.plot == 'correlation':
        viz.plot_method_correlation()
    elif args.plot == 'spatial':
        viz.plot_spatial_distribution()
    elif args.plot == 'comparison':
        viz.plot_top_pmts_comparison()
    elif args.plot == 'summary':
        viz.plot_summary_statistics()


if __name__ == "__main__":
    main()
