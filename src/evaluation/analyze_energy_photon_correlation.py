#!/usr/bin/env python3
"""
analyze_energy_photon_correlation.py

Analysiert Energie-Photon-Korrelation PRO MATERIAL
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_material_correlation(data_path, output_dir='./material_analysis'):
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Lade Daten...")
    with h5py.File(data_path, 'r') as f:
        # Lade Phi-Daten
        E_gamma = f['phi']['E_gamma_tot_keV'][()]  # (n_events,)
        mat_ids = f['phi']['matID'][()]            # (n_events,)
        
        # Lade Target (Signals)
        target_keys = sorted(f['target'].keys())
        n_events = E_gamma.shape[0]
        n_voxels = len(target_keys)
        
        print(f"  Events: {n_events:,}")
        print(f"  Voxels: {n_voxels:,}")
        
        # Berechne Photon-Summe pro Event
        photon_sums = np.zeros(n_events)
        
        for i in range(0, n_events, 10000):
            end_i = min(i + 10000, n_events)
            
            # Lade Chunk
            signals = []
            for key in target_keys:
                signals.append(f['target'][key][i:end_i])
            signals = np.array(signals).T  # (chunk_size, n_voxels)
            
            # Summiere
            photon_sums[i:end_i] = np.sum(signals, axis=1)
            
            if (i // 10000) % 10 == 0:
                print(f"  Processed {i:,} / {n_events:,} events")
    
    print("\n✓ Daten geladen")
    
    # Material-IDs mapping
    material_names = {
        0: 'no_material',
        1: 'LiquidArgon',
        2: 'Water',
        3: 'metal_steel',
        4: 'metal_copper',
        5: 'tyvek'
    }
    
    # Analysiere pro Material
    unique_mats = np.unique(mat_ids)
    print(f"\nGefundene Materialien: {unique_mats}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, mat_id in enumerate(unique_mats):
        if idx >= 6:
            break
        
        ax = axes[idx]
        
        # Filter by material
        mask = mat_ids == mat_id
        E_mat = E_gamma[mask]
        N_mat = photon_sums[mask]
        
        mat_name = material_names.get(int(mat_id), f'Unknown_{int(mat_id)}')
        
        # Scatter
        ax.scatter(E_mat, N_mat, alpha=0.3, s=5, label=f'{mat_name} (N={np.sum(mask):,})')
        
        # Linear Fit
        if len(E_mat) > 100:
            # Robust fit (RANSAC)
            from sklearn.linear_model import RANSACRegressor
            
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(E_mat.reshape(-1, 1), N_mat)
            
            E_range = np.linspace(E_mat.min(), E_mat.max(), 100)
            N_fit = ransac.predict(E_range.reshape(-1, 1))
            
            ax.plot(E_range, N_fit, 'r-', linewidth=2.5,
                   label=f'Fit: {ransac.estimator_.coef_[0]:.3f} photons/keV')
            
            # R²
            r2 = ransac.score(E_mat.reshape(-1, 1), N_mat)
            
            ax.text(0.05, 0.95, 
                   f'R² = {r2:.3f}\n'
                   f'Efficiency = {ransac.estimator_.coef_[0]:.3f} ph/keV',
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Capture Energy [keV]', fontweight='bold')
        ax.set_ylabel('Total Detected Signal', fontweight='bold')
        ax.set_title(f'{mat_name}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Energy-Photon Correlation by Material',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'energy_photon_by_material.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    data_path = "/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/outdated/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5"
    analyze_material_correlation(data_path)