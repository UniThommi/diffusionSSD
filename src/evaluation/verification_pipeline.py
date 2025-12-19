# verification_pipeline.py - Systematic verification workflow

class PhysicsVerificationPipeline:
    """
    State-of-the-art verification for physics-informed diffusion models
    
    References:
    - Karras et al. (2022): "Elucidating the Design Space of Diffusion Models"
    - Thuerey et al. (2021): "Physics-based Deep Learning" 
    - Cranmer et al. (2020): "The frontier of simulation-based inference"
    """
    
    def __init__(self, checkpoint_dir="./checkpoints_cpu"):
        # CRITICAL: Always use best validated model
        from load_best_model import load_best_model
        
        self.model, self.model_info = load_best_model(checkpoint_dir)
        self.checkpoint_epoch = self.model_info['epoch']
        self.checkpoint_loss = self.model_info['loss']
        
        print(f"✓ Loaded best model: Epoch {self.checkpoint_epoch}, Loss {self.checkpoint_loss:.6f}")
        
        # Load test data (held-out, never seen during training!)
        self.test_data = self._load_test_data()
        
        # Initialize verification modules
        self.fidelity_verifier = ReconstructionFidelityVerifier(self.model)
        self.physics_verifier = PhysicsConstraintVerifier(self.model, self.test_data)
        self.consistency_verifier = ConditionalConsistencyVerifier(self.model)
        
    def run_full_verification(self, output_dir="./verification_results"):
        """
        Complete verification pipeline with visualization-first philosophy
        
        Best Practices (ML Physics):
        1. Always use held-out test set (never training data!)
        2. Verify against ground truth physics, not just metrics
        3. Visualize everything - numbers alone can be misleading
        4. Compare to baselines (random, simple models)
        5. Test edge cases and failure modes
        """
        
        results = {}
        
        # Phase 1: Reconstruction Quality
        print("\n[Phase 1/4] Reconstruction Fidelity...")
        results['fidelity'] = self.fidelity_verifier.verify_and_visualize(
            self.test_data, output_dir
        )
        
        # Phase 2: Physics Constraints (MAIN FOCUS)
        print("\n[Phase 2/4] Physics Constraints...")
        results['physics'] = self.physics_verifier.verify_and_visualize(
            output_dir
        )
        
        # Phase 3: Conditional Consistency
        print("\n[Phase 3/4] Conditional Consistency...")
        results['consistency'] = self.consistency_verifier.verify_and_visualize(
            self.test_data, output_dir
        )
        
        # Phase 4: Summary Report
        print("\n[Phase 4/4] Generating Summary Report...")
        self._generate_summary_report(results, output_dir)
        
        return results

class VerificationPlotLibrary:
    """
    Comprehensive visualization suite for diffusion model verification
    
    References:
    - Tukey (1977): "Exploratory Data Analysis"
    - Cleveland (1993): "Visualizing Data"
    """
    
    @staticmethod
    def plot_reconstruction_quality(x_true, x_pred, voxel_ids, save_path):
        """
        Plot 1: Signal Reconstruction Comparison
        
        Purpose: Verify model can reconstruct voxel activation patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top-left: Direct Comparison (Scatter)
        ax = axes[0, 0]
        ax.scatter(x_true, x_pred, alpha=0.3, s=1, c='navy')
        ax.plot([0, x_true.max()], [0, x_true.max()], 'r--', 
                linewidth=2, label='Perfect Reconstruction')
        
        # Compute metrics
        mse = np.mean((x_true - x_pred)**2)
        r2 = 1 - (np.sum((x_true - x_pred)**2) / 
                  np.sum((x_true - np.mean(x_true))**2))
        
        ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.3f}',
                transform=ax.transAxes, fontsize=11, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('True Signal', fontweight='bold')
        ax.set_ylabel('Predicted Signal', fontweight='bold')
        ax.set_title('Voxel-Level Reconstruction', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Top-right: Residual Distribution
        ax = axes[0, 1]
        residuals = x_pred - x_true
        ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', 
                edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (Pred - True)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Residual Distribution\n(should be centered at 0)', 
                     fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Bottom-left: Per-Voxel Error Profile
        ax = axes[1, 0]
        voxel_errors = np.abs(x_pred - x_true)
        
        # Group by voxel position (spatial pattern?)
        ax.plot(voxel_ids, voxel_errors, 'o', alpha=0.3, markersize=2)
        
        # Running average
        window = 50
        running_avg = np.convolve(voxel_errors, 
                                  np.ones(window)/window, 
                                  mode='valid')
        ax.plot(voxel_ids[window-1:], running_avg, 'r-', 
                linewidth=2, label='Running Average')
        
        ax.set_xlabel('Voxel ID', fontweight='bold')
        ax.set_ylabel('Absolute Error', fontweight='bold')
        ax.set_title('Spatial Error Profile', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Bottom-right: Quantile-Quantile Plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot: Residuals vs Normal', fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()

class PhysicsConstraintVerifier:
    """
    Material-aware physics verification for multi-component detector
    
    Key Insight: Energy-light correlation depends on:
    1. Capture material (H2O vs LAr vs Steel)
    2. Optical path length through shielding
    3. Distance from capture to detector voxels
    
    References:
    - Agostinelli et al. (2003): GEANT4 - Optical physics
    - Saldanha et al. (2009): "Model Independent Approach to Precision Optical Calibration"
    """
    
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        
        # Load detector geometry
        self.voxel_positions = self._load_voxel_geometry()
        self.material_map = self._load_material_map()
        
    def verify_energy_conservation(self, output_dir):
        """
        Material-dependent energy-photon correlation analysis
        
        Strategy:
        1. Cluster events by capture material
        2. Within each material, verify E_gamma ∝ N_photons
        3. Compare optical efficiency across materials
        """
        
        results = {}
        
        # Generate predictions for all test events
        print("  Generating model predictions...")
        predictions = self._generate_predictions()
        
        # Extract physics quantities
        energies = self._extract_energies()  # E_gamma_tot_keV
        photon_yields = self._compute_photon_yields(predictions)
        materials = self._identify_capture_materials()
        shielding_depths = self._compute_shielding_depths()
        
        # Material-dependent analysis
        unique_materials = np.unique(materials)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for idx, material in enumerate(unique_materials):
            if idx >= 6:
                break
                
            ax = axes.flatten()[idx]
            
            # Filter events by material
            mask = materials == material
            E_mat = energies[mask]
            N_mat = photon_yields[mask]
            
            # Scatter plot
            ax.scatter(E_mat, N_mat, alpha=0.4, s=20, 
                      label=f'{material} (N={np.sum(mask)})')
            
            # Fit linear model (robust regression)
            from sklearn.linear_model import RANSACRegressor
            
            X = E_mat.reshape(-1, 1)
            y = N_mat
            
            ransac = RANSACRegressor()
            ransac.fit(X, y)
            
            # Plot fit
            E_range = np.linspace(E_mat.min(), E_mat.max(), 100)
            N_fit = ransac.predict(E_range.reshape(-1, 1))
            
            ax.plot(E_range, N_fit, 'r-', linewidth=2.5, 
                   label=f'Fit: N = {ransac.estimator_.coef_[0]:.2f} × E')
            
            # Compute optical efficiency
            eff = ransac.estimator_.coef_[0]  # photons/keV
            r2 = ransac.score(X, y)
            
            results[material] = {
                'efficiency': eff,
                'r2_score': r2,
                'n_events': np.sum(mask)
            }
            
            ax.set_xlabel('Capture Energy [keV]', fontweight='bold')
            ax.set_ylabel('Detected Photons', fontweight='bold')
            ax.set_title(f'{material}\nEff={eff:.3f} photons/keV, R²={r2:.3f}', 
                        fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Material-Dependent Energy-Photon Correlation', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'energy_conservation_by_material.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
        
        # Additional plot: Shielding effect
        self._plot_shielding_effect(energies, photon_yields, 
                                    shielding_depths, output_dir)
        
        return results
    
    def _plot_shielding_effect(self, energies, photon_yields, 
                               shielding_depths, output_dir):
        """
        Plot 2: Optical Attenuation vs Shielding Depth
        
        Physics: Photons traversing steel → exponential attenuation
        Expected: N_detected ∝ exp(-μ × d) where d = shielding depth
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Photon yield vs shielding depth
        ax = axes[0]
        
        # Bin by shielding depth
        depth_bins = np.linspace(0, shielding_depths.max(), 10)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        
        mean_yields = []
        std_yields = []
        
        for i in range(len(depth_bins) - 1):
            mask = (shielding_depths >= depth_bins[i]) & \
                   (shielding_depths < depth_bins[i+1])
            
            if np.sum(mask) > 0:
                mean_yields.append(np.mean(photon_yields[mask]))
                std_yields.append(np.std(photon_yields[mask]))
            else:
                mean_yields.append(np.nan)
                std_yields.append(np.nan)
        
        mean_yields = np.array(mean_yields)
        std_yields = np.array(std_yields)
        
        ax.errorbar(bin_centers, mean_yields, yerr=std_yields, 
                   fmt='o-', linewidth=2, markersize=8, 
                   capsize=5, label='Data')
        
        # Fit exponential decay
        valid = ~np.isnan(mean_yields)
        if np.sum(valid) > 3:
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, mu):
                return a * np.exp(-mu * x)
            
            try:
                popt, _ = curve_fit(exp_decay, 
                                   bin_centers[valid], 
                                   mean_yields[valid])
                
                x_fit = np.linspace(0, shielding_depths.max(), 100)
                y_fit = exp_decay(x_fit, *popt)
                
                ax.plot(x_fit, y_fit, 'r--', linewidth=2.5,
                       label=f'Fit: N = {popt[0]:.1f} × exp(-{popt[1]:.3f} × d)')
                
                # Extract attenuation coefficient
                mu = popt[1]  # cm^-1
                ax.text(0.05, 0.95, 
                       f'Attenuation: μ = {mu:.3f} cm⁻¹\n'
                       f'(Expected for steel: ~0.1-0.3 cm⁻¹)',
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except:
                pass
        
        ax.set_xlabel('Shielding Depth [cm]', fontweight='bold')
        ax.set_ylabel('Mean Detected Photons', fontweight='bold')
        ax.set_title('Optical Attenuation through Steel', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Right: 2D histogram (Energy vs Shielding)
        ax = axes[1]
        
        # Normalize photon yields by energy (optical efficiency)
        efficiency = photon_yields / (energies + 1e-6)
        
        h = ax.hist2d(shielding_depths, efficiency, 
                     bins=[20, 20], cmap='viridis', 
                     cmin=1)  # Filter out zeros
        
        plt.colorbar(h[3], ax=ax, label='Counts')
        
        ax.set_xlabel('Shielding Depth [cm]', fontweight='bold')
        ax.set_ylabel('Optical Efficiency [photons/keV]', fontweight='bold')
        ax.set_title('Efficiency vs Shielding (2D Histogram)', fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'shielding_attenuation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
    
    def _identify_capture_materials(self):
        """
        Identify capture material from phi['matID']
        
        Material IDs (from hdf5_structure.txt):
        - LiquidArgon
        - Water
        - metal_copper
        - metal_steel
        - no_material
        - tyvek
        """
        
        materials = []
        for event in self.test_data:
            mat_id = event['phi'][18]  # matID is 19th parameter
            
            # Map numerical ID to material name
            if mat_id == 0:
                materials.append('Water')
            elif mat_id == 1:
                materials.append('LiquidArgon')
            elif mat_id == 2:
                materials.append('Steel')
            elif mat_id == 3:
                materials.append('Copper')
            else:
                materials.append('Other')
        
        return np.array(materials)
    
    def _compute_shielding_depths(self):
        """
        Compute optical path length through steel cryostat
        
        Geometry:
        - Cryostat is cylindrical
        - NC position: (xNC_mm, yNC_mm, zNC_mm) - indices 19, 20, 21 in phi
        - Detector voxels are outside cryostat
        
        For each event:
        1. Find NC position
        2. Compute distance to nearest detector voxel
        3. Estimate steel thickness traversed
        """
        
        depths = []
        
        for event in self.test_data:
            # NC position
            x_nc = event['phi'][19]  # xNC_mm
            y_nc = event['phi'][20]  # yNC_mm
            z_nc = event['phi'][21]  # zNC_mm
            
            nc_pos = np.array([x_nc, y_nc, z_nc])
            
            # Radial distance from detector center
            r_nc = np.sqrt(x_nc**2 + y_nc**2)
            
            # Cryostat inner radius (from theta/inner_radius_in_mm)
            r_cryostat = 500  # mm (approximate, load from HDF5 if available)
            
            # Estimate shielding depth (simplified)
            if r_nc < r_cryostat:
                # Inside cryostat → must traverse steel to reach detectors
                # Approximate as radial distance to cryostat wall
                depth = (r_cryostat - r_nc) / 10  # Convert mm to cm
            else:
                # Outside cryostat (shouldn't happen for NC)
                depth = 0.0
            
            depths.append(depth)
        
        return np.array(depths)

class SpatialLocalityVerifier:
    """
    Quantify spatial locality in voxelized photon detection data
    
    References:
    - FWHM (Full Width at Half Maximum) - standard in experimental physics
    - Silhouette Score - ML cluster quality metric
    - Moran's I - spatial autocorrelation (geospatial statistics)
    """
    
    def verify_spatial_locality(self, predictions, test_data, output_dir):
        """
        Comprehensive spatial locality analysis
        
        Metrics:
        1. Center-of-light vs true capture position
        2. Spatial spread (FWHM, RMS)
        3. Radial distribution functions
        4. Spatial autocorrelation
        """
        
        results = {}
        
        # Extract NC positions
        nc_positions = np.array([event['phi'][19:22] for event in test_data])  # xNC, yNC, zNC
        
        # Compute center-of-light from predictions
        col_positions = self._compute_center_of_light(predictions)
        
        # Metric 1: Position Reconstruction Error
        position_errors = np.linalg.norm(col_positions - nc_positions, axis=1)
        
        results['mean_position_error'] = np.mean(position_errors)
        results['median_position_error'] = np.median(position_errors)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Position Error Distribution
        ax = axes[0, 0]
        ax.hist(position_errors, bins=50, alpha=0.7, color='steelblue', 
                edgecolor='black')
        ax.axvline(np.mean(position_errors), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(position_errors):.1f} mm')
        ax.axvline(np.median(position_errors), color='orange', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(position_errors):.1f} mm')
        ax.set_xlabel('Position Reconstruction Error [mm]', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Spatial Locality: Position Accuracy', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: 2D Position Scatter (X-Y plane)
        ax = axes[0, 1]
        ax.scatter(nc_positions[:, 0], nc_positions[:, 1], 
                  alpha=0.3, s=20, c='blue', label='True NC Position')
        ax.scatter(col_positions[:, 0], col_positions[:, 1],
                  alpha=0.3, s=20, c='red', marker='x', label='Predicted COL')
        
        # Draw connection lines for first 50 events
        for i in range(min(50, len(nc_positions))):
            ax.plot([nc_positions[i, 0], col_positions[i, 0]],
                   [nc_positions[i, 1], col_positions[i, 1]],
                   'k-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('X [mm]', fontweight='bold')
        ax.set_ylabel('Y [mm]', fontweight='bold')
        ax.set_title('Position Reconstruction (X-Y Plane)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axis('equal')
        
        # Plot 3: Radial Distribution Function
        ax = axes[0, 2]
        
        radial_distances = self._compute_radial_distributions(predictions, nc_positions)
        
        # Average over all events
        r_bins = np.linspace(0, 1000, 50)  # mm
        mean_rdf = np.mean(radial_distances, axis=0)
        
        ax.plot(r_bins[:-1], mean_rdf, 'b-', linewidth=2.5, label='Mean RDF')
        ax.axvline(np.mean(position_errors), color='red', linestyle='--',
                  linewidth=2, label='Mean Position Error')
        
        ax.set_xlabel('Distance from NC Position [mm]', fontweight='bold')
        ax.set_ylabel('Normalized Photon Density', fontweight='bold')
        ax.set_title('Radial Distribution Function', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Spatial Spread (FWHM)
        ax = axes[1, 0]
        
        spatial_spreads = self._compute_spatial_spread(predictions, nc_positions)
        
        ax.hist(spatial_spreads, bins=50, alpha=0.7, color='green',
               edgecolor='black')
        ax.axvline(np.mean(spatial_spreads), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(spatial_spreads):.1f} mm')
        
        results['mean_spatial_spread'] = np.mean(spatial_spreads)
        
        ax.set_xlabel('Spatial Spread (RMS) [mm]', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Light Distribution Width', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 5: 3D Visualization (sample)
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        
        # Show first event as example
        event_idx = 0
        signal = predictions[event_idx]
        active_voxels = signal > 0.1  # Threshold
        
        if np.sum(active_voxels) > 0:
            voxel_pos = self.voxel_positions[active_voxels]
            voxel_signals = signal[active_voxels]
            
            scatter = ax.scatter(voxel_pos[:, 0], voxel_pos[:, 1], voxel_pos[:, 2],
                               c=voxel_signals, s=50, cmap='hot', alpha=0.8)
            
            # Mark NC position
            nc_pos = nc_positions[event_idx]
            ax.scatter([nc_pos[0]], [nc_pos[1]], [nc_pos[2]],
                      c='blue', s=200, marker='*', 
                      edgecolor='black', linewidth=2,
                      label='True NC Position')
            
            # Mark COL
            col_pos = col_positions[event_idx]
            ax.scatter([col_pos[0]], [col_pos[1]], [col_pos[2]],
                      c='red', s=200, marker='X',
                      edgecolor='black', linewidth=2,
                      label='Center-of-Light')
            
            plt.colorbar(scatter, ax=ax, label='Signal Strength', shrink=0.6)
        
        ax.set_xlabel('X [mm]', fontweight='bold')
        ax.set_ylabel('Y [mm]', fontweight='bold')
        ax.set_zlabel('Z [mm]', fontweight='bold')
        ax.set_title(f'3D Light Distribution (Event {event_idx})', fontweight='bold')
        ax.legend()
        
        # Plot 6: Spatial Autocorrelation (Moran's I)
        ax = axes[1, 2]
        
        morans_i = self._compute_spatial_autocorrelation(predictions)
        
        ax.hist(morans_i, bins=30, alpha=0.7, color='purple',
               edgecolor='black')
        ax.axvline(np.mean(morans_i), color='red', linestyle='--',
                  linewidth=2, label=f"Mean Moran's I: {np.mean(morans_i):.3f}")
        ax.axvline(0, color='gray', linestyle=':', linewidth=2,
                  label='No Autocorrelation')
        
        results['mean_morans_i'] = np.mean(morans_i)
        
        ax.set_xlabel("Moran's I (Spatial Autocorrelation)", fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Spatial Clustering Strength\n(I > 0: clustered)', 
                    fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.suptitle('Spatial Locality Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'spatial_locality_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
        
        return results
    
    def _compute_center_of_light(self, predictions):
        """
        Compute center-of-light (weighted mean position) from voxel signals
        """
        
        col_positions = []
        
        for signal in predictions:
            # Weighted by signal strength
            weights = signal + 1e-10  # Avoid division by zero
            
            # Weighted mean position
            col = np.average(self.voxel_positions, axis=0, weights=weights)
            col_positions.append(col)
        
        return np.array(col_positions)
    
    def _compute_spatial_spread(self, predictions, nc_positions):
        """
        Compute RMS spatial spread around NC position
        
        RMS = sqrt(mean((r_i - r_NC)^2))
        """
        
        spreads = []
        
        for signal, nc_pos in zip(predictions, nc_positions):
            # Active voxels
            active = signal > 0.1
            if np.sum(active) == 0:
            spreads.append(np.nan)
            continue
        
        # Distances from NC position
        voxel_pos = self.voxel_positions[active]
        distances = np.linalg.norm(voxel_pos - nc_pos, axis=1)
        
        # Weighted RMS (weighted by signal strength)
        weights = signal[active]
        rms = np.sqrt(np.average(distances**2, weights=weights))
        
        spreads.append(rms)
    
    return np.array(spreads)

def _compute_radial_distributions(self, predictions, nc_positions):
    """
    Compute radial distribution function (RDF)
    
    RDF(r) = density of photons at distance r from NC position
    """
    
    rdfs = []
    r_bins = np.linspace(0, 1000, 50)  # 0-1000 mm in 50 bins
    
    for signal, nc_pos in zip(predictions, nc_positions):
        # Distances of all voxels from NC position
        distances = np.linalg.norm(self.voxel_positions - nc_pos, axis=1)
        
        # Bin photon counts by distance
        rdf, _ = np.histogram(distances, bins=r_bins, weights=signal)
        
        # Normalize by bin volume (spherical shells)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        shell_volumes = 4 * np.pi * r_centers**2 * np.diff(r_bins)
        rdf_normalized = rdf / (shell_volumes + 1e-10)
        
        rdfs.append(rdf_normalized)
    
    return np.array(rdfs)

def _compute_spatial_autocorrelation(self, predictions):
    """
    Compute Moran's I spatial autocorrelation
    
    Moran's I: measures spatial clustering
    - I > 0: positive autocorrelation (clustered)
    - I = 0: random
    - I < 0: negative autocorrelation (dispersed)
    
    Reference: Moran (1950), "Notes on Continuous Stochastic Phenomena"
    """
    
    morans_i_values = []
    
    for signal in predictions:
        # Build spatial weight matrix (inverse distance weighting)
        n_voxels = len(signal)
        
        # For efficiency, sample subset if too many voxels
        if n_voxels > 500:
            sample_idx = np.random.choice(n_voxels, 500, replace=False)
            signal_sample = signal[sample_idx]
            pos_sample = self.voxel_positions[sample_idx]
        else:
            signal_sample = signal
            pos_sample = self.voxel_positions
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(pos_sample))
        
        # Inverse distance weights (avoid self-weights)
        np.fill_diagonal(distances, np.inf)
        weights = 1.0 / (distances + 1e-10)
        np.fill_diagonal(weights, 0)
        
        # Normalize weights
        row_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (row_sums + 1e-10)
        
        # Compute Moran's I
        mean_signal = np.mean(signal_sample)
        deviations = signal_sample - mean_signal
        
        numerator = np.sum(weights * np.outer(deviations, deviations))
        denominator = np.sum(deviations**2)
        
        n = len(signal_sample)
        morans_i = (n / np.sum(weights)) * (numerator / (denominator + 1e-10))
        
        morans_i_values.append(morans_i)
    
    return np.array(morans_i_values)

