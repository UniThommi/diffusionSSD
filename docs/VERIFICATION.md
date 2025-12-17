# Diffusion Model Verification Guide

## Übersicht

Dieses Dokument erklärt die **5 Verifikationsmethoden** für dein NC→Optical Signal Diffusionsmodell, basierend auf State-of-the-Art Diffusion Papers.

---

## Warum diese Methoden?

### ⚠️ Problem: Training mit nur 5 Epochen
Dein Training:
- **5 Epochen** × 50 Steps = **250 Trainingsschritte**
- Typisch für Diffusion: **100k-1M Steps**
- **Risiko:** Underfitting, Mode Collapse, Physics nicht gelernt

### ✓ Lösung: Multi-Perspektiven Verifikation
Keine einzelne Metrik reicht! Wir prüfen aus **5 komplementären Perspektiven**.

---

## Die 5 Verifikationsmethoden

### 1️⃣ Reconstruction Fidelity (Ho et al. 2020)

**Paper:** [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)

**Was wird getestet:**
```
Input: NC-Parameter φ (Position, Energie, Richtung)
       ↓
    Diffusion Model
       ↓
Output: Optical Signal x₀
       ↓
Vergleich mit Ground Truth
```

**Metriken:**
- **MSE (Mean Squared Error):** Wie genau ist Rekonstruktion?
- **Correlation:** Strukturelle Ähnlichkeit
- **F1-Score:** Precision/Recall für aktive Voxel

**Interpretation:**
- `MSE < 0.1`: Sehr gut
- `Correlation > 0.8`: Gut
- `F1 > 0.7`: Multiplizität funktioniert

**Warum es verifiziert:**  
Zeigt, ob das Modell **überhaupt etwas Sinnvolles** generiert.

---

### 2️⃣ Conditional Consistency (Rombach et al. 2022)

**Paper:** [Stable Diffusion](https://arxiv.org/abs/2112.10752)

**Was wird getestet:**
```
Fixiere φ → Sample 50× → Analysiere Konsistenz
```

**Idee:**  
- Für **gleiche** NC-Parameter sollten Samples **ähnlich** sein
- Aber **nicht identisch** (Stochastizität ist ok)

**Metriken:**
- **Mean Consistency:** Durchschnitt aller Samples ≈ Ground Truth?
- **Variance Plausibility:** Varianz physikalisch sinnvoll?

**Interpretation:**
```
Gut:     Mean(Samples) ≈ Ground Truth, Var mittel
Schlecht: Samples kollabieren (Var→0) ODER explodieren (Var→∞)
```

**Warum es verifiziert:**  
Zeigt, ob Conditioning `φ → x₀` **gelernt** wurde (nicht ignoriert!).

---

### 3️⃣ Interpolation Smoothness (Song et al. 2021)

**Paper:** [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)

**Was wird getestet:**
```
φ_A = [x=0, y=0, E=100keV]
        ↓ Interpoliere
φ_interp(α) = (1-α)·φ_A + α·φ_B
        ↓ Interpoliere
φ_B = [x=100, y=100, E=200keV]

→ Signal sollte smooth von A nach B wandern
```

**Metrik:**
- **Perceptual Path Length (PPL):** Summe der Abstände zwischen Steps

**Interpretation:**
```
PPL ≈ 1.0:  Perfekt (gerade Linie im Signal-Space)
PPL > 2.0:  Latenter Space ist nicht gut strukturiert
```

**Warum es verifiziert:**  
Zeigt, ob Modell **physikalische Kontinuität** versteht (nicht nur Lookup-Table).

---

### 4️⃣ Physics Constraints (Karras et al. 2022)

**Paper:** [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)

**Was wird getestet:**
1. **Energieerhaltung:** `Σ(Signal) ∝ E_gamma_tot_keV`
2. **Räumliche Lokalität:** Hits um NC-Position konzentriert
3. **Multiplizitätsbedingung:** ≥6 Voxel über Threshold

**Interpretation:**
```
✓ Gut:     Physik-Gesetze erfüllt
✗ Schlecht: Energieverletzung, räumliche Artefakte
```

**Warum es verifiziert:**  
**Wichtigste Metrik!** Zeigt, ob Modell **echte Physik** lernt oder nur Patterns memoriert.

---

### 5️⃣ Classifier-Free Guidance (Ho & Salimans 2022)

**Paper:** [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

**Was wird getestet:**
```
Guidance-Stärke w:
ε̃ = (1+w)·ε_θ(x_t, φ) - w·ε_θ(x_t, ∅)

w=0:   Keine Konditionierung (Baseline)
w=1:   Normale Konditionierung
w>1:   Stärkere Konditionierung
```

**Interpretation:**
- Höheres `w` → bessere Fidelity, weniger Diversität
- Trade-off sollte existieren

**Warum es verifiziert:**  
Zeigt **Sensitivität** des Modells auf Conditioning.

---

## Usage

### Quick Start
```bash
# Basis-Verifikation (alle Tests)
python verify_diffusion_model.py \
    --checkpoint ./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5 \
    --n-test-events 100

# Mit Visualisierungen
python verify_diffusion_model.py \
    --checkpoint ./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5 \
    --n-test-events 100 \
    --visualize \
    --show-trajectory
```

### Erwartete Laufzeit
- **Tests allein:** ~30-45 min (100 Events)
- **Mit Visualisierungen:** +10 min
- **Mit Trajektorie:** +5 min

### Output
```
verification_results/
├── verification_results.json           # Numerische Ergebnisse
├── verification_visualization.png      # Hauptvisualisierung
└── denoising_trajectory.png           # Trajektorien-Plot
```

---

## Interpretation der Ergebnisse

### Overall Score Interpretation

| Score | Bedeutung | Action |
|-------|-----------|--------|
| **≥ 0.80** | ✓ **Excellent** - Model ready for PMT selection | Proceed with confidence |
| **0.60-0.79** | ⚠ **Good** - Minor issues, usable | Check which metric is low |
| **0.40-0.59** | ⚠ **Fair** - Significant issues | More training recommended |
| **< 0.40** | ✗ **Poor** - Model failed | Re-train with more epochs/data |

### Score-Komponenten Gewichte
```python
Overall = 0.3·Fidelity + 0.2·Consistency + 0.1·Smoothness + 0.4·Physics
```

**Warum Physics 40%?**  
→ Physikalische Plausibilität ist kritischer als perfekte Rekonstruktion!

---

## Häufige Probleme & Lösungen

### Problem 1: Low Fidelity Score (< 0.5)
**Symptom:** MSE hoch, Correlation niedrig

**Ursachen:**
- Zu wenig Training (5 Epochen!)
- Learning Rate zu hoch/niedrig
- Model zu klein (hidden_dim=512)

**Lösung:**
```bash
# Train länger
python train_diffusion_cpu.py \
    --epochs 20 \
    --steps-per-epoch 500 \
    --learning-rate 5e-5
```

---

### Problem 2: Low Consistency Score (< 0.5)
**Symptom:** Samples für gleiches φ zu unterschiedlich

**Ursachen:**
- Conditioning nicht gelernt (Modell ignoriert φ)
- Zu viel Rauschen im Sampling

**Lösung:**
- Prüfe Timestep-Embedding-Dimension (sollte ≥32 sein)
- Erhöhe `n_sampling_steps` von 50 auf 100

---

### Problem 3: Low Physics Score (< 0.5)
**Symptom:** Energieverletzung, Multiplizität falsch

**Ursachen:**
- Model lernt nur Patterns, nicht Physik
- Training Data zu klein/biased

**Lösung:**
- **Kritisch!** Mehr Daten sammeln
- Physics-Informed Loss hinzufügen:
```python
def physics_loss(x_gen, phi):
    energy_true = phi[:, 1]  # E_gamma_tot_keV
    energy_gen = tf.reduce_sum(x_gen, axis=1)
    
    energy_loss = tf.abs(energy_gen - energy_true)
    return energy_loss
```

---

### Problem 4: Model generiert nur Noise
**Symptom:** Alle Scores < 0.2

**Ursachen:**
- Training hat nicht funktioniert
- Checkpoint korrumpiert
- Falsches Target-Format

**Lösung:**
```bash
# Prüfe Training-Loss
python train_diffusion_cpu.py --eval-only

# Falls Loss nicht sinkt → Re-initialize
```

---

## Vergleich mit Baseline

### Random Baseline
Generiere zufällige Signale → Score sollte ~0.1 sein

### Memorization Check
Prüfe, ob Modell nur Training-Daten memoriert:
```python
# Test auf UNSEEN Events (nicht im Training)
verifier = DiffusionVerifier(model, config)
verifier.run_all_tests()  # Should still score > 0.6
```

---

## Advanced: Custom Metrics

### Beispiel: Photon-Propagation Zeit
```python
def verify_photon_timing(x_gen, phi):
    """Prüfe ob Photon-Ankunftszeiten physikalisch sind"""
    nc_position = phi[:3]  # x, y, z
    
    for voxel_idx, signal in enumerate(x_gen):
        if signal > threshold:
            voxel_pos = get_voxel_position(voxel_idx)
            distance = np.linalg.norm(voxel_pos - nc_position)
            
            # Speed of light in LAr: ~0.67c
            expected_time = distance / (0.67 * 3e8)
            
            # Check if timing matches...
```

---

## Literatur & Weiterführendes

### Diffusion Models
1. **Ho et al. (2020)** - DDPM (Grundlage)  
   https://arxiv.org/abs/2006.11239

2. **Song et al. (2021)** - Score-Based (Theorie)  
   https://arxiv.org/abs/2011.13456

3. **Rombach et al. (2022)** - Stable Diffusion (Conditional)  
   https://arxiv.org/abs/2112.10752

4. **Karras et al. (2022)** - EDM (Best Practices)  
   https://arxiv.org/abs/2206.00364

### Physics-ML
5. **Cranmer et al. (2020)** - "The frontier of simulation-based inference"  
   PNAS, https://arxiv.org/abs/1911.01429

6. **Brehmer et al. (2020)** - "Mining for Dark Matter Substructure"  
   https://arxiv.org/abs/1909.02005

---

## FAQ

**Q: Warum DDIM statt DDPM Sampling?**  
A: DDIM ist deterministisch und **50× schneller** (50 steps statt 1000). Performance identisch.

**Q: Kann ich mit 5 Epochen überhaupt gute Ergebnisse erwarten?**  
A: Nein. 5 Epochen sind ein **Proof-of-Concept**. Für Production: ≥20 Epochen.

**Q: Was ist ein "guter" Overall Score für mein Use-Case?**  
A: Für PMT-Selektion: **≥0.6** ausreichend (nur relative Importance zählt). Für exakte Rekonstruktion: **≥0.8** nötig.

**Q: Soll ich Physics-Constraints in den Loss integrieren?**  
A: **Ja!** Weighted Sum:
```python
total_loss = mse_loss + 0.1 * energy_loss + 0.05 * multiplicity_loss
```

---

## Kontakt & Contributions

Bei Fragen oder Verbesserungsvorschlägen:
- Issue erstellen mit Verification-Log
- Score-Ergebnisse als JSON anhängen
- Visualisierungen hilfreich

**Happy Verifying! 🚀**