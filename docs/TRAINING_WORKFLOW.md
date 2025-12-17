# Verbesserter Training Workflow

## Neue Features

### 1️⃣ **Best Model Tracking**
- Training speichert **automatisch** das beste Modell (niedrigster Validation Loss)
- Gespeichert als `best_model.weights.h5`
- Verifikation nutzt automatisch best model!

### 2️⃣ **Validation Split**
- **90% Training** / **10% Validation**
- Erkennt Overfitting frühzeitig
- Validation Loss entscheidet über "bestes" Modell

### 3️⃣ **Early Stopping**
- Stoppt Training wenn **5 Epochen** keine Verbesserung
- Spart Zeit und verhindert Overfitting

### 4️⃣ **Training History**
- JSON-File mit allen Losses: `logs_cpu/training_history.json`
- Plot automatisch generiert: `logs_cpu/training_curves.png`

---

## Workflow: Training → Verifikation → PMT-Selection

### Schritt 1: Training (verbessert)

```bash
python train_diffusion_cpu.py
```

**Output:**
```
Total Events: 1,946,992
  Training: 1,752,293
  Validation: 194,699

Starte Training für 20 Epochen...

Epoch 1/20
  Step 10: Loss = 0.152341
  ...
  Validation Loss: 0.145623
  ★ New best validation loss: 0.145623
  
Epoch 2/20
  ...
  Validation Loss: 0.142105
  ★ New best validation loss: 0.142105
  
Epoch 5/20
  ...
  Validation Loss: 0.141998
  No improvement (1/5)

...

Epoch 12/20
  Validation Loss: 0.142201
  No improvement (5/5)
  
⚠ Early stopping triggered (no improvement for 5 epochs)

✓ Training abgeschlossen in 142.3 Minuten
✓ Training history saved: logs_cpu/training_history.json
✓ Training curves saved: logs_cpu/training_curves.png
```

**Gespeicherte Files:**
```
checkpoints_cpu/
├── best_model.weights.h5            ← WICHTIGSTER FILE!
├── best_model_info.json             ← Metadata (Epoch, Loss, Config)
├── checkpoint_epoch_1_model.weights.h5
├── checkpoint_epoch_1_info.json
├── checkpoint_epoch_2_model.weights.h5
├── ...
└── checkpoint_epoch_12_model.weights.h5  (letztes vor Early Stop)

logs_cpu/
├── training_history.json            ← Alle Losses für Analyse
└── training_curves.png              ← Visualisierung
```

---

### Schritt 2: Checkpoint Management

#### Liste alle Checkpoints
```bash
python load_best_model.py --list
```

**Output:**
```
======================================================================
VERFÜGBARE CHECKPOINTS
======================================================================

★ BEST MODEL:
  Path: checkpoints_cpu/best_model.weights.h5
  Epoch: 7
  Loss: 0.141523
  Date: 2024-12-10T15:23:45

REGULAR CHECKPOINTS (12):
  Epoch  1: Loss=0.145623  (2024-12-10)
  Epoch  2: Loss=0.142105  (2024-12-10)
  Epoch  3: Loss=0.143001  (2024-12-10)
  ...
  Epoch  7: Loss=0.141523  (2024-12-10)  ← BEST
  ...
  Epoch 12: Loss=0.142201  (2024-12-10)
======================================================================
```

#### Vergleiche alle Checkpoints
```bash
python load_best_model.py --compare
```

**Output:**
```
======================================================================
CHECKPOINT COMPARISON
======================================================================

Rank   Epoch    Loss         File
----------------------------------------------------------------------
★ 1    7        0.141523     checkpoint_epoch_7_model.weights.h5
  2    8        0.141789     checkpoint_epoch_8_model.weights.h5
  3    6        0.142001     checkpoint_epoch_6_model.weights.h5
  4    2        0.142105     checkpoint_epoch_2_model.weights.h5
  5    9        0.142234     checkpoint_epoch_9_model.weights.h5
...

======================================================================
EMPFEHLUNG: Nutze Epoch 7 (Loss: 0.141523)
======================================================================
```

#### Lade Best Model programmatisch
```python
from load_best_model import load_best_model

# Automatisch bestes laden
model, info = load_best_model()

print(f"Geladen: Epoch {info['epoch']}, Loss {info['loss']:.6f}")

# Inferenz
predictions = model([x_noisy, phi, t], training=False)
```

---

### Schritt 3: Verifikation (nutzt automatisch best model)

```bash
python verify_diffusion_model.py --visualize
```

**Output:**
```
======================================================================
DIFFUSION MODEL VERIFICATION
======================================================================
Checkpoint: checkpoints_cpu/best_model.weights.h5

Lade Modell...
✓ Best Model geladen aus Epoch 7
  Epoch: 7
  Loss: 0.141523
  Parameters: 9,152,781

Lade Test-Daten...
  ✓ 100 Events geladen

======================================================================
TEST 1: RECONSTRUCTION FIDELITY
======================================================================
...
  → Overall Fidelity Score: 0.723/1.0

...

======================================================================
OVERALL VERIFICATION SCORE
======================================================================
  Reconstruction Fidelity:    0.723
  Conditional Consistency:    0.681
  Interpolation Smoothness:   0.654
  Physics Constraints:        0.712

  → OVERALL SCORE: 0.697/1.0

  ⚠ GOOD - Model works but has some issues
======================================================================
```

---

### Schritt 4: PMT Selection (nutzt automatisch best model)

#### Fast Methods
```bash
python fast_pmt_selection.py
```

**Nutzt automatisch:**
```python
# In fast_pmt_selection.py Zeile 560:
from load_best_model import load_best_model

model, _ = load_best_model('./checkpoints_cpu')
# → Lädt automatisch best_model.weights.h5
```

#### Shapley Values
```bash
python shapley_pmt_selection.py
```

---

## Vergleich: Alt vs. Neu

### ALT (Problematisch)
```
Training:
  ✗ Kein Validation Split
  ✗ Kein Best Model Tracking
  ✗ Keine Early Stopping
  ✗ Manuelles Epoch-Auswählen nötig

Verifikation:
  ✗ Muss manuell Checkpoint-Pfad angeben
  ✗ Weiß nicht welches Epoch am besten ist
  ✗ Könnte übertrainiertes Modell nutzen

Result:
  → Unsicher welches Modell optimal
  → Potentiell schlechte Performance
```

### NEU (Optimal)
```
Training:
  ✓ 90/10 Train/Val Split
  ✓ Best Model automatisch gespeichert
  ✓ Early Stopping nach 5 Epochen
  ✓ Training History & Plots

Verifikation:
  ✓ Lädt automatisch best_model.weights.h5
  ✓ Garantiert bestes verfügbares Modell
  ✓ Transparenz: Epoch & Loss angezeigt

Result:
  → Immer optimales Modell
  → Reproduzierbar & nachvollziehbar
```

---

## Training Curves interpretieren

### Gute Kurven
```
Loss
  |
  |  Train ----
  |       \
  |        ----___
  |  Val ------   ---___
  |              \      ---___
  |               -------     ===== (Plateau)
  +--------------------------------> Epoch
  
Interpretation: ✓ Training konvergiert, kein Overfitting
Aktion: Best Model ist valide, nutzen!
```

### Overfitting
```
Loss
  |
  |  Train ----
  |       \     ___
  |        -----   ---___
  |                      ----____
  |  Val ------                  
  |       \              ___---
  |        ----_____----
  +--------------------------------> Epoch
              ↑
          Overfitting startet hier
  
Interpretation: ✗ Validation Loss steigt nach Epoch X
Aktion: Nutze Best Model (VOR Overfitting!)
```

### Underfitting
```
Loss
  |
  |  Train ----
  |       \
  |        ----___
  |               ----___
  |                      ----___  (Noch fallend)
  |  Val ------                ----___
  |       \                          ----
  +----------------------------------------> Epoch
  
Interpretation: ⚠ Beide Losses fallen noch
Aktion: Mehr Epochen trainieren!
```

---

## Troubleshooting

### Problem: "Best model nicht gefunden"

**Ursache:** Altes Training ohne Validation

**Lösung:**
```bash
# Option 1: Re-train mit neuem Skript
python train_diffusion_cpu.py

# Option 2: Nutze letztes Checkpoint manuell
python verify_diffusion_model.py \
    --checkpoint ./checkpoints_cpu/checkpoint_epoch_5_model.weights.h5
```

---

### Problem: Early Stopping zu früh

**Symptom:** Stoppt nach 6 Epochen, aber Loss sinkt noch

**Lösung:** Erhöhe Patience in `train_diffusion_cpu.py`:
```python
# Zeile ~195
patience = 10  # statt 5
```

---

### Problem: Validation Loss höher als Training Loss

**Symptom:** 
```
Training Loss: 0.12
Validation Loss: 0.18
```

**Ursachen:**
1. **Normal:** Leichte Differenz (±10%) ist ok
2. **Overfitting:** Große Differenz (>50%) → Model memoriert Training Data
3. **Zu kleine Validation:** Nur 194k Events, könnte rauschen

**Lösung bei Overfitting:**
- Dropout erhöhen
- Model kleiner machen (hidden_dim=256 statt 512)
- Mehr Daten

---

## Best Practices

### 1. Immer mit Validation trainieren
```python
# config in train_diffusion_cpu.py
'epochs': 20,              # Genug für Konvergenz
'save_every_n_epochs': 2,  # Alle 2 Epochen speichern
```

### 2. Training Curves checken
```bash
# Nach Training
eog logs_cpu/training_curves.png
# oder
open logs_cpu/training_curves.png
```

### 3. Best Model für alles nutzen
```python
from load_best_model import load_best_model

model, info = load_best_model()
# → Garantiert bestes Modell!
```

### 4. Checkpoint Management
```bash
# Vor großem Experiment: Backup
cp -r checkpoints_cpu checkpoints_cpu_backup_$(date +%Y%m%d)

# Nach erfolgreichem Training: Cleanup alte Checkpoints
python load_best_model.py --compare
# → Behalte nur best_model + Top-3 Checkpoints
```

---

## Quick Reference

```bash
# Training (neu & verbessert)
python train_diffusion_cpu.py

# Checkpoints anzeigen
python load_best_model.py --list

# Checkpoints vergleichen
python load_best_model.py --compare

# Best Model laden & testen
python load_best_model.py --load

# Verifikation (nutzt automatisch best model)
python verify_diffusion_model.py --visualize

# PMT Selection (nutzt automatisch best model)
python fast_pmt_selection.py
python shapley_pmt_selection.py
```

---

## Zusammenfassung

**Vorher:** "Welches Checkpoint soll ich nehmen?"  
**Jetzt:** `best_model.weights.h5` → **Immer das Beste!**

✓ Automatisch  
✓ Transparent  
✓ Reproduzierbar  
✓ Wissenschaftlich solide