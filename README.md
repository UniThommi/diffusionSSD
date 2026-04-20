# NeCaDi — Generation Pipeline

## Neural Network Architecture

### Overview

The model (`scoreDiffusion`) contains two independent sub-networks that are
trained and sampled in sequence:

| Sub-model | Task | Architecture |
|---|---|---|
| `model_area` / `ema_area` | predict 4D area-hit vector | Residual MLP (ResNet) |
| `model_voxels[n]` / `ema_voxels[n]` | predict 2D voxel grid per region | 2D U-Net |

Both sub-networks are conditioned on the same physics feature vector **φ** and
diffusion timestep **t**, but use different conditioning strategies and
activations.

---

### Time embedding

The scalar timestep `t ∈ [0,1]` is mapped to a `num_embed`-dimensional vector
by `scoreDiffusion.Embedding()`:

```
freq  = exp(−log(10000) / (half_dim−1) · range(half_dim))   # [half_dim]  GaussianFourierProjection
angle = t · freq · 1000
emb   = [sin(angle), cos(angle)]                              # [num_embed]  SinCosEmbedding
emb   → Dense(2·num_embed) → Swish → Dense(num_embed)        # [num_embed]
```

`half_dim = num_embed // 2 = 64` (from `config.model.embed_dim = 128`).
The `· 1000` scaling widens the frequency range so small differences in `t`
produce distinct embeddings even late in the schedule.

---

### ResNet — area hits model

**Inputs:** `[area_hits [B,4], t [B,1], φ [B,phi_dim]]`  
**Output:** predicted velocity `v_area [B,4]`

The time embedding and φ are fused before the residual stack:

```
φ          → Dense(128) → Swish → cond [B,128]
t          → Embedding()        → time_emb [B,128]
[time_emb, cond] → Concatenate → Dense(128) → conditioning [B,128]
```

Each residual block (`resnet_dense`, repeated `num_layer−1 = 2` times) applies
**affine conditioning** (not full FiLM — no separate generator network):

```
residual = Dense(128)(x)
scale, shift = split(Dense(256)(conditioning), 2)   # [B,128] each
x = Swish(Dense(128)(LeakyReLU(x)))
x = Swish((1 + scale) · x + shift)                 # affine modulation
x = Dropout(0.1)(x)
x = Dense(128)(x)
x = Add([x, residual])
```

The `(1 + scale)` term ensures the identity mapping when `scale = 0`,
stabilising early training. Output layer: `Dense(4)`.

Activation: **LeakyReLU(α=0.01)** throughout (not Swish).

---

### U-Net — voxel distribution model (one per region)

**Inputs:** `[voxel_grid [B,H,W,1], t [B,1], area_hits [B,4], φ [B,phi_dim]]`  
**Output:** predicted velocity `v_voxel [B,H,W,1]`

#### Conditioning pipeline

Three streams are fused into a single conditioning tensor of shape `[B, num_embed]`:

```
t               → Embedding()                  → time_emb  [B,128]
φ               → Dense(128) → Swish           → cond      [B,128]
area_hits[idx]  → Dense(128) → Swish           → area_emb  [B,128]
[time_emb, area_emb, cond] → Concatenate [B,384] → Dense(128) → conditioning [B,128]
```

`area_hits[idx]` is the **scalar** hit count for this specific region, extracted
from the 4D vector via `Lambda`. Each U-Net sees only its own region's count.

The conditioning tensor is reshaped to `[B,1,1,128]` inside each residual block
and injected via a Dense projection (additive, not affine).

#### Encoder–Bottleneck–Decoder

```
Input [B, H, W, 1]
  → ZeroPadding2D(pad)                  # makes H,W divisible by 2^4=16
  → Conv2D(16, kernel=1)                # embed to 16 feature maps

Encoder (widths [32, 64, 96], attentions [F, T, T]):
  → DownBlock(32, False):  4×ResBlock → skip ×4 → AveragePooling2D(2)
  → DownBlock(64, True):   4×ResBlock → skip ×4 → AveragePooling2D(2)
  → DownBlock(96, True):   4×ResBlock → skip ×4 → AveragePooling2D(2)

Bottleneck (width 128, attention True):
  → 4×ResBlock(128, True)

Decoder (widths [96, 64, 32], attentions [T, T, F]):
  → UpBlock(96, True):  UpSampling2D(2) → 4×(Concatenate(skip) + ResBlock)
  → UpBlock(64, True):  UpSampling2D(2) → 4×(Concatenate(skip) + ResBlock)
  → UpBlock(32, False): UpSampling2D(2) → 4×(Concatenate(skip) + ResBlock)

  → Conv2D(1, kernel=1, init="zeros")   # project to 1 channel
  → Cropping2D(pad)                     # restore original H, W
```

Downsampling uses **AveragePooling2D** (not strided convolution).
Upsampling uses **nearest-neighbour UpSampling2D**.
Skip connections are added via `Concatenate` (not Add), so the decoder sees
both its upsampled features and the encoder's full-resolution features.
The output Conv2D is **zero-initialised**, so the model starts by predicting
near-zero velocity — a stable initialisation for diffusion training.

#### Residual block

```
residual ← Conv2D(width, kernel=1)   # only if channel dim changes
n_proj   ← Dense(width)(conditioning)

x → Swish → Conv2D(width, kernel=3, padding="same")
  → Add(n_proj)
  → Swish → Conv2D(width, kernel=3, padding="same")
  → Add(residual)
```

If `attention=True`, a self-attention sublayer follows:

```
residual ← x
x → GroupNormalization(groups=4)
  → MultiHeadAttention(num_heads=1, key_dim=width, axes=(1,2,3))
  → Add(residual)
```

`attention_axes=(1,2,3)` means attention operates jointly over the height,
width, and channel dimensions of the feature map — every spatial position
attends to every other position.

#### Padding

The zero-padding `pad` is **not set in config.toml** for the grid dimensions.
It is computed at runtime (in `train.py` / `generate.py`) to make each spatial
dimension divisible by `2^unet_block_depth = 2^4 = 16`:

```python
target = ceil(n / divisor) * divisor
total  = target - n
pad    = (total // 2, total - total // 2)   # symmetric split
```

Padding is then removed by `Cropping2D(pad)` after the output convolution,
so the U-Net input and output always have identical spatial shape.

---

### Hyperparameters (from `config.toml`)

| Parameter | Value | Location |
|---|---|---|
| Embedding dim (`num_embed`) | 128 | `model.embed_dim` |
| Diffusion steps | 512 | `diffusion.num_steps` |
| U-Net widths | [32, 64, 96, 128] | `model.unet_widths` |
| U-Net block depth | 4 | `model.unet_block_depth` |
| Attention per level | [F, T, T, T] | `model.unet_attentions` |
| Kernel size | 3 | `model.unet_kernel` |
| Stride | 2 | `model.unet_stride` |
| Initial feature maps | 16 | `model.unet_input_embedding_dims` |
| ResNet hidden dim | 128 | `model.resnet_hidden_dim` |
| ResNet layers | 3 (→ 2 res blocks + 1 output) | `model.num_resnet_layers` |
| EMA decay | 0.999 | `training.ema_decay` |
| Active φ features | r, φ, z, E_γ, #γ, matID (one-hot) | `features.active_phi` |

---

## Input

- Physics conditioning vector **φ** (per NC event):
  position (x, y, z, r, φ), energy, material/volume IDs, gamma kinematics, capture time, Ge77 flag
- Normalized to [0, 1]; material IDs optionally one-hot encoded

---

## Stage 1 — Area Hits (ResNet + DDIM)

- **Model:** small residual MLP (ResNet)
- **Input:** φ (physics features)
- **Output:** 4D vector `[hits_PIT, hits_BOT, hits_WALL, hits_TOP]`
- **Sampler:** DDIM (η=0), 512 steps, cosine log-SNR schedule
- **v-parameterization:** model predicts `v = α·ε − σ·x₀`; x₀ recovered deterministically at each step

---

## Stage 2 — Voxel Distributions (U-Net + DDIM, per region)

- **Model:** 4 separate 2D U-Nets (one per detector region); PIT+BOT optionally merged into one grid
- **Input:** φ + area_hits from Stage 1 + noise grid
- **Output:** 2D voxel hit map `[z × φ × 1]` per region
- **Sampler:** same DDIM as Stage 1 (512 steps, η=0)
- **Zero-skipping:** events with area_hits ≈ 0 are skipped entirely
- Periodic boundary padding along φ-axis (cylindrical geometry)

---

## Post-Processing

- **Denormalize:** voxel grids × area_hits_raw → absolute hit counts
- **Grid → Voxel remap:** 2D grid cells back to named voxel keys
- **Round** to integer hit counts (`np.rint → int32`)
- **Write ML-format HDF5:**
  - `phi_matrix   (N × n_φ)`     float32 — physics features
  - `target_matrix (N × 9583)`   int32   — voxel hit counts
  - `region_matrix (N × 4)`      int32   — summed per region

---

## Velocity Parameterization (v-score)

### What it is

The model does not predict noise `ε` or the clean data `x₀` directly.
Instead it predicts the **velocity** `v`, a rotation in (data, noise) space defined as:

```
v = α·ε − σ·x₀
```

where `α` and `σ` are the signal and noise weights at timestep `t`, satisfying `α² + σ² = 1`.
This identity is enforced by the cosine log-SNR schedule
(`scoreDiffusion.get_logsnr_alpha_sigma`):

```python
logsnr = logsnr_schedule_cosine(t)          # λ = log(α²/σ²)
alpha  = sqrt(sigmoid( logsnr))             # α = √sigmoid(λ)
sigma  = sqrt(sigmoid(-logsnr))             # σ = √sigmoid(−λ)
```

Because `sigmoid(λ) + sigmoid(−λ) = 1`, it follows that `α² + σ² = 1` exactly.

### Forward process (training)

At each training step a random `t ~ U[0,1]` is drawn and clean data `x₀` is corrupted:

```
x_t = α·x₀ + σ·ε,    ε ~ N(0, I)          # scoreDiffusion.train_step
```

The true velocity for that sample is then:

```
v = α·ε − σ·x₀
```

The model is trained with **MSE loss on `v`**:

```python
perturbed_x = alpha * x0 + z * sigma       # forward process
v           = alpha * z  - sigma * x0      # true velocity
loss        = mean((model(perturbed_x, t, φ) − v)²)
```

This is done identically for the ResNet (area hits, 4D vector) and for each U-Net
(voxel grids, 2D spatial), with `alpha`/`sigma` broadcast to the spatial dimensions
via `tf.reshape(alpha, [-1, 1, 1, 1])`.

After every gradient step the **EMA shadow weights** are updated:

```python
ema_weight = ema_decay · ema_weight + (1 − ema_decay) · weight   # decay = 0.999
```

Only the EMA weights are saved as checkpoints and used at inference.

### Inference: recovering x₀ from v

Because `α² + σ² = 1`, the system

```
x_t = α·x₀ + σ·ε
v   = α·ε  − σ·x₀
```

can be solved analytically:

```
x₀ = α·x_t − σ·v_θ          (multiply first eq. by α, second by σ, subtract)
ε  = (x_t − α·x₀) / σ       (rearrange first eq.)
```

These two lines appear verbatim in `DDPMSampler`:

```python
mean = alpha * x - sigma * score   # predicted x₀
eps  = (x - alpha * mean) / sigma  # inferred noise direction
```

### DDIM update step

Using the recovered `x₀` and `ε`, the deterministic DDIM step moves to `x_{t−1}`:

```python
x = alpha_ * mean + sigma_ * eps   # x_{t−1} = α'·x₀ + σ'·ε
```

where `alpha_`, `sigma_` correspond to `t − 1/num_steps`.
No fresh noise is injected (η=0), making every trajectory fully deterministic.
After `num_steps = 512` iterations the loop returns `mean` — the final noiseless `x₀`.

---

## Training

### Overview

Training is driven by Keras `model.fit()` (`train.py`, line 670).
The `scoreDiffusion` model overrides `train_step`, so each call to `fit` internally
runs **two independent gradient updates per batch** — one for the ResNet (area hits)
and one for each U-Net (voxel grids) — using separate optimizer instances compiled
inside `scoreDiffusion.compile`.

### Dataset pipeline

```
HDF5 file
  → voxelDataset(file, config)          # normalisation, one-hot, grid inference
  → .get_dataset(rank, size)            # sharded tf.data.Dataset of (voxels, area_hits, φ)
  → .repeat()                           # infinite stream for steps_per_epoch control
  → .prefetch(AUTOTUNE)                 # background prefetch
  → .batch(BATCH_SIZE)                  # fed to model.fit
```

Grid shapes (`SHAPE_PIT`, `SHAPE_BOT`, etc.) are **not set in config.toml**.
They are inferred by `voxelDataset` from the voxel indices in the HDF5 file and
written back into `config` before the model is constructed:

```python
for region in config['LAYER_NAMES']:
    config[f'SHAPE_{region}'] = list(train_dataset.grid_shapes[region])
```

### Merged regions

If `merged_regions = ["PIT", "BOT"]` is set in config, the two regions share a
single U-Net under the name `PITBOT`. `train.py` computes `LAYER_NAMES` by
removing the individual names and appending the merged name:

```python
layer_names = [r for r in active_regions if r not in merged_regions]
layer_names.append('PITBOT')
```

`AREA_RATIOS['PITBOT']` is read directly from `config.normalization.area_ratios`
and used to weight the PITBOT U-Net loss term in the total loss.

### Loss structure

The total loss tracked by `model.fit` is:

```
total_loss = area_loss + Σ_i (AREA_RATIOS[i] × voxel_loss_i)
```

Each term is an MSE on the velocity prediction (see *Velocity Parameterization*).
The area loss and each voxel loss are computed under **separate `GradientTape`
contexts** and their gradients applied to separate optimizer instances, so the
ResNet and each U-Net are updated independently within the same training step.
All gradients are clipped per-tensor before the update:

```python
g = [tf.clip_by_norm(grad, 1) for grad in g]
```

### Optimizer and learning-rate schedule

`train.py` supports two optimizers (selected via `config.training.optimizer`):
`Adam` (default) or `Adamax`. An optional cosine decay schedule is applied:

```python
CosineDecay(
    initial_learning_rate = LR * size,          # scaled by number of GPUs
    decay_steps           = NUM_EPOCHS * (train_size // BATCH_SIZE)
)
```

Without cosine decay, a constant learning rate `LR * size` is used.
In Horovod multi-GPU mode the optimizer is wrapped with
`hvd.DistributedOptimizer(opt, average_aggregated_gradients=True)`.

### EMA and checkpointing

After every gradient step the EMA shadow weights are updated in-place
(decay = 0.999, configured via `config.training.ema_decay`).
The `MultiModelCheckpoint` callback saves **only the EMA weights** when
`val_loss` improves:

```
checkpoints/<run_name>/ema_area_model_best.weights.h5
checkpoints/<run_name>/ema_voxel_{area_name}_best.weights.h5
```

The live (non-EMA) weights are never saved; only EMA weights are used at inference.

### Callbacks

| Callback | Trigger | Action |
|---|---|---|
| `EarlyStopping` | `val_loss` stagnates for `patience` epochs | stops training, restores best weights |
| `MultiModelCheckpoint` | `val_loss` improves | saves EMA weights for all sub-models |
| `MemoryCleanup` | end of every epoch | `gc.collect()` |
| `IncrementalPlotter` | every `plot_update_frequency` epochs | saves PNG loss plots to checkpoint dir |
| `TensorBoard` (optional) | every epoch | writes scalar metrics to `logs/<run_name>/` |
| `hvd.BroadcastGlobalVariablesCallback` | epoch 0 | broadcasts initial weights from rank 0 to all workers |
| `hvd.MetricAverageCallback` | end of epoch | averages metrics across all workers |

### Outputs

After training the checkpoint directory contains:

```
ema_area_model_best.weights.h5          — ResNet EMA weights at best val_loss
ema_voxel_{area_name}_best.weights.h5   — U-Net EMA weights per region
best_epoch.txt                          — best epoch number and all val metrics
history.json                            — full per-epoch loss history (optional)
overview_final.png / unets_final.png    — training curves (optional)
```

---

## Distillation (`distill_nc.py`)

### Goal

Progressive distillation halves the number of DDIM sampling steps per stage
without retraining from scratch.
A student U-Net learns to replicate the teacher's **two-step** denoising result
in a **single step**, so `num_steps` goes from `T → T/factor` (default `factor=2`).
Stages can be chained: 512 → 256 → 128 → …

Only the U-Nets are distilled.
The ResNet (area hits) is frozen and passed through unchanged — its EMA weights
become `self.ema_area` on the distilled model, so `generate.py` and
`MultiModelCheckpoint` work without modification.

### What is frozen vs. trained

| Component | Status | Weights initialised from |
|---|---|---|
| `teacher_area` (ResNet EMA) | frozen (`trainable=False`) | teacher checkpoint |
| `teacher_voxels[n]` (U-Net EMAs) | frozen (`trainable=False`) | teacher checkpoint |
| `model_voxels[n]` (student U-Nets) | **trained** | cloned from teacher EMA |
| `ema_voxels[n]` (student EMA shadow) | updated after each step | cloned from student |

The student U-Nets are initialised with the teacher's EMA weights (`set_weights`),
so distillation starts from a high-quality solution rather than random noise.

### Distillation target (per U-Net, per batch)

For each region, a random integer timestep index `i ~ U[0, num_steps)` is drawn.
Three continuous timesteps are derived from it:

```
u     = (i + 1) / num_steps          # current timestep
u_mid = u - 0.5 / num_steps          # midpoint
u_s   = u - 1.0 / num_steps          # student target timestep
```

The clean voxel `x₀` is corrupted to `z_v = α·x₀ + σ·ε` at time `u`.

**Teacher step 1** — DDIM step from `u` to `u_mid`:
```
v_t     = teacher(z_v,     u,     area_hits, φ)
mean_v  = α·z_v    − σ·v_t             # recover x₀
eps_v   = (z_v − α·mean_v) / σ         # recover ε
z_mid_v = α_mid·mean_v + σ_mid·eps_v   # DDIM → u_mid
```

**Teacher step 2** — DDIM step from `u_mid` to `u_s`:
```
v_t_mid     = teacher(z_mid_v, u_mid, area_hits, φ)
mean_v      = α_mid·z_mid_v − σ_mid·v_t_mid
eps_v       = (z_mid_v − α_mid·mean_v) / σ_mid
z_teacher_v = α_s·mean_v + σ_s·eps_v   # DDIM → u_s
```

**Consistency target** — recover the implied `x₀` that would produce `z_teacher_v`
in one step from `z_v` (Salimans & Ho 2022, eq. 15):
```
σ_frac    = exp(0.5·(softplus(logsnr) − softplus(logsnr_s)))
x_target  = (z_teacher_v − σ_frac·z_v) / (α_s − σ_frac·α)
```
Special case `i == 0`: `σ_s → 0` makes the denominator undefined;
`x_target` is replaced with `mean_v` (the teacher's direct denoised estimate).

The target is then converted back to velocity space:
```
eps_target = (z_v − α·x_target) / σ
v_target   = α·eps_target − σ·x_target
```

**Student loss** (unweighted MSE, per region):
```
loss_voxel = mean( (student(z_v, u, area_hits, φ) − v_target)² )
```

### Optimizer structure

`scoreDiffusion_distill.compile` creates **one independent optimizer per U-Net**,
mirroring `scoreDiffusion.compile` but without an area optimizer:

```python
self.voxel_optimizers[n] = type(optimizer).from_config(optimizer.get_config())
```

Each region's gradient tape, gradient clip (`clip_by_norm(grad, 1)`), and
`apply_gradients` call are fully independent — no coupling between regions.

The `loss` metric reported to `MultiModelCheckpoint` is the **area-weighted sum**
`Σ(AREA_RATIOS[n] × loss_voxel_n)`, computed after all gradient updates for
monitoring and early-stopping only. It has zero effect on any gradient.

### Inference after distillation

`scoreDiffusion_distill.generate()` is structurally identical to
`scoreDiffusion.generate()`:

- Stage 1 uses `self.ema_area` (= frozen teacher ResNet) at `self.num_steps`
- Stage 2 uses `self.ema_voxels[n]` (distilled student U-Net EMA) at `self.num_steps`

Because `self.num_steps = teacher.num_steps // factor`, both stages run at the
reduced step count. The ResNet is an MLP over 4 scalars so the step reduction
has negligible wall-clock impact there; the speedup comes entirely from the U-Nets.

---

## Key Design Choices

- **Hierarchical:** area counts constrain voxel distributions
- **EMA weights** used exclusively at inference (ema_decay = 0.999)
- **Fully deterministic:** DDIM η=0 → same φ → same output
- **Batch padding** to fixed size → no TF graph retracing
- GPU memory growth enabled; CPU fallback supported
