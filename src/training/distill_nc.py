#!/usr/bin/env python3
"""
distill_nc.py — Progressive distillation of scoreDiffusion U-Nets.

Implements one stage of progressive distillation (Salimans & Ho 2022):
  - Teacher: trained scoreDiffusion with frozen EMA models.
  - Student: per-region U-Nets that learn to match the teacher's 2-step
    denoising result in a single step, halving the required sampling steps.
  - ResNet (area model): frozen teacher EMA, passed through unchanged.
    It is the fragile conditioning root and not the generation bottleneck.

Distillation can be chained: the distilled model becomes the teacher for
the next stage (e.g. 512 → 256 → 128 steps with factor=2 each stage).

Usage via train.py:
    Set enable_distillation = true and distillation_factor = 2 in config.toml.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras


class scoreDiffusion_distill(keras.Model):
    """Progressive distillation of scoreDiffusion U-Nets.

    Each U-Net is distilled independently:
      - Its own GradientTape and optimizer update (unweighted loss_voxel).
      - No coupling between regions during optimization.

    The area-weighted aggregate "loss" is computed purely for logging and
    checkpoint monitoring (MultiModelCheckpoint monitors val_loss). It has
    zero effect on any gradient.
    """

    def __init__(self, teacher, factor: int, config: dict = None):
        """
        Args:
            teacher: Trained scoreDiffusion instance with EMA weights already
                     loaded from checkpoint before calling this constructor.
            factor:  Step-count compression per distillation stage.
                     factor=2 → student runs at teacher.num_steps // 2.
            config:  Preprocessed config dict (same object passed to teacher).
        """
        super().__init__()
        if config is None:
            raise ValueError("Config required")

        self.config       = config
        self.factor       = factor
        self.ema          = teacher.ema                    # EMA decay (e.g. 0.999)
        self.num_area     = teacher.num_area               # 4 (area hits dimension)
        self.active_areas = teacher.active_areas           # e.g. ['PITBOT','WALL','TOP']
        self.shapes       = teacher.shapes                 # per-region [z, phi, 1]
        self.num_steps    = teacher.num_steps // factor    # e.g. 512//2 = 256

        # ── Teacher: frozen EMA models ────────────────────────────────────────
        # Use teacher EMA (best-quality weights) as the oracle for distillation.
        self.teacher_area   = teacher.ema_area
        self.teacher_voxels = {n: teacher.ema_voxels[n] for n in self.active_areas}
        self.teacher_area.trainable = False
        for n in self.active_areas:
            self.teacher_voxels[n].trainable = False

        # ── Student U-Nets: architecture cloned from teacher EMA ─────────────
        # Keras Functional models are built on construction, so clone_model
        # works immediately without a dummy forward pass.
        self.model_voxels = {}
        for n in self.active_areas:
            self.model_voxels[n] = keras.models.clone_model(self.teacher_voxels[n])
            self.model_voxels[n].set_weights(self.teacher_voxels[n].get_weights())

        # ── EMA tracking models for student U-Nets ───────────────────────────
        self.ema_voxels = {}
        for n in self.active_areas:
            self.ema_voxels[n] = keras.models.clone_model(self.model_voxels[n])
            self.ema_voxels[n].set_weights(self.model_voxels[n].get_weights())

        # ── Expose frozen teacher ResNet as self.ema_area ─────────────────────
        # MultiModelCheckpoint and generate() both access self.model.ema_area.
        # Assigning teacher.ema_area keeps those callbacks working without
        # modification. The ResNet EMA weights are re-saved each checkpoint.
        self.ema_area = teacher.ema_area

        # ── Optimizers (set during compile) ───────────────────────────────────
        self.voxel_optimizers = {n: None for n in self.active_areas}

        # ── Metrics ───────────────────────────────────────────────────────────
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_voxel_trackers = {
            n: keras.metrics.Mean(name=f"loss_{n}") for n in self.active_areas
        }

    @property
    def metrics(self):
        return [self.loss_tracker] + list(self.loss_voxel_trackers.values())

    def compile(self, optimizer, **kwargs):
        """Create one independent optimizer per student U-Net.

        No area optimizer — the ResNet is frozen and not being trained.
        Mirrors scoreDiffusion.compile but without self.area_optimizer.
        """
        super().compile(**kwargs)
        self.voxel_optimizers = {
            n: type(optimizer).from_config(optimizer.get_config())
            for n in self.active_areas
        }
        for n in self.active_areas:
            self.voxel_optimizers[n].build(self.model_voxels[n].trainable_variables)

    # ── Diffusion schedule (copied verbatim from distillation.py) ────────────

    @tf.function
    def get_logsnr_alpha_sigma(self, time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha  = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma  = tf.sqrt(tf.math.sigmoid(-logsnr))
        return logsnr, alpha, sigma

    @tf.function
    def logsnr_schedule_cosine(self, t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t, tf.float32) + b))

    def prior_sde(self, dimensions):
        return tf.random.normal(dimensions)

    @tf.function
    def call(self, x):
        """Forward pass (not used, required by Keras API)."""
        return x

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self, inputs):
        """Distillation training step.

        One student step learns to replicate two teacher steps. All U-Nets
        share the same sampled timestep i (consistent with scoreDiffusion.train_step).

        Loss handling
        -------------
        OPTIMIZATION: each U-Net uses its own unweighted loss_voxel.
            GradientTape → apply_gradients is per-region with no cross-coupling.
        LOGGING ONLY: area-weighted Σ(loss_voxel_r * weight_r) is accumulated
            after all gradient updates and stored as "loss" for the overview
            plot and MultiModelCheckpoint. Zero effect on any gradient.
        """
        voxels, area_hits, cond = inputs

        # Single timestep index shared across all U-Nets.
        # Matches scoreDiffusion.train_step: one random_t per batch.
        i = tf.random.uniform(
            (tf.shape(cond)[0], 1), minval=0, maxval=self.num_steps, dtype=tf.int32
        )
        u     = tf.cast(i + 1, tf.float32) / tf.cast(self.num_steps, tf.float32)
        u_mid = u - 0.5 / tf.cast(self.num_steps, tf.float32)
        u_s   = u - 1.0 / tf.cast(self.num_steps, tf.float32)

        # Schedule values — all shape [batch, 1]
        logsnr,     alpha,     sigma     = self.get_logsnr_alpha_sigma(u)
        logsnr_mid, alpha_mid, sigma_mid = self.get_logsnr_alpha_sigma(u_mid)
        logsnr_s,   alpha_s,   sigma_s   = self.get_logsnr_alpha_sigma(u_s)
        sigma_frac = tf.exp(
            0.5 * (tf.math.softplus(logsnr) - tf.math.softplus(logsnr_s))
        )

        total_voxel_loss = 0.0   # accumulates area-weighted sum for logging only

        # Loop unrolled at tf.function trace time (Python list → static graph).
        for area_name in self.active_areas:
            voxel    = voxels[area_name]                             # [batch, z, phi, 1]
            shape_4d = [-1] + [1] * len(self.shapes[area_name])     # [-1, 1, 1, 1]

            # Reshape scalar schedule values to broadcast over voxel dimensions
            a,     s     = tf.reshape(alpha,     shape_4d), tf.reshape(sigma,     shape_4d)
            a_mid, s_mid = tf.reshape(alpha_mid, shape_4d), tf.reshape(sigma_mid, shape_4d)
            a_s,   s_s   = tf.reshape(alpha_s,   shape_4d), tf.reshape(sigma_s,   shape_4d)
            sfrac        = tf.reshape(sigma_frac, shape_4d)

            # Noisy voxel at time u: x_t = α*x₀ + σ*ε
            eps_v = tf.random.normal(tf.shape(voxel), dtype=tf.float32)
            z_v   = a * voxel + s * eps_v

            # ── Teacher step 1: z_v @ u → mean_v → z_mid_v @ u_mid ───────────
            v_t    = self.teacher_voxels[area_name]([z_v, u, area_hits, cond], training=False)
            mean_v = a * z_v - s * v_t
            eps_v  = (z_v - a * mean_v) / s
            z_mid_v = a_mid * mean_v + s_mid * eps_v

            # ── Teacher step 2: z_mid_v @ u_mid → mean_v → z_teacher_v @ u_s ─
            v_t_mid = self.teacher_voxels[area_name](
                [z_mid_v, u_mid, area_hits, cond], training=False
            )
            mean_v  = a_mid * z_mid_v - s_mid * v_t_mid
            eps_v   = (z_mid_v - a_mid * mean_v) / s_mid
            z_teacher_v = a_s * mean_v + s_s * eps_v

            # ── Consistency target (Salimans & Ho 2022, eq. 15) ───────────────
            x_target_v = (z_teacher_v - sfrac * z_v) / (a_s - sfrac * a)
            # Guard: at i==0, sigma_s→0 and the formula is undefined.
            # Use mean_v directly (the teacher's denoised estimate).
            i_4d = tf.reshape(i, shape_4d)          # [batch, 1, 1, 1] for broadcasting
            x_target_v = tf.where(i_4d == 0, mean_v, x_target_v)

            eps_target_v = (z_v - a * x_target_v) / s
            v_target_v   = a * eps_target_v - s * x_target_v

            # ── Student gradient update — independent per U-Net ───────────────
            with tf.GradientTape() as tape:
                v_v = self.model_voxels[area_name]([z_v, u, area_hits, cond])
                loss_voxel = tf.reduce_mean(tf.square(v_v - v_target_v))  # UNWEIGHTED

            g = tape.gradient(loss_voxel, self.model_voxels[area_name].trainable_variables)
            g = [tf.clip_by_norm(grad, 1) for grad in g]
            self.voxel_optimizers[area_name].apply_gradients(
                zip(g, self.model_voxels[area_name].trainable_variables)
            )

            # EMA update for this region's student model
            for w, ew in zip(self.model_voxels[area_name].weights,
                             self.ema_voxels[area_name].weights):
                ew.assign(self.ema * ew + (1.0 - self.ema) * w)

            # Log unweighted per-region loss (the actual optimization signal)
            self.loss_voxel_trackers[area_name].update_state(loss_voxel)

            # Accumulate area-weighted loss for the combined overview plot ONLY.
            # All gradient updates for this region are already done above.
            area_weight = self.config['AREA_RATIOS'][area_name]
            total_voxel_loss += loss_voxel * area_weight

        # "loss" drives MultiModelCheckpoint and the combined overview plot only.
        # No loss_area term — the ResNet is frozen and not being distilled.
        self.loss_tracker.update_state(total_voxel_loss)

        results = {"loss": self.loss_tracker.result()}
        results.update({
            f"loss_{n}": t.result()           # unweighted, for individual U-Net plots
            for n, t in self.loss_voxel_trackers.items()
        })
        return results

    @tf.function
    def test_step(self, inputs):
        """Validation step — same computation as train_step, no weight updates."""
        voxels, area_hits, cond = inputs

        i = tf.random.uniform(
            (tf.shape(cond)[0], 1), minval=0, maxval=self.num_steps, dtype=tf.int32
        )
        u     = tf.cast(i + 1, tf.float32) / tf.cast(self.num_steps, tf.float32)
        u_mid = u - 0.5 / tf.cast(self.num_steps, tf.float32)
        u_s   = u - 1.0 / tf.cast(self.num_steps, tf.float32)

        logsnr,     alpha,     sigma     = self.get_logsnr_alpha_sigma(u)
        logsnr_mid, alpha_mid, sigma_mid = self.get_logsnr_alpha_sigma(u_mid)
        logsnr_s,   alpha_s,   sigma_s   = self.get_logsnr_alpha_sigma(u_s)
        sigma_frac = tf.exp(
            0.5 * (tf.math.softplus(logsnr) - tf.math.softplus(logsnr_s))
        )

        total_voxel_loss = 0.0

        for area_name in self.active_areas:
            voxel    = voxels[area_name]
            shape_4d = [-1] + [1] * len(self.shapes[area_name])

            a,     s     = tf.reshape(alpha,     shape_4d), tf.reshape(sigma,     shape_4d)
            a_mid, s_mid = tf.reshape(alpha_mid, shape_4d), tf.reshape(sigma_mid, shape_4d)
            a_s,   s_s   = tf.reshape(alpha_s,   shape_4d), tf.reshape(sigma_s,   shape_4d)
            sfrac        = tf.reshape(sigma_frac, shape_4d)

            eps_v = tf.random.normal(tf.shape(voxel), dtype=tf.float32)
            z_v   = a * voxel + s * eps_v

            v_t    = self.teacher_voxels[area_name]([z_v, u, area_hits, cond], training=False)
            mean_v = a * z_v - s * v_t
            eps_v  = (z_v - a * mean_v) / s
            z_mid_v = a_mid * mean_v + s_mid * eps_v

            v_t_mid = self.teacher_voxels[area_name](
                [z_mid_v, u_mid, area_hits, cond], training=False
            )
            mean_v  = a_mid * z_mid_v - s_mid * v_t_mid
            eps_v   = (z_mid_v - a_mid * mean_v) / s_mid
            z_teacher_v = a_s * mean_v + s_s * eps_v

            x_target_v = (z_teacher_v - sfrac * z_v) / (a_s - sfrac * a)
            i_4d = tf.reshape(i, shape_4d)
            x_target_v = tf.where(i_4d == 0, mean_v, x_target_v)

            eps_target_v = (z_v - a * x_target_v) / s
            v_target_v   = a * eps_target_v - s * x_target_v

            v_v = self.model_voxels[area_name]([z_v, u, area_hits, cond], training=False)
            loss_voxel = tf.reduce_mean(tf.square(v_v - v_target_v))

            self.loss_voxel_trackers[area_name].update_state(loss_voxel)

            area_weight = self.config['AREA_RATIOS'][area_name]
            total_voxel_loss += loss_voxel * area_weight

        self.loss_tracker.update_state(total_voxel_loss)

        results = {"loss": self.loss_tracker.result()}
        results.update({
            f"loss_{n}": t.result()
            for n, t in self.loss_voxel_trackers.items()
        })
        return results

    # ── Sampling ──────────────────────────────────────────────────────────────

    @tf.function
    def DDPMSampler(self, cond, model, data_shape=None, const_shape=None, area_hits=None):
        """DDPM sampling with velocity parameterization.

        Copied from scoreDiffusion.DDPMSampler. self.num_steps is already
        set to the reduced step count (teacher.num_steps // factor).
        Both the ResNet and the U-Nets run at this reduced step count during
        generate().
        """
        batch_size = cond.shape[0]
        data_shape = np.concatenate(([batch_size], data_shape))
        x = self.prior_sde(data_shape)

        for time_step in tf.range(self.num_steps, 0, delta=-1):
            random_t  = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr,  alpha,  sigma  = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(
                tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps
            )

            if area_hits is None:
                # Area hits model (ResNet)
                score = model([x, random_t, cond], training=False)
            else:
                # Voxel model (U-Net), conditioned on area_hits
                score  = model([x, random_t, area_hits, cond], training=False)
                alpha  = tf.reshape(alpha,  const_shape)
                sigma  = tf.reshape(sigma,  const_shape)
                alpha_ = tf.reshape(alpha_, const_shape)
                sigma_ = tf.reshape(sigma_, const_shape)

            mean = alpha * x - sigma * score
            eps  = (x - alpha * mean) / sigma
            x    = alpha_ * mean + sigma_ * eps

        return mean

    def generate(self, cond):
        """Generate samples: frozen teacher ResNet + distilled student U-Nets.

        Both the ResNet and the U-Nets run at self.num_steps (the reduced count).
        The ResNet is fast (MLP on 4 scalars) so the step reduction has negligible
        effect on wall-clock time there; the speedup comes from the U-Nets.
        """
        start = time.time()

        # Stage 1: generate area hits (4D) using frozen teacher ResNet
        area_hits = self.DDPMSampler(
            cond,
            self.ema_area,
            data_shape=[self.num_area],
            const_shape=[-1, 1]
        )

        # Stage 2: generate voxel grids using distilled student U-Nets
        voxels = {}
        for area_name in self.active_areas:
            voxels[area_name] = self.DDPMSampler(
                cond,
                self.ema_voxels[area_name],
                data_shape=self.shapes[area_name],
                const_shape=[-1] + [1] * len(self.shapes[area_name]),
                area_hits=area_hits
            )

        end = time.time()
        print(f"Time for sampling {cond.shape[0]} events: {end - start:.2f}s")

        return {k: v.numpy() for k, v in voxels.items()}, area_hits.numpy()
