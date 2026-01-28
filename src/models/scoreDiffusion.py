# scoreDiffusion.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
from src.models.architectures import Unet, Resnet
import time

try:
    import horovod.tensorflow.keras as hvd
    HVD_AVAILABLE = True
except ImportError:
    hvd = None
    HVD_AVAILABLE = False

# Reproduzierbarkeit
#tf.random.set_seed(1234)

class scoreDiffusion(keras.Model):
    """Score-based generative model for calorimeter simulation.
    
    Generates optical response of neutron capture events across 4 detector areas (PIT, BOT, WALL, TOP).
    Uses separate U-Nets for voxel hit distributions per area and a Resnet for
    total hit counts per area.
    """
    class SinCosEmbedding(layers.Layer):
        """Sinusoidal positional embedding layer"""
        def __init__(self, projection, **kwargs):
            super().__init__(**kwargs)
            self.projection = projection
        
        def call(self, inputs):
            angle = inputs * self.projection * 1000
            return tf.concat([tf.math.sin(angle), tf.math.cos(angle)], -1)
        
        def get_config(self):
            config = super().get_config()
            config.update({"projection": self.projection})
            return config
    
    def __init__(self, num_area,num_cond=1,name='SGM',config=None):
        """Initialize scoreDiffusion model.
        
        Args:
            num_areas: Legacy parameter (ignored, computed from config)
            num_cond: Number of conditional features (physics)
            name: Model name
            config: Configuration dict with keys:
                - LAYER_NAMES: List of area names ['PIT', 'BOT', 'WALL', 'TOP']
                - SHAPE_PIT/BOT/WALL/TOP: Grid shapes per area [z, phi, 1]
                - EMBED: Embedding dimension for time/conditioning
                - num_steps: Number of diffusion timesteps
                - ema_decay: EMA decay rate for model weights
                - PAD: Padding mode for U-Net
                - AREA_RATIOS: Area size ratios for loss weighting
        """
        super(scoreDiffusion, self).__init__()
        if config is None:
            raise ValueError("Config file not given")
        
        self.num_cond = num_cond        
        self.config = config
        self.num_embed = self.config['EMBED']
        self.area_names = self.config['LAYER_NAMES']
        self.num_area = num_area
        # Separate optimizers (will be set during compile)
        self.area_optimizer = None
        self.voxel_optimizers = {area_name: None for area_name in self.area_names}
        self.active_areas = self.area_names  

        # Voxel grid shapes per area: [z_bins, phi_bins, 1]
        self.shapes = {
            'PIT': config['SHAPE_PIT'],
            'BOT': config['SHAPE_BOT'],
            'WALL': config['SHAPE_WALL'],
            'TOP': config['SHAPE_TOP']
        }
        # Diffusion process parameters
        self.num_steps = self.config["num_steps"]
        self.ema = self.config["ema_decay"]           
                
        self.verbose = 1 if (hvd and hvd.rank() == 0) else 1 #show progress only for first rank

        # Initialize Gaussian Fourier projection for time embedding          
        self.projection = self.GaussianFourierProjection(scale = 16)

        # Metrics: total loss, area hit loss, and per-area voxel losses
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_area_tracker = keras.metrics.Mean(name="area_loss")
        self.loss_voxel_trackers = {
            area_name: keras.metrics.Mean(name=f"voxel_loss_{area_name}")
            for area_name in self.area_names
        }

        # Swish activation
        self.activation = tf.keras.activations.swish

        # ===== Input Definitions =====
        inputs_time = Input((1,), name='time') # Diffusion timestep t ∈ [0,1]
        inputs_cond = Input((self.num_cond,), name='condition') # Physics
        inputs_area_hits = Input((self.num_area,), name='area_hits')  # Total hits per area (4D vector)

        # Embed global condition (e.g. incident particle energy)
        dense_cond = layers.Dense(self.num_embed,activation=None)(inputs_cond) 
        dense_cond = self.activation(dense_cond)  
        
        # ===== Build 4 separate U-Nets (one per detector area) =====
        self.model_voxels = {}      # Voxel distribution models
        self.ema_voxels = {}        # EMA versions for inference

        # Nur aktive Areas trainieren
        for area_name in self.active_areas:
            data_shape = self.shapes[area_name]

            # Extract scalar hit count for this specific area from 4D vector
            area_idx = self.area_names.index(area_name)
            area_hits_scalar = layers.Lambda(
                lambda x, idx: tf.expand_dims(x[:, idx], -1),
                arguments={'idx': area_idx},
                output_shape=(1,)
            )(inputs_area_hits)
            
            # Embed area hit count
            dense_area = layers.Dense(self.num_embed, activation=None)(area_hits_scalar)
            dense_area = self.activation(dense_area)
            
            # Create time embedding (sinusoidal positional encoding)
            voxel_conditional = self.Embedding(inputs_time, self.projection)

            # Combine: time + area_hits + global_condition → conditioning vector
            voxel_conditional = layers.Concatenate()(
                [voxel_conditional, dense_area, dense_cond]
            )
            voxel_conditional = layers.Dense(self.num_embed, activation=None)(voxel_conditional)
            
            # Build U-Net for this area's voxel distribution
            # Input: noisy voxel grid [batch, z, phi, 1]
            # Output: denoised voxel grid (same shape)
            inputs, outputs = Unet(
                data_shape,
                voxel_conditional,
                input_embedding_dims=16,
                stride=2,
                kernel=3,
                block_depth=4,
                widths=[32, 64, 96, 128],
                attentions=[False, True, True, True],   # Self-attention in deeper layers
                pad=self.config['PAD'][area_name],
                use_1D=False                             # Treat as 2D with channel dimension
            )

            # Wrap U-Net in Keras Model
            # Takes: [voxel_grid, time, area_hits_vector, condition]
            self.model_voxels[area_name] = keras.Model(
                inputs=[inputs, inputs_time, inputs_area_hits, inputs_cond],
                outputs=outputs,
                name=f'unet_{area_name}'
            )
            
            # Clone for EMA (used during generation)
            self.ema_voxels[area_name] = keras.models.clone_model(
                self.model_voxels[area_name]
            )
        
        # ===== Build Resnet for area hit distribution (4D vector) =====
        # Predicts total hit counts across 4 areas conditioned on global features
        
        # Time embedding for area model
        layer_conditional = self.Embedding(inputs_time, self.projection)
        layer_conditional = layers.Concatenate()([layer_conditional, dense_cond])
        layer_conditional = layers.Dense(self.num_embed, activation=None)(layer_conditional)

        # Resnet outputs 4D vector: [hits_PIT, hits_BOT, hits_WALL, hits_TOP]
        outputs = Resnet(
            inputs_area_hits,
            self.num_area,
            layer_conditional,
            num_embed=self.num_embed,
            num_layer=3,
            mlp_dim=128,
        )

        self.model_area = keras.Model(
            inputs=[inputs_area_hits, inputs_time, inputs_cond],
            outputs=outputs,
            name='resnet_area_hits'
        )
        self.ema_area = keras.models.clone_model(self.model_area)
        
        # Print model summaries on rank 0
        if self.verbose:
            print("\n=== Area Hits Model ===")
            for area_name in self.active_areas:
                print(f"\n=== {area_name} U-Net ===")
                print(self.model_voxels[area_name].summary())
        
    @property
    def metrics(self):
        """Metrics tracked during training/validation."""
        return [self.loss_tracker, self.loss_area_tracker] + \
           list(self.loss_voxel_trackers.values())
    
    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        # Clone optimizer for each sub-model
        self.area_optimizer = type(optimizer).from_config(optimizer.get_config())
        self.voxel_optimizers = {
            area_name: type(optimizer).from_config(optimizer.get_config())
            for area_name in self.area_names
        }
        
        # Build optimizers with correct variables
        self.area_optimizer.build(self.model_area.trainable_variables)
        for area_name in self.active_areas:
            self.voxel_optimizers[area_name].build(
                self.model_voxels[area_name].trainable_variables
            )

    def GaussianFourierProjection(self,scale = 30):
        """Create frequencies for Gaussian Fourier time embedding.
        
        Returns sinusoidal frequencies for positional encoding of timesteps.
        """
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb, tf.float32)
        freq = tf.exp(-emb * tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq

    def Embedding(self,inputs,projection):
        """Sinusoidal time embedding.
        
        Maps scalar timestep t to high-dimensional embedding via sin/cos functions.
        Standard approach from "Attention Is All You Need" (Vaswani et al., 2017).
        
        Args:
            inputs: Timestep tensor [batch, 1]
            projection: Frequency tensor from GaussianFourierProjection
            
        Returns:
            Embedded timestep [batch, num_embed]
        """
        embedding = self.SinCosEmbedding(projection)(inputs)
        embedding = layers.Dense(2*self.num_embed, activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding
        
    def prior_sde(self,dimensions):
        """Sample from prior distribution (standard Gaussian noise)."""
        return tf.random.normal(dimensions)

    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        """Cosine noise schedule for diffusion process.
        
        Maps timestep t ∈ [0,1] to log-SNR (signal-to-noise ratio).
        From "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
        
        Args:
            t: Timestep [batch, 1]
            logsnr_min/max: SNR bounds
            
        Returns:
            log(α²/σ²) where α=signal weight, σ=noise weight
        """
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    @tf.function
    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min=-20., logsnr_max=20.):
        """Inverse of cosine schedule (logsnr → t)."""
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a
        
    @tf.function
    def get_logsnr_alpha_sigma(self,time):
        """Convert timestep to noise schedule parameters.
        
        Returns:
            logsnr: Log signal-to-noise ratio
            alpha: Signal weight √(sigmoid(logsnr))
            sigma: Noise weight √(sigmoid(-logsnr))
        """
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        
        return logsnr, alpha, sigma    

    @tf.function
    def train_step(self, inputs):
        """Single training step using velocity parameterization.
        
        Trains two models in parallel:
        1. Area hits model: Predicts 4D vector of total hits per area
        2. Voxel models: Predict 2D hit distributions within each area
        
        Uses v-prediction: v = α*ε - σ*x₀ (interpolates between noise and data).
        From "Progressive Distillation for Fast Sampling" (Salimans & Ho, 2022).
        
        Args:
            inputs: Tuple of (voxels_dict, area_hits, condition)
                - voxels_dict: {'PIT': [batch,z,phi,1], ...}
                - area_hits: [batch, 4] total hits per area
                - condition: [batch, num_cond] global features
                
        Returns:
            Dict of losses for logging
        """
        voxels,area_hits,cond = inputs

        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)

        # ===== Train Area Hits Model =====
        with tf.GradientTape() as tape:
            # Add Gaussian noise: x_t = α*x₀ + σ*ε
            z = tf.random.normal((tf.shape(area_hits)),dtype=tf.float32)
            perturbed_x = alpha*area_hits + z * sigma

            # Predict velocity: v = α*ε - σ*x₀            
            score = self.model_area([perturbed_x, random_t,cond])
            v = alpha * z - sigma * area_hits

            # MSE loss on velocity prediction
            losses = tf.square(score - v)                        
            loss_area = tf.reduce_mean(losses)
            
        # Update area model weights
        trainable_variables = self.model_area.trainable_variables
        g = tape.gradient(loss_area, trainable_variables)
        g = [tf.clip_by_norm(grad, 1) for grad in g]
        self.area_optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_area.weights, 
                                    self.ema_area.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
    
        # ===== Train Voxel Models (parallel over 4 areas) =====
        total_voxel_loss = 0.0
    
        for area_name in self.active_areas:
            voxel = voxels[area_name]   # [batch, z, phi, 1]

            # Reshape alpha/sigma to match voxel dimensions
            shape = [-1] + [1] * len(self.shapes[area_name])   # [-1, 1, 1, 1]         
            alpha_reshape = tf.reshape(alpha, shape)
            sigma_reshape = tf.reshape(sigma, shape)
            
            with tf.GradientTape() as tape:
                # Perturb voxels with same timestep as area hits
                z = tf.random.normal(tf.shape(voxel), dtype=tf.float32)
                perturbed_x = alpha_reshape * voxel + z * sigma_reshape
                
                # Predict velocity conditioned on area_hits vector
                score = self.model_voxels[area_name](
                    [perturbed_x, random_t, area_hits, cond]
                )
                
                v = alpha_reshape * z - sigma_reshape * voxel
                losses = tf.square(score - v)
                loss_voxel = tf.reduce_mean(losses)

                # Weight loss by area size (larger areas get more weight)
                area_weight = self.config['AREA_RATIOS'][area_name]
                weighted_loss = loss_voxel * area_weight

                self.loss_voxel_trackers[area_name].update_state(weighted_loss)
                total_voxel_loss += weighted_loss
            
            # Update voxel model weights
            trainable_vars = self.model_voxels[area_name].trainable_variables
            g = tape.gradient(loss_voxel, trainable_vars)
            g = [tf.clip_by_norm(grad, 1) for grad in g]
            self.voxel_optimizers[area_name].apply_gradients(zip(g, trainable_vars))
            
            # Update EMA weights
            for weight, ema_weight in zip(
                self.model_voxels[area_name].weights,
                self.ema_voxels[area_name].weights
            ):
                ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        
        # Aggregate losses
        self.loss_tracker.update_state(total_voxel_loss + loss_area)
        self.loss_area_tracker.update_state(loss_area)
        
        results = {
            "loss": self.loss_tracker.result(),
            "loss_area": self.loss_area_tracker.result(),
        }
        results.update({
            f"loss_{name}": tracker.result() 
            for name, tracker in self.loss_voxel_trackers.items()
        })
        
        return results

    @tf.function
    def test_step(self, inputs):
        """Validation step (no weight updates)."""
        voxels, area_hits, cond = inputs
        
        random_t = tf.random.uniform((tf.shape(cond)[0], 1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        
        # ----- Area Hits Validation -----
        z = tf.random.normal(tf.shape(area_hits), dtype=tf.float32)
        perturbed_x = alpha * area_hits + z * sigma            
        score = self.model_area([perturbed_x, random_t, cond])
        v = alpha * z - sigma * area_hits
        losses = tf.square(score - v)
        loss_area = tf.reduce_mean(losses)
        
        # ----- Voxel Validation (parallel über alle Layer) -----
        total_voxel_loss = 0.0
        
        for area_name in self.active_areas:
            voxel = voxels[area_name]
            shape = [-1] + [1] * len(self.shapes[area_name])
            
            alpha_reshape = tf.reshape(alpha, shape)
            sigma_reshape = tf.reshape(sigma, shape)
            
            z = tf.random.normal(tf.shape(voxel), dtype=tf.float32)
            perturbed_x = alpha_reshape * voxel + z * sigma_reshape
            
            score = self.model_voxels[area_name](
                [perturbed_x, random_t, area_hits, cond],
                training=False
            )
            
            v = alpha_reshape * z - sigma_reshape * voxel
            losses = tf.square(score - v)
            loss_voxel = tf.reduce_mean(losses)
            
            # Area-weighted loss
            area_weight = self.config['AREA_RATIOS'][area_name]
            weighted_loss = loss_voxel * area_weight
            
            self.loss_voxel_trackers[area_name].update_state(weighted_loss)
            total_voxel_loss += weighted_loss
        
        self.loss_tracker.update_state(total_voxel_loss + loss_area)
        self.loss_area_tracker.update_state(loss_area)
        
        results = {
            "loss": self.loss_tracker.result(),
            "loss_layer": self.loss_area_tracker.result(),
        }
        results.update({
            f"loss_{name}": tracker.result() 
            for name, tracker in self.loss_voxel_trackers.items()
        })
        
        return results
            
    @tf.function
    def call(self,x):    
        """Forward pass (not used, required by Keras API)."""    
        return self.model(x)

    def generate(self,cond):
        """Generate samples via DDPM sampling.
        
        Two-stage generation:
        1. Sample area_hits vector (4D) from Resnet
        2. Sample voxel grids for each area conditioned on area_hits
        
        Args:
            cond: Global conditions [batch, num_cond]
            
        Returns:
            voxels_np: Dict of voxel grids {'PIT': array, ...}
            area_hits_np: Total hits per area [batch, 4]
        """
        start = time.time()
        
        # 1. Generate area hits (4D vector)
        area_hits = self.DDPMSampler(
            cond, 
            self.ema_area,
            data_shape=[self.num_areas],
            const_shape=[-1, 1]
        )

        # 2. Generiere Voxels parallel für alle Layer
        voxels = {}
        for area_name in self.active_areas:
            voxels[area_name] = self.DDPMSampler(
                cond,
                self.ema_voxels[area_name],
                data_shape=self.shapes[area_name],
                const_shape=[-1] + [1] * len(self.shapes[area_name]),
                layer_energy=area_hits  # Condition voxels on total hits
            )
        
        end = time.time()
        print(f"Time for sampling {cond.shape[0]} events: {end - start:.2f}s")
        
        # Convert to numpy
        voxels_np = {k: v.numpy() for k, v in voxels.items()}
        return voxels_np, area_hits.numpy()        

    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    area_hits=None):
        """DDPM sampling with velocity parameterization.
        
        Iteratively denoises from pure Gaussian noise to data distribution.
        Uses deterministic DDIM-style update: x_{t-1} = α'*mean + σ'*ε
        
        Args:
            cond: Conditioning features
            model: EMA model to use for denoising
            data_shape: Shape of output (excluding batch)
            const_shape: Shape for broadcasting alpha/sigma
            area_hits: Optional area hits for voxel generation
            
        Returns:
            Generated samples (final mean)
        """        
        batch_size = cond.shape[0]
        data_shape = np.concatenate(([batch_size],data_shape))
        # Start from pure noise
        x = self.prior_sde(data_shape)
        
        # Iterative denoising from t=1 to t=0
        for time_step in tf.range(self.num_steps, 0, delta=-1):
            # Current and next timestep
            random_t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps)

            # Predict velocity: v = α*ε - σ*x₀
            if area_hits is None:
                # Area hits model (no additional conditioning)
                score = model([x, random_t,cond],training=False)
            else:
                # Voxel model (conditioned on area_hits)
                score = model([x, random_t,area_hits,cond],training=False)

                # Reshape for broadcasting over voxel dimensions
                alpha = tf.reshape(alpha, const_shape)
                sigma = tf.reshape(sigma, const_shape)
                alpha_ = tf.reshape(alpha_, const_shape)
                sigma_ = tf.reshape(sigma_, const_shape)

            # Recover mean: x₀ = (α*x_t - σ*v) / α                
            mean = alpha * x - sigma * score

            # Recover noise: ε = (x_t - α*x₀) / σ
            eps = (x - alpha * mean) / sigma
            
            # Update: x_{t-1} = α'*x₀ + σ'*ε
            x = alpha_ * mean + sigma_ * eps

        # Return final mean (no noise at t=0)
        return mean

        