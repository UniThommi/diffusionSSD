import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def build_diffusion_model(phi_dim=23, target_dim=9583, T=1000, 
                         t_emb_dim=64, hidden_dim=512, n_layers=4,
                         use_residual=True, use_attention=False):
    """
    Physics-Aware Diffusion Model
    
    Verbesserungen:
    1. Größeres Timestep-Embedding (64 statt 32)
    2. Separate Phi-Processing Network
    3. Optional: Self-Attention Layer
    4. LayerNormalization für Stabilität
    
    Args:
        phi_dim: NC-Parameter Dimension (22)
        target_dim: Voxel Dimension (7789 oder 9583)
        T: Diffusion Steps (1000)
        t_emb_dim: Timestep Embedding Dimension (64 empfohlen)
        hidden_dim: Hidden Layer Size (512)
        n_layers: Anzahl Hidden Layers (4)
        use_residual: Residual Connections (True)
        use_attention: Self-Attention einbauen (False, zu teuer)
    """
    # Inputs
    x_input = tf.keras.Input(shape=(target_dim,), name="x_t")
    phi_input = tf.keras.Input(shape=(phi_dim,), name="phi")
    t_input = tf.keras.Input(shape=(), dtype=tf.int32, name="timestep")
    
    # 1. Timestep Embedding (größer für bessere Konditionierung)
    t_emb = tf.keras.layers.Embedding(T, t_emb_dim, name="t_embedding")(t_input)
    t_emb = tf.keras.layers.Dense(t_emb_dim, activation="swish", name="t_dense1")(t_emb)
    t_emb = tf.keras.layers.Dense(t_emb_dim, activation="swish", name="t_dense2")(t_emb)
    
    # 2. Phi Processing Network (Separate für besseres Learning)
    phi_processed = tf.keras.layers.Dense(hidden_dim//2, activation="swish", 
                                         name="phi_dense1")(phi_input)
    phi_processed = tf.keras.layers.LayerNormalization(name="phi_norm1")(phi_processed)
    phi_processed = tf.keras.layers.Dense(hidden_dim//2, activation="swish",
                                         name="phi_dense2")(phi_processed)
    phi_processed = tf.keras.layers.LayerNormalization(name="phi_norm2")(phi_processed)
    
    # 3. X Compression (Efficient)
    x_compressed = tf.keras.layers.Dense(hidden_dim, activation="swish", 
                                        name="x_compress")(x_input)
    x_compressed = tf.keras.layers.LayerNormalization(name="x_norm")(x_compressed)
    
    # 4. Concatenate alle Inputs
    x = tf.keras.layers.Concatenate(name="concat_inputs")([
        x_compressed, phi_processed, t_emb
    ])
    
    # 5. Deep Processing mit Residual Connections
    for i in range(n_layers):
        if use_residual and i > 0:
            # Residual Connection
            residual = x
            x = tf.keras.layers.Dense(hidden_dim, activation="swish", 
                                     name=f"hidden_{i}_1")(x)
            x = tf.keras.layers.LayerNormalization(name=f"norm_{i}_1")(x)
            x = tf.keras.layers.Dense(hidden_dim, activation="swish",
                                     name=f"hidden_{i}_2")(x)
            x = tf.keras.layers.LayerNormalization(name=f"norm_{i}_2")(x)
            
            # Residual Add
            x = tf.keras.layers.Add(name=f"residual_{i}")([x, residual])
        else:
            x = tf.keras.layers.Dense(hidden_dim, activation="swish", 
                                     name=f"hidden_{i}")(x)
            x = tf.keras.layers.LayerNormalization(name=f"norm_{i}")(x)
        
        # Dropout für Regularisierung
        x = tf.keras.layers.Dropout(0.1, name=f"dropout_{i}")(x)
    
    # Optional: Self-Attention (nur wenn use_attention=True)
    if use_attention:
        # Reshape für Attention: (batch, 1, hidden_dim)
        x_reshaped = tf.keras.layers.Reshape((1, hidden_dim))(x)
        
        # Multi-Head Attention
        attention_out = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=hidden_dim//4,
            name="self_attention"
        )(x_reshaped, x_reshaped)
        
        # Flatten zurück
        x = tf.keras.layers.Flatten()(attention_out)
    
    # 6. Output Layer (zurück zu target_dim)
    x = tf.keras.layers.Dense(target_dim, name="output")(x)
    
    model = tf.keras.Model(
        inputs=[x_input, phi_input, t_input], 
        outputs=x, 
        name="physics_aware_diffusion_model"
    )
    
    return model


# Beispiel für Memory-Monitoring
def monitor_memory():
    """Hilfsfunktion zum Memory-Monitoring"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Memory-effiziente Konfiguration
def set_gpu_memory_limit(memory_limit_mb=4096):
    """Begrenzt GPU-Memory"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
    except RuntimeError as e:
        print(f"GPU memory limit setting failed: {e}")


if __name__ == "__main__":
    # Memory-Optimierungen
    monitor_memory()
    
    # Teste verschiedene Modell-Varianten
    print("=== Memory-efficient Model ===")
    model1 = build_diffusion_model(hidden_dim=256)
    model1.summary()
    print(f"Parameters: {model1.count_params():,}")