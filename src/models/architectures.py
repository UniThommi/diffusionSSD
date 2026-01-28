# architectures.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import numpy as np
import keras.ops as ops
#import tensorflow_addons as tfa

def Unet(
        num_dim,
        time_embedding,
        input_embedding_dims,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        pad=((2,1),(0,0),(4,3)),
        use_1D=False,
        #pad=((1,0),(0,0),(1,0)),
):
    """
    U-Net Architektur für Diffusionsmodelle.
    
    Struktur: Encoder (Downsampling) → Bottleneck → Decoder (Upsampling)
    Skip-Connections verbinden Encoder- und Decoder-Ebenen (Feature-Erhaltung).
    
    Args:
        num_dim: Input-Dimensionalität (Shape der Voxel-Daten)
        time_embedding: Zeitliches Embedding für Diffusionsprozess
        input_embedding_dims: Anzahl Feature-Maps nach erster Convolution
        stride: Downsampling/Upsampling-Faktor (typisch 2)
        kernel: Kernel-Größe für Convolutions (typisch 3)
        block_depth: Anzahl ResidualBlocks pro U-Net-Ebene
        widths: Feature-Map-Breiten pro Ebene [32, 64, 96, 128]
        attentions: Ob Attention-Mechanismus pro Ebene [False, True, True, True]
        pad: Zero-Padding für 2D (kompensiert Convolution-Verlust)
        use_1D: Falls True, nutze 1D-Conv statt 2D-Conv
        
    Returns:
        inputs, outputs: Keras Input/Output für Model-Definition
    """
    # Referenz-Implementation basierend auf: Clear Diffusion (Keras)
    #https://github.com/beresandras/clear-diffusion-keras/blob/master/architecture.py
    #act = layers.LeakyReLU(alpha=0.01)

    act = tf.keras.activations.swish

    def ResidualBlock(width, attention):
        """
        Residual Block mit optionalem Attention-Mechanismus.
        
        Struktur:
        1. Conv → Add(time_embedding) → Conv
        2. Residual-Connection: input + output
        3. Optional: Multi-Head Self-Attention
        
        Residual-Connections helfen gegen Vanishing-Gradient-Problem bei tiefen Netzen.
        
        Args:
            width: Anzahl Feature-Maps
            attention: Bool - ob Self-Attention angewendet wird
        """
        def forward(x):
            x , n = x # x: Feature-Maps, n: Zeit-Embedding
            # Falls Dimension nicht passt, projiziere mit 1x1 Convolution
            input_width = x.shape[2] if use_1D else x.shape[3]
            if input_width == width:
                residual = x
            else:
                if use_1D:
                    residual = layers.Conv1D(width, kernel_size=1)(x)
                else:
                    residual = layers.Conv2D(width, kernel_size=1)(x)

            # === Zeit-Embedding Integration ===
            # Projiziere Zeit-Embedding in Feature-Raum
            n = layers.Dense(width)(n)
            #x = tfa.layers.GroupNormalization(groups=4)(x)

            # === Erste Convolution ===
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = layers.Conv2D(width, kernel_size=kernel, padding="same")(x)

            # Addiere Zeit-Information
            x = layers.Add()([x, n])
            #x = tfa.layers.GroupNormalization(groups=4)(x)

            # === Zweite Convolution ===
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = layers.Conv2D(width, kernel_size=kernel, padding="same")(x)

            # Residual-Connection: Output = Input + Transformation(Input)
            x = layers.Add()([residual, x])

            # === Optional: Self-Attention ===
            # Attention ermöglicht globale Abhängigkeiten (nicht nur lokale wie Conv)
            if attention:
                residual = x
                if use_1D:
                    # LayerNorm für 1D-Daten                    
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
                    # Attention über räumliche Dimension (axis=1)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1)
                    )(x, x)
                else:
                    # GroupNorm für 3D-Daten (stabiler als BatchNorm bei kleinen Batches)
                    #x = tfa.layers.GroupNormalization(groups=4, center=False, scale=False)(x)
                    x = keras.layers.GroupNormalization(groups=4, center=False, scale=False)(x)
                    # Attention über alle räumlichen Dimensionen (x, y, z)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1, 2, 3)
                    )(x, x)

                # Residual-Connection für Attention
                x = layers.Add()([residual, x])
            return x
        return forward

    def DownBlock(block_depth, width, attention):
        """
        Encoder-Block: Feature-Extraktion + Downsampling.
        
        Pipeline:
        1. Wiederhole block_depth × ResidualBlock (Feature-Verarbeitung)
        2. Speichere Features in Skip-Liste (für Decoder)
        3. Average-Pooling: Reduziere räumliche Auflösung um Faktor stride
        
        Downsampling reduziert Dimensionalität und vergrößert rezeptives Feld.
        """
        def forward(x):
            x, n, skips = x # x: Features, n: Zeit-Embedding, skips: Skip-Connection-Liste

            # Mehrere Residual-Blocks für Feature-Extraktion
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x,n])
                skips.append(x)     # Speichere für Skip-Connection im Decoder

            # Downsampling: Reduziere räumliche Dimensionen
            if use_1D:
                x = layers.AveragePooling1D(pool_size=stride)(x)
            else:
                x = layers.AveragePooling2D(pool_size=stride)(x)
            return x

        return forward

    def UpBlock(block_depth, width, attention):
        """
        Decoder-Block: Upsampling + Feature-Rekonstruktion.
        
        Pipeline:
        1. Upsampling: Vergrößere räumliche Auflösung um Faktor stride
        2. Wiederhole block_depth ×:
           a. Concatenate mit Skip-Connection (kombiniere Low- und High-Level-Features)
           b. ResidualBlock (Feature-Verarbeitung)
        
        Skip-Connections ermöglichen präzise Rekonstruktion durch direkte High-Res-Info.
        """
        def forward(x):
            x, n, skips = x
            # Upsampling: Vergrößere räumliche Dimensionen (Nearest-Neighbor-Interpolation)
            if use_1D:
                x = layers.UpSampling1D(size=stride)(x)
            else:
                x = layers.UpSampling2D(size=stride)(x)
            # Kombiniere mit Skip-Connections aus Encoder
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width, attention)([x,n])
            return x

        return forward

     # === U-Net Konstruktion ===
    
    # Input: Voxel-Daten
    inputs = keras.Input((num_dim))
    if use_1D:
        #No padding to 1D model
        x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs)
        # Zeit-Embedding: Reshape für Broadcasting über 1D-Features
        n = layers.Reshape((1,time_embedding.shape[-1]))(time_embedding)
    else:
        # 2D-Daten: Zero-Padding kompensiert Größenverlust durch Convolutions
        # Padding asymmetrisch: pad=(before, after) pro Dimension

        # NEU: Prüfe ob Padding nötig ist (pad != 0)
        if pad != 0:
            inputs_padded = layers.ZeroPadding2D(pad)(inputs)
            x = layers.Conv2D(input_embedding_dims, kernel_size=1)(inputs_padded)
        else:
            # Kein Padding (z.B. für WALL mit pad=0)
            x = layers.Conv2D(input_embedding_dims, kernel_size=1)(inputs)
            
        # Zeit-Embedding: Reshape für Broadcasting über 2D-Features
        n = layers.Reshape((1,1,time_embedding.shape[-1]))(time_embedding)
    
    # Skip-Connection-Liste (wird von Encoder gefüllt, von Decoder geleert)
    skips = []
    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, n, skips])

    # === Bottleneck ===
    # Tiefste Ebene: Höchste semantische Abstraktion, niedrigste Auflösung
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])([x,n])

    # === Decoder-Pfad ===
    # Rekonstruiere in umgekehrter Reihenfolge (symmetrisch zum Encoder)
    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, n,  skips])

    # === Output-Projektion ===
    # Projiziere auf 1 Kanal (Hits pro Voxel)
    # Zero-Initialization: Zu Beginn prädiziert Modell nahe Null (stabiler Training-Start)
    if use_1D:
        outputs = layers.Conv1D(1, kernel_size=1, kernel_initializer="zeros")(x)
    else:
        outputs = layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)
        outputs = layers.Cropping2D(pad)(outputs)


    return inputs, outputs

def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
        activation='leakyrelu'
):
    """
    Residual MLP für Area-Hit-Modellierung.
    
    Verwendet Affine Conditioning (nicht echtes FiLM) für Zeit-Konditionierung:
    - Zeit-Embedding wird direkt in scale/shift-Parameter projiziert
    - Output = (1 + scale) * Features + shift
    - Kein separates Generator-Netzwerk (würde bei echtem FiLM existieren)
    
    Die affine Transformation ermöglicht dem Modell, Features zeitabhängig
    zu verstärken/dämpfen (scale) und zu verschieben (shift). Der "+1" Term
    in (1 + scale) sorgt für Identity-Initialisierung (bei scale=0 bleibt x unverändert).
    scale/shift sind global über alle Features (nicht feature-wise individuell)
    
    --> stärkere Konditionierung ist für Area-Hits.
    
    Args:
        inputs: Input-Layer (Area-Hits)
        end_dim: Output-Dimensionalität (typisch = num_areas)
        time_embedding: Zeitliches Embedding für Diffusionsprozess
        num_embed: Embedding-Dimensionalität (nicht verwendet)
        num_layer: Anzahl Residual-Blocks
        mlp_dim: Hidden-Layer-Dimensionalität
        activation: Aktivierungsfunktion
        
    Returns:
        outputs: Keras Layer (prädizierte Area-Hits)
    """
    
    act = layers.LeakyReLU(alpha=0.01)

    def resnet_dense(input_layer,hidden_size):
        layer,time = input_layer

        # === Residual-Pfad ===
        # Projiziere Input auf korrekte Dimension
        residual = layers.Dense(hidden_size)(layer)
        # Generiere scale und shift aus Zeit-Embedding
        embed =  layers.Dense(2*hidden_size)(time)
        scale, shift = ops.split(embed, 2, axis=-1)
        
        # === Feature-Transformation mit affiner Konditionierung ===
        x = act(layer)
        x = layers.Dense(hidden_size)(x)
        # - "1 +" : Identity-Baseline (bei scale=0 → kein Scaling)
        # - "scale" : Zeitabhängige Verstärkung/Dämpfung der Features
        # - "* x" : Multiplikative Modulation (Gating-Effekt)
        # - "+ shift" : Zeitabhängiger additiver Bias
        x = act((1.0+scale)*x + shift)
        # Dropout für Regularisierung
        x = layers.Dropout(.1)(x)
        x = layers.Dense(hidden_size)(x)
        x = layers.Add()([x, residual])
        return x

    # === ResNet Konstruktion ===
    # Projeziert Zeitembedding auf MLP-dim für alle Residual Blocks
    embed = act(layers.Dense(mlp_dim)(time_embedding))
    
    layer = layers.Dense(mlp_dim)(inputs)
    # Stack von Residual-Blocks
    for i in range(num_layer-1):
        layer =  resnet_dense([layer,embed],mlp_dim)

    outputs = layers.Dense(end_dim)(layer)

    return outputs



