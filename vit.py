"""
This is a simple Vision Transformer
patch-extractort -> patch-embedding +  positional embedding -> transformer encoding -> classification
"""

import tensorflow as tf


class PatchExtractor(tf.keras.layers.Layer):
    """
    suggested images of 224x224
    """
    def __init__(self):
        super(PatchExtractor, self).__init__()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, 16, 16, 1],
            strides = [1, 16, 16, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches=196, projection_dim=768):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        """
        here, we use embedding object, we should change to cosine-sine-based positional encoding or another one
        """
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches+1, output_dim=projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embedins
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)        
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded

class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(out_features)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(AttentionBlock, self).__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = tf.keras.layers.Add()([x3, x2])
        return y

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [AttentionBlock(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y    
    
def create_vit(config_data, config_model):
    num_classes = config_data.get('N_CLASSES')
    num_patches = config_data.getint('CROP_SIZE') // 16
    num_patches = num_patches * num_patches
    projection_dim = config_model.getint('PROJECT_DIM') 
    input_shape = (config_data.getint('CROP_SIZE'), config_data.getint('CROP_SIZE'), 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Patch extractor
    patches = PatchExtractor()(inputs)
    # Patch encoder
    patches_embed = PatchEncoder(num_patches, projection_dim)(patches)
    # Transformer encoder
    representation = TransformerEncoder(projection_dim)(patches_embed)
    """
    here, global average pooling is used, but the paper says using class embedding representations[:,0,:]
    """
    representation = tf.keras.layers.GlobalAveragePooling1D()(representation)
    # MLP to classify outputs
    logits = MLP(projection_dim, num_classes, 0.5)(representation)
    # Create model
    
    model = tf.keras.Model(inputs = inputs, outputs = tf.keras.layers.Softmax()(logits))    
    return model


    
