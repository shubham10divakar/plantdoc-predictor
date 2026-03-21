# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:14:15 2026

@author: Subham Divakar
"""

import tensorflow as tf
from tensorflow.keras import layers

# ================================
# Additive Attention Layer
# ================================
class AdditiveAttention(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.units = units

        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features):

        # flatten spatial dimensions
        flattened_features = tf.reshape(
            features,
            (tf.shape(features)[0], -1, features.shape[-1])
        )

        score = self.V(
            tf.nn.tanh(
                self.W1(flattened_features) +
                self.W2(flattened_features)
            )
        )

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * flattened_features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    # IMPORTANT for model saving/loading
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config


# ================================
# Recursive Additive Attention
# ================================
class RecursiveAdditiveAttentionModule(tf.keras.layers.Layer):

    def __init__(self, num_iterations=3, units=256, **kwargs):
        super(RecursiveAdditiveAttentionModule, self).__init__(**kwargs)

        self.num_iterations = num_iterations
        self.units = units

        self.attention_layer = AdditiveAttention(units)
        self.layer_norm = layers.LayerNormalization(axis=-1)

    def call(self, inputs):

        x = inputs

        for _ in range(self.num_iterations):

            context_vector, _ = self.attention_layer(x)

            # broadcast context to spatial dims
            context_vector = context_vector[:, tf.newaxis, tf.newaxis, :]

            x = x + context_vector

            x = self.layer_norm(x)

        return x

    # IMPORTANT for serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_iterations": self.num_iterations,
            "units": self.units
        })
        return config


# ================================
# Residual Block
# ================================
def residual_block(x, filters, kernel_size=(3,3), strides=(1,1), block_number=1):

    shortcut = x

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=f'resd_block{block_number}_conv1'
    )(x)

    x = layers.BatchNormalization(name=f'resd_block{block_number}_bn1')(x)
    x = layers.ReLU(name=f'resd_block{block_number}_relu1')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=f'resd_block{block_number}_conv2'
    )(x)

    x = layers.BatchNormalization(name=f'resd_block{block_number}_bn2')(x)

    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters,
            (1,1),
            strides=strides,
            padding='same',
            use_bias=False,
            name=f'resd_block{block_number}_shortcut_conv'
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f'resd_block{block_number}_shortcut_bn'
        )(shortcut)

    x = layers.add([x, shortcut], name=f'resd_block{block_number}_add')

    x = layers.ReLU(name=f'resd_block{block_number}_relu2')(x)

    return x