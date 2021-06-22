import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import pandas as pd

from ..tools import sequences as seqtools

def local_interactions_layer(window_size, **lambda_args):

    def local_interactions(seq_pairs):
        seq_pairs.shape.assert_is_compatible_with([None, 4, None, 2])
        seq_len = seq_pairs.get_shape()[2]

        by_position = []
        for pos in range(window_size, seq_len-window_size):
            by_channel = []
            for channel in range(4):
                top = seq_pairs[:, channel:channel+1, pos-window_size:pos+window_size+1, 0]
                bot = seq_pairs[:, channel:channel+1, pos-window_size:pos+window_size+1, 1]
                mat = tf.matmul( tf.transpose(top,[0,2,1]), bot)
                by_channel.append(tf.reshape(mat, [-1, ((window_size*2)+1)**2]))
            by_channel = tf.concat(by_channel, axis=1)
            by_position.append(by_channel)
        by_position = tf.stack(by_position, axis=1)

        return by_position

    return layers.Lambda(local_interactions, **lambda_args)

class Predictor:
    """

    Predicts thermodynamic yield for a hybridization reaction between two DNA sequences
    where the second sequence will be reverse-complemented.

    Note that this Predictor is designed to be differentialable (unlike Nupack), which means it
    can be used in a neural network.

    """

    def __init__(self, model_path = None, **kwargs):

        for arg, val in kwargs.items():
            setattr(self, arg, val)

        if model_path is None:
            self.model = tf.keras.Sequential([
                local_interactions_layer(window_size=1, input_shape=[4,80,2]),
                layers.AveragePooling1D(3),
                layers.Conv1D(36, 3, activation='tanh'),
                layers.GlobalAveragePooling1D(),
                layers.Dense(1, name='logit'),
                layers.Activation('sigmoid')
            ])

        else:
            self.model = tf.keras.models.load_model(model_path)

    def __call__(self, X):
        return self.model(X)

    def trainable(self, flag):
        for layer in self.model.layers:
            layer.trainable = flag

    def seq_pairs_to_onehots(self, seq_pairs):
        # transform sequences into their one-hot representation
        onehot_pairs = np.stack([
            seqtools.seqs_to_onehots(seq_pairs.target_features.values),
            seqtools.seqs_to_onehots(seq_pairs.query_features.values)
        ], axis = 1)

        # transpose onehot pairs from (batch, pair, len, base) -> (batch, base, len, pair)
        onehot_pairs_T = onehot_pairs.transpose(0, 3, 2, 1)

        return onehot_pairs_T

    def train(self, sequences, yields, learning_rate=1e-3, **fit_kwargs):
        self.model.compile(tf.keras.optimizers.RMSprop(learning_rate), tf.keras.losses.binary_crossentropy)
        history = self.model.fit(
            sequences,
            yields,
            **fit_kwargs
        )
        return history

    def save(self, model_path):
        self.model.save(model_path)
