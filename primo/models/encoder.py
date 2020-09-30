import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from . import _default_sequences
from ..tools import sequences as seqtools

def entropy_regularizer(strength):

    def encoder_entropy(seq_probs):
        seq_probs.shape.assert_is_compatible_with([None, None, 4])

        ent_by_position = -tf.reduce_sum(
            seq_probs * tf.log(seq_probs + 1e-10),
            axis = 2
        )
        mean_ent_by_sequence = tf.reduce_mean(
            ent_by_position,
            axis = 1
        )
        mean_ent_by_batch = tf.reduce_mean(
            mean_ent_by_sequence,
            axis = 0
        )

        return strength * mean_ent_by_batch

    return encoder_entropy

class Encoder:

    defaults = {
        "input_dim": 4096,
        "output_len": 80,

        "entropy_reg_strength": 1e-2
    }

    def __init__(self, model_path = None, **kwargs):
        for arg, val in _default_sequences.items():
            setattr(self, arg, val)
        
        for arg, val in self.defaults.items():
            setattr(self, arg, val)
            
        for arg, val in kwargs.items():
            setattr(self, arg, val)

        if model_path is None:
            self.model = tf.keras.Sequential([
                layers.Dense(self.input_dim/2, activation = 'relu', input_shape=[self.input_dim]),
                layers.Dense(self.output_len * 4, activation='relu'),
                layers.Reshape([self.output_len, 4]),
                layers.Activation('softmax'),
                layers.Lambda(
                    lambda x: x,
                    activity_regularizer=entropy_regularizer(
                        self.entropy_reg_strength
                    )
                )
            ], name='encoder')
            
        else:
            self.model = tf.keras.models.load_model(model_path)
            
    def __call__(self, X):
        return self.model(X)

    def trainable(self, flag):
        for layer in self.model.layers:
            layer.trainable = flag
    
    def encode_feature_seqs(self, X):
        onehots = self.model.predict(X)
        return seqtools.onehots_to_seqs(onehots)

    # todo
    def build_targets(self, feature_seqs, barcoder):
        pass

    # todo
    def build_queries(self, feature_seqs):
        pass


    # evaluate ???
    
    def save(self, model_path):
        self.model.save(model_path)
    
