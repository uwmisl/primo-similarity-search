import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import pandas as pd

class EncoderTrainer:
    
    def __init__(self, encoder, predictor):
        
        self.encoder = encoder
        self.predictor = predictor
        
        X_pairs = layers.Input([2, encoder.input_dim])
        X1, X2 = layers.Lambda(lambda X: (X[:,0,:], X[:,1,:]))(X_pairs)
        distances = layers.Lambda(lambda Xs: tf.sqrt(tf.reduce_sum(tf.square(Xs[0]-Xs[1]), axis=1)))([X1,X2])

        S1 = encoder(X1)
        S2 = encoder(X2)

        S_pairs = layers.Lambda(
            lambda Ss: tf.stack(Ss, axis=-1)
        )([S1,S2])
        S_pairs_T = layers.Lambda(lambda S: tf.transpose(S, [0, 2, 1, 3]))(S_pairs)

        y_h = predictor(S_pairs_T)
        y_h_T = layers.Reshape([1])(y_h)

        self.calcdists = tf.keras.Model(inputs=X_pairs, outputs=distances)
        self.model = tf.keras.Model(inputs=X_pairs, outputs=y_h_T)
        self.predictor.trainable(False)
        
        
    def refit_predictor(self, predictor_batch_generator, simulator, refit_every = 1, refit_epochs = 10):
        """Generate a callback function to refit the yield predictor during encoder training.
        
        Arguments:
        predictor_batch_generator: a generator that yields a tuple (pair of indices, pair of feature vectors).
        cupyck_sess: an active cupyck session (either CPU or GPU) that will be used for simulation
        refit_every: run the callback every N encoder training epochs (default: 1)
        refit_epochs: the number of epochs to run the yield trainer for during this callback (default: 10)
        """
        
        def callback(epoch, logs):
            if epoch % refit_every == 0:
                print
                print "refitting..."
                # get batch of features
                idx_pairs, feat_pairs = next(predictor_batch_generator)
                
                # convert to sequences
                seq_pairs = pd.DataFrame({
                    "target_features": self.encoder.encode_feature_seqs(feat_pairs[:, 0]),
                    "query_features": self.encoder.encode_feature_seqs(feat_pairs[:, 1])
                })
                
                # simulate yields
                sim_results = simulator.simulate(seq_pairs)
                
                # encode onehots
                onehot_seq_pairs = self.predictor.seq_pairs_to_onehots(seq_pairs)
                
                # refit yield predictor
                self.predictor.trainable(True)
                history = self.predictor.train(onehot_seq_pairs, sim_results.duplex_yield, epochs=refit_epochs, verbose=0)
                self.predictor.trainable(False)
                
                print "predictor loss: %g" % history.history['loss'][-1]

        return tf.keras.callbacks.LambdaCallback(on_epoch_end=callback)
        

        
        