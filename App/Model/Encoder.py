import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, embeded_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(embeded_dim, activation = 'relu')
        
    def call(self, x):
        
        x = self.dense(x)
        
        return x