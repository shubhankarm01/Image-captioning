import tensorflow as tf
from tensorflow.keras.initializers import constant
from App.Model import Attention

class Decoder(tf.keras.Model):
    def __init__(self, units, embeded_mat, len_sent):
        super().__init__()
        self.units = int(units)
        self.embeded_mat = tf.keras.layers.Embedding(input_length = len_sent,
                                                     input_dim = embeded_mat.shape[0], 
                                                     output_dim = embeded_mat.shape[1],
                                                     embeddings_initializer = constant(embeded_mat), trainable = False)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences = True, return_state = True)
        self.dense_1 = tf.keras.layers.Dense(self.units, activation = 'relu')
        self.dense_2 = tf.keras.layers.Dense(embeded_mat.shape[0])
        
        self.attention = Attention.Attention(self.units)
        
    def call(self, feat, hidd_state, token):
        
        context, atten_weig = self.attention.call(feat, hidd_state)
        
        y  = self.embeded_mat(token)
        
        context = tf.expand_dims(context, axis = 1)
        y = tf.concat([context, y], axis = 2)
        
        output, hidd_state, seq = self.lstm(y)
        
        y = self.dense_1(output)
        y = tf.reshape(y, [-1, y.shape[2]])
        y = self.dense_2(y)
        
        return y, hidd_state, atten_weig