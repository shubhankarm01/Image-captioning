import tensorflow as tf

class Attention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.weig_1 = tf.keras.layers.Dense(units, activation = 'relu')
        self.weig_2 = tf.keras.layers.Dense(units, activation = 'relu')
        self.vec = tf.keras.layers.Dense(1)
        
    def call(self, feat, hidden):
        
        hidden = tf.expand_dims(hidden, axis = 1)
        score = tf.nn.tanh(self.weig_1(feat) + self.weig_2(hidden))
    
        atten_weig = tf.nn.softmax(self.vec(score), axis = 1) 
        
        context = atten_weig*feat
        context = tf.reduce_sum(context, axis = 1)
        
        return context, atten_weig
