import App.core as config
import numpy as np
import json

from App.Model import Encoder, Attention, Decoder
import tensorflow as tf

mat = np.load(config.embedded_mat_path)

word_index = dict()
with open(config.word_index_path, 'rb') as file:
    word_index = json.load(file)

index_word = {j : i for i, j in word_index.items()}

encoder = Encoder.Encoder(config.config['embeded_dim'])
attention = Attention.Attention(config.config['units'])
decoder = Decoder.Decoder(config.config['units'], mat, config.config['len_sent'])

encoder.load_weights(config.model_path/'Encoder/encoder')
decoder.load_weights(config.model_path/'Decoder/decoder')

def img_preprocess(img_path):
    
    # img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_path, expand_animations = False)
    img = tf.image.resize(img, [299, 299])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    return img


feat_model = tf.keras.models.load_model(config.feat_model_path)


def evaluate(img_path):
    
    img = img_preprocess(img_path)
    img = tf.expand_dims(img, axis = 0)
    img_feat = feat_model(img)
    img_feat = tf.reshape(img_feat, (-1, 8*8, img_feat.shape[3]))
    
    feat = encoder(img_feat)
    hidd_state = tf.zeros([1, int(config.config['units'])])
    decoder_inp = tf.expand_dims([word_index['<start>']], axis = 0)
    
    result = []
    for i in range(int(config.config['len_sent'])):
        
        pred, hidd_state, atten_weig = decoder(feat, hidd_state, decoder_inp)

        pred_id = tf.random.categorical(pred, 1).numpy()[0][0]
        result.append(index_word[pred_id])
  
        if index_word[pred_id] == '<end>':
            return result[:-1]
        
        decoder_inp = tf.expand_dims([pred_id], axis = 0)
        
    return result

# evaluate(r'F:\Git repository\Image captioning\Data\flickr30k_images\flickr30k_images/205842.jpg')