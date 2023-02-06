# Image-captioning
In this project, a encoder-decoder model is trained to caption a given input image.

flickr30k image dataset is used for training the model. 
At first, the InceptionV3 pre-trained model from tensorflow.keras API is used for extracting features of the images. 

The captions are processed and used for training word2vec model using gensim. 
Then google-news-300 w2v pre-trained model is used for expanding the vocabulary of trained model. 
Later, the model is re-trained to fune-tune the embedding vectors.
The embedding matrix is then saved for the later use.

Encoder-decoder model with attention mechanism is constructed for task of auto-captioning. 
Batches of saved image features are passed through dense layer in the encoder.
Caption tokens are sequencently passed through embedding layer to be processed later with LSTM and dense layers.
Attention mechanism based on the image-feature and word-vector of token from previous time-stamp is used to extract context from the images.
Custome training loops are then run with ADAM optimizer to decrease the learning losses.
At last, the model weights are saved for testing and building the API.

Fastapi is used for creating the backend API. Streamlit is used for creating simplistic frontend UI for the API. 
Both backend and frontend are containerized and linked together using Docker compose.
