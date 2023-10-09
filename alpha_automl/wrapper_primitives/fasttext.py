import torch
import numpy as np
from alpha_automl.base_primitive import BasePrimitive
from alpha_automl._optional_dependency import check_optional_dependency

ml_task = 'nlp'
check_optional_dependency('fasttext', ml_task)
import fasttext


class FastTextEmbedder(BasePrimitive):
    """
    FastTextEmbedderWrapper provides word embeddings word embeddings using
    fastText is a library for efficient learning of word representations 
    and sentence classification
    
    Args: 
        fasttext_model_path : the path where the pretrained fasttext model 
                              (cc.en.300.bin) is located. 
                              Use the following to download model and pass the 
                              fasttext_model_path as arg to this class

                              fasttext.util.download_model('en', if_exists='ignore')  # English
                              fasttext_model_path = '<path_to_model>/cc.en.300.bin' 
    """

    def __init__(self, fasttext_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fasttext_model_path = fasttext_model_path

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        # Load fasttext model
        fasttext_model = fasttext.load_model(self.fasttext_model_path)
        text_list = texts.tolist()
        
        embeddings = [fasttext_model.get_sentence_vector(str(text).strip()) for text in text_list]

        # Convert embeddings to numpy array
        embeddings = np.array(embeddings)

        # Move embeddings to the specified device
        embeddings = torch.from_numpy(embeddings).to(self.device)

        return embeddings.cpu().numpy()
