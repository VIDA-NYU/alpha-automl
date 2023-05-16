import numpy as np
# import fasttext
# import torch
# import importlib
from _optional_dependency import import_optional_dependency
from base_primitive import BasePrimitive


class FastTextEmbedderWrapper(BasePrimitive):
    '''
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

    '''
    
    def __init__(self, fasttext_model_path):
        '''
        # Uncomment and use for cuda processing is required
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        '''

        self.fasttext_model_path = fasttext_model_path
        self.fasttext = import_optional_dependency('fasttext')
        self.fasttext_util = import_optional_dependency('fasttext.util')
    

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        fasttext_model = self.fasttext.load_model(self.fasttext_model_path)
        embeddings = []
        text_list = texts.tolist()
        for text in text_list:
            text = str(text).strip()
            embeddings.append(fasttext_model.get_sentence_vector(text))
            
        '''
        # Uncomment and use the below code if cuda processing is required
            embeddings = torch.tensor(embeddings).to(self.device)
        return embeddings.numpy()
        
        '''

        return np.array(embeddings)
