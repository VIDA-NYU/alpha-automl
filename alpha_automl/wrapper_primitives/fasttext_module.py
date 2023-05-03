import numpy as np
import fasttext
import torch
import importlib
# from alpha_automl._optional_dependency import import_optional_dependency
from sklearn.base import BaseEstimator, TransformerMixin


class FastTextEmbedderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, fasttext_model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fasttext_model_path = fasttext_model_path
        
        try:
            dependency_module = importlib.import_module('fasttext')
        except ImportError:
            raise ImportError(f'Missing optional dependency "{dependency_name}". Use pip or conda to install it.')

        

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        fasttext_model = fasttext.load_model(self.fasttext_model_path)
        embeddings = []
        text_list = texts.tolist()
        for text in text_list:
            text = str(text).strip()
            embeddings.append(fasttext_model.get_sentence_vector(text))
        embeddings = torch.tensor(embeddings).to(self.device)
        return embeddings.numpy()







# class FastTextEmbedderWrapper(BaseEstimator, TransformerMixin):
#     def __init__(self, fasttext_model):
#         self.model_path = model_path
#         self.fasttext_model = fasttext_model
#         self.fasttext, self.fasttext_util = import_optional_dependency(['fasttext','fasttext.util'])
        
#         if self.fasttext is None:
#             raise ImportError("fasttext is not installed.")
#         if self.fasttext_util is None:
#             raise ImportError("fasttext.util is not installed.")
    

