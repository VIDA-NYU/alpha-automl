from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import numpy as np


class MyEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.embedder = SentenceTransformer('xlm-roberta-base')

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        text_list = texts.tolist()
        embeddings = self.embedder.encode(text_list)

        return np.array(embeddings)
