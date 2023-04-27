from alpha_automl.base_primitive import BasePrimitive
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('xlm-roberta-base')


class MyEmbedder(BasePrimitive):

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        text_list = texts.tolist()
        embeddings = embedder.encode(text_list)

        return np.array(embeddings)
