import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models.doc2vec import Doc2Vec, TaggedDocument




class Doc2VecEmbedder(BaseEstimator,TransformerMixin):
    
    def fit(self, X, y=None):
#         doc = X['text'].tolist()
#         documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
#         model = Doc2Vec(documents, vector_size=15, window=2, min_count=1, workers=4)
        return self

    def transform(self, texts):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
        model = Doc2Vec(documents)
        
#         embeddings = []
        embeddings = model.infer_vector(texts)
#         text_list = texts.tolist()
#         for text in text_list:
#             text = str(text).strip()
#             embeddings.append(model.infer_vector(text))
        return np.array(embeddings)


