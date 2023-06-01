import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, ElectraModel, BertModel, RobertaModel, AutoModel
import torch
from sentence_transformers import SentenceTransformer

class HuggingfaceInterface(BaseEstimator, TransformerMixin):

    def __init__(self, model, tokenizer, name, last_four_model_layers=False):
        '''
        model: Huggingface model class object, for eg: AutoModel(), make sure output_hidden_states=True when instantiating the model class before passing into this class.
        tokenizer: Huggingface tokenizer class object, for eg: AutoTokenizer()
        '''
        self.last_four_model_layers = last_four_model_layers
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, texts):

        list_texts = [text for text in texts]  # since texts is in the form of numpy array
        total_length = len(list_texts)

        if total_length < 32:
            batch_size = 16
        else:
            batch_size = 32

        steps = total_length // batch_size
        batch_embeddings = []

        for start in range(0, total_length, batch_size):
            if start == (steps * batch_size):
                batch_texts = list_texts[start: total_length]

            else:
                batch_texts = list_texts[start: start + batch_size]

            ids = self.tokenizer(batch_texts, padding=True, return_tensors="pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            ids = ids.to(device)
            self.model.eval()

            with torch.no_grad():
                out = self.model(
                    **ids)  # model output contains last_hidden_state, pooler_output, hidden_outputs of each model layer and the embedding layer

            last_hidden_states = out.last_hidden_state
            sentence_embedding = last_hidden_states[:, 0, :]

            sentence_embedding = sentence_embedding.cpu().numpy()

            torch.cuda.empty_cache()

            if start == 0:
                batch_embeddings = sentence_embedding
            else:
                batch_embeddings = np.concatenate((batch_embeddings, sentence_embedding), axis=0)

        return batch_embeddings
