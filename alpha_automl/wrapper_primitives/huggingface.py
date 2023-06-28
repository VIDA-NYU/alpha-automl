import torch
import numpy as np
from alpha_automl._optional_dependency import import_optional_dependency
from alpha_automl.base_primitive import BasePrimitive
transformers = import_optional_dependency('transformers')


class HuggingfaceEmbedder(BasePrimitive):

    def __init__(self, name, last_four_model_layers=False):
        self.name = name
        self.last_four_model_layers = last_four_model_layers

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

        # Loading tokenizer and model
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.name)
        model = transformers.AutoModel.from_pretrained(self.name, output_hidden_states=True)

        for start in range(0, total_length, batch_size):
            if start == (steps * batch_size):
                batch_texts = list_texts[start: total_length]

            else:
                batch_texts = list_texts[start: start + batch_size]

            ids = tokenizer(batch_texts, padding=True, return_tensors="pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            ids = ids.to(device)
            model.eval()

            with torch.no_grad():
                # Model output contains last_hidden_state, pooler_output, hidden_outputs of each model layer and the
                # embedding layer
                out = model(**ids)

            last_hidden_states = out.last_hidden_state
            sentence_embedding = last_hidden_states[:, 0, :]

            sentence_embedding = sentence_embedding.cpu().numpy()

            torch.cuda.empty_cache()

            if start == 0:
                batch_embeddings = sentence_embedding
            else:
                batch_embeddings = np.concatenate((batch_embeddings, sentence_embedding), axis=0)

        return batch_embeddings
