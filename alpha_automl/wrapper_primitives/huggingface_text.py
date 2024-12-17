import logging
import torch
import numpy as np
from alpha_automl.base_primitive import BasePrimitive
from alpha_automl._optional_dependency import check_optional_dependency

ml_task = 'nlp'
check_optional_dependency('transformers', ml_task)
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class HuggingfaceTextTransformer(BasePrimitive):

    def __init__(self, name, tokenizer=None, max_length=512):
        self.name = name
        self.tokenizer = tokenizer if tokenizer else name
        self.max_length = max_length

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        list_texts = [text if str(text)!='nan' else '' for text in texts]  # since texts is in the form of numpy array
        total_length = len(list_texts)

        if total_length < 32:
            batch_size = 16
        else:
            batch_size = 32

        steps = total_length // batch_size
        batch_embeddings = []

        # Loading tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        model = AutoModel.from_pretrained(self.name, output_hidden_states=True)

        for start in range(0, total_length, batch_size):
            if start == (steps * batch_size):
                batch_texts = list_texts[start: total_length]

            else:
                batch_texts = list_texts[start: start + batch_size]

#             batch_texts = [' '.join(line.split()) if str(line)!='nan' else '' for line in batch_texts]
            ids = tokenizer(batch_texts, padding=True, return_tensors="pt", max_length=self.max_length)

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
