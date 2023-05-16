from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, ElectraModel, AutoTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoModel
import torch
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator') Only for comparison
model = ElectraModel.from_pretrained('google/electra-small-discriminator', output_hidden_states=True)

tokenizer_sentiment = RobertaTokenizer.from_pretrained("allenai/reviews_roberta_base")
model_sentiment = RobertaModel.from_pretrained("allenai/reviews_roberta_base")

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = BertModel.from_pretrained("bert-base-uncased")

tokenizer_tweets = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_tweets = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", output_hidden_states=True)

sentenceTransformerModel = SentenceTransformer('bert-base-uncased')

class MySentenceEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, last_four_model_layers=False):
        self.name = 'bert-base-uncased'
        self.last_four_model_layers = last_four_model_layers

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        batch_texts = [text for text in texts]
        embedding = sentenceTransformerModel.encode(batch_texts)
        return embedding



class MyEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, last_four_model_layers=False):
        self.name = 'google/electra-small-discriminator'
        self.last_four_model_layers = last_four_model_layers

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        batch_texts = [text for text in texts] #since texts is in the form of numpy array
        ids = tokenizer(batch_texts, padding=True, return_tensors="pt")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        ids = ids.to(device)
        model.eval()

        with torch.no_grad():
            out = model(**ids)  #model output contains last_hidden_state, pooler_output, hidden_outputs of each model layer and the embedding layer

        last_hidden_states = out.last_hidden_state #torch.Size([batch_size, seq_length, 768])

        sentence_embedding = torch.mean(last_hidden_states, dim=1) #torch.Size([batch_size, 768])

        return sentence_embedding.cpu().numpy()


class MySentimentEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, last_four_model_layers=False):
        self.last_four_model_layers = last_four_model_layers
        self.name = 'allenai/reviews_roberta_base'

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        list_texts = [text for text in texts] #since texts is in the form of numpy array
        total_length = len(list_texts)

        if total_length < 32:
            batch_size = 16
        else:
            batch_size = 32

        steps = total_length // batch_size
        batch_embeddings = []

        for start in range(0, total_length, batch_size):
            if start  == (steps * batch_size):
                batch_texts = list_texts[start: total_length]
            else:
                batch_texts = list_texts[start: start+batch_size]

            ids = tokenizer_sentiment(batch_texts, padding=True, return_tensors="pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_sentiment.to(device)
            ids = ids.to(device)
            model_sentiment.eval()

            with torch.no_grad():
                out = model_sentiment(**ids)  #model output contains last_hidden_state, pooler_output, hidden_outputs of each model layer and the embedding layer

            last_hidden_states = out.last_hidden_state
            sentence_embedding = last_hidden_states[:,0,:]

            sentence_embedding = sentence_embedding.cpu().numpy()

            torch.cuda.empty_cache()

            if start == 0:
                batch_embeddings = sentence_embedding
            else:
                batch_embeddings = np.concatenate((batch_embeddings, sentence_embedding), axis=0)
        return batch_embeddings



class MyBERTEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, last_four_model_layers=False):
        self.last_four_model_layers = last_four_model_layers
        self.name = 'bert-base-uncased'

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        list_texts = [text for text in texts] #since texts is in the form of numpy array
        total_length = len(list_texts)

        if total_length < 32:
            batch_size = 16
        else:
            batch_size = 32

        steps = total_length // batch_size
        batch_embeddings = []

        for start in range(0, total_length, batch_size):
            if start  == (steps * batch_size):
                batch_texts = list_texts[start: total_length]
            else:
                batch_texts = list_texts[start: start+batch_size]

            ids = tokenizer_bert(batch_texts, padding=True, return_tensors="pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_bert.to(device)
            ids = ids.to(device)
            model_bert.eval()

            with torch.no_grad():
                out = model_bert(**ids)  #model output contains last_hidden_state, pooler_output, hidden_outputs of each model layer and the embedding layer

            last_hidden_states = out.last_hidden_state
            sentence_embedding = last_hidden_states[:,0,:]

            sentence_embedding = sentence_embedding.cpu().numpy()

            torch.cuda.empty_cache()

            if start == 0:
                batch_embeddings = sentence_embedding
            else:
                batch_embeddings = np.concatenate((batch_embeddings, sentence_embedding), axis=0)
        return batch_embeddings



class MyTweetEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, last_four_model_layers=False):
        self.last_four_model_layers = last_four_model_layers
        self.name = 'cardiffnlp/twitter-roberta-base-sentiment'

    def fit(self, X, y=None):
        return self

    def transform(self, texts):

        list_texts = [text for text in texts] #since texts is in the form of numpy array
        total_length = len(list_texts)
        if total_length < 32:
            batch_size = 16
        else:
            batch_size = 32

        steps = total_length // batch_size
        batch_embeddings = []

        for start in range(0, total_length, batch_size):
            if start  == (steps * batch_size):
                batch_texts = list_texts[start: total_length]
            else:
                batch_texts = list_texts[start: start+batch_size]

            ids = tokenizer_tweets(batch_texts, padding=True, return_tensors="pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_tweets.to(device)
            ids = ids.to(device)
            model_tweets.eval()

            with torch.no_grad():
                out = model_tweets(**ids)  #model output contains last_hidden_state, pooler_output, hidden_outputs of each model layer and the embedding layer

            last_hidden_states = out.last_hidden_state
            sentence_embedding = last_hidden_states[:,0,:]

            sentence_embedding = sentence_embedding.cpu().numpy()

            torch.cuda.empty_cache()

            if start == 0:
                batch_embeddings = sentence_embedding
            else:
                batch_embeddings = np.concatenate((batch_embeddings, sentence_embedding), axis=0)
        return batch_embeddings
