import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from alpha_automl.wrapper_primitives.huggingface import HuggingfaceInterface



class TestHuggingfaceInterface():
    X = pd.DataFrame(data={'text': ['I`d have responded, if I were going', 'Sooo SAD I will miss you here in San Diego!!!', 'my boss is bullying me..', 'what interview! leave me alone', 'Sons of ****, why couldn`t they put them on the releases we already bought'], 'sentiment': ['neutral', 'negative', 'negative', 'negative', 'negative']}) 
    y = pd.DataFrame(data={'sentiment': ['neutral', 'negative', 'negative', 'negative', 'negative']})

    test = pd.DataFrame(data={'text': ['I`d have responded, if I were going', 'Sooo SAD I will miss you here in San Diego!!!'], 'sentiment': ['neutral', 'negative'] })

    def test_hf(self):
        
        encoder = HuggingfaceInterface()
        encoder.fit(self.X, self.y)
        x = self.test['text'].values.tolist()
        pred = encoder.transform(x)
        assert pred.shape == (2,768)
        
        
