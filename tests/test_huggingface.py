import pandas as pd
from alpha_automl.wrapper_primitives.huggingface import HuggingfaceTextTransformer


class TestHuggingfaceEmbedder():
    X = pd.DataFrame(data={'text': ['I`d have responded, if I were going',
                                    'Sooo SAD I will miss you here in San Diego!!!',
                                    'my boss is bullying me..', 'what interview! leave me alone',
                                    'Sons of ****, why couldn`t they put them on the releases we already bought']
                           }
                     )
    y = pd.DataFrame(data={'sentiment': ['neutral', 'negative', 'negative', 'negative', 'negative']})

    def test_fit_transform(self):
        from huggingface_hub import snapshot_download
        model_name = 'google/electra-small-discriminator'
        snapshot_download(repo_id=model_name)

        encoder = HuggingfaceTextTransformer(model_name)
        encoder.fit(self.X, self.y)
        pred = encoder.transform(self.X['text'].values.tolist())
        assert pred.shape == (5, 256)
