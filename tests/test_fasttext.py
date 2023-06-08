import numpy as np
from alpha_automl.wrapper_primitives.fasttext import FastTextEmbedder


class TestFastTextEmbedder:
    """This is the testcases for fasttext embedder."""

    example = ["I would have responded, if I were going"]
    df = np.array([example])
    
    def test_fit_transform(self):
        fasttext_model_path = 'tests/test_data/cc.en.2.bin' 
        embedder = FastTextEmbedder(fasttext_model_path)
        embedder.fit(self.df)
        np_array = embedder.transform(self.df)
        assert np_array.shape == (len(self.df), 2)
