import numpy as np
import pandas as pd
import os
from alpha_automl._optional_dependency import import_optional_dependency
from alpha_automl.wrapper_primitives.fasttext import FastTextEmbedder


class TestFastTextEmbedder:
    """This is the testcases for fasttext embedder."""

    example = ["I would have responded, if I were going"]
    df = np.array([example])
    
    def test_fasttext_embedder(self):
        fasttext_model_path = 'tests/test_data/cc.en.2.bin' 
        embedder = FastTextEmbedder(fasttext_model_path)
        embedder.fit(self.df)
        np_array = embedder.transform(self.df)
        assert np_array.shape == (len(self.df), 2)