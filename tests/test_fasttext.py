import numpy as np
import pandas as pd
import os
from alpha_automl._optional_dependency import import_optional_dependency
from alpha_automl.wrapper_primitives.fasttext import FastTextEmbedderWrapper


class TestFastTextEmbedderWrapper:
    """This is the testcases for datetime encoder."""

    example = ["I would have responded, if I were going"]
    df = np.array([example])
    
    def test_fasttext_embedder(self):
        os.chdir("..")
        embedder = FastTextEmbedderWrapper(os.getcwd() + '/examples/cc.en.300.bin')
        embedder.fit(self.df)
        np_array = embedder.transform(self.df)
        assert np_array.shape == (len(self.df), 300)