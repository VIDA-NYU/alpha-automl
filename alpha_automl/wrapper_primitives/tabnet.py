import torch
import numpy as np
import pandas as pd
from alpha_automl.base_primitive import BasePrimitive
from alpha_automl._optional_dependency import check_optional_dependency

ml_task = 'regression'
check_optional_dependency('pytorch_tabnet', ml_task)
from pytorch_tabnet.tab_model import TabNetRegressor

class PytorchTabNetRegressor(BasePrimitive):
    """
    This is a pyTorch implementation of Tabnet
    (Arik, S. O., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular Learning. arXiv preprint arXiv:1908.07442.)
    https://arxiv.org/pdf/1908.07442.pdf.
    Please note that some different choices have been made overtime to improve the library
    which can differ from the orginal paper.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TabNetRegressor(n_d=32, verbose=False)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.model.fit(X.values, y.values)
        else:
            self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        # Load fasttext model
        if isinstance(X, pd.DataFrame):
            return self.model.predict(X.values)
        else:
            return self.model.predict(np.asarray(X))