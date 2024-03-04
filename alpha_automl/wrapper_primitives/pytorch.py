import torch
import numpy as np
from alpha_automl._optional_dependency import import_optional_dependency


class PytorchPrimitive():

    def __init__(self, input_class):
        skorch = import_optional_dependency('skorch')
        self.model = skorch.NeuralNetClassifier(input_class,
                                                criterion=torch.nn.L1Loss(),
                                                lr=0.1,
                                                max_epochs=10
                                                )

    def fit(self, X, y):
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        self.model.fit(X, y)

    def predict(self, X):
        X = X.astype(np.float32)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = X.astype(np.float32)
        return self.model.predict_proba(X)
