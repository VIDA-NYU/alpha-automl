import numpy as np
from alpha_automl.base_primitive import BasePrimitive
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)


class SkSelfTrainingClassifier(BasePrimitive):
    sdg_params = dict(alpha=1e-5, penalty="l2", loss="log_loss")
    model = SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return

    def predict(self, X):
        pred = self.model.predict(X)

        return np.array(pred)


class SkLabelSpreading(BasePrimitive):
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            self.pipe = Pipeline(
                [
                    ("sklearn.semi_supervised.LabelSpreading", LabelSpreading()),
                ]
            )
        else:
            self.pipe = Pipeline(
                [
                    ("toarray", FunctionTransformer(lambda x: x.toarray())),
                    ("sklearn.semi_supervised.LabelSpreading", LabelSpreading()),
                ]
            )
        self.pipe.fit(X, y)
        return

    def predict(self, X):
        pred = self.pipe.predict(X)

        return np.array(pred)


class SkLabelPropagation(BasePrimitive):
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            self.pipe = Pipeline(
                [
                    ("sklearn.semi_supervised.LabelPropagation", LabelPropagation()),
                ]
            )
        else:
            self.pipe = Pipeline(
                [
                    ("toarray", FunctionTransformer(lambda x: x.toarray())),
                    ("sklearn.semi_supervised.LabelPropagation", LabelPropagation()),
                ]
            )
        self.pipe.fit(X, y)
        return

    def predict(self, X):
        pred = self.pipe.predict(X)

        return np.array(pred)
