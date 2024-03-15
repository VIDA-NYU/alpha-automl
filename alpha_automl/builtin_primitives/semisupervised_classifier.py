import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading
)

from alpha_automl.base_primitive import BasePrimitive
from alpha_automl.utils import SemiSupervisedSplitter

logger = logging.getLogger(__name__)


def make_label_pipeline(method, X):
    step = None
    if method == "LabelSpreading":
        step = ("sklearn.semi_supervised.LabelSpreading", LabelSpreading())
    elif method == "LabelPropagation":
        step = ("sklearn.semi_supervised.LabelPropagation", LabelPropagation())
    else:
        raise Exception("method should be either LabelSpreading or LabelPropagation") 

    if isinstance(X, np.ndarray):
        pipe = Pipeline([step])
    else:
        pipe = Pipeline(
            [
                ("toarray", FunctionTransformer(lambda x: x.toarray())),
                step,
            ]
        )
    return pipe


class SkLabelSpreading(BasePrimitive):
    def fit(self, X, y=None):
        self.pipe = make_label_pipeline("LabelSpreading", X)
        self.pipe.fit(X, y)

    def predict(self, X):
        pred = self.pipe.predict(X)

        return np.array(pred)


class SkLabelPropagation(BasePrimitive):
    def fit(self, X, y=None):
        self.pipe = make_label_pipeline("LabelPropagation", X)
        self.pipe.fit(X, y)

    def predict(self, X):
        pred = self.pipe.predict(X)

        return np.array(pred)


class AutonBox(BasePrimitive):
    def __init__(self, base_estimator, iteration=100):
        self.base_estimator = base_estimator
        self.splitter = SemiSupervisedSplitter()
        self.calibclf = CalibratedClassifierCV(self.base_estimator, cv=self.splitter)
        self.pipeline = OneVsRestClassifier(self.calibclf)
        self.iteration = iteration
        self.frac = 1 / iteration

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y_cp = y.to_numpy()
        else:
            y_cp = y.copy()

        for labelIteration in range(self.iteration):
            labeledIx = np.where(y_cp != -1)[0]
            unlabeledIx = np.where(y_cp == -1)[0]
            if len(unlabeledIx) == 0:
                continue

            if labelIteration == 0:
                num_instances_to_label = int(self.frac * len(unlabeledIx) + 0.5)

            labeledX = X[labeledIx]
            labeledy = y_cp[labeledIx]

            self.pipeline.fit(labeledX, labeledy)
            probas = self.pipeline.predict_proba(X[unlabeledIx])

            entropies = np.sum(np.log2(probas.clip(0.0000001, 1.0)) * probas, axis=1)
            entIdx = np.rec.fromarrays((entropies, unlabeledIx))
            entIdx.sort(axis=0)

            labelableIndices = entIdx["f1"][-num_instances_to_label:].reshape((-1,))

            predictions = self.pipeline.predict(X[labelableIndices])

            y_cp[labelableIndices, 0] = predictions

        labeledIx = np.where(y_cp != -1)[0]
        labeledX = X[labeledIx]
        labeledy = y_cp[labeledIx]

        self.pipeline.fit(labeledX, labeledy)

    def predict(self, X):
        pred = self.pipeline.predict(X)

        return np.array(pred)
