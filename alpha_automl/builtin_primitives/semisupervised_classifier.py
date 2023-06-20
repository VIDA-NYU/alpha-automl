import logging

import numpy as np
import pandas as pd
from alpha_automl.base_primitive import BasePrimitive
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)
from alpha_automl.utils import SemiSupervisedSplitter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

logger = logging.getLogger(__name__)

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


class AutonBox(BasePrimitive):
    def __init__(self, base_estimator, iteration=100):
        self.base_estimator = base_estimator
        self.splitter = SemiSupervisedSplitter()
        self.calibclf = CalibratedClassifierCV(self.base_estimator, cv=self.splitter)
        self.pipeline = OneVsRestClassifier(self.calibclf)
        self.iteration = iteration
        self.frac = 1/iteration
        
    def fit(self, X, y):
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
            
            labelableIndices = entIdx['f1'][-num_instances_to_label:].reshape((-1,))
            
            predictions = self.pipeline.predict(X[labelableIndices])
            
            y_cp[labelableIndices, 0] = predictions
            
        labeledIx = np.where(y_cp != -1)[0]
        labeledX = X[labeledIx]
        labeledy = y_cp[labeledIx]
        
        self.pipeline.fit(labeledX, labeledy)
    
    def predict(self, X):
        pred = self.pipeline.predict(X)

        return np.array(pred)