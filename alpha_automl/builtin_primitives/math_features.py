import numpy as np
import pandas as pd
from alpha_automl.base_primitive import BasePrimitive
from feature_engine.creation import MathFeatures

class MathFeaturesSum(BasePrimitive):
    def __init__(self, columns):
        self.columns = columns
        self.math_features = MathFeatures(variables=self.columns, func='sum')

    def fit(self, X, y=None):
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        return self.math_features.transform(X)
    
class MathFeaturesMean(BasePrimitive):
    def __init__(self, columns):
        self.columns = columns
        self.math_features = MathFeatures(variables=self.columns, func='mean')

    def fit(self, X, y=None):
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        return self.math_features.transform(X)
    
class MathFeaturesStd(BasePrimitive):
    def __init__(self, columns):
        self.columns = columns
        self.math_features = MathFeatures(variables=self.columns, func='std')

    def fit(self, X, y=None):
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        return self.math_features.transform(X)
    
class MathFeaturesProd(BasePrimitive):
    def __init__(self, columns):
        self.columns = columns
        self.math_features = MathFeatures(variables=self.columns, func='prod')

    def fit(self, X, y=None):
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        return self.math_features.transform(X)