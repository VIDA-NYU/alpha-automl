import numpy as np
import pandas as pd
from alpha_automl.base_primitive import BasePrimitive
from feature_engine.creation import MathFeatures

class MathFeaturesSum(BasePrimitive):
    def __init__(self, numeric_columns, column_names):
        self.column_names = column_names
        self.numeric_columns = numeric_columns
        self.math_features = MathFeatures(variables=self.numeric_columns, func='sum')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        return self.math_features.transform(X)
    
class MathFeaturesMean(BasePrimitive):
    def __init__(self, numeric_columns, column_names):
        self.column_names = column_names
        self.numeric_columns = numeric_columns
        self.math_features = MathFeatures(variables=self.numeric_columns, func='mean')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        return self.math_features.transform(X)
    
class MathFeaturesStd(BasePrimitive):
    def __init__(self, numeric_columns, column_names):
        self.column_names = column_names
        self.numeric_columns = numeric_columns
        self.math_features = MathFeatures(variables=self.numeric_columns, func='std')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        return self.math_features.transform(X)
    
class MathFeaturesProd(BasePrimitive):
    def __init__(self, numeric_columns, column_names):
        self.column_names = column_names
        self.numeric_columns = numeric_columns
        self.math_features = MathFeatures(variables=self.numeric_columns, func='prod')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        self.math_features.fit(X)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.column_names)
        return self.math_features.transform(X)