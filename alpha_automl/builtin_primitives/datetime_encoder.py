import numpy as np
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeature(BaseEstimator, TransformerMixin):
    """This is a datetime encoder using feature_engine.CyclicalFeatures(TM)
    Input:
        method: "cosine" or "sine", determines fitting the timestamp to cosine wave or sine wave. Cosine by default.
    Methods:
        fit: fit the timestamps to the grid.
        transform: return the numpy array fitted to the trigonometic waves.
        convert_to_datetime64_and_process: call pandas DataFrame, unify the timestamp.
    """

    def __init__(self, expand=True):
        self.embedder = CyclicalFeatures()
        self.method = "cosine"
        self.expand = expand

    def fit(self, X, y=None):
        if self.expand:
            df = self.expand_and_convert(X)
        else:
            df = self.convert_to_datetime64_and_process(X)
        self.embedder.fit(df)
        return self

    def transform(self, datetimes):
        if self.expand:
            df = self.expand_and_convert(datetimes)
        else:
            df = self.convert_to_datetime64_and_process(datetimes)
        embeddings = self.embedder.transform(df)
        embeddings = embeddings.filter(regex="_cos$|_sin$", axis=1)
        return np.array(embeddings)

    def convert_to_datetime64_and_process(self, datetimes):
        df = pd.DataFrame(datetimes)
        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: pd.Timestamp(x))
        df[df.columns[0]] = df[df.columns[0]].apply(
            lambda x: int(x.timestamp())
            - int(x.replace(month=12, day=12, hour=0, minute=0, second=0).timestamp())
        )
        return df

    def expand_and_convert(self, datetimes):
        df = pd.DataFrame(datetimes)
        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: pd.Timestamp(x))
        col_name = df.columns.values[0]
        df[f"{col_name}_year"] = df[col_name].dt.year
        df[f"{col_name}_month"] = df[col_name].dt.month
        df[f"{col_name}_day"] = df[col_name].dt.day
        df[f"{col_name}_hour"] = df[col_name].dt.hour
        df[f"{col_name}_minute"] = df[col_name].dt.minute
        df[f"{col_name}_dayofweek"] = df[col_name].dt.dayofweek
        df[f"{col_name}_dayofyear"] = df[col_name].dt.day_of_year
        df[f"{col_name}_quarter"] = df[col_name].dt.quarter
        df = df.drop(col_name, axis=1)
        for column in df.columns:
            while df[column].min() == 0 or df[column].max == 0:
                df[column] = df[column].apply(lambda x: x + 1)
        return df


class Datetime64ExpandEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, datetimes):
        df = self.convert_to_datetime64(datetimes)
        col_name = df.columns.values[0]
        df[f"{col_name}_year"] = df[col_name].dt.year
        df[f"{col_name}_month"] = df[col_name].dt.month
        df[f"{col_name}_day"] = df[col_name].dt.day
        df[f"{col_name}_hour"] = df[col_name].dt.hour
        df[f"{col_name}_minute"] = df[col_name].dt.minute
        df[f"{col_name}_dayofweek"] = df[col_name].dt.dayofweek
        df[f"{col_name}_dayofyear"] = df[col_name].dt.day_of_year
        df[f"{col_name}_quarter"] = df[col_name].dt.quarter
        df = df.drop(col_name, axis=1)
        return np.array(df)

    def convert_to_datetime64(self, datetimes):
        df = pd.DataFrame(datetimes)
        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: pd.Timestamp(x))
        return df


class DummyEncoder(BaseEstimator, TransformerMixin):
    """dummy variables approach"""

    def fit(self, X, y=None):
        return self

    def transform(self, datetimes):
        df = self.convert_to_datetime64(datetimes)
        col_name = df.columns.values[0]
        df[f"{col_name}_month"] = df[col_name].dt.month
        df[f"{col_name}_hour"] = df[col_name].dt.hour
        df[f"{col_name}_dayofweek"] = df[col_name].dt.day_of_week
        df[f"{col_name}_quarter"] = df[col_name].dt.quarter

        df = pd.get_dummies(
            df,
            columns=[
                f"{col_name}_month",
                f"{col_name}_hour",
                f"{col_name}_dayofweek",
                f"{col_name}_quarter",
            ],
        )
        df = df.drop(col_name, axis=1)
        return np.array(df)

    def convert_to_datetime64(self, datetimes):
        df = pd.DataFrame(datetimes)
        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: pd.Timestamp(x))
        return df
