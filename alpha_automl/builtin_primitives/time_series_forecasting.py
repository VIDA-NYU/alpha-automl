import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from alpha_automl.base_primitive import BasePrimitive
from alpha_automl._optional_dependency import check_optional_dependency

ml_task = 'timeseries'
check_optional_dependency('gluonts', ml_task)
check_optional_dependency('neuralforecast', ml_task)
check_optional_dependency('pmdarima', ml_task)

from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from pmdarima.arima import auto_arima, ndiffs


logger = logging.getLogger(__name__)


# Arima
class ArimaEstimator(BasePrimitive):
    def __init__(
        self,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=6,
        trace=True,
    ):
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.max_p = max_p
        self.max_order = None
        self.trace = trace

    def fit(self, train, y=None):
        self.arima_model = auto_arima(
            train[train.columns[1]],
            d=self.get_ndiffs(train[train.columns[1]]),
            seasonal=self.seasonal,
            stepwise=self.stepwise,
            suppress_warnings=self.suppress_warnings,
            error_action=self.error_action,
            max_p=self.max_p,
            max_order=self.max_order,
            trace=self.trace,
        )

        logger.info(
            f"Making predictions for ARIMA model order: {self.arima_model.order}"
        )

    def predict(self, X):
        n_periods = len(X)
        fc = self.arima_model.predict(n_periods=n_periods, return_conf_int=False)
        if isinstance(fc, pd.Series):
            fc = fc.values
        fc = fc.reshape(fc.shape[0], -1)
        return fc

    def get_ndiffs(self, y_train):
        kpss_diffs = ndiffs(y_train, alpha=0.05, test="kpss", max_d=6)
        adf_diffs = ndiffs(y_train, alpha=0.05, test="adf", max_d=6)

        n_diffs = max(adf_diffs, kpss_diffs)
        logger.info(f"Estimated differencing term: {n_diffs}")
        return n_diffs


# DeepAR
class DeeparEstimator(BasePrimitive):
    def __init__(self):
        self.features = [
            "hour",
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "sin_day",
            "cos_day",
            "dayofmonth",
            "weekofyear",
        ]
        self.freq = "d"
        self.prediction_length = 1
        self.context_length = 30
        self.epochs = 5
        self.estimator = DeepAREstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            trainer=Trainer(epochs=self.epochs),
        )

    def fit(self, train, test=None):
        date_column = train.columns[0]
        target_column = train.columns[1]
        train = self.create_time_features(train, date_column)
        train = self.standard_scaling(train)
        train_list = ListDataset(
            [
                {
                    "start": train.index[0],
                    "target": train[target_column],
                    "feat_dynamic_real": [train[feature] for feature in self.features],
                }
            ],
            freq=self.freq,
        )

        self.predictor = self.estimator.train(training_data=train_list)

    def predict(self, X):
        date_column = X.columns[0]
        target_column = X.columns[1]
        test = self.create_time_features(X, date_column)
        test_list = ListDataset(
            [
                {
                    "start": test.index[0],
                    "target": test[target_column],
                    "feat_dynamic_real": [test[feature] for feature in self.features],
                }
            ],
            freq=self.freq,
        )
        _, ts_it = make_evaluation_predictions(
            test_list, predictor=self.predictor, num_samples=len(test)
        )

        return np.asarray(list(ts_it)[0])

    def create_time_features(self, df, date_col=None):
        """
        Creates time series features from datetime index
        """
        df[date_col] = df[date_col].apply(lambda x: pd.Timestamp(x))
        df = df.reset_index(drop=True)
        df = df.set_index(date_col)
        df["date"] = df.index
        df["hour"] = df["date"].dt.hour
        df["dayofweek"] = df["date"].dt.dayofweek
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["dayofyear"] = df["date"].dt.dayofyear
        df["sin_day"] = np.sin(df["dayofyear"])
        df["cos_day"] = np.cos(df["dayofyear"])
        df["dayofmonth"] = df["date"].dt.day
        df["weekofyear"] = df["date"].dt.weekofyear
        df = df.drop(["date"], axis=1)
        return df

    def standard_scaling(self, df):
        scaler = StandardScaler()
        scaler.fit(df[self.features])  # never scale on the training+test!
        df[self.features] = scaler.transform(df[self.features])

        return df


# NBEATS
class NBEATSEstimator(BasePrimitive):
    def __init__(self, freq="d", max_epochs=10):
        self.freq = freq
        self.max_epochs = max_epochs

    def fit(self, X, y):
        horizon = len(y)
        date_column = X.columns[0]
        X[date_column] = X[date_column].apply(lambda x: pd.Timestamp(x))
        nbeats = [
            NBEATS(input_size=horizon // 2, h=horizon, max_epochs=self.max_epochs)
        ]
        self.nbeats_nf = NeuralForecast(models=nbeats, freq=self.freq)

        tmp_X = X.copy()
        tmp_X.columns = ["ds", "y"]
        tmp_X.insert(loc=0, column="unique_id", value=1.0)
        self.nbeats_nf.fit(tmp_X)

    def predict(self, X):
        Y_hat_nbeats = self.nbeats_nf.predict().reset_index()
        return np.array(Y_hat_nbeats[["NBEATS"]])[: len(X)]


# NHITS
class NHITSEstimator(BasePrimitive):
    def __init__(self, freq="d", max_epochs=10):
        self.freq = freq
        self.max_epochs = max_epochs

    def fit(self, X, y):
        horizon = len(y)
        date_column = X.columns[0]
        X[date_column] = X[date_column].apply(lambda x: pd.Timestamp(x))
        nhits = [NHITS(input_size=horizon // 2, h=horizon, max_epochs=self.max_epochs)]
        self.nhits_nf = NeuralForecast(models=nhits, freq=self.freq)

        tmp_X = X.copy()
        tmp_X.columns = ["ds", "y"]
        tmp_X.insert(loc=0, column="unique_id", value=1.0)
        self.nhits_nf.fit(tmp_X)

    def predict(self, X):
        Y_hat_nhits = self.nhits_nf.predict().reset_index()
        return np.array(Y_hat_nhits[["NHITS"]])[: len(X)]
