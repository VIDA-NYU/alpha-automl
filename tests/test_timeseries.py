import pandas as pd
from sklearn.pipeline import Pipeline

from alpha_automl.builtin_primitives.time_series_forecasting import (
    ArimaEstimator,
    DeeparEstimator,
    NBEATSEstimator,
    NHITSEstimator,
)


class TestTimeSeriesForecasting:
    X = pd.DataFrame(
        data={
            "Date": [
                "1999/02/07",
                "1999/02/08",
                "1999/02/09",
                "1999/02/10",
                "1999/02/11",
                "1999/02/12",
                "1999/02/13",
                "1999/02/14",
                "1999/02/15",
                "1999/02/16",
            ],
            "Value": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        }
    )
    y = pd.DataFrame(data={"Value": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]})

    test = pd.DataFrame(data={"Date": ["1999/02/17", "1999/02/18"], "Value": [2, 3]})

    def test_arima(self):
        pipe = Pipeline(
            steps=[("alpha_automl.builtin_primitives.ArimaEstimator", ArimaEstimator())]
        )
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.test)
        assert pred.shape == (2, 1)

    def test_nbeats(self):
        pipe = Pipeline(
            steps=[
                ("alpha_automl.builtin_primitives.NBEATSEstimator", NBEATSEstimator())
            ]
        )
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.test)
        assert pred.shape == (2, 1)

    def test_nhits(self):
        pipe = Pipeline(
            steps=[("alpha_automl.builtin_primitives.NHITSEstimator", NHITSEstimator())]
        )
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.test)
        assert pred.shape == (2, 1)

    def test_deepar(self):
        pipe = Pipeline(
            steps=[
                ("alpha_automl.builtin_primitives.DeeparEstimator", DeeparEstimator())
            ]
        )
        pipe.fit(self.X, self.y)
        pred = pipe.predict(self.test)
        assert pred.shape == (2, 1)
