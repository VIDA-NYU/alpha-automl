import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from alpha_automl.automl_api import (
    AutoMLClassifier,
    AutoMLRegressor,
    AutoMLSemiSupervisedClassifier,
    AutoMLTimeSeries,
)
from alpha_automl.utils import SemiSupervisedLabelEncoder, SemiSupervisedSplitter


class TestAutoMLTimeSeries:
    """This is the testcases for AutoMLTimeSeries API object."""

    api = AutoMLTimeSeries(
        output_folder="tmp/",
        time_bound=10,
        verbose=True,
        date_column="Date",
        target_column="Value",
    )

    def test_init(self):
        assert self.api.date_column == "Date"
        assert self.api.target_column == "Value"
        assert isinstance(self.api.splitter, TimeSeriesSplit)
        assert self.api.splitter.get_n_splits() == 5

    def test_column_parser(self):
        test_X = pd.DataFrame(
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
        X, y = self.api._column_parser(test_X)
        cols = list(X.columns.values)
        assert cols[0] == "Date"
        assert cols[1] == "Value"

    def test_pass_splitter_args(self):
        test_api = AutoMLTimeSeries(
            output_folder="tmp/",
            date_column="Date",
            target_column="Value",
            split_strategy_kwargs={"n_splits": 3, "test_size": 20},
        )

        assert test_api.splitter.get_n_splits() == 3


class TestAutoMLSemiSupervisedClassifier:
    """This is the testcases for AutoMLSemiSupervisedClassifier API object."""

    api = AutoMLSemiSupervisedClassifier(
        output_folder="tmp/",
        time_bound=10,
        verbose=True,
        split_strategy_kwargs={'test_size': .1},
    )

    def test_init(self):
        assert isinstance(self.api.splitter, SemiSupervisedSplitter)
        assert isinstance(self.api.label_encoder, SemiSupervisedLabelEncoder)
        assert self.api.splitter.get_n_splits() == 1
