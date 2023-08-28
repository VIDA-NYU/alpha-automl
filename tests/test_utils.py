from os.path import dirname, join

import numpy as np
import pandas as pd

from alpha_automl.utils import (
    SemiSupervisedLabelEncoder,
    SemiSupervisedSplitter,
    create_object,
    sample_dataset,
)


def test_create_object():
    from sklearn.ensemble import RandomForestClassifier

    import_path = "sklearn.ensemble.RandomForestClassifier"

    actual_object = create_object(import_path)
    expected_object = RandomForestClassifier()

    assert type(actual_object) == type(expected_object)


def test_sample_dataset():
    dataset_path = join(dirname(__file__), "./test_data/movies.csv")
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=["rating"])
    y = dataset[["rating"]]
    sample_size = 10

    actual_X, actual_y, actual_is_sampled = sample_dataset(
        X, y, sample_size, "CLASSIFICATION"
    )
    expected_X_len = sample_size
    expected_y_len = sample_size
    expected_is_sampled = True

    assert actual_is_sampled == expected_is_sampled
    assert len(actual_X) == expected_X_len
    assert len(actual_y) == expected_y_len


class TestSemiSupervisedLabelEncoder:
    y = pd.DataFrame(
        data={
            "type": [
                "type 4",
                "type 3",
                "type 2",
                "type 1",
                np.NaN,
                "type 2",
                np.NaN,
                "type 1",
                "type 3",
                "type 3",
                np.NaN,
            ]
        }
    )
    test_y = pd.DataFrame(data={"type": ["type 1", "type 2", "type 3", np.NaN]})
    test_y_df = pd.DataFrame(data={"type": [0, 1, 2, 3, -1]})
    encoder = SemiSupervisedLabelEncoder()

    def test_fit_transform(self):
        labeled_y = self.encoder.fit_transform(self.y)
        assert labeled_y[0] == 3
        assert labeled_y[-1] == -1

    def test_transform(self):
        self.encoder.fit_transform(self.y)
        labeled_y = self.encoder.transform(self.test_y)
        assert labeled_y[0] == 0
        assert labeled_y[-1] == -1

    def test_inverse_transform_np(self):
        labeled_y = self.encoder.fit_transform(self.y)
        inversed_y = self.encoder.inverse_transform(labeled_y)
        assert inversed_y[0] == "type 4"
        assert pd.isnull(inversed_y)[-1]

    def test_inverse_transform_df(self):
        self.encoder.fit_transform(self.y)
        inversed_y = self.encoder.inverse_transform(self.test_y_df)
        assert inversed_y[0] == "type 1"
        assert pd.isnull(inversed_y)[-1]


class TestSemiSupervisedSplitter:
    X = pd.DataFrame(
        data={
            "data": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "output": [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8],
        }
    )
    y = pd.DataFrame(data={"label": [4, 3, 2, -1, 4, 3, 2, -1, 4, 3, 2, -1]})
    splitter = SemiSupervisedSplitter()

    def test_split(self):
        split = self.splitter.split(self.X, self.y)
        test_split = list(split)[0][1]
        assert not pd.isnull(self.y["label"][test_split]).all()
