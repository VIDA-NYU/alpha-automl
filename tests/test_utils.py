import pandas as pd
from os.path import join, dirname
from alpha_automl.utils import create_object, sample_dataset


def test_create_object():
    from sklearn.ensemble import RandomForestClassifier
    import_path = 'sklearn.ensemble.RandomForestClassifier'

    actual_object = create_object(import_path)
    expected_object = RandomForestClassifier()

    assert type(actual_object) == type(expected_object)


def test_sample_dataset():
    dataset_path = join(dirname(__file__), './test_data/movies.csv')
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=['rating'])
    y = dataset[['rating']]
    sample_size = 10

    actual_X, actual_y, actual_is_sampled = sample_dataset(X, y, sample_size, 'CLASSIFICATION')
    expected_X_len = sample_size
    expected_y_len = sample_size
    expected_is_sampled = True

    assert actual_is_sampled == expected_is_sampled
    assert len(actual_X) == expected_X_len
    assert len(actual_y) == expected_y_len
