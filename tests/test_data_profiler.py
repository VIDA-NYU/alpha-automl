import pandas as pd
from os.path import join, dirname
from alpha_automl.data_profiler import profile_data


def test_profile_data():
    dataset_path = join(dirname(__file__), './test_data/movies.csv')
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(columns=['rating'])

    actual_metadata = profile_data(X)
    expected_metadata = {'nonnumeric_columns': {'CATEGORICAL_ENCODER': [(1, 'type')],
                                                'TEXT_ENCODER': [(2, 'title'), (3, 'director'), (4, 'cast'),
                                                                 (5, 'country'), (8, 'duration'), (9, 'listed_in'),
                                                                 (10, 'description')],
                                                'DATETIME_ENCODER': [(6, 'date_added')]},
                         'useless_columns': [], 'missing_values': True}

    assert actual_metadata == expected_metadata
