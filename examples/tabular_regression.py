from alpha_automl import AutoMLRegressor
from os.path import join, dirname
import pandas as pd

if __name__ == '__main__':
    # Read the datasets
    output_path = join(dirname(__file__), 'tmp/')
    train_dataset = pd.read_csv(join(dirname(__file__), 'datasets/196_autoMpg/train_data.csv'))
    test_dataset = pd.read_csv(join(dirname(__file__), 'datasets/196_autoMpg/test_data.csv'))

    X_train = train_dataset.drop(columns=['class'])
    y_train = train_dataset[['class']]
    X_test = test_dataset.drop(columns=['class'])
    y_test = test_dataset[['class']]

    # Add settings
    automl = AutoMLRegressor(output_path, time_bound=10)

    # Perform the search
    automl.fit(X_train, y_train)

    # Plot leaderboard
    automl.plot_leaderboard(use_print=True)

    # Evaluate best model
    automl.score(X_test, y_test)
