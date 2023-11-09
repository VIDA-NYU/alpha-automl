from alpha_automl import AutoMLClassifier
from os.path import join, dirname
import pandas as pd

if __name__ == '__main__':
    # If running it in Windows or CUDA environment, Alpha-AutoML should be used inside of "if __name__ == '__main__':"
    # Read the datasets
    train_dataset = pd.read_csv(join(dirname(__file__), 'datasets/299_libras_move/train_data.csv'))
    test_dataset = pd.read_csv(join(dirname(__file__), 'datasets/299_libras_move/test_data.csv'))

    target_column = 'class'
    X_train = train_dataset.drop(columns=[target_column])
    y_train = train_dataset[[target_column]]
    X_test = test_dataset.drop(columns=[target_column])
    y_test = test_dataset[[target_column]]

    # Add settings
    automl = AutoMLClassifier(time_bound=10)

    # Perform the search
    automl.fit(X_train, y_train)

    # Plot leaderboard
    automl.plot_leaderboard(use_print=True)

    # Evaluate best model
    automl.score(X_test, y_test)
