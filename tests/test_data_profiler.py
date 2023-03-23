from alpha_automl import AutoMLClassifier
import pandas as pd

if __name__ == '__main__':
    # Read the datasets
    output_path = '/Users/rlopez/D3M/tmp/'
    train_dataset = pd.read_csv('/Users/rlopez/D3M/examples/datasets/299_libras_move/train_data.csv')
    test_dataset = pd.read_csv('/Users/rlopez/D3M/examples/datasets/299_libras_move/test_data.csv')

    X_train = train_dataset.drop(columns=['class'])
    y_train = train_dataset[['class']]
    X_test = test_dataset.drop(columns=['class'])
    y_test = test_dataset[['class']]

    # Add settings
    automl = AutoMLClassifier(output_path, time_bound=10)

    # Perform the search
    automl.fit(X_train, y_train)

    # Plot leaderboard
    automl.plot_leaderboard(use_print=True)

    # Evaluate best model
    automl.score(X_test, y_test)
