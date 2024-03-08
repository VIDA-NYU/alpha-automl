import argparse
from os.path import dirname, join

import openml
import pandas as pd
from alpha_automl import AutoMLClassifier

if __name__ == "__main__":
    # If running it in Windows or CUDA environment, Alpha-AutoML should be used inside of "if __name__ == '__main__':"

    # argparser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-s",
        "--save",
        type=bool,
        default=False,
        help="If AlphaAutoML PPO need to learn from the weights from this task.",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        default=None,
        help="The specific dataset directory, should contain train_data.csv and test_data.csv.",
    )
    parser.add_argument(
        "-c",
        "--target-column",
        default="class",
        help="The target column name for local dataset csv files, 'class' by defualt",
    )

    parser.add_argument(
        "-t",
        "--task",
        metavar="task_id",
        type=int,
        default=None,
        help="The specific task name (as defined in the benchmark file) to run."
        "\nWhen an OpenML reference is used as benchmark, the dataset name should be used instead."
        "\nIf not provided, then all tasks from the benchmark will be run.",
    )
    parser.add_argument(
        "-T",
        "--time-bound",
        type=int,
        default=1,
        help="The time bound for running AlphaAutoML task, unit by minute, 1 minute by default.",
    )

    args = parser.parse_args()

    # Read the datasets
    if args.dataset_dir:
        train_dataset = pd.read_csv(
            join(dirname(__file__), args.dataset_dir, "train_data.csv")
        )
        test_dataset = pd.read_csv(
            join(dirname(__file__), args.dataset_dir, "test_data.csv")
        )
        target_column = args.target_column
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]
        X_test = test_dataset.drop(columns=[target_column])
        y_test = test_dataset[[target_column]]
    elif args.task:
        task = openml.tasks.get_task(args.task, download_qualities=False)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices(
            repeat=0,
            fold=0,
            sample=0,
        )
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

    # Add settings
    automl = AutoMLClassifier(time_bound=args.time_bound, save_checkpoint=args.save)

    # Perform the search
    automl.fit(X_train, y_train)

    # Plot leaderboard
    automl.plot_leaderboard(use_print=True)

    # Evaluate best model
    automl.score(X_test, y_test)
