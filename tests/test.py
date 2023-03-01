from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from alpha_automl import AutoMLClassifier

if __name__ == '__main__':
    diabetes = load_iris()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # Add settings
    automl = AutoMLClassifier('/Users/rlopez/D3M/tmp/', time_bound=10, metric='accuracy', split_strategy='holdout', verbose=False)

    # Perform the search
    automl.fit(X_train, y_train)

    # Plot
    #print(automl.plot_leaderboard())

    # Evaluate best model
    y_hat = automl.predict(X_test)
    automl.score(X_test, y_test)

    # Evaluate best model externally
    acc = accuracy_score(y_test, y_hat)
    print("Accuracy: %.3f" % acc)
