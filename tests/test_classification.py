from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from alpha_automl import AutoMLClassifier

if __name__ == '__main__':
    diabetes = load_digits()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    print(set(y_train))

    # Add settings
    automl = AutoMLClassifier('/Users/rlopez/D3M/tmp/', time_bound=10, metric='accuracy', split_strategy='holdout', verbose=False)

    # Perform the search
    from xgboost import XGBClassifier
    xgboost_object = XGBClassifier()
    automl.add_primitives([(xgboost_object, 'new_xgboost', 'CLASSIFICATION')])

    automl.fit(X_train, y_train)

    # Evaluate best model
    y_hat = automl.predict(X_test)
    automl.score(X_test, y_test)

    # Evaluate best model externally
    acc = accuracy_score(y_test, y_hat)
    print("Accuracy: %.3f" % acc)

