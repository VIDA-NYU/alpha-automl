import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from alpha_automl.builtin_primitives.semisupervised_classifier import (
    SkLabelPropagation,
    SkLabelSpreading,
    SkSelfTrainingClassifier,
    AutonBox,
)

class TestSemiSupervisedClassifier:
    """This is the testcases for semi-supervised classifier."""

    df = fetch_20newsgroups(
        subset="train",
        categories=[
            "talk.religion.misc",
        ],
    )
    X, y = df.data[:10], df.target[:10]
    y[-1] = 2
    X = pd.DataFrame([[x] for x in X], columns=["email"])
    y = pd.DataFrame([[y] for y in y], columns=["category"])
    vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
    st_pipeline = Pipeline(
        steps=[
            (
                "sklearn.compose.ColumnTransformer",
                ColumnTransformer(
                    remainder="passthrough",
                    transformers=[
                        (
                            "sklearn.feature_extraction.text.TfidfVectorizer-email",
                            TfidfVectorizer(),
                            0,
                        )
                    ],
                ),
            )
        ]
    )
    X = st_pipeline.fit_transform(X)

    def test_self_training_classifier(self):
        encoder = SkSelfTrainingClassifier()

        encoder.fit(self.X, self.y)
        pred = encoder.predict(self.X)
        print(pred, self.y)
        assert pred[0] == 0

    def test_label_spreading(self):
        encoder = SkLabelSpreading()
        encoder.fit(self.X, self.y)
        pred = encoder.predict(self.X)
        print(pred, self.y)
        assert pred[0] == 0

    def test_label_propagation(self):
        encoder = SkLabelPropagation()
        encoder.fit(self.X, self.y)
        pred = encoder.predict(self.X)
        print(pred, self.y)
        assert pred[0] == 0
    
    def test_autonbox(self):
        base_estimator = SGDClassifier(alpha=1e-5,
                                       penalty="l2",
                                       loss="log_loss")
        encoder = AutonBox(base_estimator=base_estimator)
        encoder.fit(self.X, self.y)
        pred = encoder.predict(self.X)
        print(pred, self.y)
        assert pred[0] == 0