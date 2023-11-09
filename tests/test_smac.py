import pandas as pd

from ConfigSpace import ConfigurationSpace
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from alpha_automl.scorer import make_scorer, make_splitter
from alpha_automl.hyperparameter_tuning.smac import SmacOptimizer, gen_configspace


class TestSmacOptimizer:
    """This is the testcases for SMAC optimizer."""
    X = pd.DataFrame(
        data={
            "Feature": [3, 2, 1, 2, 3, 2, 1, 2, 3, 2],
            "Value": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        }
    )
    y = pd.DataFrame(data={"Value": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]})
    splitter = make_splitter('holdout', None)
    scorer = make_scorer('accuracy_score', None)

    pipeline = Pipeline([("sklearn.linear_model.SGDClassifier", SGDClassifier())])

    def test_smac_init(self):
        smac = SmacOptimizer(X=self.X, y=self.y, splitter=self.splitter, scorer=self.scorer, n_trials=200)
        assert smac.n_trials == 200
    
    def test_gen_configspace(self):
        conf_space = gen_configspace(self.pipeline)
        assert isinstance(conf_space, ConfigurationSpace)
        assert list(conf_space.keys())[0] == 'alpha'

    def test_optimize_pipeline(self):
        smac = SmacOptimizer(X=self.X, y=self.y, splitter=self.splitter, scorer=self.scorer, n_trials=10)
        new_pipeline = smac.optimize_pipeline(self.pipeline)
        assert new_pipeline.get_params() != self.pipeline.get_params()
        new_pipeline.fit(self.X, self.y)
        self.pipeline.fit(self.X, self.y)
        assert new_pipeline.score(self.X, self.y) >= self.pipeline.score(self.X, self.y)

