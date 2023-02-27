import logging
import datetime
from alpha_automl.automl_manager import AutoMLManager
from alpha_automl.utils import make_scorer, make_splitter


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

id_best_pipeline = 'pipeline_1'


class AutoML():

    def __init__(self, output_folder, time_bound=15, metric=None, split_strategy='holdout', time_bound_run=5,
                 metric_kwargs=None, split_strategy_kwargs=None):
        """
        Create/instantiate an AutoML object

        :param output_folder: Path to the output directory
        :param time_bound: Limit time in minutes to perform the search
        :param metric: A str (see model evaluation documentation in sklearn) or a scorer callable object/function
        :param split_strategy: Method to score the pipeline: `holdout, cross_validation or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits. `
        :param time_bound_run: Limit time in minutes to score a pipeline
        :param metric_kwargs: Additional arguments for metric
        :param split_strategy_kwargs: Additional arguments for splitting_strategy
        """

        self.output_folder = output_folder
        self.time_bound = time_bound
        self.time_bound_run = time_bound_run
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.split_strategy = split_strategy
        self.split_strategy_kwargs = split_strategy_kwargs
        self.pipelines = {}
        self.scorer = None
        self.splitter = None
        self.automl_manager = AutoMLManager(output_folder, time_bound, time_bound_run)

    def fit(self, X, y):
        """
        Search for pipelines and fit the best pipeline

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        """
        self.scorer = make_scorer(self.metric, self.metric_kwargs)
        self.splitter = make_splitter(self.split_strategy, self.split_strategy_kwargs, y)
        start_time = datetime.datetime.utcnow()
        pipelines = []

        for pipeline in self.automl_manager.search_pipelines(X, y, self.scorer, self.splitter):
            end_time = datetime.datetime.utcnow()

            if pipeline['message'] == 'FOUND':
                duration = str(end_time - start_time)
                print(f'Found pipeline, time={duration}, scoring...')
            elif pipeline['message'] == 'SCORED':
                print(f'Scored pipeline, score={pipeline["pipeline_score"]}')
                pipelines.append(pipeline)

        sorted_pipelines = sorted(pipelines, key=lambda x: x['pipeline_score'])  # TODO: Improve this, sort by score

        for index, pipeline_data in enumerate(sorted_pipelines, start=1):
            pipeline_id = f'pipeline_{index}'
            self.pipelines[pipeline_id] = {'pipeline': pipeline_data['pipeline_object'],
                                           'score': pipeline_data['pipeline_score']}

        self._fit(X, y, id_best_pipeline)

    def predict(self, X):
        """
        Predict classes for X using the best pipeline
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :return: The predictions
        """

        return self._predict(X, id_best_pipeline)

    def score(self, X, y):
        """
        Return the performance (using the chosen metric) on the given test data and labels using the best pipeline.
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        :return: A dict with metric and performance
        """
        return self._score(X, y, id_best_pipeline)

    def fit_pipeline(self, X, y, pipeline_id):
        """
        Fit a pipeline given its id
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        :param pipeline_id: Id of a pipeline
        """
        self._fit(X, y, pipeline_id)

    def predict_pipeline(self, X, pipeline_id):
        """
        Predict classes for X given the id of a pipeline
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param pipeline_id: Id of a pipeline
        :return: The predictions
        """
        return self._predict(X, pipeline_id)

    def score_pipeline(self, X, y, pipeline_id):
        """
        Return the performance (using the chosen metric) on the given test data and labels using a given pipeline.
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        :param pipeline_id: Id of a pipeline
        :return: A dict with metric and performance
        """
        return self._score(X, y, pipeline_id)

    def get_pipeline(self, pipeline_id=None):
        """
        Return a pipeline, if pipeline_id is None, return the best pipeline
        :param pipeline_id: Id of a pipeline
        :return: A Pipeline object
        """
        if pipeline_id is None:
            pipeline_id = id_best_pipeline

        return self.pipelines[pipeline_id][pipeline_id]

    def plot_leaderboard(self):
        """
        Plot the leaderboard
        """
        pass

    def plot_comparison_pipelines(self):
        """
        Plot PipelineProfiler visualization
        """
        pass

    def _fit(self, X, y, pipeline_id):
        self.pipelines[pipeline_id]['pipeline'].fit(X, y)

    def _predict(self, X, pipeline_id):
        predictions = self.pipelines[pipeline_id]['pipeline'].predict(X)

        return predictions

    def _score(self, X, y, id_pipeline):
        predictions = self.pipelines[id_pipeline]['pipeline'].predict(X)
        score = self.scorer._score_func(y, predictions)

        print(f'Metric: {self.metric}, Score: {score}')

        return {'metric': self.metric, 'score': score}
