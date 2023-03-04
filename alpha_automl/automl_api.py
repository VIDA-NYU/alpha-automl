import sys
import logging
import datetime
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from alpha_automl.automl_manager import AutoMLManager
from alpha_automl.utils import make_scorer, make_splitter, make_str_metric
from alpha_automl.visualization import plot_comparison_pipelines


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


AUTOML_NAME = 'AlphaAutoML'
PIPELINE_PREFIX = 'AlphaAutoML #'


class BaseAutoML():

    def __init__(self, output_folder, time_bound=15, metric=None, split_strategy='holdout', time_bound_run=5, task=None,
                 metric_kwargs=None, split_strategy_kwargs=None, verbose=False):
        """
        Create/instantiate an BaseAutoML object

        :param output_folder: Path to the output directory
        :param time_bound: Limit time in minutes to perform the search
        :param metric: A str (see model evaluation documentation in sklearn) or a scorer callable object/function
        :param split_strategy: Method to score the pipeline: `holdout, cross_validation or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits`
        :param time_bound_run: Limit time in minutes to score a pipeline
        :param task: The task to be solved
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
        self.new_primitives = {}
        self.scorer = None
        self.splitter = None
        self.leaderboard = None
        self.automl_manager = AutoMLManager(output_folder, time_bound, time_bound_run, task)

        if not verbose:
            # Hide all warnings and logs
            warnings.filterwarnings('ignore')
            for logger_name in logging.root.manager.loggerDict:
                if logger_name not in ['alpha_automl', 'alpha_automl.automl_api']:
                    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    def fit(self, X, y):
        """
        Search for pipelines and fit the best pipeline

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        """
        self.scorer = make_scorer(self.metric, self.metric_kwargs)
        self.splitter = make_splitter(self.split_strategy, self.split_strategy_kwargs)
        automl_hyperparams = {'new_primitives': self.new_primitives}
        start_time = datetime.datetime.utcnow()
        pipelines = []

        for pipeline in self.automl_manager.search_pipelines(X, y, self.scorer, self.splitter, automl_hyperparams):
            end_time = datetime.datetime.utcnow()

            if pipeline['message'] == 'FOUND':
                duration = str(end_time - start_time).split('.')[0]
                logger.info(f'Found pipeline, time={duration}, scoring...')
            elif pipeline['message'] == 'SCORED':
                logger.info(f'Scored pipeline, score={pipeline["pipeline_score"]}')
                pipelines.append(pipeline)

        if len(pipelines) == 0:
            logger.info('No pipelines were found')
            return

        logger.info(f'Found {len(pipelines)} pipelines')
        sorted_pipelines = sorted(pipelines, key=lambda x: x['pipeline_score'], reverse=True)  # TODO: Improve this, sort by score

        leaderboard_data = []
        for index, pipeline_data in enumerate(sorted_pipelines, start=1):
            pipeline_id = PIPELINE_PREFIX + str(index)
            pipeline_summary = self._get_pipeline_summary(pipeline_data['pipeline_object'].named_steps.keys())
            self.pipelines[pipeline_id] = {'pipeline_object': pipeline_data['pipeline_object'],
                                           'pipeline_score': pipeline_data['pipeline_score'],
                                           'pipeline_summary': pipeline_summary}

            leaderboard_data.append([index, pipeline_summary, pipeline_data['pipeline_score']])
            metric_str = make_str_metric(self.metric)
            self.leaderboard = pd.DataFrame(leaderboard_data, columns=['ranking', 'summary', metric_str])

        best_pipeline_id = PIPELINE_PREFIX + '1'
        self._fit(X, y, best_pipeline_id)

    def predict(self, X):
        """
        Predict classes for X using the best pipeline
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :return: The predictions
        """
        best_pipeline_id = PIPELINE_PREFIX + '1'

        return self._predict(X, best_pipeline_id)

    def score(self, X, y):
        """
        Return the performance (using the chosen metric) on the given test data and labels using the best pipeline.
        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        :return: A dict with metric and performance
        """
        best_pipeline_id = PIPELINE_PREFIX + '1'

        return self._score(X, y, best_pipeline_id)

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
            best_pipeline_id = PIPELINE_PREFIX + '1'
            pipeline_id = best_pipeline_id

        return self.pipelines[pipeline_id]['pipeline_object']

    def add_primitives(self, new_primitives):
        for primitive_object, primitive_name, primitive_type in new_primitives:
            self.new_primitives[primitive_name] = {'primitive_object': primitive_object, 'primitive_type': primitive_type}

    def get_leaderboard(self):
        """
        Return the leaderboard
        """
        return self.leaderboard

    def plot_leaderboard(self, use_print=False):
        """
        Plot the leaderboard
        """

        if len(self.pipelines) > 0:
            if use_print:
                print(self.leaderboard.to_string(index=False))
            else:
                return self.leaderboard.style.hide_index()
        else:
            logger.info('No pipelines were found')

    def plot_pipeline(self, pipeline_id=None, use_print=False):
        """
        Plot a pipeline, if pipeline_id is None, return the best pipeline
        :param pipeline_id: Id of a pipeline
        """
        if pipeline_id is None:
            best_pipeline_id = PIPELINE_PREFIX + '1'
            pipeline_id = best_pipeline_id

        if use_print:
            print(self.pipelines[pipeline_id]['pipeline_object'])
        else:
            return self.pipelines[pipeline_id]['pipeline_object']

    def plot_comparison_pipelines(self):
        """
        Plot PipelineProfiler visualization
        """
        pipelines = make_pipelineprofiler_inputs(self.pipelines, AUTOML_NAME)
        plot_comparison_pipelines(pipelines)

    def _fit(self, X, y, pipeline_id):
        self.pipelines[pipeline_id]['pipeline_object'].fit(X, y)

    def _predict(self, X, pipeline_id):
        predictions = self.pipelines[pipeline_id]['pipeline_object'].predict(X)

        return predictions

    def _score(self, X, y, pipeline_id):
        predictions = self.pipelines[pipeline_id]['pipeline_object'].predict(X)
        score = self.scorer._score_func(y, predictions)

        logger.info(f'Metric: {self.metric}, Score: {score}')

        return {'metric': self.metric, 'score': score}

    def _get_pipeline_summary(self, pipeline_steps):
        step_names = []

        for step in pipeline_steps:
            step_name = step.split('.')[-1]
            step_names.append(step_name)

        return ', '.join(step_names)


class AutoMLClassifier(BaseAutoML):

    def __init__(self, output_folder, time_bound=15, metric='accuracy', split_strategy='holdout', time_bound_run=5,
                 metric_kwargs=None, split_strategy_kwargs=None, verbose=False):
        """
        Create/instantiate an AutoMLClassifier object

        :param output_folder: Path to the output directory
        :param time_bound: Limit time in minutes to perform the search
        :param metric: A str (see model evaluation documentation in sklearn) or a scorer callable object/function
        :param split_strategy: Method to score the pipeline: `holdout, cross_validation or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits. `
        :param time_bound_run: Limit time in minutes to score a pipeline
        :param metric_kwargs: Additional arguments for metric
        :param split_strategy_kwargs: Additional arguments for splitting_strategy
        """

        self.label_enconder = LabelEncoder()
        task = 'CLASSIFICATION'
        super().__init__(output_folder, time_bound, metric, split_strategy, time_bound_run, task, metric_kwargs,
                         split_strategy_kwargs, verbose)

    def fit(self, X, y):
        y = self.label_enconder.fit_transform(y)
        super().fit(X, y)

    def predict(self, X):
        predictions = super().predict(X)

        return self.label_enconder.inverse_transform(predictions)

    def score(self, X, y):
        y = self.label_enconder.transform(y)

        return super().score(X, y)


class AutoMLRegressor(BaseAutoML):

    def __init__(self, output_folder, time_bound=15, metric='max_error', split_strategy='holdout', time_bound_run=5,
                 metric_kwargs=None, split_strategy_kwargs=None, verbose=False):
        """
        Create/instantiate an AutoMLRegressor object

        :param output_folder: Path to the output directory
        :param time_bound: Limit time in minutes to perform the search
        :param metric: A str (see model evaluation documentation in sklearn) or a scorer callable object/function
        :param split_strategy: Method to score the pipeline: `holdout, cross_validation or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits. `
        :param time_bound_run: Limit time in minutes to score a pipeline
        :param metric_kwargs: Additional arguments for metric
        :param split_strategy_kwargs: Additional arguments for splitting_strategy
        """

        task = 'REGRESSION'
        super().__init__(output_folder, time_bound, metric, split_strategy, time_bound_run, task, metric_kwargs,
                         split_strategy_kwargs, verbose)
