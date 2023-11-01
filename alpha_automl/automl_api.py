import os
import sys
import logging
import datetime
import tempfile
import pandas as pd
from multiprocessing import set_start_method
from sklearn.preprocessing import LabelEncoder
from alpha_automl.automl_manager import AutoMLManager
from alpha_automl.scorer import make_scorer, make_splitter, make_str_metric, get_sign_sorting
from alpha_automl.utils import make_d3m_pipelines, hide_logs, get_start_method, check_input_for_multiprocessing, SemiSupervisedSplitter, SemiSupervisedLabelEncoder
from alpha_automl.visualization import plot_comparison_pipelines


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

AUTOML_NAME = 'AlphaAutoML'
PIPELINE_PREFIX = 'Pipeline #'


class BaseAutoML():

    def __init__(self, time_bound=15, metric=None, split_strategy='holdout', time_bound_run=5, task=None,
                 score_sorting='auto', metric_kwargs=None, split_strategy_kwargs=None, start_mode='auto',
                 verbose=logging.INFO):
        """
        Create/instantiate an BaseAutoML object.

        :param time_bound: Limit time in minutes to perform the search
        :param metric: A str (see in the documentation the list of available metrics) or a callable object/function
        :param split_strategy: Method to score the pipeline: `holdout`, `cross_validation` or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits
        :param time_bound_run: Limit time in minutes to score a pipeline
        :param task: The task to be solved
        :param score_sorting: The sort used to order the scores. It could be `auto`, `ascending` or `descending`.
            `auto` is used for the built-in metrics. For the user-defined metrics, this param must be passed.
        :param metric_kwargs: Additional arguments for metric
        :param split_strategy_kwargs: Additional arguments for splitting_strategy.
        :param start_mode: The mode to start the multiprocessing library. It could be `auto`, `fork` or `spawn`.
        :param verbose: Whether or not to show additional logs
        """
        tmpdirname = tempfile.mkdtemp(prefix="alpha_automl", suffix="_log")
        logger.info(f'created temporary directory: {tmpdirname}')
        self.output_folder = tmpdirname
        self.time_bound = time_bound
        self.time_bound_run = time_bound_run
        self.metric = make_str_metric(metric)
        self.metric_kwargs = metric_kwargs
        self.split_strategy = split_strategy
        self.split_strategy_kwargs = split_strategy_kwargs
        self.scorer = make_scorer(metric, metric_kwargs)
        self.score_sorting = score_sorting
        self.splitter = make_splitter(split_strategy, split_strategy_kwargs)
        self.pipelines = {}
        self.new_primitives = {}
        self.X = None
        self.y = None
        self.leaderboard = None
        self.automl_manager = AutoMLManager(self.output_folder, time_bound, time_bound_run, task, verbose)

        hide_logs(verbose)
        
        self._start_method = get_start_method(start_mode)
        set_start_method(self._start_method, force=True)
        check_input_for_multiprocessing(self._start_method, self.scorer._score_func, 'metric')
        check_input_for_multiprocessing(self._start_method, self.splitter, 'split strategy')

    def fit(self, X, y):
        """
        Search for pipelines and fit the best pipeline.

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        """
        self.X = X
        self.y = y
        automl_hyperparams = {'new_primitives': self.new_primitives}
        pipelines = []
        start_time = datetime.datetime.utcnow()

        for pipeline_data in self.automl_manager.search_pipelines(X, y, self.scorer, self.splitter, automl_hyperparams):
            end_time = datetime.datetime.utcnow()
            pipeline = pipeline_data['pipeline']
            message = pipeline_data['message']

            if message == 'FOUND':
                duration = str(end_time - start_time).split('.')[0]
                logger.info(f'Found pipeline, time={duration}, scoring...')
            elif message == 'SCORED':
                logger.info(f'Scored pipeline, score={pipeline.get_score()}')
                pipelines.append(pipeline)

        if len(pipelines) == 0:
            logger.warning('No pipelines were found')
            return

        logger.info(f'Found {len(pipelines)} pipelines')
        sign = get_sign_sorting(self.scorer._score_func, self.score_sorting)
        sorted_pipelines = sorted(pipelines, key=lambda x: x.get_score() * sign, reverse=True)

        leaderboard_data = []
        for index, pipeline in enumerate(sorted_pipelines, start=1):
            pipeline_id = PIPELINE_PREFIX + str(index)
            self.pipelines[pipeline_id] = pipeline
            if (
                pipeline.get_pipeline().steps[-1][0]
                == 'sklearn.semi_supervised.SelfTrainingClassifier'
                or pipeline.get_pipeline().steps[-1][0]
                == 'alpha_automl.builtin_primitives.semisupervised_classifier.AutonBox'
            ):
                leaderboard_data.append(
                    [
                        index,
                        f'{pipeline.get_summary()}, {pipeline.get_pipeline().steps[-1][1].base_estimator.__class__.__name__}',
                        pipeline.get_score(),
                    ]
                )
            else:
                leaderboard_data.append([index, pipeline.get_summary(), pipeline.get_score()])
            

        self.leaderboard = pd.DataFrame(leaderboard_data, columns=['ranking', 'pipeline', self.metric])

        best_pipeline_id = PIPELINE_PREFIX + '1'
        self._fit(X, y, best_pipeline_id)

    def predict(self, X):
        """
        Predict classes for X using the best pipeline.

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :return: The predictions
        """
        if len(self.pipelines) == 0:
            logger.warning('No pipelines were found')
            return

        best_pipeline_id = PIPELINE_PREFIX + '1'

        return self._predict(X, best_pipeline_id)

    def score(self, X, y):
        """
        Return the performance (using the chosen metric) on the given test data and labels using the best pipeline.

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        :return: A dict with metric and performance
        """
        if len(self.pipelines) == 0:
            logger.warning('No pipelines were found')
            return

        best_pipeline_id = PIPELINE_PREFIX + '1'

        return self._score(X, y, best_pipeline_id)

    def fit_pipeline(self, pipeline_id):
        """
        Fit a pipeline given its id.

        :param pipeline_id: Id of a pipeline
        """
        self._fit(self.X, self.y, pipeline_id)

    def predict_pipeline(self, X, pipeline_id):
        """
        Predict classes for X given the id of a pipeline.

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param pipeline_id: Id of a pipeline
        :return: The predictions
        """
        self._fit(self.X, self.y, pipeline_id)
        return self._predict(X, pipeline_id)

    def score_pipeline(self, X, y, pipeline_id):
        """
        Return the performance (using the chosen metric) on the given test data and labels using a given pipeline.

        :param X: The training input samples, array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: The target classes, array-like, shape = [n_samples] or [n_samples, n_outputs]
        :param pipeline_id: Id of a pipeline
        :return: A dict with metric and performance
        """
        self._fit(self.X, self.y, pipeline_id)
        return self._score(X, y, pipeline_id)

    def get_pipeline(self, pipeline_id=None):
        """
        Return a pipeline given its pipeline id, if pipeline_id is None, return the best pipeline.

        :param pipeline_id: Id of a pipeline
        :return: A Pipeline object
        """
        if pipeline_id is None:
            best_pipeline_id = PIPELINE_PREFIX + '1'
            pipeline_id = best_pipeline_id

        return self.pipelines[pipeline_id].get_pipeline()

    def add_primitives(self, new_primitives):
        """
        Add new primitives.

        :param new_primitives: Set of new primitives, tuples of name and object primitive
        """
        for primitive_object, primitive_type in new_primitives:
            check_input_for_multiprocessing(self._start_method, primitive_object, 'primitive')
            primitive_name = f'{primitive_object.__module__}.{primitive_object.__class__.__name__}'
            primitive_name = primitive_name.replace('__', '')  # Sklearn restriction on estimator names
            self.new_primitives[primitive_name] = {'primitive_object': primitive_object,
                                                   'primitive_type': primitive_type}

    def get_leaderboard(self):
        """
        Return the leaderboard.

        :return: The leaderboard
        """
        return self.leaderboard

    def plot_leaderboard(self, use_print=False):
        """
        Plot the leaderboard.

        :param use_print: Whether or not to use a regular print
        :return: The leaderboard
        """
        if len(self.pipelines) > 0:
            if use_print:
                print(self.leaderboard.to_string(index=False))
            else:
                decimal_format = {self.metric: '{:.3f}'}
                try:
                    return self.leaderboard.style.format(decimal_format).hide_index()
                except Exception:  # For newer version of Pandas
                    return self.leaderboard.style.format(decimal_format).hide()
        else:
            logger.info('No pipelines were found')

    def plot_pipeline(self, pipeline_id=None, use_print=False):
        """
        Plot a pipeline, if pipeline_id is None, return the best pipeline.

        :param pipeline_id: Id of a pipeline
        :param use_print: Whether or not to use a regular print
        """
        if pipeline_id is None:
            best_pipeline_id = PIPELINE_PREFIX + '1'
            pipeline_id = best_pipeline_id

        if use_print:
            print(self.pipelines[pipeline_id].get_pipeline())
        else:
            return self.pipelines[pipeline_id].get_pipeline()

    def plot_comparison_pipelines(self, precomputed_pipelines=None, precomputed_primitive_types=None):
        """
        Plot PipelineProfiler visualization.

        :param precomputed_pipelines: Pre-calculated list of pipelines
        :param precomputed_primitive_types: Pre-calculated list of primitive types
        """
        if precomputed_pipelines is None and precomputed_primitive_types is None:
            sign = get_sign_sorting(self.scorer._score_func, self.score_sorting)
            pipelines, primitive_types = make_d3m_pipelines(self.pipelines, self.new_primitives, self.metric, sign)
            plot_comparison_pipelines(pipelines, primitive_types)
        else:
            plot_comparison_pipelines(precomputed_pipelines, precomputed_primitive_types)

    def _fit(self, X, y, pipeline_id):
        self.pipelines[pipeline_id].get_pipeline().fit(X, y)

    def _predict(self, X, pipeline_id):
        predictions = self.pipelines[pipeline_id].get_pipeline().predict(X)

        return predictions

    def _score(self, X, y, pipeline_id):
        predictions = self.pipelines[pipeline_id].get_pipeline().predict(X)
        score = self.scorer._score_func(y, predictions)

        logger.info(f'Metric: {self.metric}, Score: {score}')

        return {'metric': self.metric, 'score': score}


class AutoMLClassifier(BaseAutoML):

    def __init__(self, time_bound=15, metric='accuracy_score', split_strategy='holdout',
                 time_bound_run=5, score_sorting='auto', metric_kwargs=None, split_strategy_kwargs=None,
                 start_mode='auto', verbose=logging.INFO):
        """
        Create/instantiate an AutoMLClassifier object.

        :param time_bound: Limit time in minutes to perform the search.
        :param metric: A str (see in the documentation the list of available metrics) or a callable object/function.
        :param split_strategy: Method to score the pipeline: `holdout`, `cross_validation` or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits.
        :param time_bound_run: Limit time in minutes to score a pipeline.
        :param score_sorting: The sort used to order the scores. It could be `auto`, `ascending` or `descending`.
            `auto` is used for the built-in metrics. For the user-defined metrics, this param must be passed.
        :param metric_kwargs: Additional arguments for metric.
        :param split_strategy_kwargs: Additional arguments for splitting_strategy.
        :param start_mode: The mode to start the multiprocessing library. It could be `auto`, `fork` or `spawn`.
        :param verbose: Whether or not to show additional logs.
        """

        self.label_enconder = LabelEncoder()
        task = 'CLASSIFICATION'
        super().__init__(time_bound, metric, split_strategy, time_bound_run, task, score_sorting,
                         metric_kwargs, split_strategy_kwargs, start_mode, verbose)

    def fit(self, X, y):
        y = self.label_enconder.fit_transform(y)
        super().fit(X, y)

    def predict(self, X):
        predictions = super().predict(X)

        return self.label_enconder.inverse_transform(predictions)

    def score(self, X, y):
        y = self.label_enconder.transform(y)

        return super().score(X, y)

    def fit_pipeline(self, pipeline_id):
        super().fit_pipeline(pipeline_id)

    def predict_pipeline(self, X, pipeline_id):
        predictions = super().predict_pipeline(X, pipeline_id)

        return self.label_enconder.inverse_transform(predictions)

    def score_pipeline(self, X, y, pipeline_id):
        y = self.label_enconder.transform(y)

        return super().score_pipeline(X, y, pipeline_id)


class AutoMLRegressor(BaseAutoML):

    def __init__(self, time_bound=15, metric='mean_absolute_error', split_strategy='holdout',
                 time_bound_run=5, score_sorting='auto', metric_kwargs=None, split_strategy_kwargs=None,
                 start_mode='auto', verbose=logging.INFO):
        """
        Create/instantiate an AutoMLRegressor object.

        :param time_bound: Limit time in minutes to perform the search.
        :param metric: A str (see in the documentation the list of available metrics) or a callable object/function.
        :param split_strategy: Method to score the pipeline: `holdout`, `cross_validation` or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits.
        :param time_bound_run: Limit time in minutes to score a pipeline.
        :param score_sorting: The sort used to order the scores. It could be `auto`, `ascending` or `descending`.
            `auto` is used for the built-in metrics. For the user-defined metrics, this param must be passed.
        :param metric_kwargs: Additional arguments for metric.
        :param split_strategy_kwargs: Additional arguments for splitting_strategy.
        :param start_mode: The mode to start the multiprocessing library. It could be `auto`, `fork` or `spawn`.
        :param verbose: Whether or not to show additional logs.
        """

        task = 'REGRESSION'
        super().__init__(time_bound, metric, split_strategy, time_bound_run, task, score_sorting,
                         metric_kwargs, split_strategy_kwargs, start_mode, verbose)

        
class AutoMLTimeSeries(BaseAutoML):
    def __init__(self, time_bound=15, metric='mean_squared_error', split_strategy='timeseries',
                 time_bound_run=5, score_sorting='auto', metric_kwargs=None, split_strategy_kwargs=None,
                 verbose=logging.INFO, date_column=None, target_column=None):
        """
        Create/instantiate an AutoMLTimeSeries object.

        :param time_bound: Limit time in minutes to perform the search.
        :param metric: A str (see in the documentation the list of available metrics) or a callable object/function.
        :param split_strategy: Method to score the pipeline: `holdout`, `cross_validation` or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits.
        :param time_bound_run: Limit time in minutes to score a pipeline.
        :param score_sorting: The sort used to order the scores. It could be `auto`, `ascending` or `descending`.
            `auto` is used for the built-in metrics. For the user-defined metrics, this param must be passed.
        :param metric_kwargs: Additional arguments for metric.
        :param split_strategy_kwargs: Additional arguments for TimeSeriesSplit, E.g. n_splits and test_size(int).
        :param start_mode: The mode to start the multiprocessing library. It could be `auto`, `fork` or `spawn`.
        :param verbose: Whether or not to show additional logs.
        """

        task = 'TIME_SERIES_FORECAST'
        self.date_column = date_column
        self.target_column = target_column
        super().__init__(time_bound, metric, split_strategy, time_bound_run, task, score_sorting,
                         metric_kwargs, split_strategy_kwargs, verbose)
        
    def _column_parser(self, X):
        cols = list(X.columns.values)
        cols.remove(self.date_column)
        cols.remove(self.target_column)
        X = X[[self.date_column, self.target_column] + cols]
        y = X[[self.target_column]]
        return X, y
        
    def fit(self, X, y=None):
        X, y = self._column_parser(X)
        super().fit(X, y)


class AutoMLSemiSupervisedClassifier(BaseAutoML):

    def __init__(self, time_bound=15, metric='f1_score', split_strategy='holdout',
                 time_bound_run=5, score_sorting='auto', metric_kwargs={'average': 'micro'}, split_strategy_kwargs=None,
                 start_mode='auto', verbose=logging.INFO):
        """
        Create/instantiate an AutoMLSemiSupervisedClassifier object.
        
        :param time_bound: Limit time in minutes to perform the search.
        :param metric: A str (see in the documentation the list of available metrics) or a callable object/function.
        :param split_strategy: Method to score the pipeline: `holdout`, `cross_validation` or an instance of
            BaseCrossValidator, BaseShuffleSplit, RepeatedSplits.
        :param time_bound_run: Limit time in minutes to score a pipeline.
        :param score_sorting: The sort used to order the scores. It could be `auto`, `ascending` or `descending`.
            `auto` is used for the built-in metrics. For the user-defined metrics, this param must be passed.
        :param metric_kwargs: Additional arguments for metric.
        :param split_strategy_kwargs: Additional arguments for splitting_strategy. In SemiSupervised case, `n_splits`
            and `test_size`(test proportion from 0 to 1) can be pass to the splitter.
        :param start_mode: The mode to start the multiprocessing library. It could be `auto`, `fork` or `spawn`.
        :param verbose: Whether or not to show additional logs.
        """
        self.label_enconder = SemiSupervisedLabelEncoder()
        task = 'SEMISUPERVISED'
        super().__init__(time_bound, metric, split_strategy, time_bound_run, task, score_sorting,
                         metric_kwargs, split_strategy_kwargs, start_mode, verbose)

        if split_strategy_kwargs is None:
            split_strategy_kwargs = {'test_size': 0.2}

        self.splitter = SemiSupervisedSplitter(**split_strategy_kwargs)

    def fit(self, X, y):
        y = self.label_enconder.fit_transform(y)
        super().fit(X, y)

    def predict(self, X):
        predictions = super().predict(X)

        return self.label_enconder.inverse_transform(predictions)

    def score(self, X, y):
        y = self.label_enconder.transform(y)

        return super().score(X, y)

    def fit_pipeline(self, pipeline_id):
        super().fit_pipeline(pipeline_id)

    def predict_pipeline(self, X, pipeline_id):
        predictions = super().predict_pipeline(X, pipeline_id)

        return self.label_enconder.inverse_transform(predictions)

    def score_pipeline(self, X, y, pipeline_id):
        y = self.label_enconder.transform(y)

        return super().score_pipeline(X, y, pipeline_id)
