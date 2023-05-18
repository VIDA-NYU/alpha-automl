import logging
import datetime
import numpy as np
from sklearn.metrics import make_scorer as make_scorer_sk
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, cross_val_score
from alpha_automl.utils import RANDOM_SEED
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score,\
    max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score,\
    adjusted_mutual_info_score, rand_score, mutual_info_score, normalized_mutual_info_score


logger = logging.getLogger(__name__)

METRICS = {
    # Classification metrics
    'accuracy_score': accuracy_score,
    'f1_score': f1_score,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'jaccard_score': jaccard_score,
    # Regression metrics
    'max_error': max_error,
    'mean_absolute_error': mean_absolute_error,
    'mean_squared_error': mean_squared_error,
    'mean_squared_log_error': mean_squared_log_error,
    'median_absolute_error': median_absolute_error,
    'r2_score': r2_score,
    # Clustering metrics
    'adjusted_mutual_info_score': adjusted_mutual_info_score,
    'rand_score': rand_score,
    'mutual_info_score': mutual_info_score,
    'normalized_mutual_info_score': normalized_mutual_info_score
}

# How metrics should be order to get the best scores
METRICS_ORDERING = {
    # Classification metrics
    accuracy_score: 'ascending',
    f1_score: 'ascending',
    precision_score: 'ascending',
    recall_score: 'ascending',
    jaccard_score: 'ascending',
    # Regression metrics
    max_error: 'descending',
    mean_absolute_error: 'descending',
    mean_squared_error: 'descending',
    mean_squared_log_error: 'descending',
    median_absolute_error: 'descending',
    r2_score: 'ascending',
    # Clustering metrics
    adjusted_mutual_info_score: 'ascending',
    rand_score: 'ascending',
    mutual_info_score: 'ascending',
    normalized_mutual_info_score: 'ascending'
}


def make_scorer(metric, metric_kwargs=None):
    if metric_kwargs is None:
        metric_kwargs = {}

    if isinstance(metric, str) and metric in METRICS:
        return make_scorer_sk(METRICS[metric], **metric_kwargs)

    elif callable(metric):
        module = getattr(metric, '__module__', '')
        if module.startswith('sklearn.metrics._scorer'):  # Heuristic to know if it is a sklearn scorer
            return metric

        else:
            return make_scorer_sk(metric, **metric_kwargs)

    else:
        raise ValueError(f'Unknown "{metric}" metric, you should choose among: {list(METRICS.keys())} or a callable '
                         f'object/function')


def make_splitter(splitting_strategy, splitting_strategy_kwargs=None):
    if splitting_strategy_kwargs is None:
        splitting_strategy_kwargs = {}

    if isinstance(splitting_strategy, str):
        if splitting_strategy == 'holdout':
            if 'test_size' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['test_size'] = 0.25
            if 'random_state' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['random_state'] = RANDOM_SEED
            if 'n_splits' in splitting_strategy_kwargs:
                raise ValueError('You sent the keyword argument "n_splits" for holdout, but it is not needed.'
                                 'Use "cv" (cross-validation) or do not send it for holdout.')

            holdout_split = ShuffleSplit(n_splits=1, **splitting_strategy_kwargs)

            return holdout_split

        elif splitting_strategy == 'cv':
            if 'n_splits' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['n_splits'] = 5
            kfold_split = KFold(**splitting_strategy_kwargs)

            return kfold_split

        else:
            raise ValueError(f'Unknown "{splitting_strategy}" splitting strategy, you should choose "holdout", "cv" or '
                             f'an instance of BaseCrossValidator, BaseShuffleSplit, RepeatedSplits.')

    elif isinstance(splitting_strategy, (BaseCrossValidator, BaseShuffleSplit, _RepeatedSplits)):
        return splitting_strategy

    else:
        raise ValueError(f'Unknown "{splitting_strategy}" splitting strategy, you should choose "holdout", "cv" or an '
                         f'instance of BaseCrossValidator, BaseShuffleSplit, RepeatedSplits.')


def score_pipeline(pipeline, X, y, scoring, splitting_strategy):
    score = None
    start_time = None
    end_time = None

    try:
        start_time = datetime.datetime.utcnow().isoformat() + 'Z'
        scores = cross_val_score(pipeline, X, y, cv=splitting_strategy, scoring=scoring, error_score='raise')
        end_time = datetime.datetime.utcnow().isoformat() + 'Z'
        score = np.average(scores)
        logger.info(f'Score: {score}')
    except Exception:
        logger.warning('Exception scoring a pipeline')
        logger.warning('Detailed error:', exc_info=True)

    return score, start_time, end_time


def make_str_metric(metric):
    if isinstance(metric, str):
        return metric
    elif callable(metric):
        return metric.__name__
    else:
        return str(metric)


def get_sign_sorting(metric, suggested_sorting_mode):
    # This sign is used to follow the convention that higher values are better values
    sorting_mode = None
    if suggested_sorting_mode == 'auto' and metric in METRICS_ORDERING:
        sorting_mode = METRICS_ORDERING[metric]
    elif suggested_sorting_mode in {'ascending', 'descending'} and metric in METRICS_ORDERING:
        logger.warning('You are specifying a mode to order a built-in metric')
        sorting_mode = suggested_sorting_mode
    elif suggested_sorting_mode == 'auto' and metric not in METRICS_ORDERING:
        logger.warning('You should specify the mode to order the scores for your defined metric. Using "ascending".')
        sorting_mode = 'ascending'
    elif suggested_sorting_mode in {'ascending', 'descending'} and metric not in METRICS_ORDERING:
        sorting_mode = suggested_sorting_mode
    else:
        raise ValueError(f'Unknown "{suggested_sorting_mode}" sorting mode.')

    sign = 1 if sorting_mode == 'ascending' else -1

    return sign
