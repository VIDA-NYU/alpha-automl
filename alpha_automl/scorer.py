import logging
import datetime
import numpy as np
from sklearn.metrics import SCORERS, get_scorer, make_scorer as make_scorer_sk
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, train_test_split, cross_val_score
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits
from alpha_automl.utils import RANDOM_SEED

logger = logging.getLogger(__name__)


def make_scorer(metric, metric_kwargs=None):
    if isinstance(metric, str) and metric in SCORERS.keys():
        return get_scorer(metric)

    elif callable(metric):
        if metric_kwargs is None:
            metric_kwargs = {}

        module = getattr(metric, '__module__', '')
        if module.startswith('sklearn.metrics._scorer'):  # Heuristic to know if it is a sklearn scorer
            return metric

        else:
            return make_scorer_sk(metric, **metric_kwargs)

    else:
        raise ValueError(f'Unknown "{metric}" metric, you should choose among: {list(SCORERS.keys())} or a scorer '
                         f'callable object/function')


def make_splitter(splitting_strategy, splitting_strategy_kwargs=None):
    if splitting_strategy_kwargs is None:
        splitting_strategy_kwargs = {}

    if isinstance(splitting_strategy, str):
        if splitting_strategy == 'holdout':
            if 'test_size' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['test_size'] = 0.25
            if 'random_state' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['random_state'] = RANDOM_SEED

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


def score_pipeline(pipeline, X, y, scoring, splitting_strategy, verbose=True):
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
        if verbose:
            logger.warning('Detailed error:', exc_info=True)

    return score, start_time, end_time


def make_str_metric(metric):
    if isinstance(metric, str):
        return metric
    elif callable(metric):
        return metric.__name__
    else:
        return str(metric)
