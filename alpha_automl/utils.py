import logging
import importlib
import numpy as np
from sklearn.metrics import SCORERS, get_scorer, make_scorer as make_scorer_sk
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, train_test_split, cross_val_score
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits

logger = logging.getLogger(__name__)

RANDOM_SEED = 0


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


def create_object(import_path, class_params=None):
    if class_params is None:
        class_params = {}

    modules = import_path.split('.')
    class_name = modules[-1]
    import_module = '.'.join(modules[:-1])
    class_ = getattr(importlib.import_module(import_module), class_name)
    object_ = class_(**class_params)  # Instantiates the object

    return object_


def score_pipeline(pipeline, X, y, scoring, splitting_strategy, verbose=True):
    score = None

    try:
        scores = cross_val_score(pipeline, X, y, cv=splitting_strategy, scoring=scoring, error_score='raise')
        score = np.average(scores)
        logger.info(f'Score: {score}')
    except Exception:
        logger.warning('Exception scoring a pipeline')
        if verbose:
            logger.warning('Detailed error:', exc_info=True)

    return score


def sample_dataset(X, y, sample_size):
    original_size = len(X)

    if original_size > sample_size:
        ratio = sample_size / original_size
        try:
            _, X_test, _, y_test = train_test_split(X, y, random_state=RANDOM_SEED, test_size=ratio, stratify=y)
        except:
            # Not using stratified sampling when the minority class has few instances, not enough for all the folds
            _, X_test, _, y_test = train_test_split(X, y, random_state=RANDOM_SEED, test_size=ratio)
        logger.info(f'Sampling down data from {original_size} to {len(X_test)}')
        return X_test, y_test, True

    else:
        logger.info('Not doing sampling for small dataset (size = %d)', original_size)
        return X, y, False


def make_str_metric(metric):
    if isinstance(metric, str):
        return metric
    elif callable(metric):
        return metric.__name__
    else:
        return str(metric)
