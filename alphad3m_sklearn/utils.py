import logging
import importlib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS, get_scorer, make_scorer as make_scorer_sk
from sklearn.model_selection import BaseCrossValidator, KFold, train_test_split
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits

logger = logging.getLogger(__name__)


def make_scorer(metric, metric_kwargs):
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


def make_splitter(splitting_strategy, splitting_strategy_kwargs, array):
    if splitting_strategy_kwargs is None:
        splitting_strategy_kwargs = {}

    if isinstance(splitting_strategy, str):
        if splitting_strategy == 'holdout':
            if 'test_size' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['test_size'] = 0.25
            if 'random_state' not in splitting_strategy_kwargs:
                splitting_strategy_kwargs['random_state'] = 1
            train_indices, test_indices = train_test_split(array, **splitting_strategy_kwargs)
            holdout_split = [(train_indices, test_indices)]

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


def score_pipeline(pipeline, X, y, scoring, splitting_strategy, verbose=False):
    score = None
    try:
        scores = cross_val_score(pipeline, X, y, cv=splitting_strategy, scoring=scoring, error_score='raise')
        score = np.average(scores)
    except Exception:
        if verbose:
            logging.warning("Exception scoring pipeline", exc_info=True)

    return score
