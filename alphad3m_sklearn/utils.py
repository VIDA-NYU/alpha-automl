from sklearn.metrics import SCORERS, make_scorer
from sklearn.model_selection import BaseCrossValidator, KFold, train_test_split
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits


def format_metric(metric, metric_kwargs):
    if isinstance(metric, str) and metric in SCORERS.keys():
        return metric

    elif callable(metric):  # TODO: Add more conditions like https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/metrics/_scorer.py#L480
        if metric_kwargs is None:
            metric_kwargs = {}

        return make_scorer(metric, **metric_kwargs)

    else:
        raise ValueError(f'Unknown "{metric}" metric, you should choose among: {list(SCORERS.keys())} or a scorer '
                         f'callable object/function')


def format_splitting_strategy(splitting_strategy, splitting_strategy_kwargs, array):
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
