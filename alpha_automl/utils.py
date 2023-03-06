import logging
import json
import inspect
import importlib
import numpy as np
from os.path import join, dirname
from sklearn.metrics import SCORERS, get_scorer, make_scorer as make_scorer_sk
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, train_test_split, cross_val_score
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits

logger = logging.getLogger(__name__)

RANDOM_SEED = 0
PRIMITIVE_TYPES = {}

with open(join(dirname(__file__), 'resource', 'primitives_hierarchy.json')) as fin:
    primitives = json.load(fin)
    for primitive_type, primitive_names in primitives.items():
        for primitive_name in primitive_names:
            PRIMITIVE_TYPES[primitive_name] = primitive_type


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


def make_pipelineprofiler_inputs(pipelines, new_primitives, metric, source_name='Pipeline'):
    profiler_inputs = []
    primitive_types = {}

    for primitive_name, primitive_type in PRIMITIVE_TYPES.items():
        primitive_path = '.'.join(primitive_name.split('.')[-2:])
        primitive_name = f'alpha_automl.primitives.{primitive_path}'
        primitive_types[primitive_name] = primitive_type.replace('_', ' ').title()

    for new_primitive in new_primitives:
        primitive_path = '.'.join(new_primitive.split('.')[-2:])
        primitive_name = f'alpha_automl.primitives.{primitive_path}'
        primitive_types[primitive_name] = new_primitives[new_primitive]['primitive_type'].replace('_', ' ').title()

    # TODO: Read these primitive types from grammar
    ordered_types = ['IMPUTATION', 'TEXT_FEATURIZER', 'DATETIME_ENCODER', 'CATEGORICAL_ENCODER',
                     'FEATURE_SCALING', 'FEATURE_SELECTION', 'REGRESSION', 'CLASSIFICATION']
    ordered_types = [i.replace('_', ' ').title() for i in ordered_types]

    for pipeline_id, pipeline_data in pipelines.items():
        profiler_data = {
            'pipeline_id': pipeline_id,
            'inputs': [{'name': 'input dataset'}],
            'steps': [],
            'outputs': [],
            'pipeline_digest': pipeline_id,
            'start': '2023-03-05T03:05:50.788926Z',  # TODO: Calculate these values on scoring function
            'end': '2023-03-05T03:05:51.788926Z',
            'scores': [{'metric': {'metric': metric}, 'value': pipeline_data['pipeline_score'],
                        'normalized': pipeline_data['pipeline_score']}],
            'pipeline_source': {'name': source_name},
        }
        #  The code below is based on the function "import_autosklearn" of PipelineProfiler
        prev_list = ['inputs.0']
        cur_step_idx = 0
        for primitive_type in ordered_types:
            steps_in_type = []
            for step_id, step_object in pipeline_data['pipeline_object'].steps:
                primitive_path = '.'.join(step_id.split('.')[-2:])
                primitive_id = f'alpha_automl.primitives.{primitive_path}'
                if primitive_types[primitive_id] == primitive_type:
                    steps_in_type.append((primitive_id, step_object))

            if len(steps_in_type) == 0:
                continue

            new_prev_list = []
            for step_id, step_object in steps_in_type:
                step_ref = f'steps.{cur_step_idx}.produce'
                cur_step_idx += 1
                new_prev_list.append(step_ref)

                step = {
                    'primitive': {'python_path': step_id, 'name': ''},
                    'arguments': {},
                    'outputs': [{'id': 'produce'}],
                    'reference': {'type': 'CONTAINER', 'data': step_ref},
                    'hyperparams': {}
                }

                for param_name, param_value in get_primitive_params(step_object).items():
                    step['hyperparams'][param_name] = {
                        'type': 'VALUE',
                        'data': param_value
                    }
                for idx, prev in enumerate(prev_list):
                    cur_argument_idx = f'input{idx}'
                    step['arguments'][cur_argument_idx] = {
                        'data': prev
                    }
                profiler_data['steps'].append(step)

            prev_list = new_prev_list

        profiler_data['outputs'] = []
        for prev in prev_list:
            profiler_data['outputs'].append({'data': prev})

        profiler_inputs.append(profiler_data)

    return profiler_inputs, primitive_types


def get_primitive_params(primitive_object):
    constructor_params = set(inspect.signature(primitive_object.__init__).parameters)
    params = {}

    for param_name, param_value in primitive_object.__dict__.items():
        if param_name in constructor_params:
            if callable(param_value):
                param_value = param_value.__name__
            params[param_name] = param_value

    return params
