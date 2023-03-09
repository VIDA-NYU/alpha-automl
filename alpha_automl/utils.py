import logging
import inspect
import importlib
import datamart_profiler
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import SCORERS, get_scorer, make_scorer as make_scorer_sk
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, train_test_split, cross_val_score
from sklearn.model_selection._split import BaseShuffleSplit, _RepeatedSplits
from alpha_automl.primitive_loader import PRIMITIVE_TYPES

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
        return X_test.reset_index(drop=True), y_test.reset_index(drop=True), True

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
    ordered_types = ['IMPUTATION', 'COLUMN_TRANSFORMER', 'TEXT_ENCODER', 'DATETIME_ENCODER', 'CATEGORICAL_ENCODER',
                     'FEATURE_SCALING', 'FEATURE_SELECTION', 'REGRESSION', 'CLUSTERING', 'CLASSIFICATION']
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

        all_steps = []
        for step_id, step_object in pipeline_data['pipeline_object'].steps:
            all_steps.append((step_id, step_object))
            if isinstance(step_object, ColumnTransformer):
                for transformer_name, transformer_object, _ in step_object.transformers:
                    step_id = transformer_name.split('-')[0]
                    all_steps.append((step_id, transformer_object))

        #  The code below is based on the function "import_autosklearn" of PipelineProfiler
        prev_list = ['inputs.0']
        cur_step_idx = 0
        for primitive_type in ordered_types:
            steps_in_type = []
            for step_id, step_object in all_steps:
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

                if isinstance(step_object, ColumnTransformer):
                    step['hyperparams'] = {}  # Ignore hyperparameters of ColumnTransformer

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


def has_missing_values(X):
    return X.isnull().values.any()


def select_encoders(X):
    selected_encoders = {}
    mapping_encoders = {'http://schema.org/Enumeration': 'CATEGORICAL_ENCODER',
                        'http://schema.org/Text': 'TEXT_ENCODER',
                        'http://schema.org/DateTime': 'DATETIME_ENCODER'}

    data_profile = datamart_profiler.process_dataset(X, coverage=False)

    for index, column_profile in enumerate(data_profile['columns']):
        column_name = column_profile['name']
        semantic_types = column_profile['semantic_types'] if len(column_profile['semantic_types']) > 0 else [column_profile['structural_type']]

        for semantic_type in semantic_types:
            if semantic_type == 'http://schema.org/identifier':
                semantic_type = 'http://schema.org/Integer'
            elif semantic_type in {'http://schema.org/longitude', 'http://schema.org/latitude'}:
                semantic_type = 'http://schema.org/Float'

            if semantic_type in mapping_encoders:
                encoder = mapping_encoders[semantic_type]
                if encoder not in selected_encoders:
                    selected_encoders[encoder] = []
                selected_encoders[encoder].append((index, column_name))

    return selected_encoders
