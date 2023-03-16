import logging
import inspect
import importlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, train_test_split, cross_val_score
from alpha_automl.primitive_loader import PRIMITIVE_TYPES

logger = logging.getLogger(__name__)

COLUMN_TRANSFORMER_ID = 'sklearn.compose.ColumnTransformer'
COLUMN_SELECTOR_ID = 'ColumnSelector'
RANDOM_SEED = 0


def create_object(import_path, class_params=None):
    if class_params is None:
        class_params = {}

    modules = import_path.split('.')
    class_name = modules[-1]
    import_module = '.'.join(modules[:-1])
    class_ = getattr(importlib.import_module(import_module), class_name)
    object_ = class_(**class_params)  # Instantiates the object

    return object_


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
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.reset_index(drop=True)

        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.reset_index(drop=True)

        return X_test, y_test, True

    else:
        logger.info('Not doing sampling for small dataset (size = %d)', original_size)
        return X, y, False


def is_equal_splitting(strategy1, strategy2):
    if strategy1.__class__ == strategy2.__class__ and strategy1.__dict__ == strategy2.__dict__:
        return True
    else:
        return False


def make_d3m_pipelines(pipelines, new_primitives, metric, source_name='Pipeline'):
    d3m_pipelines = []
    primitive_types = {}

    for primitive_name, primitive_type in PRIMITIVE_TYPES.items():
        primitive_path = '.'.join(primitive_name.split('.')[-2:])
        primitive_name = f'alpha_automl.primitives.{primitive_path}'
        primitive_types[primitive_name] = primitive_type.replace('_', ' ').title()

    for new_primitive in new_primitives:
        primitive_path = '.'.join(new_primitive.split('.')[-2:])
        primitive_name = f'alpha_automl.primitives.{primitive_path}'
        primitive_types[primitive_name] = new_primitives[new_primitive]['primitive_type'].replace('_', ' ').title()

    for pipeline_id, pipeline in pipelines.items():
        new_pipeline = {
            'pipeline_id': pipeline_id,
            'inputs': [{'name': 'input dataset'}],
            'steps': [],
            'outputs': [],
            'pipeline_digest': pipeline_id,
            'start': pipeline.get_start_time(),
            'end': pipeline.get_end_time(),
            'scores': [{'metric': {'metric': metric}, 'value': pipeline.get_score(), 'normalized': pipeline.get_score()}],
            'pipeline_source': {'name': source_name},
        }

        #  The code below is an adaptation of the function "import_autosklearn" of PipelineProfiler
        prev_list = ['inputs.0']
        cur_step_idx = 0

        for step_id, step_object in pipeline.get_pipeline().steps:
            steps_in_type = []
            primitive_path = '.'.join(step_id.split('.')[-2:])
            primitive_id = f'alpha_automl.primitives.{primitive_path}'
            steps_in_type.append((primitive_id, step_object))

            new_prev_list = []
            cur_step_idx = add_d3m_step(steps_in_type, cur_step_idx, prev_list, new_prev_list, new_pipeline)
            prev_list = new_prev_list

            if isinstance(step_object, ColumnTransformer):
                steps_in_type = []
                for transformer_name, transformer_object, _ in step_object.transformers:
                    if transformer_name == COLUMN_SELECTOR_ID: continue
                    primitive_path = '.'.join(transformer_name.split('-')[0].split('.')[-2:])
                    primitive_id = f'alpha_automl.primitives.{primitive_path}'
                    steps_in_type.append((primitive_id, transformer_object))

                new_prev_list = []
                cur_step_idx = add_d3m_step(steps_in_type, cur_step_idx, prev_list, new_prev_list, new_pipeline)
                prev_list = new_prev_list

        new_pipeline['outputs'] = []
        for prev in prev_list:
            new_pipeline['outputs'].append({'data': prev})

        d3m_pipelines.append(new_pipeline)

    return d3m_pipelines, primitive_types


def add_d3m_step(steps_in_group, cur_step_idx, prev_list, new_prev_list, new_pipeline):
    for step_id, step_object in steps_in_group:
        step_ref = f'steps.{cur_step_idx}.produce'
        cur_step_idx += 1
        new_prev_list.append(step_ref)

        step = {
            'primitive': {'python_path': step_id, 'name': 'primitive'},
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
        new_pipeline['steps'].append(step)

    return cur_step_idx


def get_primitive_params(primitive_object):
    constructor_params = set(inspect.signature(primitive_object.__init__).parameters)
    params = {}

    for param_name, param_value in primitive_object.__dict__.items():
        if param_name in constructor_params:
            if callable(param_value):
                param_value = param_value.__name__
            params[param_name] = param_value

    return params
