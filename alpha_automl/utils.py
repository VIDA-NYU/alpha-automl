import sys
import importlib
import inspect
import logging
import platform
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit, train_test_split

from alpha_automl.primitive_loader import PRIMITIVE_TYPES as INSTALLED_PRIMITIVES

logger = logging.getLogger(__name__)

COLUMN_TRANSFORMER_ID = 'sklearn.compose.ColumnTransformer'
COLUMN_SELECTOR_ID = 'ColumnSelector'
NATIVE_PRIMITIVE = 'native'
ADDED_PRIMITIVE = 'added'
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


def sample_dataset(X, y, sample_size, task):
    original_size = len(X)
    shuffle = True
    if task == 'TIME_SERIES_FORECAST':
        shuffle = False

    if original_size > sample_size:
        ratio = sample_size / original_size
        try:
            _, X_test, _, y_test = train_test_split(X, y, random_state=RANDOM_SEED, test_size=ratio, stratify=y, shuffle=shuffle)
        except Exception:
            # Not using stratified sampling when the minority class has few instances, not enough for all the folds
            _, X_test, _, y_test = train_test_split(X, y, random_state=RANDOM_SEED, test_size=ratio, shuffle=shuffle)
        logger.debug(f'Sampling down data from {original_size} to {len(X_test)}')
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.reset_index(drop=True)

        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.reset_index(drop=True)

        return X_test, y_test, True

    else:
        logger.debug('Not doing sampling for small dataset (size = %d)', original_size)
        return X, y, False


def is_equal_splitting(strategy1, strategy2):
    if strategy1.__class__ == strategy2.__class__ and strategy1.__dict__ == strategy2.__dict__:
        return True
    else:
        return False


def make_d3m_pipelines(pipelines, new_primitives, metric, ordering_sign, source_name='Pipeline'):
    d3m_pipelines = []
    d3m_primitive_types = {}
    all_primitive_types = {}
    
    for primitive_name, primitive_type in INSTALLED_PRIMITIVES.items():
        all_primitive_types[primitive_name] = primitive_type
        primitive_path = '.'.join(primitive_name.split('.')[-2:])
        d3m_primitive_name = f'alpha_automl.primitives.{primitive_path}'
        d3m_primitive_types[d3m_primitive_name] = primitive_type.replace('_', ' ').title()

    for new_primitive in new_primitives:
        primitive_type = new_primitives[new_primitive]['primitive_type']
        all_primitive_types[new_primitive] = primitive_type
        primitive_path = '.'.join(new_primitive.split('.')[-2:])
        d3m_primitive_name = f'alpha_automl.primitives.{primitive_path}'
        d3m_primitive_types[d3m_primitive_name] = primitive_type.replace('_', ' ').title()

    for pipeline_id, pipeline in pipelines.items():
        new_pipeline = {
            'pipeline_id': pipeline_id,
            'inputs': [{'name': 'input dataset'}],
            'steps': [],
            'outputs': [],
            'pipeline_digest': pipeline_id,
            'start': pipeline.get_start_time(),
            'end': pipeline.get_end_time(),
            'scores': [{'metric': {'metric': metric}, 'value': pipeline.get_score(),
                        'normalized': pipeline.get_score() * ordering_sign}],
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
                    if transformer_name == COLUMN_SELECTOR_ID:
                        continue
                    primitive_path = '.'.join(transformer_name.split('-')[0].split('.')[-2:])
                    primitive_id = f'alpha_automl.primitives.{primitive_path}'
                    steps_in_type.append((primitive_id, transformer_object))

                new_prev_list = []
                cur_step_idx = add_d3m_step(steps_in_type, cur_step_idx, prev_list, new_prev_list, new_pipeline)
                prev_list = new_prev_list
            
            if all_primitive_types[step_id] == 'SEMISUPERVISED_CLASSIFIER':
                classifier_object = step_object.base_estimator
                classifier_path = f'classifier.{classifier_object.__class__.__name__}'
                for primitive_name, primitive_type in all_primitive_types.items():
                    if primitive_type != 'CLASSIFIER':
                        continue
                    if classifier_object.__class__.__name__ in primitive_name:
                        classifier_path = '.'.join(primitive_name.split('.')[-2:])
                        break
                
                classifier_step = [(f'alpha_automl.primitives.{classifier_path}', classifier_object)]
                new_prev_list = []
                cur_step_idx = add_d3m_step(classifier_step, cur_step_idx, prev_list, new_prev_list, new_pipeline)
                prev_list = new_prev_list
                
        new_pipeline['outputs'] = []
        for prev in prev_list:
            new_pipeline['outputs'].append({'data': prev})

        d3m_pipelines.append(new_pipeline)
    
    return d3m_pipelines, d3m_primitive_types


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
                'data': str(param_value)
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


def hide_logs(level):
    """
    Three levels of logs:
    - verbose == logging.DEBUG: show all logs
    - verbose == logging.INFO: show find and scored pipelines
    - verbose <= logging.WARNING: show no logs
    """
    warnings.filterwarnings('ignore')
    logging.root.setLevel(logging.CRITICAL)
    logging.getLogger('alpha_automl').setLevel(level)


def get_start_method(suggested_method):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    operating_system = platform.system()

    if suggested_method == 'auto':
        if device != 'cuda' and operating_system != 'Windows':
            return 'fork'

        elif device == 'cuda':
            return 'spawn'

        elif operating_system == 'Windows':
            return 'spawn'

    elif suggested_method == 'fork':
        if device == 'cuda':
            raise ValueError('Cuda does not support "fork" method. Use "spawn".')

        elif operating_system == 'Windows':
            raise ValueError('Windows does not support "fork" method. Use "spawn".')

        else:
            return suggested_method

    elif suggested_method == 'spawn':
        if device != 'cuda' and operating_system != 'Windows':
            logger.debug('We recommend to use "fork" in non-Windows platforms.')

        return suggested_method


def check_input_for_multiprocessing(start_method, callable_input, input_type):
    if start_method == 'spawn':
        module_name = getattr(callable_input, '__module__', '')
        object_name = getattr(callable_input, '__name__', callable_input.__class__.__name__)
        if module_name == '__main__':
            raise ImportError(f'The input {input_type} must be implemented in an external module and be called like '
                              f'from my_external_module import {object_name}"')


class SemiSupervisedSplitter:
    """
    SemiSupervisedSplitter makes sure that unlabeled rows not being
    selected as test/validation data for semi-supervised classification tasks.
    """
    def __init__(self, n_splits=1, test_size=.2, random_state=0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y, groups=None):
        if isinstance(y, pd.DataFrame):
            y_array = y.to_numpy()
        else:
            y_array = y
        for rx, tx in ShuffleSplit(n_splits=self.n_splits,
                                   test_size=self.test_size,
                                   random_state=self.random_state).split(X,y):
            unlabeled_t = np.where(y_array[tx] == -1)[0]
            labeled_r = np.where(y_array[rx] != -1)[0]           
            tbr_r = np.random.choice(labeled_r, size=unlabeled_t.shape[0], replace=False)
            tx[unlabeled_t], rx[tbr_r] = rx[tbr_r], tx[unlabeled_t]

            yield rx, tx

    def get_n_splits(self, groups=None):
        return self.n_splits


class SemiSupervisedLabelEncoder:
    """
    SemiSupervisedLabelEncoder ignores the unlabeled values (-1 or nan) and only apply
    LabelEncoder transformation to columns with clear label.
    """
    label_encoder = LabelEncoder()

    def fit_transform(self, df):
        if isinstance(df, pd.DataFrame):
            df = df.replace(-1, np.NaN)
            df[df.columns[0]] = pd.Series(
                self.label_encoder.fit_transform(df[df.columns[0]][df[df.columns[0]].notnull()]),
                index=df[df.columns[0]][df[df.columns[0]].notnull()].index
            )
            df = df.fillna(-1).astype(int)
            return df.to_numpy()
        else:
            raise TypeError("Only pd.DataFrame are allowed")

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            df = df.replace(-1, np.NaN)
            df[df.columns[0]] = pd.Series(
                self.label_encoder.transform(df[df.columns[0]][df[df.columns[0]].notnull()]),
                index=df[df.columns[0]][df[df.columns[0]].notnull()].index
            )
            df = df.fillna(-1).astype(int)
            return df.to_numpy()
        else:
            raise TypeError("Only pd.DataFrame are allowed")

    def inverse_transform(self, df):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        df = df.replace(-1, np.NaN)

        df[df.columns[0]] = pd.Series(
            self.label_encoder.inverse_transform(df[df.columns[0]][df[df.columns[0]].notnull()].astype(int)),
            index=df[df.columns[0]][df[df.columns[0]].notnull()].index
        )
        return df.to_numpy()
