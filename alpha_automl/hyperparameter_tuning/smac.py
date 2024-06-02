import json
import logging
import copy
from os.path import dirname, join

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
    Float,
    Integer,
    
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from smac import HyperparameterOptimizationFacade, Scenario

from alpha_automl.scorer import make_scorer, make_splitter
from alpha_automl.utils import create_object
from alpha_automl.primitive_loader import PRIMITIVE_TYPES
from alpha_automl.pipeline_synthesis.pipeline_builder import extract_estimators

logger = logging.getLogger(__name__)
SMAC_PARAMETERS_PATH = join(dirname(__file__), 'smac_parameters.json')


def load_smac_parameters():
    with open(SMAC_PARAMETERS_PATH) as fin:
        primitives = json.load(fin)
    logger.info('[SMAC] smac_parameters loaded')

    return primitives


SMAC_DICT = load_smac_parameters()


def gen_pipeline(config, pipeline):
    new_pipeline = make_pipeline()
    for step_name, step_obj in pipeline.steps:
        if "feature_engine.creation" in step_name:
            step_type = "FEATURE_GENERATOR"
        elif "feature_engine.selection" in step_name:
            step_type = "FEATURE_SELECTOR"
        else:
            step_type = PRIMITIVE_TYPES[step_name]

        if step_type == 'COLUMN_TRANSFORMER':
            transformers = []
            for trans_name, _, trans_index in step_obj.__dict__['transformers']:
                trans_prim_name = trans_name.split('-')[0]
                trans_obj = create_object(trans_prim_name, get_primitive_params(config, trans_prim_name))
                transformers.append((trans_name, trans_obj, trans_index))
                step_obj.__dict__['transformers'] = transformers
            new_pipeline.steps.append([step_name, create_object(step_name, step_obj.__dict__)])
        elif step_type == 'CLASSIFICATION_SINGLE_ENSEMBLER' or step_type == 'REGRESSION_SINGLE_ENSEMBLER':
            estimator = step_obj.estimator
            estimator_name = estimator.__class__.__name__
            for smac_name in SMAC_DICT.keys():
                if estimator_name == smac_name.split(".")[-1]:
                    estimator = create_object(smac_name, get_primitive_params(config, smac_name))
            primitive_object = create_object(step_name, {'estimator': estimator})
            new_pipeline.steps.append([step_name, primitive_object])
        elif step_type == 'CLASSIFICATION_MULTI_ENSEMBLER' or step_type == 'REGRESSION_MULTI_ENSEMBLER':
            estimators = extract_estimators_smac(step_obj, config)
            logger.critical(f"[YFW] =========== {config} --- {estimators} ==========")
            primitive_object = create_object(step_name, {'estimators': estimators})
            new_pipeline.steps.append([step_name, primitive_object])
        else:
            new_pipeline.steps.append([step_name, create_object(step_name, get_primitive_params(config, step_name))])

    return new_pipeline


def extract_estimators_smac(step_obj, config):
    new_estimators = []
    estimators = copy.deepcopy(step_obj.estimators)
    while estimators:
        estimator_name, estimator_obj = estimators.pop()
        estimator_name_lookup, estimator_name_counter = estimator_name.split('-')
        new_estimators.append((estimator_name, create_object(estimator_name_lookup, get_primitive_params(config, estimator_name_lookup))))

    return new_estimators


def get_primitive_params(config, step_name):
    params = list(SMAC_DICT[step_name].keys())
    class_params = {}
    for param in params:
        class_params[param] = config[param]
    logger.critical(f'[SMAC] {step_name}: {class_params}')
    return class_params


def gen_configspace(pipeline):
    # (from build_configspace) Build Configuration Space which defines all parameters and their ranges
    configspace = ConfigurationSpace(seed=0)
    all_params = {}
    for primitive, prim_obj in pipeline.steps:
        step_type = PRIMITIVE_TYPES[primitive]
        try:
            params = SMAC_DICT[primitive]
            add_params(params, all_params)
            if step_type == 'COLUMN_TRANSFORMER':
                for trans_name, _, _ in prim_obj.__dict__['transformers']:
                    trans_prim_name = trans_name.split('-')[0]
                    params = SMAC_DICT[trans_prim_name]
                    add_params(params, all_params)
            elif step_type == 'CLASSIFICATION_SINGLE_ENSEMBLER' or step_type == 'REGRESSION_SINGLE_ENSEMBLER':
                estimator_obj = prim_obj.estimator
                for smac_name, params in SMAC_DICT.items():
                    if estimator_obj.__class__.__name__ == smac_name.split(".")[-1]:
                        add_params(params, all_params)
            elif step_type == 'CLASSIFICATION_MULTI_ENSEMBLER' or step_type == 'REGRESSION_MULTI_ENSEMBLER':
                for estimator_name, _ in prim_obj.estimators:
                    estimator_name_lookup, _ = estimator_name.split('-')
                    params = SMAC_DICT[estimator_name_lookup]
                    add_params(params, all_params)
        except Exception as e:
            logger.critical(f'[SMAC] {str(e)}')
    configspace.add_hyperparameters(cast_primitive(all_params))
    return configspace


def add_params(params, all_params):
    for param_name, param_conf in params.items():
        if param_name in all_params:
            pass
        else:
            all_params[param_name] = param_conf


def cast_primitive(params):
    new_hyperparameters = []
    for name, conf in params.items():
        config_space = cast_hyperparameter(name, conf)
        if config_space is not None:
            new_hyperparameters.append(config_space)

    return new_hyperparameters


def cast_hyperparameter(param_name, param_conf):
    param_type, param_value, param_default = '', '', ''
    config_space = None
    try:
        param_type = param_conf['type']
        param_value = param_conf['value']
        param_default = param_conf['default']
    except Exception as e:
        logger.critical(f'[SMAC] {str(e)}')
        return
    if param_type == 'Categorical':
        config_space = Categorical(param_name, param_value, default=param_default)
    elif param_type == 'Integer':
        min_value = int(param_value[0])
        max_value = int(param_value[1])
        config_space = Integer(
            param_name, (min_value, max_value), default=param_default
        )
    elif param_type == 'Float':
        min_value = float(param_value[0])
        max_value = float(param_value[1])
        config_space = Float(param_name, (min_value, max_value), default=param_default)
    elif param_type == 'Constant':
        config_space = Constant(param_name, param_value)
    elif param_type == 'Boolean':
        config_space = Categorical(param_name, param_value, default=param_default)
    else:
        logger.error(f'Unknown param_type {param_type}')

    return config_space


class SmacOptimizer:
    def __init__(
        self,
        X=None,
        y=None,
        n_trials=50,
        splitter=make_splitter('holdout'),
        scorer=make_scorer('accuracy_score'),
    ):
        self.pipeline = None
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.splitter = splitter
        self.scorer = scorer
        return

    def train(self, config: Configuration, seed: int = 0) -> float:
        self.pipeline = gen_pipeline(config, self.pipeline)
        scores = cross_val_score(
            self.pipeline,
            self.X,
            self.y,
            cv=self.splitter,
            scoring=self.scorer,
            error_score='raise',
        )
        logger.critical(f"[WWWWWWWWWWWWWWWW] {self.pipeline} ~~~~~ {scores}")
        
        return 1 - np.mean(scores)

    def optimize_pipeline(self, pipeline):
        self.pipeline = pipeline
        logger.critical(f"????????????????????????????{pipeline}????????????????????????????")
        if self.pipeline is None:
            logger.critical('[SMAC] get_pipeline return None value!')
            return
        optimized_conf = self._optimize_pipeline(self.pipeline)
        logger.critical(f"[YFW] ----------------- {optimized_conf} --- {pipeline}")
        if optimized_conf:
            optimized_pipeline = gen_pipeline(optimized_conf, self.pipeline)
            logger.debug(f'[SMAC] {pipeline} successfully optimized!')
            return optimized_pipeline
        else:
            return self.pipeline
        

    def _optimize_pipeline(self, pipeline):
        scenario = Scenario(
            gen_configspace(pipeline), deterministic=True, n_trials=self.n_trials
        )

        smac = HyperparameterOptimizationFacade(scenario, self.train, overwrite=True)
        return smac.optimize()
