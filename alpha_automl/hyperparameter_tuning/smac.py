import json
import logging
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
from sklearn.compose import ColumnTransformer
from smac import HyperparameterOptimizationFacade, Scenario

from alpha_automl.primitive_loader import PRIMITIVE_TYPES
from alpha_automl.scorer import make_scorer, make_splitter
from alpha_automl.utils import create_object

logger = logging.getLogger(__name__)
SMAC_PARAMETERS_PATH = join(dirname(__file__), "smac_parameters.json")


def load_smac_parameters():
    with open(SMAC_PARAMETERS_PATH) as fin:
        primitives = json.load(fin)
    logger.info("[SMAC] smac_parameters loaded")

    return primitives


SMAC_DICT = load_smac_parameters()


def gen_pipeline(config, pipeline):
    new_pipeline = make_pipeline()
    for step_name, step_obj in pipeline.steps:
        step_type = PRIMITIVE_TYPES[step_name]

        if step_type == "COLUMN_TRANSFORMER":
            transformers = []
            for trans_name, _, trans_index in step_obj.__dict__["transformers"]:
                trans_prim_name = trans_name.split("-")[0]
                trans_obj = create_object(
                    trans_prim_name, get_primitive_params(config, trans_prim_name)
                )
                transformers.append((trans_name, trans_obj, trans_index))
            transformer_obj = ColumnTransformer(transformers, remainder='passthrough')
            new_pipeline.steps.append(
                [step_name, transformer_obj]
            )
        elif step_type == "SEMISUPERVISED_CLASSIFIER":
            classifier_name = find_classifier_prim_name(
                step_obj.__dict__["base_estimator"]
            )
            classifier_obj = create_object(
                classifier_name, get_primitive_params(config, classifier_name)
            )
            step_obj.base_estimator = classifier_obj
            new_pipeline.steps.append(
                [step_name, step_obj]
            )
        else:
            new_pipeline.steps.append(
                [
                    step_name,
                    create_object(step_name, get_primitive_params(config, step_name)),
                ]
            )

    return new_pipeline


def get_primitive_params(config, step_name):
    params = list(SMAC_DICT[step_name].keys())
    class_params = {}
    for param in params:
        class_params[param] = config[param]
    logger.debug(f"[SMAC] {step_name}: {class_params}")
    return class_params


def gen_configspace(pipeline):
    # (from build_configspace) Build Configuration Space which defines all parameters and their ranges
    configspace = ConfigurationSpace(seed=0)
    for primitive, prim_obj in pipeline.steps:
        step_type = PRIMITIVE_TYPES[primitive]
        try:
            params = SMAC_DICT[primitive]
            configspace.add_hyperparameters(cast_primitive(params))
            if step_type == "COLUMN_TRANSFORMER":
                for trans_name, _, _ in prim_obj.__dict__["transformers"]:
                    trans_prim_name = trans_name.split("-")[0]
                    params = SMAC_DICT[trans_prim_name]
                    configspace.add_hyperparameters(cast_primitive(params))
            elif step_type == "SEMISUPERVISED_CLASSIFIER":
                logger.critical(prim_obj.__dict__)
                classifier_name = find_classifier_prim_name(
                    prim_obj.__dict__["base_estimator"]
                )
                params = SMAC_DICT[classifier_name]
                configspace.add_hyperparameters(cast_primitive(params))

        except Exception as e:
            logger.critical(f"[SMAC] {str(e)}")
    return configspace


def find_classifier_prim_name(classifier_obj):
    classifier_name = classifier_obj.__class__.__name__
    for prim_name in SMAC_DICT.keys():
        if classifier_name in prim_name:
            classifier_name = prim_name
    return classifier_name


def cast_primitive(params):
    new_hyperparameters = []
    for name, conf in params.items():
        config_space = cast_hyperparameter(name, conf)
        if config_space is not None:
            new_hyperparameters.append(config_space)

    return new_hyperparameters


def cast_hyperparameter(param_name, param_conf):
    param_type, param_value, param_default = "", "", ""
    config_space = None
    try:
        param_type = param_conf["type"]
        param_value = param_conf["value"]
        param_default = param_conf["default"]
    except Exception as e:
        logger.critical(f"[SMAC] {str(e)}")
        return
    if param_type == "Categorical":
        config_space = Categorical(param_name, param_value, default=param_default)
    elif param_type == "Integer":
        min_value = int(param_value[0])
        max_value = int(param_value[1])
        config_space = Integer(
            param_name, (min_value, max_value), default=param_default
        )
    elif param_type == "Float":
        min_value = float(param_value[0])
        max_value = float(param_value[1])
        config_space = Float(param_name, (min_value, max_value), default=param_default)
    elif param_type == "Constant":
        config_space = Constant(param_name, param_value)
    else:
        logger.error(f"Unknown param_type {param_type}")

    return config_space


class SmacOptimizer:
    def __init__(
        self,
        X=None,
        y=None,
        n_trials=50,
        splitter=make_splitter("holdout"),
        scorer=make_scorer("accuracy_score"),
        time_limit=None,
    ):
        self.pipeline = None
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.splitter = splitter
        self.scorer = scorer
        self.time_limit = time_limit
        return

    def train(self, config: Configuration, seed: int = 0) -> float:
        pipeline = gen_pipeline(config, self.pipeline)
        scores = cross_val_score(
            pipeline,
            self.X,
            self.y,
            cv=self.splitter,
            scoring=self.scorer,
            error_score="raise",
        )
        return 1 - np.mean(scores)

    def optimize_pipeline(self, pipeline):
        self.pipeline = pipeline
        if self.pipeline is None:
            logger.critical("[SMAC] get_pipeline return None value!")
            return
        optimized_conf = self._optimize_pipeline(self.pipeline)
        optimized_pipeline = gen_pipeline(optimized_conf, self.pipeline)
        logger.critical(f"[SMAC][!!!!!!!!!!!!!!!!!!!!!!]\n {pipeline.get_params()} successfully optimized!\n\n")
        return optimized_pipeline

    def _optimize_pipeline(self, pipeline):
        scenario = Scenario(
            gen_configspace(pipeline),
            deterministic=True,
            n_trials=self.n_trials,
            walltime_limit=self.time_limit,
        )

        smac = HyperparameterOptimizationFacade(scenario, self.train)
        return smac.optimize()
