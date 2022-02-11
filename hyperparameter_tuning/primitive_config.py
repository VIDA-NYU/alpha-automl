import typing
import logging
import json
import os
from d3m import index
from alphad3m.primitive_loader import load_primitives_hierarchy
from d3m.metadata.hyperparams import Bounded, Enumeration, UniformInt, UniformBool, Uniform, Normal, Union, \
    Constant as ConstantD3M
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
     UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, NormalFloatHyperparameter

logger = logging.getLogger(__name__)

HYPERPARAMETERS_FROM_METALEARNING_PATH = os.path.join(os.path.dirname(__file__), '../resource/hyperparams.json')
PRIMITIVES = {}
primitives_info = load_primitives_hierarchy()
for primitive_type in primitives_info:
    for primitive_name in primitives_info[primitive_type]:
        PRIMITIVES[primitive_name] = primitive_type


def is_tunable(primitive_name):
    primitive_type = PRIMITIVES.get(primitive_name, None)

    if primitive_type in {'CLASSIFICATION', 'REGRESSION', 'TIME_SERIES_CLASSIFICATION', 'TIME_SERIES_FORECASTING',
                          'SEMISUPERVISED_CLASSIFICATION', 'COMMUNITY_DETECTION', 'GRAPH_MATCHING', 'LINK_PREDICTION',
                          'VERTEX_CLASSIFICATION', 'OBJECT_DETECTION', 'FEATURE_SELECTION'}:
        return True

    return False


def load_hyperparameters(primitive_name):
    primitive = index.get_primitive(primitive_name)
    hyperparameters_metadata = primitive.metadata.query()['primitive_code']['hyperparams']
    hyperparameter_class = typing.get_type_hints(primitive.__init__)['hyperparams']
    hyperparameters = {}

    if hyperparameter_class:
        for hp_name, hp_value in hyperparameter_class.configuration.items():
            if 'https://metadata.datadrivendiscovery.org/types/TuningParameter' in hyperparameters_metadata[hp_name]['semantic_types']:
                    hyperparameters[hp_name] = hp_value

    return hyperparameters


def load_primitive_configspace(configspace, primitive_name):
    default_configspace = load_default_configspace(primitive_name)
    all_hyperparameters = load_hyperparameters(primitive_name)
    default_hyperparameters = set(default_configspace.get_hyperparameter_names())
    casted_hyperparameters = []
    union_conditions = []
    for hp_name in all_hyperparameters:
        new_hp_name = primitive_name + '|' + hp_name
        if not isinstance(all_hyperparameters[hp_name], Union):
            new_hp = default_configspace.get_hyperparameter(new_hp_name) if new_hp_name in default_hyperparameters \
                else None
            casted_hp = cast_hyperparameter(all_hyperparameters[hp_name], new_hp_name)
            if casted_hp is not None:
                if new_hp is not None and casted_hp.is_legal(new_hp.default_value):
                    casted_hyperparameters.append(new_hp)
                else:
                    casted_hyperparameters.append(casted_hp)
        elif isinstance(all_hyperparameters[hp_name], Union):
            hyperparameter_class = all_hyperparameters[hp_name]
            add_case = False
            case_choices = []
            cases_param_space = []
            for case, case_class in hyperparameter_class.configuration.items():
                new_hp = None
                if new_hp_name + '|' + case in default_hyperparameters:
                    new_hp = default_configspace.get_hyperparameter(new_hp_name + '|' + case)
                casted_hp = cast_hyperparameter(case_class, new_hp_name + '|' + case)
                if casted_hp is not None:
                    add_case = True
                    case_choices.append(case)
                    if new_hp is not None and casted_hp.is_legal(new_hp.default_value):
                        casted_hyperparameters.append(new_hp)
                        cases_param_space.append(new_hp)
                    else:
                        casted_hyperparameters.append(casted_hp)
                        cases_param_space.append(casted_hp)
            if add_case:
                case_hyperparameter = CategoricalHyperparameter(
                    name=new_hp_name + '|case',
                    choices=case_choices,
                    default_value=case_choices[0])
                for i in range(len(case_choices)):
                    union_conditions.append(EqualsCondition(cases_param_space[i], case_hyperparameter, case_choices[i]))
                casted_hyperparameters.append(case_hyperparameter)

    configspace.add_hyperparameters(casted_hyperparameters)

    for condition in default_configspace.get_conditions():
        try:
            configspace.add_condition(condition)
        except Exception as e:
            logger.warning('Not possible to add condition', e)

    for condition in union_conditions:
        try:
            configspace.add_condition(condition)
        except Exception as e:
            logger.warning('Not possible to add condition', e)

    for forbidden in default_configspace.get_forbiddens():
        try:
            configspace.add_forbidden_clause(forbidden)
        except Exception as e:
            logger.warning('Not possible to add forbidden clause', e)


def load_default_configspace(primitive):
    default_config = ConfigurationSpace()

    if primitive in get_hyperparameters_from_metalearnig():
        default_config.add_configuration_space(
            primitive,
            get_configspace_from_metalearning(get_hyperparameters_from_metalearnig()[primitive]),
            '|'
        )

    return default_config


def cast_hyperparameter(hyperparameter, name):
    # From D3M hyperparameters to ConfigSpace hyperparameters
    # TODO: 'Choice' and  'Set' (D3M hyperparameters)
    new_hyperparameter = None

    try:
        if isinstance(hyperparameter, Bounded):
            lower = hyperparameter.lower 
            upper = hyperparameter.upper
            default = hyperparameter.get_default()
            if lower is None:
                lower = default
            if upper is None:
                upper = default * 2 if default > 0 else 10
            if hyperparameter.structural_type == int:
                if not hyperparameter.lower_inclusive:
                    lower += 1
                if not hyperparameter.upper_inclusive:
                    upper -= 1
                new_hyperparameter = UniformIntegerHyperparameter(name, lower, upper, default_value=default)
            else:
                if not hyperparameter.lower_inclusive:
                    lower += 1e-20
                if not hyperparameter.upper_inclusive:
                    upper -= 1e-20
                new_hyperparameter = UniformFloatHyperparameter(name, lower, upper, default_value=default)
        elif isinstance(hyperparameter, UniformBool):
            default = hyperparameter.get_default()
            new_hyperparameter = CategoricalHyperparameter(name, [True, False], default_value=default)
        elif isinstance(hyperparameter, UniformInt):
            lower = hyperparameter.lower
            upper = hyperparameter.upper
            default = hyperparameter.get_default()
            new_hyperparameter = UniformIntegerHyperparameter(name, lower, upper, default_value=default)
        elif isinstance(hyperparameter, Uniform):
            lower = hyperparameter.lower
            upper = hyperparameter.upper
            default = hyperparameter.get_default()
            new_hyperparameter = UniformFloatHyperparameter(name, lower, upper, default_value=default)
        elif isinstance(hyperparameter, Normal):
            default = hyperparameter.get_default()
            new_hyperparameter = NormalFloatHyperparameter(name, default_value=default)
        elif isinstance(hyperparameter, Enumeration):
            values = hyperparameter.values
            default = hyperparameter.get_default()
            new_hyperparameter = CategoricalHyperparameter(name, values, default_value=default)
        elif isinstance(hyperparameter, ConstantD3M):
            default = 'None' if hyperparameter.get_default() is None else hyperparameter.get_default()
            new_hyperparameter = Constant(name, default)
    except Exception as e:
        logger.error(e)

    return new_hyperparameter


def get_hyperparameters_from_metalearnig():
    with open(HYPERPARAMETERS_FROM_METALEARNING_PATH) as fin:
        search_space = json.load(fin)
        return search_space


def get_configspace_from_metalearning(metalearning_entry):
    cs = ConfigurationSpace()
    categorical_and_none_hyperparams = []

    for hyperparam in metalearning_entry:
        if len(metalearning_entry[hyperparam]['choices']) == 1 and metalearning_entry[hyperparam]['default'] is None:
            categorical_and_none_hyperparams.append(
                Constant(hyperparam, 'None')
            )
        else:
            categorical_and_none_hyperparams.append(
                CategoricalHyperparameter(
                    name=hyperparam,
                    choices=metalearning_entry[hyperparam]['choices'],
                    default_value=metalearning_entry[hyperparam]['default'])
            )
    cs.add_hyperparameters(categorical_and_none_hyperparams)

    return cs
