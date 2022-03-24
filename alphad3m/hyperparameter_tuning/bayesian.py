import logging
import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from alphad3m.hyperparameter_tuning.primitive_config import load_primitive_configspace, load_hyperparameters


MAX_RUNS = 100
logger = logging.getLogger(__name__)


def build_configspace(primitives):
    # Build Configuration Space which defines all parameters and their ranges
    configspace = ConfigurationSpace()
    for primitive in primitives:
        load_primitive_configspace(configspace, primitive)

    return configspace


def get_new_hyperparameters(primitive_name, configspace):
    hyperparameters = load_hyperparameters(primitive_name)
    new_hyperparameters = {}

    for hyperparameter_name in hyperparameters:
        hyperparameter_config_name = primitive_name + '|' + hyperparameter_name
        hyperparameter_config_name_case = hyperparameter_config_name + '|case'
        if hyperparameter_config_name in configspace:
            value = None if configspace[hyperparameter_config_name] == 'None' \
                else configspace[hyperparameter_config_name]
            new_hyperparameters[hyperparameter_name] = value
            logger.info('New value for %s=%s', hyperparameter_config_name, new_hyperparameters[hyperparameter_name])
        elif hyperparameter_config_name_case in configspace:
            case = configspace[hyperparameter_config_name_case]
            value = None if configspace[hyperparameter_config_name + '|' + case] == 'None' \
                else configspace[hyperparameter_config_name + '|' + case]
            new_hyperparameters[hyperparameter_name] = {'case': case,
                                                        'value': value}
            logger.info('New value for %s=%s', hyperparameter_config_name, new_hyperparameters[hyperparameter_name])

    return new_hyperparameters


class HyperparameterTuning(object):
    def __init__(self, primitives):
        self.configspace = build_configspace(primitives)
        # Avoiding too many iterations
        self.runcount = 1

        for param in self.configspace.get_hyperparameters():
            if isinstance(param, IntegerHyperparameter):
                self.runcount *= (param.upper - param.lower)
            elif isinstance(param, CategoricalHyperparameter):
                self.runcount *= len(param.choices)
            elif isinstance(param, OrdinalHyperparameter):
                self.runcount *= len(param.sequence)
            elif isinstance(param, FloatHyperparameter):
                self.runcount = MAX_RUNS
                break

        self.runcount = min(self.runcount, MAX_RUNS)

    def tune(self, runner, wallclock, output_dir):
        # Scenario object
        cutoff = wallclock / (self.runcount / 10)  # Allow long pipelines to try to execute one fourth of the iterations limit
        scenario = Scenario({'run_obj': 'quality',  # We optimize quality (alternatively runtime)
                             'runcount-limit': self.runcount,  # Maximum function evaluations
                             'wallclock-limit': wallclock,
                             'cutoff_time': cutoff,
                             'cs': self.configspace,  # Configuration space
                             'deterministic': 'true',
                             'output_dir': output_dir,
                             'abort_on_first_run_crash': False
                             })
        smac = SMAC4AC(scenario=scenario, rng=np.random.RandomState(0), tae_runner=runner)
        best_configuration = smac.optimize()
        min_cost = float('inf')
        best_scores = {}

        for _, run_data in smac.get_runhistory().data.items():
            if run_data.cost < min_cost:
                min_cost = run_data.cost
                best_scores = run_data.additional_info

        return best_configuration, best_scores
