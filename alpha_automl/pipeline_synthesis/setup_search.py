import signal
import os
import logging
from os.path import join
from alpha_automl.scorer import score_pipeline
from alpha_automl.data_profiler import profile_data
from alpha_automl.pipeline import Pipeline
from alpha_automl.pipeline_search.Coach import Coach
from alpha_automl.pipeline_search.pipeline.PipelineGame import PipelineGame
from alpha_automl.pipeline_search.pipeline.NNet import NNetWrapper
from alpha_automl.grammar_loader import load_manual_grammar, load_automatic_grammar
from alpha_automl.pipeline_synthesis.pipeline_builder import BaseBuilder


logger = logging.getLogger(__name__)


config = {
    'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                      'REGRESSION': 2,
                      'CLUSTERING': 3,
                      'NA': 4
                      },

    'DATA_TYPES': {'TABULAR': 1,
                   'GRAPH': 2,
                   'IMAGE': 3},

    'PIPELINE_SIZE': 8,

    'ARGS': {
        'numIters': 25,
        'numEps': 5,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 5,
        'arenaCompare': 40,
        'cpuct': 1,
        'load_model': False,
        'metafeatures_path': '/d3m/data/metafeatures',
        'verbose': True
    }
}


def signal_handler(queue):
    logger.info('Receiving signal, terminating process')
    signal.alarm(0)  # Disable the alarm
    queue.put('DONE')
    # TODO: Should it save the last status of the NN model?


def search_pipelines(X, y, scoring, splitting_strategy, task_name, time_bound, automl_hyperparams, output_folder, queue):
    signal.signal(signal.SIGALRM, lambda signum, frame: signal_handler(queue))
    signal.alarm(time_bound)


    metadata = profile_data(X)
    builder = BaseBuilder(metadata, automl_hyperparams)

    def evaluate_pipeline(primitives, origin):
        pipeline = builder.make_pipeline(primitives)
        score = None

        if pipeline is not None:
            score, start_time, end_time = score_pipeline(pipeline, X, y, scoring, splitting_strategy)
            if score is not None:
                pipeline_alphaautoml = Pipeline(pipeline, score, start_time, end_time)
                queue.put(pipeline_alphaautoml)  # Only send valid pipelines

        return score

    if task_name is None:
        task_name = 'NA'

    task_name_id = task_name + '_TASK'
    use_automatic_grammar = automl_hyperparams['use_automatic_grammar']
    include_primitives = automl_hyperparams['include_primitives']
    exclude_primitives = automl_hyperparams['exclude_primitives']
    new_primitives = automl_hyperparams['new_primitives']
    grammar = None

    if use_automatic_grammar:
        logger.info('Creating an automatic grammar')
        prioritize_primitives = automl_hyperparams['prioritize_primitives']
        target_column = ''
        dataset_path = ''
        grammar = load_automatic_grammar(task_name_id, dataset_path, target_column, include_primitives,
                                         exclude_primitives, prioritize_primitives)

    if grammar is None:
        logger.info('Creating a manual grammar')
        use_imputer = metadata['missing_values']
        nonnumeric_columns = metadata['nonnumeric_columns']
        grammar = load_manual_grammar(task_name_id, nonnumeric_columns, use_imputer, new_primitives, include_primitives,
                                      exclude_primitives)

    metric = scoring._score_func.__name__
    config_updated = update_config(task_name, metric, output_folder, grammar)
    game = PipelineGame(config_updated, evaluate_pipeline)
    nnet = NNetWrapper(game)

    if config['ARGS'].get('load_model'):
        model_file = join(config['ARGS'].get('load_folder_file')[0],
                          config['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(config['ARGS'].get('load_folder_file')[0],
                                 config['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, config['ARGS'])
    c.learn()
    logger.info('Search completed')
    queue.put('DONE')


def update_config(task_name, metric, output_folder, grammar):
    config['PROBLEM'] = task_name
    config['DATA_TYPE'] = 'TABULAR'
    config['METRIC'] = metric
    config['DATASET'] = f'DATASET_{task_name}'
    config['ARGS']['stepsfile'] = join(output_folder, f'DATASET_{task_name}_pipeline_steps.txt')
    config['ARGS']['checkpoint'] = join(output_folder, 'nn_models')
    config['ARGS']['load_folder_file'] = join(output_folder, 'nn_models', 'best.pth.tar')
    config['GRAMMAR'] = grammar
    # metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
    config['DATASET_METAFEATURES'] = [0] * 50  # metafeatures_extractor.compute_metafeatures('Compute_metafeatures')

    return config
