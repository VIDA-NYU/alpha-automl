import signal
import os
import sys
import logging
from os.path import join, dirname
from alphad3m_sklearn.pipeline_search.Coach import Coach
from alphad3m_sklearn.pipeline_search.pipeline.PipelineGame import PipelineGame
from alphad3m_sklearn.pipeline_search.pipeline.NNet import NNetWrapper
from alphad3m_sklearn.grammar_loader import load_manual_grammar, load_automatic_grammar
#from alphad3m_sklearn.data_ingestion.data_profiler import get_privileged_data, select_encoders
from alphad3m_sklearn.pipeline_synthesis.pipeline_builder import *
from alphad3m_sklearn.utils import score_pipeline

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


def signal_handler(signal_num, frame):
    logger.info(f'Receiving signal {signal.Signals(signal_num).name}, terminating process')
    signal.alarm(0)  # Disable the alarm
    sys.exit(0)


def search_pipelines(X, y, scoring, splitting_strategy, task_name, hyperparameters, time_bound,  dataset, output_folder, queue):
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(time_bound))

    builder = BaseBuilder()

    def evaluate_pipeline(primitives, origin):
        pipeline = builder.make_pipeline(primitives)
        score = score_pipeline(pipeline, X, y, scoring, splitting_strategy)
        print('score', score)
        #queue.send((pipeline, score))
        return score

    metric = 'accuracy'
    config_updated = update_config(task_name, metric, dataset, output_folder, hyperparameters)

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

    sys.exit(0)


def update_config(task_name, metric, dataset, output_folder, hyperparameters):
    config['PROBLEM'] = task_name
    config['DATA_TYPE'] = 'TABULAR'
    config['METRIC'] = metric
    config['DATASET'] = dataset
    config['ARGS']['stepsfile'] = join(output_folder, f'{dataset}_pipeline_steps.txt')
    config['ARGS']['checkpoint'] = join(output_folder, 'nn_models')
    config['ARGS']['load_folder_file'] = join(output_folder, 'nn_models', 'best.pth.tar')

    task_name_id = task_name + '_TASK'
    use_automatic_grammar = hyperparameters['use_automatic_grammar']
    include_primitives = hyperparameters['include_primitives']
    exclude_primitives = hyperparameters['exclude_primitives']

    task_keyword_ids = []
    grammar = None

    if use_automatic_grammar:
        logger.info('Creating an automatic grammar')
        prioritize_primitives = hyperparameters['prioritize_primitives']
        dataset_path = join(dirname(dataset[7:]), 'tables', 'learningData.csv')
        target_column = ''
        grammar = load_automatic_grammar(task_name_id, dataset_path, target_column, task_keyword_ids,
                                         include_primitives, exclude_primitives, prioritize_primitives)

    if grammar is None:
        logger.info('Creating a manual grammar')
        encoders = []#select_encoders(metadata['only_attribute_types'])
        use_imputer = True
        grammar = load_manual_grammar(task_name_id, task_keyword_ids, encoders, use_imputer, include_primitives,
                                      exclude_primitives)

    config['GRAMMAR'] = grammar
    # metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
    config['DATASET_METAFEATURES'] = [0] * 50  # metafeatures_extractor.compute_metafeatures('AlphaD3M_compute_metafeatures')

    return config