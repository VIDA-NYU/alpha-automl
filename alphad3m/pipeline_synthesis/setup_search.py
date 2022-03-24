import signal
import os
import sys
import logging
from os.path import join, dirname
from alphad3m.pipeline_search.Coach import Coach
from alphad3m.pipeline_search.pipeline.PipelineGame import PipelineGame
from alphad3m.pipeline_search.pipeline.NNet import NNetWrapper
from alphad3m.grammar_loader import load_manual_grammar, load_automatic_grammar
from alphad3m.data_ingestion.data_profiler import get_privileged_data, select_encoders
from alphad3m.pipeline_synthesis.d3mpipeline_builder import *
from alphad3m.metafeature.metafeature_extractor import ComputeMetafeatures
from alphad3m.utils import get_collection_type
from d3m.metadata.problem import TaskKeyword, TaskKeywordBase


logger = logging.getLogger(__name__)


config = {
    'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                      'REGRESSION': 2,
                      'CLUSTERING': 3,
                      'TIME_SERIES_FORECASTING': 4,
                      'TIME_SERIES_CLASSIFICATION': 5,
                      'COMMUNITY_DETECTION': 6,
                      'GRAPH_MATCHING': 7,
                      'COLLABORATIVE_FILTERING': 8,
                      'LINK_PREDICTION': 9,
                      'VERTEX_CLASSIFICATION': 10,
                      'OBJECT_DETECTION': 11,
                      'SEMISUPERVISED_CLASSIFICATION': 12,
                      'TEXT_CLASSIFICATION': 13,
                      'IMAGE_CLASSIFICATION': 14,
                      'AUDIO_CLASSIFICATION': 15,
                      'TEXT_REGRESSION': 16,
                      'IMAGE_REGRESSION': 17,
                      'AUDIO_REGRESSION': 18,
                      'VIDEO_CLASSIFICATION': 19,
                      'LUPI': 20,
                      'NA': 21
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

        'checkpoint': join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'nn_models'),
        'load_model': False,
        'load_folder_file': (join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'nn_models'), 'best.pth.tar'),
        'metafeatures_path': '/d3m/data/metafeatures',
        'verbose': True
    }
}


def signal_handler(signal_num, frame):
    logger.info('Receiving signal %s, terminating process' % signal.Signals(signal_num).name)
    signal.alarm(0)  # Disable the alarm
    sys.exit(0)


@database.with_sessionmaker
def generate_pipelines(task_keywords, dataset, metrics, problem, targets, features, hyperparameters, metadata,
                       pipeline_template, time_bound, msg_queue, DBSession):
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(time_bound))

    builder = None
    task_name = 'CLASSIFICATION' if TaskKeyword.CLASSIFICATION in task_keywords else 'REGRESSION'
    # Primitives for LUPI problems are no longer available. So, just exclude privileged data
    privileged_data = get_privileged_data(problem, task_keywords)
    metadata['exclude_columns'] += privileged_data

    def eval_pipeline(primitive_names, origin):
        pipeline_id = builder.make_d3mpipeline(primitive_names, origin, dataset, pipeline_template, targets,
                                               features, metadata, metrics, DBSession=DBSession)
        # Evaluate the pipeline if syntax is correct:
        if pipeline_id:
            msg_queue.send(('eval', pipeline_id))
            return msg_queue.recv()
        else:
            return None

    if TaskKeyword.CLUSTERING in task_keywords:
        task_name = 'CLUSTERING'
        builder = BaseBuilder()
    elif TaskKeyword.SEMISUPERVISED in task_keywords:
        task_name = 'SEMISUPERVISED_CLASSIFICATION'
        builder = BaseBuilder()
    elif TaskKeyword.COLLABORATIVE_FILTERING in task_keywords:
        task_name = 'COLLABORATIVE_FILTERING'
        builder = BaseBuilder()
    elif TaskKeyword.FORECASTING in task_keywords:
        task_name = 'TIME_SERIES_FORECASTING'
        builder = BaseBuilder()
    elif TaskKeyword.COMMUNITY_DETECTION in task_keywords:
        task_name = 'COMMUNITY_DETECTION'
        builder = BaseBuilder()
    elif TaskKeyword.LINK_PREDICTION in task_keywords:
        task_name = 'LINK_PREDICTION'
        builder = BaseBuilder()
    elif TaskKeyword.OBJECT_DETECTION in task_keywords:
        task_name = 'OBJECT_DETECTION'
        builder = BaseBuilder()
    elif TaskKeyword.GRAPH_MATCHING in task_keywords:
        task_name = 'GRAPH_MATCHING'
        builder = BaseBuilder()
    elif TaskKeyword.TIME_SERIES in task_keywords and TaskKeyword.CLASSIFICATION in task_keywords:
        task_name = 'TIME_SERIES_CLASSIFICATION'
        builder = BaseBuilder()
    elif TaskKeyword.VERTEX_CLASSIFICATION in task_keywords or TaskKeyword.VERTEX_NOMINATION in task_keywords:
        task_name = 'VERTEX_CLASSIFICATION'
        builder = BaseBuilder()
    elif get_collection_type(dataset[7:]) == 'text' or TaskKeyword.TEXT in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'TEXT_' + task_name
        builder = BaseBuilder()
    elif get_collection_type(dataset[7:]) == 'image' or TaskKeyword.IMAGE in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        if TaskKeyword.IMAGE not in task_keywords: task_keywords.append(TaskKeyword.IMAGE)
        task_name = 'IMAGE_' + task_name
        builder = BaseBuilder()
    elif TaskKeyword.AUDIO in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'AUDIO_' + task_name
        builder = AudioBuilder()
    elif TaskKeyword.VIDEO in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'VIDEO_' + task_name
        builder = BaseBuilder()
    elif TaskKeyword.CLASSIFICATION in task_keywords or TaskKeyword.REGRESSION in task_keywords:
        builder = BaseBuilder()
    else:
        logger.warning('Task %s doesnt exist in the grammar, using default NA_TASK' % task_name)
        task_name = 'NA'
        builder = BaseBuilder()

    use_automatic_grammar = hyperparameters['use_automatic_grammar']
    include_primitives = hyperparameters['include_primitives']
    exclude_primitives = hyperparameters['exclude_primitives']

    def update_config(task_name):
        config['PROBLEM'] = task_name
        config['DATA_TYPE'] = 'TABULAR'
        config['METRIC'] = metrics[0]['metric'].name
        config['DATASET'] = problem['inputs'][0]['dataset_id']
        config['ARGS']['stepsfile'] = join(os.environ.get('D3MOUTPUTDIR'), 'temp', config['DATASET'] + '_pipeline_steps.txt')

        task_name_id = task_name + '_TASK'
        task_keywords_mapping = {v: k for k, v in TaskKeywordBase.get_map().items()}
        task_keyword_ids = [task_keywords_mapping[t] for t in task_keywords]
        grammar = None

        if use_automatic_grammar:
            logger.info('Creating an automatic grammar')
            prioritize_primitives = hyperparameters['prioritize_primitives']
            dataset_path = join(dirname(dataset[7:]), 'tables', 'learningData.csv')
            target_column = problem['inputs'][0]['targets'][0]['column_name']
            grammar = load_automatic_grammar(task_name_id, dataset_path, target_column, task_keyword_ids,
                                             include_primitives, exclude_primitives, prioritize_primitives)

        if grammar is None:
            logger.info('Creating a manual grammar')
            encoders = select_encoders(metadata['only_attribute_types'])
            use_imputer = metadata['missing_values']
            grammar = load_manual_grammar(task_name_id, task_keyword_ids, encoders, use_imputer, include_primitives,
                                          exclude_primitives)
        config['GRAMMAR'] = grammar
        metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
        config['DATASET_METAFEATURES'] = [0] * 50 #metafeatures_extractor.compute_metafeatures('AlphaD3M_compute_metafeatures')

        return config

    config_updated = update_config(task_name)

    game = PipelineGame(config_updated, eval_pipeline)
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
