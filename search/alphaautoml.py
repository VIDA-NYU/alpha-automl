import signal
import os
import sys

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'

from os.path import join
from alphaAutoMLEdit.Coach import Coach
from alphaAutoMLEdit.pipeline.PipelineGame import PipelineGame
from alphaAutoMLEdit.pipeline.NNet import NNetWrapper
from alphad3m.primitive_loader import get_primitives_by_type
from alphad3m.grammar_loader import create_game_grammar
from alphad3m.data_ingestion.data_profiler import get_privileged_data, select_encoders
from alphad3m.search.d3mpipeline_builder import *
from alphad3m.metafeature.metafeature_extractor import ComputeMetafeatures
from alphad3m.utils import get_collection_type
from d3m.metadata.problem import TaskKeyword


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


@database.with_sessionmaker
def generate_pipelines(task_keywords, dataset, metrics, problem, targets, features, metadata, pipeline_template, msg_queue, DBSession):
    builder = None
    task_name = 'CLASSIFICATION' if TaskKeyword.CLASSIFICATION in task_keywords else 'REGRESSION'
    privileged_data = get_privileged_data(problem, task_keywords)

    def eval_pipeline(primitive_names, origin):
        pipeline_id = builder.make_d3mpipeline(primitive_names, origin, dataset, pipeline_template, targets,
                                               features, metadata, privileged_data, metrics, DBSession=DBSession)
        #execute(pipeline_id, dataset, problem, join(os.environ.get('D3MOUTPUTDIR'), 'output_dataframe.csv'), None,
        #        db_filename=join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'db.sqlite3'))
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
        builder = CollaborativeFilteringBuilder()
    elif TaskKeyword.COMMUNITY_DETECTION in task_keywords:
        task_name = 'COMMUNITY_DETECTION'
        builder = CommunityDetectionBuilder()
    elif TaskKeyword.LINK_PREDICTION in task_keywords:
        task_name = 'LINK_PREDICTION'
        builder = LinkPredictionBuilder()
    elif TaskKeyword.OBJECT_DETECTION in task_keywords:
        task_name = 'OBJECT_DETECTION'
        builder = ObjectDetectionBuilder()
    elif TaskKeyword.GRAPH_MATCHING in task_keywords:
        task_name = 'GRAPH_MATCHING'
        builder = GraphMatchingBuilder()
    elif TaskKeyword.FORECASTING in task_keywords:
        task_name = 'TIME_SERIES_FORECASTING'
        builder = BaseBuilder()
    elif TaskKeyword.TIME_SERIES in task_keywords and TaskKeyword.CLASSIFICATION in task_keywords:
        task_name = 'TIME_SERIES_CLASSIFICATION'
        builder = TimeseriesClassificationBuilder()
    elif TaskKeyword.VERTEX_CLASSIFICATION in task_keywords or TaskKeyword.VERTEX_NOMINATION in task_keywords:
        task_name = 'VERTEX_CLASSIFICATION'
        builder = VertexClassificationBuilder()
    elif get_collection_type(dataset[7:]) == 'text' or TaskKeyword.TEXT in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'TEXT_' + task_name
        builder = BaseBuilder()
    elif get_collection_type(dataset[7:]) == 'image' or TaskKeyword.IMAGE in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
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

    encoders = select_encoders(metadata['only_attribute_types'])
    use_imputer = metadata['use_imputer']

    def update_config(primitives, task_name):
        metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
        config['GRAMMAR'] = create_game_grammar(task_name + '_TASK', primitives, encoders, use_imputer)
        config['PROBLEM'] = task_name
        config['DATA_TYPE'] = 'TABULAR'
        config['METRIC'] = metrics[0]['metric'].name
        config['DATASET_METAFEATURES'] = [0] * 50 #metafeatures_extractor.compute_metafeatures('AlphaD3M_compute_metafeatures')
        config['DATASET'] = problem['inputs'][0]['dataset_id']
        config['ARGS']['stepsfile'] = join(os.environ.get('D3MOUTPUTDIR'), 'temp', config['DATASET'] + '_pipeline_steps.txt')

        return config

    def signal_handler(signal, frame):
        logger.info('Receiving SIGTERM signal')
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    primitives = get_primitives_by_type()
    config_updated = update_config(primitives, task_name)

    ############
    '''metalearning_pipelines = create_vectors_from_metalearningdb(task_name, config_updated['GRAMMAR'])
    def my_eval(primitive_names, origin):
        pipeline_representation = ' '.join(sorted(primitive_names))
        if pipeline_representation in metalearning_pipelines:
            score = metalearning_pipelines[pipeline_representation]
            print('>>>>>>>>>>yes', pipeline_representation, score)
        else:
            print('>>>>>>>>>>no', pipeline_representation)
            score = None
        return score

    game = PipelineGame(config_updated, my_eval)'''
    ############
    game = PipelineGame(config_updated, eval_pipeline)
    nnet = NNetWrapper(game)

    ########
    '''train_examples = create_vectors_from_metalearningdb(task_name, config_updated['GRAMMAR'])
    nnet.train(train_examples)
    nnet.save_checkpoint(join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'nn_models'))
    print('saved')
    sys.exit(0)'''
    #######

    if config['ARGS'].get('load_model'):
        model_file = join(config['ARGS'].get('load_folder_file')[0],
                          config['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(config['ARGS'].get('load_folder_file')[0],
                                 config['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, config['ARGS'])
    c.learn()

    sys.exit(0)