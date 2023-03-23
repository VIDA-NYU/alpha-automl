import re
import ast
import json
import gzip
import pickle
import logging
from os.path import join, dirname, exists
from alphad3m.primitive_loader import load_primitives_by_name
from alphad3m.metalearning.dataset_profiler import extract_dataprofiles, extract_metafeatures

logger = logging.getLogger(__name__)

METALEARNING_DB_PATH = join(dirname(__file__), '../resource/metalearning_db.json.gz')
PRECALCULATED_TASKKEYWORDS_PATH = join(dirname(__file__), '../resource/precalculated_taskkeywords.json')
PRECALCULATED_METAFEATURES_PATH = join(dirname(__file__), '../resource/precalculated_metafeatures.json')
PRECALCULATED_DATAPROFILES_PATH = join(dirname(__file__), '../resource/precalculated_dataprofiles.json')


IGNORE_PRIMITIVES = {
    # These primitives are static elements in the pipelines, not considered as part of the pattern
    'd3m.primitives.data_transformation.construct_predictions.Common',
    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
    'd3m.primitives.data_transformation.denormalize.Common',
    'd3m.primitives.data_transformation.flatten.DataFrameCommon',
    'd3m.primitives.data_transformation.column_parser.Common',
    'd3m.primitives.data_transformation.simple_column_parser.DataFrameCommon',
    'd3m.primitives.data_transformation.do_nothing.DSBOX',
    'd3m.primitives.data_transformation.do_nothing_for_dataset.DSBOX',
    'd3m.primitives.data_transformation.add_semantic_types.Common',
    'd3m.primitives.schema_discovery.profiler.Common',
    'd3m.primitives.schema_discovery.profiler.DSBOX',
    'd3m.primitives.data_cleaning.column_type_profiler.Simon',
    'd3m.primitives.operator.compute_unique_values.Common',
    'd3m.primitives.data_transformation.construct_confidence.Common',
    # We add these primitives internally because they require special connections
    'd3m.primitives.data_transformation.text_reader.Common',
    'd3m.primitives.data_transformation.image_reader.Common'
}


def create_compressed_metalearningdb(metalearningdb_pickle_path):
    logger.info('Compressing Meta-Learning DB...')
    primitives_by_name = load_primitives_by_name(only_installed_primitives=False)
    available_datasets = load_precalculated_data('task_keywords')
    ignore_primitives_ids = {primitives_by_name[ignore_primitive]['id'] for ignore_primitive in IGNORE_PRIMITIVES}
    pipelines_by_dataset = {}
    pipelines_hashing = {}

    with open(metalearningdb_pickle_path, 'rb') as fin:
        all_pipelines = pickle.load(fin)

    for pipeline_run in all_pipelines:
        dataset_id = get_dataset_id(pipeline_run['problem']['id'])

        if dataset_id not in available_datasets:
            continue

        pipeline_primitives = pipeline_run['steps']
        pipeline_primitives = filter_primitives(pipeline_primitives, ignore_primitives_ids)  # Get the IDs of primitives

        if dataset_id not in pipelines_by_dataset:
            pipelines_by_dataset[dataset_id] = {}

        if len(pipeline_primitives) > 0:
            score = pipeline_run['scores'][0]['normalized']
            metric = pipeline_run['scores'][0]['metric']['metric']
            pipeline_str = str(pipeline_primitives)

            if pipeline_str not in pipelines_hashing:
                hashing_value = 'P' + str(len(pipelines_hashing))
                pipelines_hashing[pipeline_str] = hashing_value

            pipeline_id = pipelines_hashing[pipeline_str]

            if pipeline_id not in pipelines_by_dataset[dataset_id]:
                pipelines_by_dataset[dataset_id][pipeline_id] = {'score': [], 'metric': []}

            pipelines_by_dataset[dataset_id][pipeline_id]['score'].append(score)
            pipelines_by_dataset[dataset_id][pipeline_id]['metric'].append(metric)

    pipeline_structure = {}
    for pipeline_str, pipeline_id in pipelines_hashing.items():
        primitives = [primitive for primitive in ast.literal_eval(pipeline_str)]  # Convert str to list
        pipeline_structure[pipeline_id] = primitives

    metalearning_db = {}
    metalearning_db['pipeline_performances'] = pipelines_by_dataset
    metalearning_db['pipeline_structure'] = pipeline_structure

    with gzip.open(METALEARNING_DB_PATH, 'wt', encoding='UTF-8') as zipfile:
        json.dump(json.dumps(metalearning_db), zipfile)  # Convert to str and then compress it

    logger.info('Compressing process ended')


def extract_taskkeywords_metalearningdb(datasets_path):
    datasets = get_unique_datasets()
    task_keywords = load_precalculated_data('task_keywords')

    for dataset_id in datasets:
        logger.info('Calculating task keywords for dataset %s...', dataset_id)
        if dataset_id not in task_keywords:
            try:
                _, _, keywords = load_task_info(dataset_id, datasets_path)
                task_keywords[dataset_id] = {'task_keywords': keywords}
                logger.info('Task keywords successfully calculated for dataset %s', dataset_id)
                with open(PRECALCULATED_TASKKEYWORDS_PATH, 'w') as fout:
                    json.dump(task_keywords, fout, indent=4, sort_keys=True)
            except Exception as e:
                logger.error(str(e))
        else:
            logger.info('Using pre-calculated task keywords for dataset %s', dataset_id)

    return task_keywords


def extract_metafeatures_metalearningdb(datasets_path):
    datasets = get_unique_datasets()
    metafeatures = load_precalculated_data('metafeatures')

    for dataset_id in datasets:
        logger.info('Calculating metafeatures for dataset %s...', dataset_id)
        if dataset_id not in metafeatures:
            try:
                dataset_path, target_column, _ = load_task_info(dataset_id, datasets_path, 'SCORE')
                mfs = extract_metafeatures(dataset_path, target_column)
                metafeatures[dataset_id] = mfs
                logger.info('Metafeatures successfully calculated for dataset %s', dataset_id)
                with open(PRECALCULATED_METAFEATURES_PATH, 'w') as fout:
                    json.dump(metafeatures, fout, indent=4, sort_keys=True)
            except Exception as e:
                logger.error(str(e))
        else:
            logger.info('Using pre-calculated metafeatures for dataset %s', dataset_id)

    return metafeatures


def extract_dataprofiles_metalearningdb(datasets_path):
    datasets = get_unique_datasets()
    dataprofiles = load_precalculated_data('dataprofiles')

    for dataset_id in datasets:
        logger.info('Calculating data profiles for dataset %s...', dataset_id)
        if dataset_id not in dataprofiles:
            try:
                dataset_path, target_column, _ = load_task_info(dataset_id, datasets_path)
                dps = extract_dataprofiles(dataset_path, target_column)
                dataprofiles[dataset_id] = dps
                logger.info('Data profiles successfully calculated for dataset %s', dataset_id)
                with open(PRECALCULATED_DATAPROFILES_PATH, 'w') as fout:
                    json.dump(dataprofiles, fout, indent=4, sort_keys=True)
            except Exception as e:
                logger.error(str(e))
        else:
            logger.info('Using pre-calculated data profiles for dataset %s', dataset_id)

    return dataprofiles


def load_task_info(dataset_id, datasets_path, suffix='TRAIN'):
    possible_names = [join(datasets_path, dataset_id), join(datasets_path, dataset_id + '_MIN_METADATA'),
                      join(datasets_path, dataset_id.replace('_MIN_METADATA', ''))]
    # All possible names of the datasets on disk, with/without the suffix 'MIN_METADATA'

    for dataset_folder_path in possible_names:
        if exists(dataset_folder_path):
            break
    else:
        raise FileNotFoundError('Dataset %s not found' % dataset_id)

    dataset_path = join(dataset_folder_path, suffix, 'dataset_%s/tables/learningData.csv' % suffix)
    problem_path = join(dataset_folder_path, suffix, 'problem_%s/problemDoc.json' % suffix)

    with open(problem_path) as fin:
        problem_doc = json.load(fin)
        task_keywords = problem_doc['about']['taskKeywords']
        target_column = problem_doc['inputs']['data'][0]['targets'][0]['colName']

    return dataset_path, target_column, task_keywords


def get_dataset_id(problem_id):
    # Remove suffixes 'TRAIN' and 'problem' from the dataset name
    dataset_id = re.sub('_TRAIN$', '', problem_id)
    dataset_id = re.sub('_problem$', '', dataset_id)

    return dataset_id


def get_unique_datasets():
    metalearning_db = load_metalearningdb()
    datasets = metalearning_db['pipeline_performances'].keys()

    return sorted(datasets)


def filter_primitives(pipeline_steps, ignore_primitives):
    primitives = []

    for pipeline_step in pipeline_steps:
        if pipeline_step['primitive']['id'] not in ignore_primitives:
            primitives.append(pipeline_step['primitive']['id'])

    if len(primitives) > 0 and primitives[0] == '7ddf2fd8-2f7f-4e53-96a7-0d9f5aeecf93':
        # Special case: Primitive to_numeric.DSBOX
        # This primitive should not be first because it only takes the numeric features, ignoring the remaining ones
        primitives = primitives[1:]

    return primitives


def load_metalearningdb():
    all_pipelines = []
    logger.info('Loading pipelines from metalearning database...')

    with gzip.open(METALEARNING_DB_PATH, 'rt', encoding='UTF-8') as zipfile:
        all_pipelines = json.loads(json.load(zipfile))  # Uncompress as str and then convert to dict

    logger.info('Found %d unique pipelines in metalearning database' % len(all_pipelines['pipeline_structure']))

    return all_pipelines


def load_precalculated_data(mode):
    if mode == 'metafeatures':
        file_path = PRECALCULATED_METAFEATURES_PATH
    elif mode == 'dataprofiles':
        file_path = PRECALCULATED_DATAPROFILES_PATH
    elif mode == 'task_keywords':
        file_path = PRECALCULATED_TASKKEYWORDS_PATH
    else:
        raise ValueError('Unknown mode "%s" to load data' % mode)

    if exists(file_path):
        with open(file_path) as fin:
            return json.load(fin)

    return {}


if __name__ == '__main__':
    # Run this to create the meta-learning DB, task keywords, data profiles, and meta-features files
    # Download the metalearningdb.pkl file from https://drive.google.com/file/d/1WjY7iKkkKMZFeoiCqzamA_iqVOwQidXS/view
    # and D3M datasets from https://datasets.datadrivendiscovery.org/d3m/datasets/-/tree/master/seed_datasets_current
    metalearningdb_pickle_path = '/Users/rlopez/D3M/metalearning_db/metalearningdb.pkl'
    datasets_root_path = '/Users/rlopez/D3M/datasets/seed_datasets_current/'

    create_compressed_metalearningdb(metalearningdb_pickle_path)
    extract_taskkeywords_metalearningdb(datasets_root_path)
    extract_dataprofiles_metalearningdb(datasets_root_path)
    extract_metafeatures_metalearningdb(datasets_root_path)
