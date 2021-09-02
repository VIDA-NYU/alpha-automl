import re
import json
import logging
import pandas as pd
from os.path import join, dirname, exists
from metalearn import Metafeatures
from alphad3m.metalearning.database import load_metalearningdb

logger = logging.getLogger(__name__)

DATASETS_FOLDER_PATH = '/Users/rlopez/D3M/datasets/seed_datasets_current/'
PRECALCULATED_METAFEATURES_PATH = join(dirname(__file__), '../../resource/precalculated_metafeatures.json')


def get_unique_datasets(remove_suffix=True):
    pipeline_runs = load_metalearningdb()
    datasets = set()

    for pipeline_run in pipeline_runs:
        dataset_id = pipeline_run['problem']['id']
        if remove_suffix:
            dataset_id = re.sub('_TRAIN$', '', dataset_id)
            dataset_id = re.sub('_problem$', '', dataset_id)
        datasets.add(dataset_id)

    logger.info('Found %d unique datasets', len(datasets))

    return sorted(datasets)


def get_X_Y(dataset_folder_path):
    problem_path = join(dataset_folder_path, 'TRAIN/problem_TRAIN/problemDoc.json')
    csv_path = join(dataset_folder_path, 'TRAIN/dataset_TRAIN/tables/learningData.csv')

    with open(problem_path) as fin:
        problem_doc = json.load(fin)
        target_name = problem_doc['inputs']['data'][0]['targets'][0]['colName']

    data = pd.read_csv(csv_path)
    Y = data[target_name]
    X = data.drop(columns=[target_name])

    return X, Y


def load_dataset(dataset_id):
    dataset_folder_path = join(DATASETS_FOLDER_PATH, dataset_id)

    if not exists(dataset_folder_path):
        # Add the suffix '_MIN_METADATA' to the name of the dataset
        dataset_folder_path = join(DATASETS_FOLDER_PATH, dataset_id + '_MIN_METADATA')
        if not exists(dataset_folder_path):
            logger.error('Dataset %s not found', dataset_id)
            return None

    try:
        X, Y = get_X_Y(dataset_folder_path)
        return {'X': X, 'Y': Y}
    except:
        logger.error('Reading dataset %s', dataset_id)
        return None


def load_precalculated_metafeatures():
    if exists(PRECALCULATED_METAFEATURES_PATH):
        with open(PRECALCULATED_METAFEATURES_PATH) as fin:
            return json.load(fin)

    return {}


def extract_metafeatures(X, Y):
    metafeatures = Metafeatures()
    mfs = None
    try:
        mfs = metafeatures.compute(X, Y, seed=0, timeout=300)
    except:
        logger.error('Calculating metafeatures')

    return mfs


def extract_metafeatures_all():
    datasets = get_unique_datasets()
    metafeatures = load_precalculated_metafeatures()

    for dataset_id in datasets:
        logger.info('Calculating metafeatures for dataset %s...', dataset_id)
        if dataset_id not in metafeatures:
            dataset = load_dataset(dataset_id)
            if dataset:
                mfs = extract_metafeatures(dataset['X'], dataset['Y'])
                if mfs:
                    metafeatures[dataset_id] = mfs
                    logger.info('Metafeatures successfully calculated')
        else:
            logger.info('Using pre-calculated metafeatures')

    with open(PRECALCULATED_METAFEATURES_PATH, 'w') as fout:
        json.dump(metafeatures, fout, indent=4)

    return metafeatures

extract_metafeatures_all()
