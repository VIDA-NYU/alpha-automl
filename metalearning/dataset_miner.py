import re
import json
import logging
import math
import hashlib
import pandas as pd
from os.path import join, dirname, exists
from metalearn import Metafeatures
from alphad3m.metalearning.database import load_metalearningdb
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DATASETS_FOLDER_PATH = '/Users/rlopez/D3M/datasets/seed_datasets_current/'
PRECALCULATED_METAFEATURES_PATH = join(dirname(__file__), '../../resource/precalculated_metafeatures.json')


def load_precalculated_metafeatures():
    if exists(PRECALCULATED_METAFEATURES_PATH):
        with open(PRECALCULATED_METAFEATURES_PATH) as fin:
            return json.load(fin)

    return {}


def get_unique_datasets():
    pipeline_runs = load_metalearningdb()
    datasets = set()

    for pipeline_run in pipeline_runs:
        problem_id = pipeline_run['problem']['id']
        dataset_id = get_dataset_id(problem_id)
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
    Y = pd.Series([str(i) for i in Y], name=target_name)  # Cast to string to get metalearn lib working correctly
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


def extract_metafeatures(X, Y):
    metafeatures = Metafeatures()
    mfs = None
    try:
        mfs = metafeatures.compute(X, Y, seed=0, timeout=300)
    except:
        logger.error('Calculating metafeatures')

    return mfs


def create_metafeatures_vector(metafeatures):
    metafeatures_ids = Metafeatures.list_metafeatures(group='all')
    vector = []

    for metafeatures_id in metafeatures_ids:
        value = metafeatures[metafeatures_id]['value']
        if isinstance(value, str):
            value = int(hashlib.sha256(value.encode('utf-8')).hexdigest(), 16) % 256
        elif math.isnan(value) or math.isinf(value):
            value = 0
        vector.append(value)

    return vector


def extract_metafeatures_metalearningdb():
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
                    with open(PRECALCULATED_METAFEATURES_PATH, 'w') as fout:
                        json.dump(metafeatures, fout, indent=4)
        else:
            logger.info('Using pre-calculated metafeatures')

    return metafeatures


def create_metalearningdb_vectors():
    vectors = {}
    metafeature_datasets = load_precalculated_metafeatures()

    for id_dataset, metafeatures in metafeature_datasets.items():
        vector = create_metafeatures_vector(metafeatures)
        vectors[id_dataset] = vector

    return vectors


def get_similar_datasets(dataset_folder, threshold=0.8):
    X, Y = get_X_Y(dataset_folder)
    mfs = extract_metafeatures(X, Y)
    target_metafeatures_vector = create_metafeatures_vector(mfs)
    metalearningdb_vectors = create_metalearningdb_vectors()
    similar_datasets = {}

    for id_dataset, vector in metalearningdb_vectors.items():
        similarity = cosine_similarity([target_metafeatures_vector], [vector]).flat[0]
        if similarity > threshold:
            similar_datasets[id_dataset] = round(similarity, 5)
    logger.info('Found %d similar datasets', len(similar_datasets))
    logger.info('Similar datasets:\n%s', str(sorted(similar_datasets.items(), key=lambda x: x[1], reverse=True)))

    return similar_datasets


def get_dataset_id(problem_id):
    dataset_id = re.sub('_TRAIN$', '', problem_id)
    dataset_id = re.sub('_problem$', '', dataset_id)

    return dataset_id
