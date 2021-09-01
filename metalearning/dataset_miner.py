import re
import json
import logging
import pandas as pd
from os.path import join, isfile
from metalearn import Metafeatures
from alphad3m.metalearning.database import load_metalearningdb

logger = logging.getLogger(__name__)
DATASETS_FOLDER_PATH = '/Users/rlopez/D3M/datasets/training_datasets/seed_datasets_archive'


def get_unique_datasets(remove_sufix=True):
    pipeline_runs = load_metalearningdb()
    datasets = set()

    for pipeline_run in pipeline_runs:
        dataset_id = pipeline_run['problem']['id']
        if remove_sufix:
            dataset_id = re.sub('_TRAIN$', '', dataset_id)
            dataset_id = re.sub('_problem$', '', dataset_id)
        datasets.add(dataset_id)

    logger.info('Found %d unique datasets', len(datasets))

    return datasets


def get_target_name(dataset_id):
    problem_path = join(DATASETS_FOLDER_PATH, dataset_id, 'TRAIN/problem_TRAIN/problemDoc.json')

    with open(problem_path) as fin:
        problem_doc = json.load(fin)
        target_name = problem_doc['inputs']['data'][0]['targets'][0]['colName']

        return target_name


def load_dataset(dataset_id):
    csv_file = join(DATASETS_FOLDER_PATH, dataset_id, 'TRAIN/dataset_TRAIN/tables/learningData.csv')

    if isfile(csv_file):
        try:
            data = pd.read_csv(csv_file)
            target_name = get_target_name(dataset_id)
            Y = data[target_name]
            X = data.drop(columns=[target_name])
            return {'X': X, 'Y': Y}
        except:
            logger.error('Error reading dataset')
            return None
    logger.error('File %s does not exist', csv_file)
    return None


def extract_metafeatures(X, Y):
    metafeatures = Metafeatures()
    mfs = metafeatures.compute(X, Y, timeout=10)

    return mfs


def extract_metafeatures_all():
    datasets = get_unique_datasets()
    precalculated_metafeatures = {}

    for dataset_id in datasets:
        print(dataset_id)
        if dataset_id in precalculated_metafeatures:
            pass
        else:
            dataset = load_dataset(dataset_id)
            if dataset:
                try:
                    mfs = extract_metafeatures(dataset['X'], dataset['Y'])
                except:
                    logger.error('Error calculating metafeatures')

extract_metafeatures_all()
