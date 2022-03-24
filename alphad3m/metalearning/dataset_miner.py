import re
import json
import logging
import math
import hashlib
import datamart_profiler
import pandas as pd
from os.path import join, dirname, exists
from metalearn import Metafeatures
from alphad3m.metalearning.database import load_metalearningdb
from sklearn.metrics.pairwise import cosine_similarity
from d3m.metadata.problem import TaskKeywordBase

logger = logging.getLogger(__name__)

DATASETS_FOLDER_PATH = '/Users/rlopez/D3M/datasets/seed_datasets_current/'
PRECALCULATED_METAFEATURES_PATH = join(dirname(__file__), '../resource/precalculated_metafeatures.json')
PRECALCULATED_DATAPROFILES_PATH = join(dirname(__file__), '../resource/precalculated_dataprofiles.json')
PRECALCULATED_TASKKEYWORDS_PATH = join(dirname(__file__), '../resource/precalculated_taskkeywords.json')


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


def get_unique_datasets():
    pipeline_runs = load_metalearningdb()
    datasets = set()

    for pipeline_run in pipeline_runs:
        problem_id = pipeline_run['problem']['id']
        dataset_id = get_dataset_id(problem_id)
        datasets.add(dataset_id)

    logger.info('Found %d unique datasets', len(datasets))

    return sorted(datasets)


def get_dataset_id(problem_id):
    # Remove suffixes 'TRAIN' and 'problem' from the dataset name
    dataset_id = re.sub('_TRAIN$', '', problem_id)
    dataset_id = re.sub('_problem$', '', dataset_id)

    return dataset_id


def load_task_info(dataset_id):
    possible_names = [join(DATASETS_FOLDER_PATH, dataset_id), join(DATASETS_FOLDER_PATH, dataset_id + '_MIN_METADATA'),
                      join(DATASETS_FOLDER_PATH, dataset_id.replace('_MIN_METADATA', ''))]
    # All possible names of the datasets on disk, with/without the suffix 'MIN_METADATA'

    for dataset_folder_path in possible_names:
        if exists(dataset_folder_path):
            break
    else:
        raise FileNotFoundError('Dataset %s not found' % dataset_id)

    dataset_path = join(dataset_folder_path, 'TRAIN/dataset_TRAIN/tables/learningData.csv')
    problem_path = join(dataset_folder_path, 'TRAIN/problem_TRAIN/problemDoc.json')

    with open(problem_path) as fin:
        problem_doc = json.load(fin)
        task_keywords = problem_doc['about']['taskKeywords']
        target_column = problem_doc['inputs']['data'][0]['targets'][0]['colName']

    return dataset_path, target_column, task_keywords


def extract_metafeatures(dataset_path, target_column):
    data = pd.read_csv(dataset_path)
    Y = data[target_column]
    Y = pd.Series([str(i) for i in Y], name=target_column)  # Cast to string to get metalearn lib working correctly
    X = data.drop(columns=[target_column])
    metafeatures = Metafeatures()
    mfs = metafeatures.compute(X, Y, seed=0, timeout=300)

    return mfs


def extract_dataprofiles(dataset_path, target_column, ignore_target_column=False):
    metadata = datamart_profiler.process_dataset(dataset_path, coverage=False)
    feature_types = set()
    missing_values = False
    for item in metadata['columns']:
        if ignore_target_column and item['name'] == target_column:
            continue
        identified_types = item['semantic_types'] if len(item['semantic_types']) > 0 else [item['structural_type']]
        for feature_type in identified_types:
            feature_types.add(feature_type)

        if 'missing_values_ratio' in item and item['name'] != target_column:
            missing_values = True

    dps = {'feature_types': sorted(feature_types), 'missing_values': missing_values}

    return dps


def extract_metafeatures_mldb():
    datasets = get_unique_datasets()
    metafeatures = load_precalculated_data('metafeatures')

    for dataset_id in datasets:
        logger.info('Calculating metafeatures for dataset %s...', dataset_id)
        if dataset_id not in metafeatures:
            try:
                dataset_path, target_column, _ = load_task_info(dataset_id)
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


def extract_dataprofiles_mldb():
    datasets = get_unique_datasets()
    dataprofiles = load_precalculated_data('dataprofiles')

    for dataset_id in datasets:
        logger.info('Calculating data profiles for dataset %s...', dataset_id)
        if dataset_id not in dataprofiles:
            try:
                dataset_path, target_column, _ = load_task_info(dataset_id)
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


def extract_taskkeywords_mldb():
    datasets = get_unique_datasets()
    task_keywords = load_precalculated_data('task_keywords')

    for dataset_id in datasets:
        logger.info('Calculating task keywords for dataset %s...', dataset_id)
        if dataset_id not in task_keywords:
            try:
                _, _, keywords = load_task_info(dataset_id)
                task_keywords[dataset_id] = {'task_keywords': keywords}
                logger.info('Task keywords successfully calculated for dataset %s', dataset_id)
                with open(PRECALCULATED_TASKKEYWORDS_PATH, 'w') as fout:
                    json.dump(task_keywords, fout, indent=4, sort_keys=True)
            except Exception as e:
                logger.error(str(e))
        else:
            logger.info('Using pre-calculated task keywords for dataset %s', dataset_id)

    return task_keywords


def create_metafeatures_vector(metafeatures, metafeature_indices):
    vector = []

    for metafeatures_id in metafeature_indices:
        value = metafeatures[metafeatures_id]['value']
        if isinstance(value, str):
            value = int(hashlib.sha256(value.encode('utf-8')).hexdigest(), 16) % 256
        elif math.isnan(value) or math.isinf(value):
            value = 0
        vector.append(value)

    return vector


def create_dataprofiles_vector(dataprofiles, dataprofile_indices):
    vector = []

    for dataprofile_id in dataprofile_indices:
        if dataprofile_id in dataprofiles['feature_types']:
            value = 1
        else:
            value = 0
        vector.append(value)

    value = 1 if dataprofiles['missing_values'] else 0
    vector.append(value)  # Add an extra value corresponding to the missing values data

    return vector


def create_taskkeywords_vector(task_keywords, taskkeyword_indices):
    vector = []

    for taskkeyword_id in taskkeyword_indices:
        if taskkeyword_id in task_keywords['task_keywords']:
            value = 1
        else:
            value = 0
        vector.append(value)

    return vector


def create_metafeatures_vectors_mldb(metafeature_indices):
    vectors = {}
    metafeature_datasets = load_precalculated_data('metafeatures')

    for id_dataset, metafeatures in metafeature_datasets.items():
        vector = create_metafeatures_vector(metafeatures, metafeature_indices)
        vectors[id_dataset] = vector

    return vectors


def create_dataprofiles_vectors_mldb(dataprofile_indices):
    vectors = {}
    dataprofile_datasets = load_precalculated_data('dataprofiles')

    for id_dataset, dataprofiles in dataprofile_datasets.items():
        vector = create_dataprofiles_vector(dataprofiles, dataprofile_indices)
        vectors[id_dataset] = vector

    return vectors


def create_taskkeywords_vectors_mldb(taskkeyword_indices):
    vectors = {}
    taskkeyword_datasets = load_precalculated_data('task_keywords')

    for id_dataset, task_keywords in taskkeyword_datasets.items():
        vector = create_taskkeywords_vector(task_keywords, taskkeyword_indices)
        vectors[id_dataset] = vector

    return vectors


def load_metafeatures_vectors(dataset_path, target_column):
    mfs = extract_metafeatures(dataset_path, target_column)
    metafeature_indices = Metafeatures.list_metafeatures(group='all')
    target_metafeatures_vector = create_metafeatures_vector(mfs, metafeature_indices)
    metalearningdb_vectors = create_metafeatures_vectors_mldb(metafeature_indices)

    return metalearningdb_vectors, target_metafeatures_vector


def load_profiles_vectors(dataset_path, target_column):
    dps = extract_dataprofiles(dataset_path, target_column)
    dataprofile_indices = [v for k, v in datamart_profiler.types.__dict__.items() if not k.startswith('_')]
    target_dataprofile_vector = create_dataprofiles_vector(dps, dataprofile_indices)
    metalearningdb_vectors = create_dataprofiles_vectors_mldb(dataprofile_indices)

    return metalearningdb_vectors, target_dataprofile_vector


def load_taskkeyword_vectors(task_keywords):
    taskkeyword_indices = sorted([keyword for keyword in TaskKeywordBase.get_map().keys() if keyword is not None])
    target_dataprofile_vector = create_taskkeywords_vector({'task_keywords': task_keywords}, taskkeyword_indices)
    metalearningdb_vectors = create_taskkeywords_vectors_mldb(taskkeyword_indices)

    return metalearningdb_vectors, target_dataprofile_vector


def calculate_similarity(metalearningdb_vectors, target_vector, threshold):
    similar_datasets = {}
    for id_dataset, vector in metalearningdb_vectors.items():
        similarity = cosine_similarity([target_vector], [vector]).flat[0]
        if similarity > threshold:
            similar_datasets[id_dataset] = round(similarity, 5)

    return similar_datasets


def similarity_repr(dataset_similarities):
    similarity_string = []

    for dataset_similarity in sorted(dataset_similarities.items(), key=lambda x: x[1], reverse=True):
        pretty_string = '%s=%.2f' % dataset_similarity
        similarity_string.append(pretty_string)

    return ', '.join(similarity_string)


def get_similar_datasets(mode, dataset_path, target_column, task_keywords, threshold=0.8, combined=False):
    vectors_taskkeywords, target_vector_taskkeywords = load_taskkeyword_vectors(task_keywords)

    if mode == 'metafeatures':
        vectors_dataset, target_vector_dataset = load_metafeatures_vectors(dataset_path, target_column)
    elif mode == 'dataprofiles':
        vectors_dataset, target_vector_dataset = load_profiles_vectors(dataset_path, target_column)
    else:
        raise ValueError('Unknown mode "%s" to load data' % mode)

    if combined:
        # Concatenate the vectors of the dataset and task keywords
        for id_dataset in vectors_dataset:
            vectors_dataset[id_dataset] += vectors_taskkeywords[id_dataset]
        target_vector_dataset += target_vector_taskkeywords
        similar_datasets = calculate_similarity(vectors_dataset, target_vector_dataset, threshold)
        logger.info('Similar datasets found using both information:\n%s', similarity_repr(similar_datasets))
    else:
        similar_datasets = calculate_similarity(vectors_taskkeywords, target_vector_taskkeywords, threshold)
        logger.info('Similar datasets found using task_keywords features:\n%s', similarity_repr(similar_datasets))
        vectors_dataset = {k: vectors_dataset[k] for k in similar_datasets}  # Use only the similar datasets
        similar_datasets = calculate_similarity(vectors_dataset, target_vector_dataset, threshold)
        logger.info('Similar datasets found using %s features:\n%s', mode, similarity_repr(similar_datasets))

    return similar_datasets
