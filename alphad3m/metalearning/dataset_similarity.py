import logging
import math
import hashlib
import datamart_profiler
from metalearn import Metafeatures
from alphad3m.metalearning.resource_builder import load_precalculated_data
from alphad3m.metalearning.dataset_profiler import extract_dataprofiles, extract_metafeatures
from sklearn.metrics.pairwise import cosine_similarity
from d3m.metadata.problem import TaskKeywordBase

logger = logging.getLogger(__name__)


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
