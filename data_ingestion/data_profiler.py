import re
import os
import json
import logging
import datamart_profiler
import pandas as pd
from os.path import join, dirname
from d3m import index as d3m_index
from d3m.container import Dataset
from d3m.metadata.problem import TaskKeyword
from d3m.container.dataset import D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES, D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES
from alphad3m.utils import need_denormalize

logger = logging.getLogger(__name__)


def select_annotated_feature_types(dataset_doc_path):
    feature_types = {}

    with open(dataset_doc_path) as fin:
        dataset_doc = json.load(fin)

    try:
        for data_res in dataset_doc['dataResources']:
            if data_res['resID'] == 'learningData' and data_res['resType'] == 'table':
                for column in data_res['columns']:
                    if column['colType'] != 'unknown':
                        semantic_type = [D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[column['colType']]]
                        role = D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES[column['role'][0]]
                        if 'refersTo' in column:  # It's a foreign key
                            role = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
                        feature_types[column['colName']] = (role, semantic_type, column['colIndex'])
    except:
        logger.exception('Error reading the type of attributes')

    logger.info('Features with annotated types: [%s]', ', '.join(feature_types.keys()))

    return feature_types


def select_unkown_feature_types(csv_path, annotated_features):
    all_features = pd.read_csv(csv_path, index_col=0, nrows=0).columns
    unkown_feature_types = []

    for feature_name in all_features:
        match = re.match('(.*)\.\d$', feature_name)  # To verify if pandas renames features with same names
        if match:
            feature_name = match.group(1)
        if feature_name not in annotated_features:
            unkown_feature_types.append(feature_name)

    logger.info('Features with unknown types: [%s]', ', '.join(unkown_feature_types))

    return unkown_feature_types


def select_identified_feature_types(metadata, unkown_feature_types, target_names):
    identified_feature_types = {}

    for index, item in enumerate(metadata['columns']):
        feature_name = item['name']
        if feature_name in unkown_feature_types:
            semantic_types = item['semantic_types'] if len(item['semantic_types']) > 0 else [item['structural_type']]
            d3m_semantic_types = []
            for semantic_type in semantic_types:
                if semantic_type == 'http://schema.org/Enumeration':  # Changing to D3M format
                    semantic_type = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
                elif semantic_type == 'http://schema.org/identifier':  # Changing to D3M format
                    semantic_type = 'http://schema.org/Integer'
                elif semantic_type in {'http://schema.org/longitude', 'http://schema.org/latitude'}:
                    semantic_type = 'http://schema.org/Float'
                d3m_semantic_types.append(semantic_type)

            role = 'https://metadata.datadrivendiscovery.org/types/Attribute'
            if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in d3m_semantic_types:
                role = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
            elif feature_name in target_names:
                role = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
            identified_feature_types[feature_name] = (role, d3m_semantic_types, index)

    logger.info('Inferred feature types:\n%s',
                '\n'.join(['%s = [%s]' % (k, ', '.join([i for i in v[1]])) for k, v in identified_feature_types.items()]))

    return identified_feature_types


def get_extra_metadata(metadata, exclude_columns):
    extra_metadata = {'missing_values': False, 'large_rows': False, 'large_columns': False, 'exclude_columns': []}

    for index, item in enumerate(metadata['columns']):
        if 'missing_values_ratio' in item and item['name'] not in exclude_columns:
            extra_metadata['missing_values'] = True

        if 'structural_type' in item and 'https://metadata.datadrivendiscovery.org/types/MissingData' in item['structural_type']:
            extra_metadata['exclude_columns'].append(index)

    if 'nb_rows' in metadata and metadata['nb_rows'] > 1000000:
        extra_metadata['large_rows'] = True
        extra_metadata['sample_size'] = metadata['nb_profiled_rows']

    if 'nb_columns' in metadata and metadata['nb_columns'] > 1000:
        extra_metadata['large_columns'] = True

    return extra_metadata


def profile_data(dataset_uri, targets):
    dataset_doc_path = dataset_uri[7:]
    csv_path = join(dirname(dataset_doc_path), 'tables', 'learningData.csv')
    if need_denormalize(dataset_doc_path):
        csv_path = denormalize_dataset(dataset_uri)

    target_names = [x[1] for x in targets]
    annotated_feature_types = select_annotated_feature_types(dataset_doc_path)
    unkown_feature_types = select_unkown_feature_types(csv_path, annotated_feature_types.keys())
    metadata = datamart_profiler.process_dataset(csv_path, coverage=False)
    identified_feature_types = select_identified_feature_types(metadata, unkown_feature_types, target_names)
    only_attribute_types = set()
    semantictypes_by_index = {}

    for role, semantic_types, index in annotated_feature_types.values():
        if role == 'https://metadata.datadrivendiscovery.org/types/Attribute':
            only_attribute_types.update(semantic_types)

    for role, semantic_types, index in identified_feature_types.values():
        if role == 'https://metadata.datadrivendiscovery.org/types/Attribute':
            only_attribute_types.update(semantic_types)

        for semantic_type in set(semantic_types + [role]):
            if semantic_type not in semantictypes_by_index:
                semantictypes_by_index[semantic_type] = []
            semantictypes_by_index[semantic_type].append(index)

    features_metadata = {'semantictypes_indices': semantictypes_by_index, 'only_attribute_types': only_attribute_types}
    extra_metadata = get_extra_metadata(metadata, target_names)
    features_metadata.update(extra_metadata)

    return features_metadata


def denormalize_dataset(dataset_uri):
    logger.info('Denormalizing the dataset')
    csv_path = None
    dataset = Dataset.load(dataset_uri)

    try:
        primitive_class = d3m_index.get_primitive('d3m.primitives.data_transformation.denormalize.Common')
        primitive_hyperparams = primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        primitive_denormalize = primitive_class(hyperparams=primitive_hyperparams.defaults())
        primitive_output = primitive_denormalize.produce(inputs=dataset).value

        primitive_class = d3m_index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
        primitive_hyperparams = primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        primitive_dataframe = primitive_class(hyperparams=primitive_hyperparams.defaults())
        primitive_output = primitive_dataframe.produce(inputs=primitive_output).value

        csv_path = join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'denormalized_dataset.csv')
        primitive_output.to_csv(csv_path)
    except:
        csv_path = join(dirname(dataset_uri[7:]),  'tables', 'learningData.csv')
        logger.exception('Error denormalizing dataset, using only learningData.csv file')

    return csv_path


def get_privileged_data(problem, task_keywords):
    privileged_data = []
    if TaskKeyword.LUPI in task_keywords and 'privileged_data' in problem['inputs'][0]:
        for column in problem['inputs'][0]['privileged_data']:
            privileged_data.append(column['column_index'])

    return privileged_data


def select_encoders(feature_types):
    encoders = []
    mapping_feature_types = {'https://metadata.datadrivendiscovery.org/types/CategoricalData': 'CATEGORICAL_ENCODER',
                             'http://schema.org/Text': 'TEXT_FEATURIZER', 'http://schema.org/DateTime': 'DATETIME_ENCODER'}

    for features_type in feature_types:
        if features_type in mapping_feature_types:
            encoders.append(mapping_feature_types[features_type])

    return sorted(encoders, reverse=True)
