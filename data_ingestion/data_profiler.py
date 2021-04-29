import re
import logging
import datamart_profiler
import pandas as pd
from d3m.container.dataset import D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES, D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES

logger = logging.getLogger(__name__)


def select_annotated_feature_types(dataset_doc):
    feature_types = {}
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
    all_features = pd.read_csv(csv_path).columns
    unkown_feature_types = []

    for feature_name in all_features:
        match = re.match('(.*)\.\d$', feature_name)  # To verify if pandas renames features with same names
        if match:
            feature_name = match.group(1)
        if feature_name not in annotated_features:
            unkown_feature_types.append(feature_name)

    logger.info('Features with unknown types: [%s]', ', '.join(unkown_feature_types))

    return unkown_feature_types


def indentify_feature_types(csv_path, unkown_feature_types, target_names):
    metadata = datamart_profiler.process_dataset(csv_path)
    inferred_feature_types = {}
    has_missing_values = False

    for index, item in enumerate(metadata['columns']):
        feature_name = item['name']
        if feature_name in unkown_feature_types:
            semantic_types = item['semantic_types'] if len(item['semantic_types']) > 0 else [item['structural_type']]
            d3m_semantic_types = []
            for semantic_type in semantic_types:
                if semantic_type == 'http://schema.org/Enumeration':  # Changing to D3M format
                    semantic_type = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
                elif semantic_type == 'http://schema.org/identifier':  # Changing to D3M format
                    #semantic_type = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
                    semantic_type = 'http://schema.org/Integer'
                elif semantic_type == 'https://metadata.datadrivendiscovery.org/types/MissingData':
                    semantic_type = 'http://schema.org/Text'
                d3m_semantic_types.append(semantic_type)

            role = 'https://metadata.datadrivendiscovery.org/types/Attribute'
            if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in d3m_semantic_types:
                role = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
            elif feature_name in target_names:
                role = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
            inferred_feature_types[feature_name] = (role, d3m_semantic_types, index)
        if 'missing_values_ratio' in item and feature_name not in target_names:
            has_missing_values = True

    logger.info('Inferred feature types:\n%s',
                '\n'.join(['%s = [%s]' % (k, ', '.join([i for i in v[1]])) for k, v in inferred_feature_types.items()]))

    return inferred_feature_types, has_missing_values


def profile_data(csv_path, target_names, dataset_doc):
    annotated_feature_types = select_annotated_feature_types(dataset_doc)
    unkown_feature_types = select_unkown_feature_types(csv_path, annotated_feature_types.keys())
    inferred_feature_types, has_missing_values = indentify_feature_types(csv_path, unkown_feature_types, target_names)
    only_attribute_types = set()
    semantictypes_by_index = {}

    for role, semantic_types, index in annotated_feature_types.values():
        if role == 'https://metadata.datadrivendiscovery.org/types/Attribute':
            only_attribute_types.update(semantic_types)

    for role, semantic_types, index in inferred_feature_types.values():
        if role == 'https://metadata.datadrivendiscovery.org/types/Attribute':
            only_attribute_types.update(semantic_types)

        for semantic_type in set(semantic_types + [role]):
            if semantic_type not in semantictypes_by_index:
                semantictypes_by_index[semantic_type] = []
            semantictypes_by_index[semantic_type].append(index)
    features_metadata = {'semantictypes_indices': semantictypes_by_index, 'only_attribute_types': only_attribute_types,
                         'use_imputer': has_missing_values}

    return features_metadata
