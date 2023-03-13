import logging
import datamart_profiler

logger = logging.getLogger(__name__)


def has_missing_values(X):
    return X.isnull().values.any()


def select_encoders(X):
    selected_encoders = {}
    mapping_encoders = {'http://schema.org/Enumeration': 'CATEGORICAL_ENCODER',
                        'http://schema.org/Text': 'TEXT_ENCODER',
                        'http://schema.org/DateTime': 'DATETIME_ENCODER'}

    data_profile = datamart_profiler.process_dataset(X, coverage=False)

    for index, column_profile in enumerate(data_profile['columns']):
        column_name = column_profile['name']
        semantic_types = column_profile['semantic_types'] if len(column_profile['semantic_types']) > 0 else [column_profile['structural_type']]

        for semantic_type in semantic_types:
            if semantic_type == 'http://schema.org/identifier':
                semantic_type = 'http://schema.org/Integer'
            elif semantic_type in {'http://schema.org/longitude', 'http://schema.org/latitude'}:
                semantic_type = 'http://schema.org/Float'

            if semantic_type in mapping_encoders:
                encoder = mapping_encoders[semantic_type]
                if encoder not in selected_encoders:
                    selected_encoders[encoder] = []
                selected_encoders[encoder].append((index, column_name))

    return selected_encoders
