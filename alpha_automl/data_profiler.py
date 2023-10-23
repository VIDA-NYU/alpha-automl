import logging
import datamart_profiler

CATEGORICAL_COLUMN = 'http://schema.org/Enumeration'
DATETIME_COLUMN = 'http://schema.org/DateTime'
TEXT_COLUMN = 'http://schema.org/Text'
EMPTY_COLUMN = 'https://metadata.datadrivendiscovery.org/types/MissingData'
IMAGE_COLUMN = 'https://schema.org/ImageObject'


logger = logging.getLogger(__name__)


def profile_data(X):
    metadata = {'nonnumeric_columns': {}, 'useless_columns': [], 'missing_values': False}
    mapping_encoders = {CATEGORICAL_COLUMN: 'CATEGORICAL_ENCODER', DATETIME_COLUMN: 'DATETIME_ENCODER',
                        TEXT_COLUMN: 'TEXT_ENCODER', IMAGE_COLUMN: 'IMAGE_ENCODER'}

    profiled_data = datamart_profiler.process_dataset(X, coverage=False, indexes=False)

    for index_column, profiled_column in enumerate(profiled_data['columns']):
        column_name = profiled_column['name']
        if EMPTY_COLUMN == profiled_column['structural_type']:
            metadata['useless_columns'].append((index_column, column_name))
            continue

        if CATEGORICAL_COLUMN in profiled_column['semantic_types']:
            column_type = mapping_encoders[CATEGORICAL_COLUMN]
            add_nonnumeric_column(column_type, metadata, index_column, column_name)

        elif DATETIME_COLUMN in profiled_column['semantic_types']:
            column_type = mapping_encoders[DATETIME_COLUMN]
            add_nonnumeric_column(column_type, metadata, index_column, column_name)

        elif TEXT_COLUMN == profiled_column['structural_type']:
            samples = X[column_name].dropna().sample(5)
            if samples.apply(lambda x: x.endswith(('jpg', 'png', 'jpeg', 'gif'))).all():
                column_type = mapping_encoders[IMAGE_COLUMN]
            else:
                column_type = mapping_encoders[TEXT_COLUMN]
            add_nonnumeric_column(column_type, metadata, index_column, column_name)

        if 'missing_values_ratio' in profiled_column:
            metadata['missing_values'] = True

    logger.debug(f'Results of profiling data: non-numeric features = {str(metadata["nonnumeric_columns"].keys())}, '
                f'useless columns = {str(metadata["useless_columns"])}, '
                f'missing values = {str(metadata["missing_values"])}')

    return metadata


def add_nonnumeric_column(column_type, metadata, index_column, column_name):
    if column_type not in metadata['nonnumeric_columns']:
        metadata['nonnumeric_columns'][column_type] = []

    metadata['nonnumeric_columns'][column_type].append((index_column, column_name))
