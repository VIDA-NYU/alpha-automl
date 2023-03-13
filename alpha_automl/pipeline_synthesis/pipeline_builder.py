import logging
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from alpha_automl.utils import create_object
from alpha_automl.primitive_loader import PRIMITIVE_TYPES

logger = logging.getLogger(__name__)

COLUMN_TRANSFORMER_ID = 'sklearn.compose.ColumnTransformer'


def change_default_hyperparams(primitive_object):
    if isinstance(primitive_object, OneHotEncoder):
        primitive_object.set_params(handle_unknown='ignore')
    elif isinstance(primitive_object, OrdinalEncoder):
        primitive_object.set_params(handle_unknown='use_encoded_value', unknown_value=-1)
    elif isinstance(primitive_object, SimpleImputer):
        primitive_object.set_params(strategy='most_frequent')


class BaseBuilder:

    def make_pipeline(self, primitives, automl_hyperparams, non_numeric_columns):
        pipeline_primitives = self.format_primitves(primitives, automl_hyperparams, non_numeric_columns)
        pipeline = self.make_linear_pipeline(pipeline_primitives)
        logger.info(f'New pipelined created:\n{pipeline}')

        return pipeline

    def make_linear_pipeline(self, pipeline_primitives):
        pipeline = Pipeline(pipeline_primitives)

        return pipeline

    def make_graph_pipeline(self, pipeline_primitives):
        pass

    def format_primitves(self, primitives, automl_hyperparams, non_numeric_columns):
        pipeline_primitives = []
        transformers = []

        for primitive in primitives:
            primitive_name = primitive
            if primitive_name.startswith('sklearn.'):  # It's a regular sklearn primitive
                primitive_object = create_object(primitive)
            else:
                primitive_object = automl_hyperparams['new_primitives'][primitive_name]['primitive_object']

            change_default_hyperparams(primitive_object)

            if primitive_name in PRIMITIVE_TYPES:
                primitive_type = PRIMITIVE_TYPES[primitive_name]
            else:
                primitive_type = automl_hyperparams['new_primitives'][primitive_name]['primitive_type']

            if primitive_type in non_numeric_columns:  # Add a transformer
                transformers += self.create_transformers(primitive_object, primitive_name, primitive_type, non_numeric_columns)
            else:
                if len(transformers) > 0:  # Add previous transformers
                    transformer_obj = ColumnTransformer(transformers, remainder='passthrough')
                    pipeline_primitives.append((COLUMN_TRANSFORMER_ID, transformer_obj))
                    transformers = []
                pipeline_primitives.append((primitive_name, primitive_object))

        return pipeline_primitives

    def create_transformers(self, primitive_object, primitive_name, primitive_type, non_numeric_columns):
        column_transformers = []

        if primitive_type == 'TEXT_ENCODER':
            column_transformers = [(f'{primitive_name}-{col_name}', primitive_object, col_index) for
                                   col_index, col_name in non_numeric_columns[primitive_type]]
        elif primitive_type == 'CATEGORICAL_ENCODER' or primitive_type == 'DATETIME_ENCODER':
            column_transformers = [(primitive_name, primitive_object, [col_index for col_index, _
                                                                       in non_numeric_columns[primitive_type]])]

        return column_transformers
