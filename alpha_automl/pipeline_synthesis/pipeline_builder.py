import logging
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from alpha_automl.utils import create_object
from alpha_automl.primitive_loader import PRIMITIVE_TYPES

logger = logging.getLogger(__name__)


def change_default_hyperparams(pipeline_primitives):
    for i in range(len(pipeline_primitives)):
        if isinstance(pipeline_primitives[i], OneHotEncoder):
            pipeline_primitives[i] = OneHotEncoder(handle_unknown='ignore')
        elif isinstance(pipeline_primitives[i], SimpleImputer):
            pipeline_primitives[i] = SimpleImputer(strategy='most_frequent', error_on_no_input=False)


class BaseBuilder:

    def make_pipeline(self, primitives, automl_hyperparams, non_numeric_columns):
        pipeline_primitives = self.format_primitves(primitives, automl_hyperparams, non_numeric_columns)
        change_default_hyperparams(pipeline_primitives)
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

        for primitive in primitives:
            primitive_name = primitive
            if primitive_name.startswith('sklearn.'):  # It's a regular sklearn primitive
                primitive_object = create_object(primitive)
            else:
                primitive_object = automl_hyperparams['new_primitives'][primitive_name]['primitive_object']

            if primitive_name in PRIMITIVE_TYPES:
                primitive_type = PRIMITIVE_TYPES[primitive_name]
            else:
                primitive_type = automl_hyperparams['new_primitives'][primitive_name]['primitive_type']

            if primitive_type in non_numeric_columns:  # Add a transformer
                primitive_object = self.add_transformer(primitive_object, primitive_type, non_numeric_columns)

            pipeline_primitives.append((primitive_name, primitive_object))

        return pipeline_primitives

    def add_transformer(self, primitive_object, primitive_type, non_numeric_columns):
        column_transformer = None

        if primitive_type == 'TEXT_ENCODER':
            column_transformer = ColumnTransformer([(f'column_{column}', primitive_object, column)
                                                    for column in non_numeric_columns[primitive_type]],
                                                   remainder='passthrough')
        elif primitive_type == 'CATEGORICAL_ENCODER':
            column_transformer = ColumnTransformer([('categorical_encoder', primitive_object,
                                                     non_numeric_columns[primitive_type])],
                                                   remainder='passthrough')

        return column_transformer
