import logging
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from alpha_automl.utils import create_object

logger = logging.getLogger(__name__)


def change_default_hyperparams(pipeline_primitives):
    for pipeline_primitive in pipeline_primitives:
        if isinstance(pipeline_primitive, OneHotEncoder):
            pipeline_primitive = OneHotEncoder(handle_unknown='ignore')
        elif isinstance(pipeline_primitive, SimpleImputer):
            pipeline_primitive = SimpleImputer(strategy='most_frequent', error_on_no_input=False)

def is_linear_pipeline(pipeline_primitives):
    return True


class BaseBuilder:

    def make_pipeline(self, primitives, automl_hyperparams):
        pipeline_primitives = self.format_primitves(primitives, automl_hyperparams)
        change_default_hyperparams(pipeline_primitives)
        pipeline = None

        if is_linear_pipeline(pipeline_primitives):
            pipeline = self.make_linear_pipeline(pipeline_primitives)
        else:
            pass  # TODO: Use Column Transformer

        logger.info(f'New pipelined created:\n{pipeline}')
        return pipeline

    def make_linear_pipeline(self, pipeline_primitives):
        pipeline = Pipeline(pipeline_primitives)

        return pipeline

    def make_graph_pipeline(self, pipeline_primitives):
        pass

    def format_primitves(self, primitives, automl_hyperparams):
        pipeline_primitives = []

        for primitive in primitives:
            primitive_name = primitive
            if primitive_name.startswith('sklearn.'):  # It's a regular sklearn class
                primitive_object = create_object(primitive)
            else:
                primitive_object = deepcopy(automl_hyperparams['new_primitives'][primitive_name]['primitive_object'])

            pipeline_primitives.append((primitive_name, primitive_object))

        return pipeline_primitives
