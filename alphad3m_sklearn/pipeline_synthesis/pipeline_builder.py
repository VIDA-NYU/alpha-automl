import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from alphad3m_sklearn.utils import create_object

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

    def make_pipeline(self, primitives):
        pipeline_primitives = self.format_primitves(primitives)
        change_default_hyperparams(pipeline_primitives)
        pipeline = None

        if is_linear_pipeline(pipeline_primitives):
            pipeline = self.make_linear_pipeline(pipeline_primitives)
        else:
            pass # TODO: Use Column Transformer

        logger.info(f'New pipelined created:\n{pipeline}')
        return pipeline

    def make_linear_pipeline(self, pipeline_primitives):
        pipeline = Pipeline(pipeline_primitives)

        return pipeline

    def make_graph_pipeline(self, pipeline_primitives):
        pass

    def format_primitves(self, primitives):
        pipeline_primitives = []

        for primitive in primitives:
            if isinstance(primitive, str):
                primitive_name = primitive
                primitive_object = create_object(primitive)
            elif isinstance(primitive, tuple):
                primitive_name = primitive[0]
                primitive_object = primitive[1]
            else:
                primitive_name = str(primitive)
                primitive_object = primitive

            pipeline_primitives.append((primitive_name, primitive_object))

        return pipeline_primitives
