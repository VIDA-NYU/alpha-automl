import logging
# from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from alpha_automl.utils import create_object, COLUMN_TRANSFORMER_ID, COLUMN_SELECTOR_ID, NATIVE_PRIMITIVE, \
    ADDED_PRIMITIVE
from alpha_automl.primitive_loader import PRIMITIVE_TYPES

logger = logging.getLogger(__name__)

SEMI_CLASSIFIER_PARAMS = {
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis": {},
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis": {},
    "sklearn.ensemble.BaggingClassifier": {},
    "sklearn.ensemble.ExtraTreesClassifier": {},
    "sklearn.ensemble.GradientBoostingClassifier": {},
    "sklearn.ensemble.RandomForestClassifier": {},
    "sklearn.naive_bayes.BernoulliNB": {},
    "sklearn.naive_bayes.GaussianNB": {},
    "sklearn.naive_bayes.MultinomialNB": {},
    "sklearn.neighbors.KNeighborsClassifier": {},
    "sklearn.linear_model.LogisticRegression": {},
    "sklearn.linear_model.PassiveAggressiveClassifier": {},
    "sklearn.linear_model.SGDClassifier": dict(alpha=1e-5, penalty="l2", loss="log_loss"),
    "sklearn.svm.LinearSVC": {},
    "sklearn.svm.SVC": {},
    "sklearn.tree.DecisionTreeClassifier": {},
    "xgboost.XGBClassifier": {},
    "lightgbm.LGBMClassifier": dict(verbose=-1),
}


EXTRA_PARAMS = {
    "lightgbm.LGBMClassifier": dict(verbose=-1),
    "lightgbm.LGBMRegressor": dict(verbose=-1),
}


def change_default_hyperparams(primitive_object):
    if isinstance(primitive_object, OneHotEncoder):
        primitive_object.set_params(handle_unknown='ignore')
    elif isinstance(primitive_object, OrdinalEncoder):
        primitive_object.set_params(handle_unknown='use_encoded_value', unknown_value=-1)
    elif isinstance(primitive_object, SimpleImputer):
        primitive_object.set_params(strategy='most_frequent', keep_empty_features=True)


class BaseBuilder:

    def __init__(self, metadata, automl_hyperparams):
        self.metadata = metadata
        self.automl_hyperparams = automl_hyperparams
        self.all_primitives = {}

        for primitive_name in PRIMITIVE_TYPES:
            primitive_type = PRIMITIVE_TYPES[primitive_name]
            self.all_primitives[primitive_name] = {'type': primitive_type, 'origin': NATIVE_PRIMITIVE}

        for primitive_name in automl_hyperparams['new_primitives']:
            primitive_type = automl_hyperparams['new_primitives'][primitive_name]['primitive_type']
            self.all_primitives[primitive_name] = {'type': primitive_type, 'origin': ADDED_PRIMITIVE}

    def make_pipeline(self, primitives):
        if self.all_primitives[primitives[-1]]['type'] == 'ENSEMBLER':
            ensembler_name = primitives[-1]
            pipeline_primitives = self.make_primitive_objects(primitives[:-1])
            pipeline = self.make_linear_pipeline(pipeline_primitives)
            ensembler_obj = create_object(ensembler_name, {'estimator': pipeline})
            pipeline = Pipeline([(ensembler_name, ensembler_obj)])
        else:
            pipeline_primitives = self.make_primitive_objects(primitives)
            pipeline = self.make_linear_pipeline(pipeline_primitives)

        logger.debug(f'New pipelined created:\n{pipeline}')

        return pipeline

    def make_linear_pipeline(self, pipeline_primitives):
        pipeline = Pipeline(pipeline_primitives)

        return pipeline

    def make_graph_pipeline(self, pipeline_primitives):
        pass

    def make_primitive_objects(self, primitives):
        pipeline_primitives = []
        transformers = []
        nonnumeric_columns = self.metadata['nonnumeric_columns']
        useless_columns = self.metadata['useless_columns']

        if len(useless_columns) > 0 and len(nonnumeric_columns) == 0:  # Add the transformer to the first step
            selector = (COLUMN_SELECTOR_ID, 'drop', [col_index for col_index, _ in useless_columns])
            transformer_obj = ColumnTransformer([selector], remainder='passthrough')
            pipeline_primitives.append((COLUMN_TRANSFORMER_ID, transformer_obj))

        for primitive in primitives:
            primitive_name = primitive
            primitive_type = self.all_primitives[primitive_name]['type']

            # Make sure that SEMISUPERVISED_CLASSIFIER primitive has a classifier primitive behind
            if primitive_type == 'SEMISUPERVISED_CLASSIFIER':
                if self.all_primitives[primitives[-1]]['type'] != 'CLASSIFIER':
                    return
                classifier_obj = create_object(primitives[-1], SEMI_CLASSIFIER_PARAMS[primitives[-1]])
                primitive_object = create_object(primitive_name, {'base_estimator': classifier_obj})
            elif self.all_primitives[primitive_name]['origin'] == NATIVE_PRIMITIVE:  # It's an installed primitive
                if primitive in EXTRA_PARAMS:
                    primitive_object = create_object(primitive, EXTRA_PARAMS[primitive])
                else:
                    primitive_object = create_object(primitive)
            else:
                primitive_object = self.automl_hyperparams['new_primitives'][primitive_name]['primitive_object']

            change_default_hyperparams(primitive_object)

            if primitive_type in nonnumeric_columns:  # Create a  new transformer and add it to the list
                transformers += self.create_transformers(primitive_object, primitive_name, primitive_type)
            else:
                if len(transformers) > 0:  # Add previous transformers to the pipeline
                    if len(useless_columns) > 0:
                        selector = (COLUMN_SELECTOR_ID, 'drop', [col_index for col_index, _ in useless_columns])
                        transformers = [selector] + transformers
                    transformer_obj = ColumnTransformer(transformers, remainder='passthrough')
                    pipeline_primitives.append((COLUMN_TRANSFORMER_ID, transformer_obj))
                    transformers = []
                pipeline_primitives.append((primitive_name, primitive_object))
                if primitive_type == 'SEMISUPERVISED_CLASSIFIER':
                    break
            
        return pipeline_primitives

    def create_transformers(self, primitive_object, primitive_name, primitive_type):
        column_transformers = []
        nonnumeric_columns = self.metadata['nonnumeric_columns']

        if primitive_type == 'TEXT_ENCODER':
            column_transformers = [(f'{primitive_name}-{col_name}', primitive_object, col_index) for
                                   col_index, col_name in nonnumeric_columns[primitive_type]]
        elif primitive_type == 'CATEGORICAL_ENCODER' or primitive_type == 'DATETIME_ENCODER' or primitive_type == 'IMAGE_ENCODER':
            column_transformers = [(primitive_name, primitive_object, [col_index for col_index, _
                                                                       in nonnumeric_columns[primitive_type]])]

        return column_transformers

