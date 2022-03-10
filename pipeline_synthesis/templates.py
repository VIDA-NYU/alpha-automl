import logging
from d3m.metadata.problem import TaskKeyword
from alphad3m.pipeline_synthesis.d3mpipeline_builder import BaseBuilder
from alphad3m.data_ingestion.data_profiler import get_privileged_data
from alphad3m.utils import load_primitives_types

logger = logging.getLogger(__name__)

TEMPLATES = {
    'CLASSIFICATION':
        [
            'd3m.primitives.classification.random_forest.SKlearn',
            'd3m.primitives.classification.xgboost_gbtree.Common',
            'd3m.primitives.classification.extra_trees.SKlearn',
            'd3m.primitives.classification.gradient_boosting.SKlearn',
            'd3m.primitives.classification.ada_boost.SKlearn',
            'd3m.primitives.classification.linear_svc.SKlearn'
        ],
    'DEBUG_CLASSIFICATION':
        [
            'd3m.primitives.classification.random_forest.SKlearn',
            'd3m.primitives.classification.xgboost_gbtree.Common'
        ],
    'REGRESSION':
        [
            'd3m.primitives.regression.random_forest.SKlearn',
            'd3m.primitives.regression.xgboost_gbtree.Common',
            'd3m.primitives.regression.extra_trees.SKlearn',
            'd3m.primitives.regression.gradient_boosting.SKlearn',
            'd3m.primitives.regression.ada_boost.SKlearn',
            'd3m.primitives.regression.linear_svr.SKlearn'
        ],
    'DEBUG_REGRESSION':
        [
            'd3m.primitives.regression.random_forest.SKlearn',
            'd3m.primitives.regression.xgboost_gbtree.Common'
        ]
}


def generate_pipelines(task_keywords, dataset, problem, targets, features, hyperparameters, metadata, metrics, DBSession):
    # Primitives for LUPI problems are no longer available. So, just exclude privileged data
    privileged_data = get_privileged_data(problem, task_keywords)
    metadata['exclude_columns'] += privileged_data
    task_keywords_set = set(task_keywords)
    #  Verify if two sets have at-least one element common
    if task_keywords_set & {TaskKeyword.GRAPH_MATCHING, TaskKeyword.LINK_PREDICTION, TaskKeyword.VERTEX_NOMINATION,
                            TaskKeyword.VERTEX_CLASSIFICATION, TaskKeyword.CLUSTERING, TaskKeyword.OBJECT_DETECTION,
                            TaskKeyword.COMMUNITY_DETECTION, TaskKeyword.SEMISUPERVISED, TaskKeyword.LUPI}:
        template_name = 'DEBUG_CLASSIFICATION'
    elif task_keywords_set & {TaskKeyword.COLLABORATIVE_FILTERING, TaskKeyword.FORECASTING}:
        template_name = 'DEBUG_REGRESSION'
    elif TaskKeyword.REGRESSION in task_keywords_set:
        template_name = 'REGRESSION'
        if task_keywords_set & {TaskKeyword.IMAGE, TaskKeyword.TEXT, TaskKeyword.AUDIO, TaskKeyword.VIDEO}:
            template_name = 'DEBUG_REGRESSION'
    else:
        template_name = 'CLASSIFICATION'
        if task_keywords_set & {TaskKeyword.IMAGE, TaskKeyword.TEXT, TaskKeyword.AUDIO, TaskKeyword.VIDEO}:
            template_name = 'DEBUG_CLASSIFICATION'

    logger.info('Creating pipelines from template %s' % template_name)

    feature_types = metadata['only_attribute_types']
    preprocessing_primitives = ['d3m.primitives.data_cleaning.imputer.SKlearn']

    if 'http://schema.org/Text' in feature_types:
        preprocessing_primitives.append('d3m.primitives.feature_extraction.tfidf_vectorizer.SKlearn')
    if 'http://schema.org/DateTime' in feature_types:
        preprocessing_primitives.append('d3m.primitives.data_transformation.enrich_dates.DistilEnrichDates')
    if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in feature_types:
        preprocessing_primitives.append('d3m.primitives.data_transformation.encoder.DSBOX')
    if len(preprocessing_primitives) == 1:  # Encoders were not applied, so use to_numeric for all features
        preprocessing_primitives.append('d3m.primitives.data_transformation.to_numeric.DSBOX')

    estimator_primitives = TEMPLATES.get(template_name, [])
    include_primitives = hyperparameters['include_primitives']
    exclude_primitives = hyperparameters['exclude_primitives']
    preprocessing_primitives, estimator_primitives = modify_search_space(preprocessing_primitives, estimator_primitives,
                                                                         include_primitives, exclude_primitives)

    builder = BaseBuilder()
    pipeline_ids = []

    for estimator_primitive in estimator_primitives:
        template = preprocessing_primitives + [estimator_primitive]
        pipeline_id = builder.make_d3mpipeline(template, 'Template', dataset, None, targets, features, metadata,
                                               metrics, DBSession=DBSession)
        pipeline_ids.append(pipeline_id)

    return pipeline_ids


def modify_search_space(preprocessing_primitives, estimator_primitives, include_primitives, exclude_primitives):
    primitives_types = load_primitives_types()

    for exclude_primitive in exclude_primitives:
        primitive_type = primitives_types.get(exclude_primitive, None)
        if primitive_type in {'CLASSIFICATION', 'REGRESSION'}:
            estimator_primitives = [i for i in estimator_primitives if i != exclude_primitive]
        elif primitive_type is not None:
            preprocessing_primitives = [i for i in preprocessing_primitives if i != exclude_primitive]

    for include_primitive in include_primitives:
        primitive_type = primitives_types.get(include_primitive, None)
        if primitive_type in {'CLASSIFICATION', 'REGRESSION'}:
            if include_primitive not in estimator_primitives:
                estimator_primitives.append(include_primitive)
        elif primitive_type is not None:
            if include_primitive not in preprocessing_primitives:
                preprocessing_primitives.append(include_primitive)

    return preprocessing_primitives, estimator_primitives
