import logging
import itertools
from d3m.metadata.problem import TaskKeyword
from alphad3m.search.d3mpipeline_builder import BaseBuilder
from alphad3m.data_ingestion.data_profiler import get_privileged_data

logger = logging.getLogger(__name__)

TEMPLATES = {
    'CLASSIFICATION': list(itertools.product(
        # Imputer
        ['d3m.primitives.data_cleaning.imputer.SKlearn'],
        # Classifier
        [
            'd3m.primitives.classification.random_forest.SKlearn',
            'd3m.primitives.classification.xgboost_gbtree.Common',
            'd3m.primitives.classification.extra_trees.SKlearn',
            'd3m.primitives.classification.gradient_boosting.SKlearn',
            'd3m.primitives.classification.ada_boost.SKlearn',
            'd3m.primitives.classification.linear_svc.SKlearn'
        ],
    )),
    'DEBUG_CLASSIFICATION': list(itertools.product(
        # Imputer
        ['d3m.primitives.data_cleaning.imputer.SKlearn'],
        # Classifier
        [
            'd3m.primitives.classification.random_forest.SKlearn',
            'd3m.primitives.classification.xgboost_gbtree.Common'
        ],
    )),
    'REGRESSION': list(itertools.product(
        # Imputer
        ['d3m.primitives.data_cleaning.imputer.SKlearn'],
        # Regressor
        [
            'd3m.primitives.regression.random_forest.SKlearn',
            'd3m.primitives.regression.xgboost_gbtree.Common',
            'd3m.primitives.regression.extra_trees.SKlearn',
            'd3m.primitives.regression.gradient_boosting.SKlearn',
            'd3m.primitives.regression.ada_boost.SKlearn',
            'd3m.primitives.regression.linear_svr.SKlearn'
        ],
    )),
    'DEBUG_REGRESSION': list(itertools.product(
        # Imputer
        ['d3m.primitives.data_cleaning.imputer.SKlearn'],
        # Regressor
        [
            'd3m.primitives.regression.random_forest.SKlearn',
            'd3m.primitives.regression.xgboost_gbtree.Common'
        ],
    )),
}


def generate_pipelines(task_keywords, dataset, problem, targets, features, metadata, metrics, DBSession):
    privileged_data = get_privileged_data(problem, task_keywords)
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

    logger.info("Creating pipelines from template %s" % template_name)

    templates = TEMPLATES.get(template_name, [])
    pipeline_ids = []

    for imputer, classifier in templates:
        pipeline_id = BaseBuilder.make_template(imputer, classifier, dataset, targets, features, metadata,
                                                privileged_data, metrics, DBSession=DBSession)
        pipeline_ids.append(pipeline_id)

    return pipeline_ids
