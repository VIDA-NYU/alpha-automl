import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from alphad3m_sklearn.utils import create_object


def build_pipelines(X, y, scoring, splitting_strategy):
    pipelines = []

    for string_pipeline in search_pipelines():
        pipeline_object = build_pipeline_object(string_pipeline)
        pipeline_score = score_pipeline(pipeline_object, X, y, scoring, splitting_strategy)
        pipelines.append({'pipeline_object': pipeline_object, 'pipeline_score': pipeline_score})

    return pipelines


def search_pipelines():
    # Here we call to AlphaD3M engine
    return [
        ['sklearn.preprocessing.StandardScaler', 'sklearn.svm.SVC'],
        ['sklearn.preprocessing.MaxAbsScaler', 'sklearn.svm.SVC']
    ]


def build_pipeline_object(string_pipeline):
    pipeline_primitives = []

    for primitive_string in string_pipeline:
        primitive_object = create_object(primitive_string)
        pipeline_primitives.append((primitive_string, primitive_object))

    pipeline = Pipeline(pipeline_primitives)

    return pipeline


def score_pipeline(pipeline, X, y, scoring, splitting_strategy):
    scores = cross_val_score(pipeline, X, y, cv=splitting_strategy, scoring=scoring)
    score = np.average(scores)

    return score
