from alphad3m_sklearn.pipeline_synthesis.setup_search import generate_pipelines as gp


def build_pipelines(X, y, scoring, splitting_strategy):
    pipelines = []

    for pipeline in generate_pipelines(X, y, scoring, splitting_strategy):
        pipelines.append(pipeline)

    return pipelines


def generate_pipelines(X, y, scoring, splitting_strategy):
    # Here we call to AlphaD3M engine
    from alphad3m_sklearn.utils import score_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    pipelines = []
    pipeline1 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", LogisticRegression())])

    score1 = score_pipeline(pipeline1, X, y, scoring, splitting_strategy)
    pipelines.append({'pipeline_object': pipeline1, 'pipeline_score': score1})

    pipeline2 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", SVC())])
    score2 = score_pipeline(pipeline2, X, y, scoring, splitting_strategy)
    pipelines.append({'pipeline_object': pipeline2, 'pipeline_score': score2})

    return pipelines

