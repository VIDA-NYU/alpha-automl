import logging
import os
import json
import pkg_resources
import random
import d3m.metadata.base
import d3m.runtime
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from alphad3m.schema import database, convert
from alphad3m.utils import is_collection, get_dataset_sample
from d3m.metrics import class_map
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import PerformanceMetric, TaskKeyword

logger = logging.getLogger(__name__)


with pkg_resources.resource_stream('alphad3m', 'resource/pipelines/kfold_tabular_split.yaml') as fp:
    kfold_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream('alphad3m', 'resource/pipelines/kfold_timeseries_split.yaml') as fp:
    kfold_timeseries_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream('alphad3m', 'resource/pipelines/train-test-tabular-split.yaml') as fp:
    train_test_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream('alphad3m', 'resource/pipelines/scoring.yaml') as fp:
    scoring_pipeline = Pipeline.from_yaml(fp)


def check_timeindicator(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)

    columns = dataset_doc['dataResources'][0]['columns']
    timeindicator_index = None
    has_timeindicator = False
    for item in columns:
        if item['colType'] == 'dateTime':
            timeindicator_index = item['colIndex']
        if 'timeIndicator' in item['role']:
            has_timeindicator = True
            break

    if not has_timeindicator:
        dataset_doc['dataResources'][0]['columns'][timeindicator_index]['role'].append('timeIndicator')
        try:
            with open(dataset_path, 'w') as fout:
                json.dump(dataset_doc, fout, indent=4)
        except:
            logger.error('Saving timeIndicator on dataset')


@database.with_db
def score(pipeline_id, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config, report_rank, msg_queue, db):
    # Get pipeline from database
    logger.info('About to score pipeline, id=%s, metrics=%s', pipeline_id, metrics)
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    scores = calculate_scores(pipeline, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config)
    logger.info("Evaluation results:\n%s", scores)

    if scoring_config['method'] == 'RANKING':  # Only for TA2 evaluation
        report_rank = True

    save_scores(pipeline_id, scores, metrics, report_rank, db)


def calculate_scores(pipeline, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config):
    dataset_uri_touse = dataset_uri

    if sample_dataset_uri:
        dataset_uri_touse = sample_dataset_uri
    if TaskKeyword.FORECASTING in problem['problem']['task_keywords']:
        check_timeindicator(dataset_uri_touse[7:])

    dataset = Dataset.load(dataset_uri_touse)
    scores = {}
    pipeline_split = None

    if TaskKeyword.FORECASTING in problem['problem']['task_keywords']:
        pipeline_split = kfold_timeseries_split

    elif scoring_config['method'] == 'K_FOLD':
        pipeline_split = kfold_tabular_split

    elif scoring_config['method'] == 'HOLDOUT':
        pipeline_split = train_test_tabular_split

    elif scoring_config['method'] == 'RANKING':
        pipeline_split = kfold_tabular_split
    else:
        logger.warning('Unknown evaluation method, using K_FOLD')
        pipeline_split = kfold_tabular_split

    # FIXME: Splitting pipeline fails when works with F1 and semisupervised task, so use F1_MACRO instead
    # See https://gitlab.com/datadrivendiscovery/common-primitives/-/issues/92#note_520784899
    if metrics[0]['metric'] == PerformanceMetric.F1 and TaskKeyword.SEMISUPERVISED in problem['problem']['task_keywords']:
        new_metrics = [{'metric': PerformanceMetric.F1_MACRO}]
        scores = evaluate(pipeline, pipeline_split, dataset, new_metrics, problem, scoring_config, dataset_uri)
        scores = change_name_metric(scores, new_metrics, new_metric=metrics[0]['metric'].name)
    else:
        scores = evaluate(pipeline, pipeline_split, dataset, metrics, problem, scoring_config, dataset_uri)

    return scores


def save_scores(pipeline_id, scores, metrics, report_rank, db):
    scores_db = []

    if len(scores) > 0:  # It's a valid pipeline
        scores_db = add_scores_db(scores, scores_db)
        if report_rank:  # For TA2 only evaluation
            scores = create_rank_metric(scores, metrics)
            scores_db = add_scores_db(scores, scores_db)
            logger.info("Evaluation results for RANK metric: \n%s", scores)

    record_db = database.Evaluation(pipeline_id=pipeline_id, scores=scores_db)  # Store scores
    db.add(record_db)
    db.commit()


def evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_config, dataset_uri):
    if is_collection(dataset_uri[7:]) or TaskKeyword.GRAPH in problem['problem']['task_keywords']:
        # Sampling in memory
        dataset = get_dataset_sample(dataset, problem)

    json_pipeline = convert.to_d3m_json(pipeline)

    logger.info("Pipeline to be scored:\n\t%s", '\n\t'.join(['step_%02d: %s' % (i, x['primitive']['python_path'])
                                                             for i, x in enumerate(json_pipeline['steps'])]))

    d3m_pipeline = Pipeline.from_json_structure(json_pipeline, )

    run_scores, run_results = d3m.runtime.evaluate(
        pipeline=d3m_pipeline,
        data_pipeline=data_pipeline,
        scoring_pipeline=scoring_pipeline,
        problem_description=problem,
        inputs=[dataset],
        data_params=scoring_config,
        metrics=metrics,
        volumes_dir=os.environ.get('D3MSTATICDIR', None),
        context=d3m.metadata.base.Context.TESTING,
        random_seed=0,
    )

    for result in run_results:
        if result.has_error():
            if TaskKeyword.GRAPH in problem['problem']['task_keywords']:
                # FIXME: Splitting pipeline fails for some graph datasets (e.g. 49_facebook). So, score the same dataset
                logger.warning('Evaluation failed for graph dataset, trying to score over the whole input dataset')
                return fit_score(d3m_pipeline, problem, dataset, dataset, metrics, scoring_config)
            else:
                raise RuntimeError(result.pipeline_run.status['message'])

    #save_pipeline_runs(run_results.pipeline_runs)
    combined_folds = d3m.runtime.combine_folds([fold for fold in run_scores])
    scores = {}

    for _, row in combined_folds.iterrows():
        if row['fold'] not in scores:
            scores[row['fold']] = {}
        scores[row['fold']][row['metric']] = row['value']

    return scores


def create_rank_metric(scores, metrics):
    scores_tmp = {}

    for fold, fold_scores in scores.items():
        scores_tmp[fold] = {}
        for metric, current_score in fold_scores.items():
            if metric == metrics[0]['metric'].name:
                new_score = (1.0 - metrics[0]['metric'].normalize(current_score)) + random.random() * 1.e-12
                scores_tmp[fold]['RANK'] = new_score

    return scores_tmp


def change_name_metric(scores, metrics, new_metric):
    scores_tmp = {}

    for fold, fold_scores in scores.items():
        scores_tmp[fold] = {}
        for metric, current_score in fold_scores.items():
            if metric == metrics[0]['metric'].name:
                scores_tmp[fold][new_metric] = current_score

    return scores_tmp


def add_scores_db(scores_dict, scores_db):
    for fold, fold_scores in scores_dict.items():
        for metric, value in fold_scores.items():
            scores_db.append(database.EvaluationScore(fold=fold, metric=metric, value=value))

    return scores_db


def save_pipeline_runs(pipelines_runs):
    for pipeline_run in pipelines_runs:
        save_run_path = os.path.join(os.environ['D3MOUTPUTDIR'], 'pipeline_runs',
                                     pipeline_run.to_json_structure()['id'] + '.yml')

        with open(save_run_path, 'w') as fin:
            pipeline_run.to_yaml(fin, indent=2)


def fit_score(d3m_pipeline, problem, dataset_train, dataset_test, metrics, scoring_config):
    fitted_pipeline, predictions, result = d3m.runtime.fit(
        pipeline=d3m_pipeline,
        problem_description=problem,
        inputs=[dataset_train],
        data_params=scoring_config,
        volumes_dir=os.environ.get('D3MSTATICDIR', None),
        context=d3m.metadata.base.Context.TESTING,
        random_seed=0
    )

    if result.has_error():
        raise RuntimeError(result.pipeline_run.status['message'])

    run_scores, _ = d3m.runtime.score(
        predictions, [dataset_test],
        scoring_pipeline=scoring_pipeline,
        problem_description=problem,
        metrics=metrics,
        predictions_random_seed=fitted_pipeline.random_seed,
        volumes_dir=fitted_pipeline.volumes_dir,
        scratch_dir=fitted_pipeline.scratch_dir,
        context=d3m.metadata.base.Context.TESTING,
        random_seed=0,
    )
    score = run_scores['value'][0]
    metric = run_scores['metric'][0]
    scores = {i: {metric: score} for i in range(int(scoring_config['number_of_folds']))}

    return scores
