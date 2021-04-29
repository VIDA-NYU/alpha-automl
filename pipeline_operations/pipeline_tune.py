import logging
import os
import sys
import shutil
import pickle
from os.path import join
from copy import deepcopy
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from alphad3m.pipeline_operations.pipeline_score import evaluate, kfold_tabular_split, score
from alphad3m.schema import database
from alphad3m.hyperparameter_tuning.primitive_config import is_tunable
from alphad3m.hyperparameter_tuning.bayesian import HyperparameterTuning, get_new_hyperparameters
from d3m.metadata.problem import PerformanceMetric, TaskKeyword
from alphad3m.utils import create_outputfolders

logger = logging.getLogger(__name__)


@database.with_db
def tune(pipeline_id, metrics, problem, dataset_uri, sample_dataset_uri, report_rank, timeout_tuning, msg_queue, db):
    # FIXME: Save 10% of timeout to score the best config. It shouldn't run the best config twice
    timeout_tuning = timeout_tuning * 0.9
    # Load pipeline from database
    pipeline = (
        db.query(database.Pipeline)
        .filter(database.Pipeline.id == pipeline_id)
        .options(joinedload(database.Pipeline.modules),
                 joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to tune pipeline, id=%s, dataset=%r, timeout=%d secs', pipeline_id, dataset_uri, timeout_tuning)
    tunable_primitives = {}

    for primitive in pipeline.modules:
        if is_tunable(primitive.name):
            tunable_primitives[primitive.id] = primitive.name

    if len(tunable_primitives) == 0:
        logger.info('No primitives to be tuned for pipeline %s', pipeline_id)
        sys.exit(1)

    logger.info('Tuning primitives: %s', ', '.join(tunable_primitives.values()))

    if sample_dataset_uri:
        dataset = Dataset.load(sample_dataset_uri)
    else:
        dataset = Dataset.load(dataset_uri)

    task_keywords = problem['problem']['task_keywords']
    scoring_config = {'shuffle': 'true',
                      'stratified': 'true' if TaskKeyword.CLASSIFICATION in task_keywords else 'false',
                      'method': 'K_FOLD',
                      'number_of_folds': '2'}

    metrics_to_use = deepcopy(metrics)
    if metrics[0]['metric'] == PerformanceMetric.F1 and TaskKeyword.SEMISUPERVISED in problem['problem']['task_keywords']:
        metrics_to_use = [{'metric': PerformanceMetric.F1_MACRO}]

    def evaluate_tune(hyperparameter_configuration):
        new_hyperparams = []
        for primitive_id, primitive_name in tunable_primitives.items():
            hy = get_new_hyperparameters(primitive_name, hyperparameter_configuration)
            db_hyperparams = database.PipelineParameter(
                pipeline=pipeline,
                module_id=primitive_id,
                name='hyperparams',
                value=pickle.dumps(hy),
            )
            new_hyperparams.append(db_hyperparams)

        pipeline.parameters += new_hyperparams
        scores = evaluate(pipeline, kfold_tabular_split, dataset, metrics_to_use, problem, scoring_config, dataset_uri)
        first_metric = metrics_to_use[0]['metric'].name
        score_values = []
        for fold_scores in scores.values():
            for metric, score_value in fold_scores.items():
                if metric == first_metric:
                    score_values.append(score_value)

        avg_score = sum(score_values) / len(score_values)
        cost = 1.0 - metrics_to_use[0]['metric'].normalize(avg_score)
        logger.info('Tuning results:\n%s, cost=%s', scores, cost)

        return cost

    # Run tuning, gets best configuration
    tuning = HyperparameterTuning(tunable_primitives.values())
    create_outputfolders(join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'tuning'))
    best_configuration = tuning.tune(evaluate_tune, wallclock=timeout_tuning,
                                     output_dir=join(os.environ.get('D3MOUTPUTDIR'),
                                                     'temp', 'tuning', str(pipeline_id)))

    # Duplicate pipeline in database
    new_pipeline = database.duplicate_pipeline(db, pipeline, 'HyperparameterTuning from pipeline %s' % pipeline_id)

    for primitive in new_pipeline.modules:
        if is_tunable(primitive.name):
            best_hyperparameters = get_new_hyperparameters(primitive.name, best_configuration)
            query = db.query(database.PipelineParameter).filter(database.PipelineParameter.module_id == primitive.id)\
                .filter(database.PipelineParameter.pipeline_id == new_pipeline.id)\
                .filter(database.PipelineParameter.name == 'hyperparams')
            if query.first():
                original_parameters = pickle.loads(query.first().value)
                original_parameters.update(best_hyperparameters)
                query.update({database.PipelineParameter.value: pickle.dumps(original_parameters)})
            else:
                db.add(database.PipelineParameter(
                    pipeline=new_pipeline,
                    module_id=primitive.id,
                    name='hyperparams',
                    value=pickle.dumps(best_hyperparameters),
                ))
    db.commit()

    logger.info('Tuning done, generated new pipeline %s', new_pipeline.id)
    shutil.rmtree(join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'tuning', str(pipeline_id)))
    # FIXME: We should remove this score process to avoid execute twice the best configuration
    score(new_pipeline.id, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config, report_rank, None,
          db_filename=join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'db.sqlite3'))
    # TODO: Change this static string path

    msg_queue.send(('tuned_pipeline_id', new_pipeline.id))
