import logging
import os
import sys
import shutil
import pickle
from os.path import join
from sqlalchemy.orm import joinedload
from alphad3m.pipeline_operations.pipeline_score import calculate_scores, save_scores
from alphad3m.schema import database
from alphad3m.hyperparameter_tuning.primitive_config import is_tunable
from alphad3m.hyperparameter_tuning.bayesian import HyperparameterTuning, get_new_hyperparameters
from alphad3m.utils import create_outputfolders

logger = logging.getLogger(__name__)


@database.with_db
def tune(pipeline_id, dataset_uri, sample_dataset_uri, storage_dir, metrics, problem, scoring_config, report_rank, timeout_tuning, msg_queue, db):
    # Load pipeline from database
    pipeline = (
        db.query(database.Pipeline)
        .filter(database.Pipeline.id == pipeline_id)
        .options(joinedload(database.Pipeline.modules),
                 joinedload(database.Pipeline.connections))
    ).one()

    timeout_tuning = timeout_tuning - 90  # Reduce 1.5 minutes to finish safely the tuning process
    logger.info('About to tune pipeline, id=%s, timeout=%d secs', pipeline_id, timeout_tuning)
    tunable_primitives = {}

    for primitive in pipeline.modules:
        if is_tunable(primitive.name):
            tunable_primitives[primitive.id] = primitive.name

    if len(tunable_primitives) == 0:
        logger.info('No primitives to be tuned for pipeline %s', pipeline_id)
        sys.exit(1)

    logger.info('Tuning primitives: %s', ', '.join(tunable_primitives.values()))

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
        scores = calculate_scores(pipeline, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config)
        first_metric = metrics[0]['metric'].name
        score_values = []

        for fold_scores in scores.values():
            for metric, score_value in fold_scores.items():
                if metric == first_metric:
                    score_values.append(score_value)

        avg_score = sum(score_values) / len(score_values)
        cost = 1.0 - metrics[0]['metric'].normalize(avg_score)
        logger.info('Tuning results:\n%s, cost=%s', scores, cost)

        return cost, scores

    # Run tuning, gets best configuration
    tuning = HyperparameterTuning(tunable_primitives.values())
    output_directory = join(storage_dir, 'tuning', str(pipeline_id))
    create_outputfolders(output_directory)
    best_configuration, best_scores = tuning.tune(evaluate_tune, timeout_tuning, output_directory)

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

    save_scores(new_pipeline.id, best_scores, metrics, report_rank, db)
    shutil.rmtree(output_directory)
    logger.info('Tuning done, generated new pipeline %s', new_pipeline.id)
    msg_queue.send(('tuned_pipeline_id', new_pipeline.id))
