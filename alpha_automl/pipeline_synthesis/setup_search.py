import logging
import sys
from datetime import datetime
from os.path import dirname, join

from alpha_automl.grammar_loader import load_manual_grammar
from alpha_automl.pipeline_search.agent_lab import (contain_checkpoints,
                                                    dump_result_to_json,
                                                    pipeline_search_rllib,
                                                    read_result_to_pipeline)
from alpha_automl.pipeline_search.game import PipelineGame
from alpha_automl.pipeline_synthesis.pipeline_builder import BaseBuilder
from alpha_automl.scorer import score_pipeline
from alpha_automl.utils import hide_logs

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_PATH = join(dirname(__file__), "../resource/checkpoints/")


config = {
    "PROBLEM_TYPES": {
        "CLASSIFICATION": 1,
        "REGRESSION": 2,
        "CLUSTERING": 3,
        "TIME_SERIES_FORECAST": 4,
        "SEMISUPERVISED": 5,
        "NA": 6,
    },
    "DATA_TYPES": {"TABULAR": 1, "TEXT": 2, "IMAGE": 3, "VIDEO": 4, "MULTIMODAL": 5},
    "PIPELINE_SIZE": 10,
}


def signal_handler(queue, signum):
    logger.debug(f"Receiving signal {signum}, terminating process")
    queue.append("DONE")
    # TODO: Should it save the last status of the NN model?
    sys.exit(0)


def search_pipelines(
    X,
    y,
    scoring,
    splitting_strategy,
    task_name,
    time_bound,
    automl_hyperparams,
    metadata,
    output_folder,
    checkpoints_folder,
    verbose,
):
    # signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(queue, signum))
    hide_logs(
        verbose
    )  # Hide logs here too, since multiprocessing has some issues with loggers

    builder = BaseBuilder(metadata, automl_hyperparams)
    all_primitives = builder.all_primitives
    ensemble_pipelines_hash = set()

    task_start = datetime.now()

    def evaluate_pipeline(primitives):
        has_repeated_classifiers = check_repeated_classifiers(
            primitives, all_primitives, ensemble_pipelines_hash
        )

        if has_repeated_classifiers:
            logger.info("Repeated classifiers detected in ensembles, ignoring pipeline")
            return None

        pipeline = builder.make_pipeline(primitives)
        score = None

        if pipeline is not None:
            alphaautoml_pipeline = score_pipeline(
                pipeline, X, y, scoring, splitting_strategy, task_name, verbose
            )
            if alphaautoml_pipeline is not None:
                score = alphaautoml_pipeline.get_score()
                if score is not None:
                    dump_result_to_json(primitives, task_start, score, output_folder)
        return score

    if task_name is None:
        task_name = "NA"

    task_name_id = task_name + "_TASK"
    include_primitives = automl_hyperparams["include_primitives"]
    exclude_primitives = automl_hyperparams["exclude_primitives"]
    new_primitives = automl_hyperparams["new_primitives"]
    use_imputer = metadata["missing_values"]
    nonnumeric_columns = metadata["nonnumeric_columns"]

    logger.debug("Creating a manual grammar")
    grammar = load_manual_grammar(
        task_name_id,
        nonnumeric_columns,
        use_imputer,
        new_primitives,
        include_primitives,
        exclude_primitives,
    )

    metric = scoring._score_func.__name__
    config_updated = update_config(task_name, metric, grammar, metadata)
    checkpoint_load_folder = (
        checkpoints_folder
        if contain_checkpoints(checkpoints_folder)
        else DEFAULT_CHECKPOINT_PATH
    )
    checkpoint_save_folder = (
        checkpoints_folder
        if checkpoints_folder is not None
        else DEFAULT_CHECKPOINT_PATH
    )
    game = PipelineGame(config_updated, evaluate_pipeline)
    pipeline_search_rllib(
        game, time_bound, checkpoint_load_folder, checkpoint_save_folder
    )
    logger.debug("Search completed")
    results = read_result_to_pipeline(builder, output_folder)

    # queue.put('DONE')
    return results


def update_config(task_name, metric, grammar, metadata):
    config["PROBLEM"] = task_name
    config["DATA_TYPE"] = "TABULAR"
    config["METRIC"] = metric
    config["DATASET"] = f"DATASET_{task_name}"
    config["GRAMMAR"] = grammar
    metafeatures = compute_metafeatures(metric, metadata)
    config["DATASET_METAFEATURES"] = metafeatures + [0] * (8 - len(metafeatures))

    return config


def check_repeated_classifiers(
    pipeline_primitives, all_primitives, ensemble_pipelines_hash
):
    # Verify if the classifiers are repeated in the ensembles (regardless of the order)
    classifiers = []
    pipeline_hash = ""
    has_ensemble_primitive = False
    has_repeated_classifiers = False

    for primitive_name in pipeline_primitives:
        primitive_type = all_primitives[primitive_name]["type"]

        if primitive_type == "CLASSIFIER":
            classifiers.append(primitive_name)
        elif primitive_type == "MULTI_ENSEMBLER":
            has_ensemble_primitive = True
            pipeline_hash += primitive_name
            if len(classifiers) != len(
                set(classifiers)
            ):  # All classifiers should be different
                has_repeated_classifiers = True
        else:
            pipeline_hash += primitive_name

    if not has_ensemble_primitive:
        return False

    if has_repeated_classifiers:
        return True

    pipeline_hash += "".join(sorted(classifiers))

    if pipeline_hash in ensemble_pipelines_hash:
        return True
    else:
        ensemble_pipelines_hash.add(pipeline_hash)
        return False


def compute_metafeatures(metric, metadata):
    metafeatures = []
    # SCORING METRIC
    scoring_type = 0
    if metric in [
        "accuracy_score",
        "f1_score",
        "precision_score",
        "recall_score",
        "jaccard_score",
    ]:
        scoring_type = 1
    elif metric in [
        "max_error",
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
        "r2_score",
    ]:
        scoring_type = 2
    elif metric in [
        "adjusted_mutual_info_score",
        "rand_score",
        "mutual_info_score",
        "normalized_mutual_info_score",
    ]:
        scoring_type = 3
    metafeatures.append(scoring_type)

    # IMPUTE
    metafeatures.append(1 if metadata["missing_values"] else 0)
    # ENCODE
    nonnumeric_columns = metadata["nonnumeric_columns"]
    if nonnumeric_columns != {}:
        metafeatures.append(1)
        # TEXT
        metafeatures.append(
            len(nonnumeric_columns["TEXT_ENCODER"])
            if "TEXT_ENCODER" in nonnumeric_columns
            else 0
        )
        # CATEGORICAL
        metafeatures.append(
            len(nonnumeric_columns["CATEGORICAL_ENCODER"])
            if "CATEGORICAL_ENCODER" in nonnumeric_columns
            else 0
        )
        # DATETIME
        metafeatures.append(
            len(nonnumeric_columns["DATETIME_ENCODER"])
            if "DATETIME_ENCODER" in nonnumeric_columns
            else 0
        )
        # IMAGE
        metafeatures.append(
            len(nonnumeric_columns["IMAGE_ENCODER"])
            if "IMAGE_ENCODER" in nonnumeric_columns
            else 0
        )
    else:
        metafeatures.append(0)

    return metafeatures