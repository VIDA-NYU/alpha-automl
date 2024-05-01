import json
import logging
import os
import time
from datetime import datetime

import ray
from alpha_automl.pipeline_search.agent_environment import AutoMLEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

logger = logging.getLogger(__name__)


def pipeline_search_rllib(game, time_bound, checkpoint_load_folder, checkpoint_save_folder):
    """
    Search for pipelines using Rllib
    """
    ray.init(local_mode=True, num_cpus=8, logging_level=logging.CRITICAL, log_to_driver=False)
    num_cpus = int(ray.available_resources()["CPU"])

    # load checkpoint or create a new one
    algo = load_rllib_checkpoint(game, checkpoint_load_folder, num_rollout_workers=7)
    logger.debug("Create Algo object done")

    # train model
    train_rllib_model(algo, time_bound, checkpoint_load_folder, checkpoint_save_folder)
    logger.debug("Training done")
    ray.shutdown()


def load_rllib_checkpoint(game, checkpoint_load_folder, num_rollout_workers):
    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        # or "corridor" if registered above
        .environment(AutoMLEnv, env_config={"game": game})
        .framework("torch")
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(
            # num_gpus=1,
            # num_gpus_per_worker=1 / (num_rollout_workers + 1),
            num_cpus_per_worker=1,
        )
        .rollouts(num_rollout_workers=num_rollout_workers)
        .training(
            gamma=0.99,
            clip_param=0.3,
            kl_coeff=0.3,
            entropy_coeff=0.05,
            train_batch_size=10000,
        )
    )
    config.lr = 1e-5
    config.simple_optimizer = True
    logger.debug("Create Config done")

    # Checking if the list is empty or not
    if not contain_checkpoints(checkpoint_load_folder):
        logger.debug("Cannot read checkpoint, create a new one.")
        return config.build()
    else:
        algo = config.build()
        weights = load_rllib_policy_weights(checkpoint_load_folder)

        algo.set_weights(weights)
        # Restore the old state.
        # algo.restore(load_folder)
        # checkpoint_info = get_checkpoint_info(load_folder)
        return algo


def train_rllib_model(algo, time_bound, checkpoint_load_folder, checkpoint_save_folder):
    timeout = time.time() + time_bound
    result = algo.train()
    last_best = result["episode_reward_mean"]
    best_unchanged_iter = 1
    logger.debug(pretty_print(result))

    while True:
        if (
            time.time() > timeout
            or (best_unchanged_iter >= 600 and result["episode_reward_mean"] >= 0)
            # or result["episode_reward_mean"] >= 70
        ):
            logger.debug(f"Training timeout reached")
            break

        if contain_checkpoints(checkpoint_save_folder):
            weights = load_rllib_policy_weights(checkpoint_save_folder)
            algo.set_weights(weights)
        elif contain_checkpoints(checkpoint_load_folder):
            weights = load_rllib_policy_weights(checkpoint_load_folder)
            algo.set_weights(weights)
        result = algo.train()
        logger.debug(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if result["episode_reward_mean"] > last_best:
            last_best = result["episode_reward_mean"]
            best_unchanged_iter = 1
            save_rllib_checkpoint(algo, checkpoint_save_folder)
        else:
            best_unchanged_iter += 1
    algo.stop()


def load_rllib_policy_weights(checkpoint_folder):
    logger.debug(f"Synchronizing model weights...")
    policy = Policy.from_checkpoint(checkpoint_folder)
    policy = policy["default_policy"]
    weights = policy.get_weights()

    weights = {"default_policy": weights}

    return weights


def save_rllib_checkpoint(algo, checkpoint_save_folder):
    save_result = algo.save(checkpoint_dir=checkpoint_save_folder)
    path_to_checkpoint = save_result.checkpoint.path

    logger.debug(
        f"An Algorithm checkpoint has been created inside directory: '{path_to_checkpoint}'."
    )


def dump_result_to_json(primitives, task_start, score, output_folder=None):
    output_path = generate_json_path(output_folder)
    # Read JSON data from input file
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        with open(output_path, "w") as f:
            json.dump({}, f)
    with open(output_path, "r") as f:
        data = json.load(f)

    timestamp = str(datetime.now() - task_start)
    # strftime("%Y-%m-%d %H:%M:%S")

    # Check for duplicate elements
    if primitives in data.values():
        return
    data[score] = primitives

    # Write unique elements to output file
    with open(output_path, "w") as f:
        json.dump(data, f)


def read_result_to_pipeline(builder, output_folder=None):
    output_path = generate_json_path(output_folder)

    pipelines = []
    # Read JSON data from input file
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return []
    with open(output_path, "r") as f:
        data = json.load(f)

    # Check for duplicate elements
    for score, primitives in sorted(data.items()):
        pipeline = builder.make_pipeline(primitives)
        if pipeline:
            pipelines.append(pipeline)

    return pipelines


def generate_json_path(output_folder=None):
    output_path = os.path.join(output_folder, "result.json")

    return output_path


def contain_checkpoints(folder_path):
    if folder_path is None:
        return False

    file_list = os.listdir(folder_path)

    if [f for f in file_list if not f.startswith(".")] == []:
        return False

    if (
        "algorithm_state.pkl" in file_list
        and "policies" in file_list
        and "rllib_checkpoint.json" in file_list
    ):
        return True
    else:
        logger.debug(
            f"Checkpoint folder {folder_path} does not contain all necessary files, files: {file_list}."
        )

    return False
