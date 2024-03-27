import json
import logging
import os
import time
import json
from datetime import datetime

import ray
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from alpha_automl.pipeline_search.AlphaAutoMLEnv import AlphaAutoMLEnv

logger = logging.getLogger(__name__)

PATH_TO_CHECKPOINT = "rllib/ppo_model"
PATH_TO_RESULT_JSON = "rllib/result.json"


def pipeline_search_rllib(game, time_bound, save_checkpoint=False):
    """
    Search for pipelines using Rllib
    """
    ray.init(local_mode=True)
    num_cpus = int(ray.available_resources()["CPU"])
    logger.debug("[RlLib] Ready")

    # load checkpoint or create a new one
    algo = load_rllib_checkpoint(game, num_rollout_workers=num_cpus)
    logger.debug("[RlLib] Create Algo object done")

    # train model
    train_rllib_model(algo, time_bound, save_checkpoint=save_checkpoint)
    if save_checkpoint:
        save_rllib_checkpoint(algo)
    logger.debug("[RlLib] Done")
    ray.shutdown()


def load_rllib_checkpoint(game, num_rollout_workers):
    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        # or "corridor" if registered above
        .environment(AlphaAutoMLEnv, env_config={"game": game})
        .framework("torch")
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(
            num_gpus=1,
            num_gpus_per_worker=1 / (num_rollout_workers + 1),
            num_cpus_per_worker=1,
        )
        .rollouts(num_rollout_workers=num_rollout_workers)
        .training(
            gamma=0.99,
            clip_param=0.2,
            kl_coeff=0.2,
            entropy_coeff=0.01,
            train_batch_size=5000,
        )
    )
    config.lr = 1e-4
    logger.debug("[RlLib] Create Config done")

    # Checking if the list is empty or not
    if [f for f in os.listdir(PATH_TO_CHECKPOINT) if not f.startswith(".")] == []:
        logger.info("[RlLib] Cannot read RlLib checkpoint, create a new one.")
        return config.build()
    else:
        algo = config.build()

        # Restore the old (checkpointed) state.
        algo.restore(PATH_TO_CHECKPOINT)
        # checkpoint_info = get_checkpoint_info(PATH_TO_CHECKPOINT)
        return algo


def train_rllib_model(algo, time_bound, save_checkpoint=False):
    timeout = time.time() + time_bound
    result = algo.train()
    last_best = result["episode_reward_mean"]
    best_unchanged_iter = 1
    logger.info(pretty_print(result))
    while True:
        if (
            time.time() > timeout
            or (best_unchanged_iter >= 6 and result["episode_reward_mean"] >= 0)
            # or result["episode_reward_mean"] >= 70
        ):
            logger.info(f"[RlLib] Train Timeout")
            break
        result = algo.train()
        logger.info(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if result["episode_reward_mean"] > last_best:
            last_best = result["episode_reward_mean"]
            best_unchanged_iter = 1
            if save_checkpoint:
                save_rllib_checkpoint(algo)
        else:
            best_unchanged_iter += 1
    algo.stop()


def save_rllib_checkpoint(algo):
    save_result = algo.save(checkpoint_dir=PATH_TO_CHECKPOINT)
    path_to_checkpoint = save_result.checkpoint.path

    logger.info(
        f"[RlLib] An Algorithm checkpoint has been created inside directory: '{path_to_checkpoint}'."
    )


def dump_result_to_json(primitives, task_start):
    # Read JSON data from input file
    if not os.path.exists(PATH_TO_RESULT_JSON) or os.path.getsize(PATH_TO_RESULT_JSON) == 0:
        with open(PATH_TO_RESULT_JSON, 'w') as f:
            json.dump({}, f)
    with open(PATH_TO_RESULT_JSON, 'r') as f:
        data = json.load(f)
    
    
    timestamp = str(datetime.now() - task_start)
    # strftime("%Y-%m-%d %H:%M:%S")
    
    # Check for duplicate elements
    if primitives in data.values():
        return
    data[timestamp] = primitives

    # Write unique elements to output file
    with open(PATH_TO_RESULT_JSON, "w") as f:
        json.dump(data, f)


def read_result_to_pipeline(builder):
    pipelines = []
    # Read JSON data from input file
    if (
        not os.path.exists(PATH_TO_RESULT_JSON)
        or os.path.getsize(PATH_TO_RESULT_JSON) == 0
    ):
        return []
    with open(PATH_TO_RESULT_JSON, "r") as f:
        data = json.load(f)

    # Check for duplicate elements
    for primitives in data.values():
        pipeline = builder.make_pipeline(primitives)
        if pipeline:
            pipelines.append(pipeline)

    return pipelines
