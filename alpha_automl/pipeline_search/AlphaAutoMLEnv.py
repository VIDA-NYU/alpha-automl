import logging

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.env.env_context import EnvContext

logger = logging.getLogger(__name__)


class AlphaAutoMLEnv(gym.Env):
    """
    Customized environment for RLlib Reinforcement Learning.
    reset: reset the environment to the initial state
    step: take an action and return the next state, reward, done, and info
    rewards in detail:
        - win:
            - CLASSIFICATION: 10 + (pipeline score) ^ 5 * 100
            - REGRESSION: 10 + (100 / pipeline score)
        - not end: 1
        - invalid: 10
        - bad: -100
    """

    def __init__(self, config: EnvContext):
        self.game = config["game"]
        self.board = self.game.getInitBoard()
        self.metadata = self.board[: self.game.m]
        self.observation_space = Dict(
            {
                "board": Box(
                    0, 80, shape=(self.game.p + self.game.m,), dtype=np.uint8
                ),  # board
            }
        )
        self.action_space = Discrete(80)  # primitives to choose from

        self.cur_player = 1  # NEVER USED - ONLY ONE PLAYER

    def reset(self, *, seed=None, options=None):
        # init number of steps
        self.num_steps = 0

        self.board = self.game.getInitBoard()
        self.metadata = self.board[: self.game.m]
        self.found = set()

        #         print(f"metadata: {self.metadata}\n board: {self.board}")
        return {"board": np.array(self.board).astype(np.uint8)}, {}

    def step(self, action):
        valid_moves = self.game.getValidMoves(self.board, self.cur_player)
        if action >= len(valid_moves) or valid_moves[action] != 1:
            return (
                {"board": np.array(self.board).astype(np.uint8)},
                -100,
                True,
                False,
                {},
            )

        # update number of steps
        self.num_steps += 1

        # update board with new action
        #         print(f"action: {action}\n board: {self.board}")
        self.board, _ = self.game.getNextState(self.board, self.cur_player, action)
        # reward: win(1) - pipeline score, not end(0) - 0, bad(2) - 0
        reward = 0
        game_end = self.game.getGameEnded(self.board, self.cur_player)
        if game_end == 1:  # pipeline score over threshold
            try:
                if self.game.problem == "REGRESSION":
                    reward = 10 + (100 / self.game.getEvaluation(self.board))
                else:
                    reward = 10 + (self.game.getEvaluation(self.board)) ** 5 * 100
                if tuple(self.board[self.game.m :]) not in self.found:
                    self.found.add(tuple(self.board[self.game.m :]))
                    logger.debug(
                        f"[PIPELINE FOUND] {self.board[self.game.m:]} -> {reward}"
                    )
            except:
                logger.critical(f"[PIPELINE FOUND] Error happened")
        elif game_end == 2:  # finished but invalid
            reward = 10
        else:
            reward = 1

        # done & truncated
        truncated = self.num_steps >= 200
        done = game_end or truncated

        return (
            {"board": np.array(self.board).astype(np.uint8)},
            reward,
            done,
            truncated,
            {},
        )
