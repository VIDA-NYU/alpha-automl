import logging

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.env.env_context import EnvContext

logger = logging.getLogger(__name__)


class AutoMLEnv(gym.Env):
    """
    Customized environment for RLlib Reinforcement Learning.
    reset: reset the environment to the initial state
    step: take an action and return the next state, reward, done, and info
    rewards in detail:
        - win:
            - CLASSIFICATION: 10 + (pipeline score) ^ 2 * 100
            - REGRESSION: 10 + (100 / pipeline score)
        - not end: 1
        - invalid: 10
        - bad: -1
    """

    def __init__(self, config: EnvContext):
        self.game = config["game"]  # PipelineGame
        self.board = self.game.getInitBoard()  # initial board
        self.step_stack = ["S"]  # stack for steps
        self.metadata = self.board[: self.game.m]
        self.observation_space = Dict(
            {
                "board": Box(
                    0, 85, shape=(self.game.p + self.game.m,), dtype=np.uint8
                ),  # Ray env board contains pipeline and metadata
            }
        )
        self.max_actions = 24  # max number of actions (depends on the largest step in the grammar, i.e. CLASSIFIER)
        self.action_spaces = (
            self.generate_action_spaces()
        )  # number of actions for each step
        self.action_offsets = (
            self.generate_action_offsets()
        )  # offset for each step, for translating action to PipelineGame action
        self.action_space = Discrete(self.max_actions)  # Ray env action space

    def reset(self, *, seed=None, options=None):
        self.num_steps = 0
        self.step_stack = ["S"]
        self.board = self.game.getInitBoard()
        self.metadata = self.board[: self.game.m]

        return {"board": np.array(self.board).astype(np.uint8)}, {}

    def step(self, action):
        curr_step = self.step_stack.pop()
        offseted_action = self.action_offsets[curr_step] + action
        valid_action_size = self.action_spaces[curr_step]
        # Check the action is illegal
        valid_moves = self.game.getValidMoves(self.board)
        if action >= valid_action_size or valid_moves[offseted_action - 1] != 1:
            return (
                {"board": np.array(self.board).astype(np.uint8)},
                -1,
                True,
                False,
                {},
            )

        # Check the action is out of order
        move_type, non_terminals_moves = self.extract_action_details(offseted_action)
        if move_type != curr_step:
            return (
                {"board": np.array(self.board).astype(np.uint8)},
                -100,
                True,
                False,
                {},
            )
        if (
            non_terminals_moves[0] != "E"
            and non_terminals_moves[0].upper() == non_terminals_moves[0]
        ):
            self.step_stack.extend(non_terminals_moves[::-1])

        # update number of steps
        self.num_steps += 1

        # update board with new action
        self.board = self.game.getNextState(self.board, offseted_action - 1)

        # reward: win(1) - pipeline score, not end(0) - 1, bad(2) - -1
        reward = 0
        game_end = self.game.getGameEnded(self.board)
        if game_end == 1:  # pipeline score over threshold
            try:
                if self.game.problem == "REGRESSION":
                    reward = 10 + (100 / self.game.getEvaluation(self.board))
                else:
                    reward = 10 + (self.game.getEvaluation(self.board)) ** 2 * 100
            except Exception as e:
                logger.critical(f"[PIPELINE FOUND] Error happened: {str(e)}")
        elif game_end == 2:  # finished but invalid
            reward = 10
        else:
            reward = 1

        # done & truncated
        truncated = self.num_steps >= 20
        done = game_end or truncated

        return (
            {"board": np.array(self.board).astype(np.uint8)},
            reward,
            done,
            truncated,
            {},
        )

    def extract_action_details(self, action):
        rules = self.game.grammar["RULES"]
        move_string = list(rules.keys())[list(rules.values()).index(action)]
        split_move = move_string.split("->")
        move_type = split_move[0].strip()
        non_terminals_moves = split_move[1].strip().split(" ")
        return move_type, non_terminals_moves

    def generate_action_spaces(self):
        action_spaces = {}
        for action in self.game.grammar["RULES"].values():
            move_type, non_terminals_moves = self.extract_action_details(action)

            if move_type not in action_spaces:
                action_spaces[move_type] = 1
            else:
                action_spaces[move_type] += 1

        return action_spaces

    def generate_action_offsets(self):
        action_offsets = {}
        for action in self.game.grammar["RULES"].values():
            move_type, non_terminals_moves = self.extract_action_details(action)

            if move_type not in action_offsets:
                action_offsets[move_type] = action

        return action_offsets
