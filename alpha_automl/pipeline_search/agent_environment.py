import logging
import random
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
            - CLASSIFICATION: 10 + (pipeline score) ^ 5 * 100
            - REGRESSION: 10 + (100 / pipeline score)
        - not end: 1
        - invalid: 10
        - bad: -100
    """

    def __init__(self, config: EnvContext):
        self.game = config["game"]
        self.board = self.game.getInitBoard()
        self.step_stack = ["S"]
        self.metadata = self.board[: self.game.m]
        self.observation_space = Dict(
            {
                "board": Box(
                    0, 85, shape=(self.game.p + self.game.m,), dtype=np.uint8
                ),  # board
            }
        )
        # self.action_space = Discrete(85)  # primitives to choose from
        self.max_actions = 24
        self.action_spaces = self.generate_action_spaces()
        self.action_offsets = self.generate_action_offsets()
        self.action_space = Discrete(self.max_actions)

        
        self.cur_player = 1  # NEVER USED - ONLY ONE PLAYER

    def reset(self, *, seed=None, options=None):
        # init number of steps
        self.num_steps = 0

        self.step_stack = ["S"]
        self.board = self.game.getInitBoard()
        self.metadata = self.board[: self.game.m]

        #         print(f"metadata: {self.metadata}\n board: {self.board}")
        return {"board": np.array(self.board).astype(np.uint8)}, {}

    def step(self, action):
        curr_step = self.step_stack.pop()
        offseted_action = self.action_offsets[curr_step]+action
        valid_action_size = self.action_spaces[curr_step]
        # Check the action is illegal
        valid_moves = self.game.getValidMoves(self.board, self.cur_player)
        if action >= valid_action_size or valid_moves[offseted_action-1] != 1:
            return (
                {"board": np.array(self.board).astype(np.uint8)},
                -1,
                True,
                False,
                {},
            )

        # Check the action is out of order
        move_type, non_terminals_moves = self.extract_action_details(offseted_action)
        # logger.critical(f"offseted_action: {offseted_action} ===> curr_step: {curr_step}")
        if move_type != curr_step:
            return (
                {"board": np.array(self.board).astype(np.uint8)},
                -100,
                True,
                False,
                {},
            )
        if non_terminals_moves[0] != "E" and non_terminals_moves[0].upper() == non_terminals_moves[0]:
            self.step_stack.extend(non_terminals_moves[::-1])
        

        # update number of steps
        self.num_steps += 1

        # update board with new action
        #         print(f"action: {action}\n board: {self.board}")
        self.board, _ = self.game.getNextState(self.board, self.cur_player, offseted_action-1)

        if self.num_steps > 9:
            logger.info(f"[YFW]================={self.board[self.game.m:]}")
        # reward: win(1) - pipeline score, not end(0) - 0, bad(2) - 0
        reward = 0
        game_end = self.game.getGameEnded(self.board, self.cur_player)
        if game_end == 1:  # pipeline score over threshold
            try:
                if self.game.problem == "REGRESSION":
                    reward = 10 + (100 / self.game.getEvaluation(self.board))
                else:
                    reward = 10 + (self.game.getEvaluation(self.board)) ** 2 * 100
            except:
                logger.critical(f"[PIPELINE FOUND] Error happened")
        elif game_end == 2:  # finished but invalid
            reward = 10
        else:
            #     if move_type == "S":
            #         reward = 1
            #     elif move_type == "ENCODERS":
            #         reward = 1
            # else:
            #     if move_type == "IMPUTER" or move_type == "CATEGORICAL_ENCODER":
            #         reward = 1
            reward = 1
            # if move_string.upper() != move_string:
            #     reward = random.uniform(0, 1)
            # else:
            #     split_move = move_string.split("->")
            #     non_terminals_moves = move_string.split("->")[1].strip().split(" ")
                    
            #     if split_move[0].strip() == "ENSEMBLER":
            #         if "E" in non_terminals_moves:
            #             rewards = 5
            #         else:
            #             rewards = 5 - len(non_terminals_moves)
            #     else:
            #         rewards = random.uniform(0, 1)
                
                

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
            