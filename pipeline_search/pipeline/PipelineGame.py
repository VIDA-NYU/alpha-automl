from __future__ import print_function
import os
import pickle
import math
import logging
from copy import deepcopy

from alphad3m.pipeline_search.Game import Game
from alphad3m.pipeline_search.pipeline.PipelineLogic import Board
import numpy as np
import traceback
import time

logger = logging.getLogger(__name__)

class PipelineGame(Game):
    # FIXEME: Maybe the input parameters can be in json
    def __init__(self, input={}, eval_pipeline=None):
        self.steps = 0
        self.args = input['ARGS']
        self.evaluations = {}
        self.eval_times = {}

        self.grammar = input['GRAMMAR']
        self.pipeline_size = input['PIPELINE_SIZE']
        self.problem_types = input['PROBLEM_TYPES']
        self.data_types = input['DATA_TYPES']
        self.eval_pipeline = eval_pipeline

        self.problem = input['PROBLEM'].upper()
        self.data_type = input['DATA_TYPE'].upper()
        self.metric = input['METRIC']
        self.dataset = input['DATASET']

        self.dataset_metafeatures = input['DATASET_METAFEATURES']
        if self.dataset_metafeatures is None:
            metafeatures_path = args.get('metafeatures_path')
            if not metafeatures_path is None:
                metafeatures_file = os.path.join(metafeatures_path, args['dataset'] + '_metafeatures.pkl')
                if os.path.isfile(metafeatures_file):
                    m_f = open(metafeatures_file, 'rb')
                    self.dataset_metafeatures = pickle.load(m_f)[args['dataset']]

        if self.dataset_metafeatures is None:
            logger.warning('No Dataset Metafeatures specified - Initializing to empty')
            self.dataset_metafeatures = []
        else:
            self.dataset_metafeatures = list(np.nan_to_num(np.asarray(self.dataset_metafeatures)))
            
        self.m = len(self.dataset_metafeatures)+2
        self.p = input['PIPELINE_SIZE']
        self.action_size = 0

                
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        b.set_metafeatures(self.dataset_metafeatures+[self.data_types[self.data_type]]+[self.problem_types[self.problem]])
        self.action_size = len(b.valid_moves)
        return b.pieces_m + b.pieces_p

    def getBoardSize(self):
        # (a,b) tuple
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        return b.get_board_size()

    def getActionSize(self):
        # return number of actions
        board = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        return len(board.valid_moves)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        b.set_metafeatures(board)
        b.set_pipeline(board)
        #logger.info('PREV STATE %s', b.pieces_p)
        b.execute_move(action, player)
        #logger.info('NEXT STATE %s', b.pieces_p)
        return (b.pieces_m+b.pieces_p, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        b.set_metafeatures(board)
        b.set_pipeline(board)
        #logger.info('CURR STATE %s', b.pieces_p)
        legalMoves =  b.get_legal_moves()
        #logger.info('VALID MOVES %s', [b.valid_moves[i] for i in range(0, len(legalMoves)) if legalMoves[i] == 1])
        return np.array(legalMoves)

    def getEvaluation(self, board):

        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        pipeline_enums = b.get_pipeline(board)
        if not any(pipeline_enums):
            return 0.0
        pipeline = b.get_pipeline_primitives(pipeline_enums)
        eval_val = self.evaluations.get(",".join(pipeline))

        if eval_val is None:
            self.steps = self.steps + 1
            try:
                eval_val = self.eval_pipeline(pipeline, 'AlphaAutoML')
            except:
                logger.warning('Error in Pipeline Execution %s', eval_val)
                traceback.print_exc()
            if eval_val is None:
                eval_val = float('inf')
            self.evaluations[",".join(pipeline)] = eval_val
            self.eval_times[",".join(pipeline)] = time.time()

        return eval_val
    
    def getGameEnded(self, board, player, eval_val=None):
        # return 0 if not ended, 1 if x won, -1 if x lost
        # player = 1

        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        b.set_metafeatures(board)
        b.set_pipeline(board)
        if not b.is_terminal_pipeline():
            return 0
        if len(self.evaluations) > 0:
            sorted_evals = sorted([eval for eval in list(self.evaluations.values()) if eval != float('inf')])
            if len(sorted_evals) > 0:
                if 'error' in self.metric.lower():
                   win_threshold = sorted_evals[0]
                else:
                   win_threshold = sorted_evals[-1]
                b.win_threshold = win_threshold

        eval_val = self.getEvaluation(board)

        if b.findWin(player, eval_val):
            logger.info('findwin %s', player)
            return 1
        if b.findWin(-player, eval_val):
            logger.info('findwin %', -player)
            return -1
        if b.has_legal_moves():
            return 0

        return 2

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return deepcopy(board)

    def stringRepresentation(self, board):
        # 3x3 numpy array (canonical board)
        return np.asarray(board).tostring()

    def getTrainBoard(self, board):
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        b.set_metafeatures(board)
        b.set_pipeline(board)
        return b.get_train_board()

    def getTrainExamples(self, board, pi):
        assert(len(pi) == self.getActionSize())  # 1 for pass
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        b.set_metafeatures(board)
        b.set_pipeline(board)
        if not b.is_terminal_pipeline():
            eval_val = float('inf')
        else:
            eval_val = self.getEvaluation(board)
        train_board = b.get_train_board()
        if 'error' in self.metric.lower():
            return (train_board, pi, eval_val if eval_val!= float('inf') and eval_val <= math.pow(10,15) else math.pow(10,15))
        else:
            return (train_board, pi, eval_val if eval_val != float('inf') else 0)

    def get_pipeline_primitives(self, board):
        b = Board(self.m, self.grammar, self.pipeline_size, self.metric)
        return b.get_pipeline_primitives(b.get_pipeline(board))
        
    def display(self, b):
        board = self.get_pipeline_primitives(b)
        logger.info("PIPELINE: %s", '|'.join(e for e in board))

