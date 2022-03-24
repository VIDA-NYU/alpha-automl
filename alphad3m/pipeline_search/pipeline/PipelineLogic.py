'''
Board class.
Board data:
  1=x, -1=o, 0=empty
  first dim is column , 2nd is row:
     pieces[1][2] is the square in column 2, row 3.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Board():

    def __init__(self, m=30, grammar={}, pipeline_size=6, metric='f1macro', win_threshold=0.6):
        "Set up initial board configuration."

        self.terminals = grammar['TERMINALS']
        self.non_terminals = grammar['NON_TERMINALS']
        self.start = grammar['START']
        self.m = m  # Number of metafeatures
        self.p = pipeline_size #max length of pipeline
        self.valid_moves = [i for i, j in sorted(grammar['RULES'].items(), key=lambda x: x[1])]
        self.rules = grammar['RULES']
        self.rules_lookup = grammar['RULES_LOOKUP']
        #logger.info('NUMBER of VALID MOVES %s', len(self.valid_moves))
        
        # Create the empty board array.
        self.pieces_m = [0] * self.m
        start_pipeline = [self.terminals[p] if p in self.terminals else self.non_terminals[p] for p in
                    [self.start]]

        self.pieces_p = [0] * (self.p - len(start_pipeline)) + start_pipeline

        if 'error' in metric.lower():  # TODO: change it
            win_threshold = -1 * win_threshold
        self.win_threshold = win_threshold

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces_p[index]

    def get_pipeline(self, board):
        return board[self.m:]

    def get_metafeatures(self, board):
        return board[0:self.m]

    def set_metafeatures(self, board):
        self.pieces_m = board[0:self.m]

    def set_pipeline(self, board):
        self.pieces_p = board[self.m:]

    def is_terminal_pipeline(self):
        for p in self.pieces_p:
            if p != 0 and p in list(self.non_terminals.values()):  # Empty symbol ID = 0
                return False
        return True
    
    def findWin(self, player, eval_val=None):
        """Find win of the given color in row, column, or diagonal
        (1 for x, -1 for o)"""
        if not any(self[0:]):
            return False
        if eval_val == float('inf'):
            return False

        return eval_val >= self.win_threshold
    
    def get_legal_moves(self):
        """Returns all the legal moves.
        """

        valid_moves = np.asarray([0]*len(self.valid_moves))
        pipeline = [p for p in self.pieces_p if p != 0]

        for p in pipeline:
            if p in list(self.non_terminals.values()):
                #logger.info('GET LEGAL MOVES %s', p)
                rules = [self.rules[key]-1 for key in self.rules_lookup[list(self.non_terminals.keys())[p-1]] if len(self.next_state(self.rules[key]-1)) <= self.p and len(set(self.next_state(self.rules[key]-1))) == len(self.next_state(self.rules[key]-1))]
                np.put(valid_moves, rules, [1]*len(rules))

        return valid_moves.tolist()
        
    def has_legal_moves(self):
        return len(np.where(np.asarray(self.get_legal_moves()) == 1)[0]) > 0

    def next_state(self, action):
        s = self.valid_moves[action]
        nt = self.non_terminals[s[:s.index('-')].strip()]
        r = [self.non_terminals[p] if p in self.non_terminals.keys() else self.terminals[p] for p in s[s.index('-')+2:].strip().split(' ')]
        r = [x for x in r if x != 0]
        s = []
        for p in self.pieces_p:
            if p == 0:
                continue
            
            if p == nt:
                s += r
            else:
                s.append(p)

        return s

    def get_pipeline_primitives(self, pipeline):
        #logger.info('PIPELINE PRIMITIVES FOR %s', pipeline)
        return [list(self.terminals.keys())[list(self.terminals.values()).index(i)] if i in list(self.terminals.values()) else list(self.non_terminals.keys())[list(self.non_terminals.values()).index(i)] for i in pipeline if not i == 0]

    def get_train_board(self):
        logger.info('TRAIN BOARD: %s', '|'.join(self.get_pipeline_primitives(self.pieces_p)))
        pipeline = [0]*(len(self.terminals)+len(self.non_terminals))

        for p in self.pieces_p:
            if p != 0:
                 pipeline[p] = 1
            #pipeline[p-1] = 1

        return self.pieces_m + pipeline

    def get_board_size(self):
        return self.m+(len(self.terminals)+len(self.non_terminals))
    
    def execute_move(self, action, player):
        """Perform the given move on the board;
        color gives the color of the piece to play (1=x,-1=o)
        """
        logger.info('MOVE ACTION: %s', self.valid_moves[action])
        s = self.next_state(action)
        
        s = [0] * (self.p - len(s)) + s

        self.pieces_p = s


