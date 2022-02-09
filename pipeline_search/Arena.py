import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)

NUM_IT = 5
import logging

logger = logging.getLogger(__name__)

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None, logfile=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        if not logfile is None:
            self.f = open(logfile, 'a')

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            winner: player who won the game (1 if player1, -1 if player2)
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0 and it <= NUM_IT:
            it+=1
            if verbose:
                self.f.write(','.join(self.game.get_pipeline_primitives(board))+'\n')
                assert(self.display)
                logger.info("Turn %s",it)
                logger.info("Player %s", curPlayer)
                self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
            #print('ACTION ', action)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            #print('VALIDS ', valids)
            if valids[action]!=0:
                #print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            self.f.write(','.join(self.game.get_pipeline_primitives(board))+'\n')            
            assert(self.display)
            logger.info("Turn %s", it)
            logger.info("Player %s", curPlayer)
            self.display(board)
        game_ended = self.game.getGameEnded(board, 1)
        if verbose:
            if game_ended == 1:
                self.f.write('Working Pipeline\n')
            elif game_ended == 2:
                self.f.write('Non-Working Pipeline\n')
        #return game_ended==1 and curPlayer == 1
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
        """
        num = int(num/2)
        oneWon = 0
        twoWon = 0
        if verbose:
            self.f.write('Round 1\n')
        for i in range(num):
            if verbose:
                self.f.write('Game '+str(i)+'\n')
            if self.playGame(verbose=verbose)==1:
                oneWon+=1
            else:
                twoWon+=1
        self.player1, self.player2 = self.player2, self.player1
        if verbose:
            self.f.write('Round 2 - Players Swapped\n')
        for i in range(num):
            if verbose:
                self.f.write('Game '+str(i)+'\n')
            if self.playGame(verbose=verbose)==-1:
                oneWon+=1
            else:
                twoWon+=1
        return oneWon, twoWon
