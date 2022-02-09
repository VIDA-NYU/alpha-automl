import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

np.random.seed(0)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vals = {}
        self.Vs = {}  # stores game.getValidMoves for board s
        self.count = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.get('numMCTSSims')):
            logger.info('MCTS SIMULATION %s', i + 1)
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [0] * self.game.getActionSize()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            if np.sum(probs) == 0:
                logger.info('PROB ZERO')
            return probs

        counts = [x ** (1. / temp) for x in counts]
        if np.sum(counts) == 0:
            probs = [1 / (len(counts))] * len(counts)
        else:
            non_zero_args = list(np.where(np.asarray(counts) > 0)[0])
            probs = [0] * len(counts)
            for index in non_zero_args:
                probs[index] = counts[index] / float(sum(counts))
        if np.sum(probs) == 0:
            logger.info('PROB ZERO')
        return probs

    def search(self, canonicalBoard, player=1):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        self.game.display(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        game_ended = self.game.getGameEnded(canonicalBoard, player)
        # logger.info('GAME ENDED %s', game_ended)

        if s not in self.Es:
            self.Es[s] = game_ended
        if self.Es[s] != 0:
            # terminal node
            # Clear all previous moves
            return self.Vals[s] if not self.Vals.get(s) is None else 0

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(self.game.getTrainBoard(canonicalBoard))
            logger.info('Prediction %s', v)
            # logger.info('CALLING VALID MOVES')
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            self.Ps[s] /= np.sum(self.Ps[s])  # renormalize
            self.Vals[s] = v
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]

        # Check if valid moves are available. Quit if no more legal moves are possible
        if not any(valids):
            return 0

        cur_best = -float('inf')
        best_act = -1
        current_primitives = self.game.get_pipeline_primitives(canonicalBoard)
        current_pattern = ' '.join(
            [self.game.grammar['RULES_PROBA']['TYPES'].get(p, {'type': p})['type'] for p in current_primitives])
        # pick the action with the highest upper confidence bound
        actions = []
        alpha = 0.1
        if len(self.game.grammar['RULES_PROBA']['GLOBAL']) == 0:  # It's a manual grammar, so only use the NN info
            alpha = 1
        for a in range(self.game.getActionSize()):
            if valids[a]:
                # logger.info('MCTS ACTION %s', a)
                global_proba = self.game.grammar['RULES_PROBA']['GLOBAL'].get(a + 1, (0, 0))[1]
                local_proba = self.game.grammar['RULES_PROBA']['LOCAL'].get(current_pattern, {}).get(a + 1, (0, 0))[1]
                correlation = global_proba * local_proba

                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.get('cpuct') * (
                                alpha * self.Ps[s][a] + (1 - alpha) * correlation) * math.sqrt(self.Ns[s]) / (
                                    1 + self.Nsa[(s, a)])
                    # u = (global_proba + local_proba) + self.Qsa[(s, a)] + self.args.get('cpuct') * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                    # u = self.Qsa[(s, a)] + self.args.get('cpuct') * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.get('cpuct') * (alpha * self.Ps[s][a] + (1 - alpha) * correlation) * math.sqrt(
                        self.Ns[s])  # Q = 0 ?
                    # u = (global_proba + local_proba) + self.args.get('cpuct') * math.sqrt(self.Ns[s]) # Q = 0 ?
                    # u = self.args.get('cpuct') * self.Ps[s][a] * math.sqrt(self.Ns[s])  # Q = 0 ?
                # print(self.game.grammar['RULES_PROBA']['GLOBAL'].get(a + 1, ('None', 0))[0], correlation)
                # print('u', u)
                if u > cur_best:
                    cur_best = u
                    best_act = a
                    actions = [a]
                elif u == cur_best:
                    actions.append(a)
        if len(actions) > 1:
            # print('random')
            a = np.random.choice(np.asarray(actions))
        else:
            a = best_act
        # print('>>>>>>>>>>best a=%d' % a, 'global', self.game.grammar['RULES_PROBA']['GLOBAL'].get(a + 1, ('None', 0)))
        # print('>>>>>>>>>>best a=%d' % a, 'local', self.game.grammar['RULES_PROBA']['LOCAL'].get(current_pattern, {}).get(a + 1, ('None', 0)))
        # print('>>>>>>>>>>best a=%d' % a, 'total',  self.game.grammar['RULES_PROBA']['GLOBAL'].get(a+1, (0, 0))[1] * self.game.grammar['RULES_PROBA']['LOCAL'].get(current_pattern, {}).get(a + 1, (0, 0))[1])
        # logger.info('BEST ACTIONS %s', actions)
        # logger.info('MCTS BEST ACTION %s', best_act)
        next_s, next_player = self.game.getNextState(canonicalBoard, player, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        self.game.display(next_s)

        # logger.info('NEXT STATE SEARCH RECURSION')
        v = self.search(next_s, next_player)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return v
