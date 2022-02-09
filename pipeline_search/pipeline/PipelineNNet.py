import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PipelineNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.action_size = game.getActionSize()
        self.board_size = game.getBoardSize()
        self.problem = game.problem
        self.args = args

        super(PipelineNNet, self).__init__()
        hlayer = 512
        torch.manual_seed(1)
        self.lstm = nn.LSTM(self.board_size, hlayer, 2)
        self.probFC = nn.Linear(hlayer, self.action_size)
        self.valueFC = nn.Linear(hlayer, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_size)
        lstm_out, hidden = self.lstm(s)
        s = lstm_out[:,-1]
        pi = self.probFC(s)                                                                         # batch_size x 512
        v = self.valueFC(s)                                                                          # batch_size x 512
                
        if self.problem == 'CLASSIFICATION':
            return F.log_softmax(pi, 1), F.sigmoid(v)
        else:
            return F.log_softmax(pi, 1), v
