import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class PolicyValueNet(object):
    def __init__(self) -> None:
        pass

    def policy_value(self, state_batch):
        pass

    def policy_value_fn(self, board):
        pass

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        pass

    def get_policy_param(self):
        pass

    def save_model(self, model_file):
        pass


class Net(nn.modules):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, X):
        pass


class ResBlock(nn.modules):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, X):
        pass