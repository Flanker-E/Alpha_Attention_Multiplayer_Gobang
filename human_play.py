# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net_numpy import PolicyValueNetNumpy
import argparse
from pathlib import Path
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(opt):
    n, number_player, board_size, save_dir, weights, start_player= \
        opt.number_in_row, opt.number_player, opt.width, Path(opt.save_dir), opt.weights, opt.start
    # n = 5
    pure_mcts_playout_num=5000
    width, height = board_size, board_size
    model_file = save_dir / weights #'best_policy_8_8_5.model'
    # print(model_file)
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNetNumpy(width, height, policy_param)
        best_policy = PolicyValueNet(width, height, model_file)
        mcts_player1 = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        mcts_player2 = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        if(number_player==3):
            mcts_player3 = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)
        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human1 = Human()
        human2 = Human()
        human3 = Human()

        pure_mcts_player1 = MCTS_Pure(c_puct=5,
                                     n_playout=pure_mcts_playout_num)
        pure_mcts_player2 = MCTS_Pure(c_puct=5,
                                     n_playout=pure_mcts_playout_num)
        # set start_player=0 for human first
        # game.start_play(human, mcts_player, start_player=1, is_shown=1)
        # game.start_play(mcts_player2, mcts_player, start_player=1, is_shown=1)
        # game.start_play(human1, human2, human3, start_player=1, is_shown=1)
        if(number_player==3):
            # game.start_play(human1, pure_mcts_player1, pure_mcts_player2, start_player=start_player, is_shown=1)
            game.start_play(mcts_player1, mcts_player2, mcts_player3, start_player=start_player, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights','-w', type=str, default='best_policy.model', help='initial weights path')
    parser.add_argument('--save_dir', type=str, default='models', help='save to project/name')
    parser.add_argument('--number_player','-np', type=int, default=3, help='number of players')
    parser.add_argument('--width', type=int, default=6, help='width of board')
    parser.add_argument('--number_in_row','-n', type=int, default=4, help='win condition')
    parser.add_argument('--start','-st', type=int, default=0, help='start number of players')
    opt = parser.parse_args()

    run(opt)
