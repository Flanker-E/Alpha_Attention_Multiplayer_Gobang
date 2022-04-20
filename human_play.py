# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game
# from mcts_pure import MCTSPlayer as MCTS_Pure
from MCTS import MCTSPlayer as MCTS_Pure
# from mcts_alphaZero import MCTSPlayer
import MCTS_alpha
from MCTS_alpha import MCTSPlayerAlpha as MCTSPlayer
# from policy_value_net_numpy import PolicyValueNetNumpy
import argparse
from pathlib import Path
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from Net_util_pytorch import PolicyValueNet
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
from minmax_gobang_AI import *


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


def run():
    n, number_player, board_size, weights, start_player= \
        opt.number_in_row, opt.number_player, opt.width, opt.weights, opt.start
    # n = 5
    pure_mcts_playout_num=opt.pure_num
    mcts_playout_num=opt.alpha_num
    width, height = board_size, board_size
    res_num = opt.res_num
    m = opt.min_max
    # print(model_file)
    try:
        if m == True:
            go()
        else:
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
            if weights!='':
                # pass
                model_file = Path(weights) #'best_policy_8_8_5.model'
                best_policy = PolicyValueNet(width, 
                                            height, 
                                            res_num=res_num, 
                                            model_file=model_file)
                mcts_player1 = MCTSPlayer(policy_value_fn = best_policy.policy_value_fn,
                                        c_puct=5,
                                        n_playout=mcts_playout_num)  # set larger n_playout for better performance
                mcts_player2 = MCTSPlayer(policy_value_fn = best_policy.policy_value_fn,
                                        c_puct=5,
                                        n_playout=mcts_playout_num)  # set larger n_playout for better performance
                if(number_player==3):
                    mcts_player3 = MCTSPlayer(policy_value_fn = best_policy.policy_value_fn,
                                        c_puct=5,
                                        n_playout=mcts_playout_num)
            else:
                print("pure MCTS") 
                mcts_player2 = MCTS_Pure(c_puct=5,
                                        n_playout=pure_mcts_playout_num)
                mcts_player3 = MCTS_Pure(c_puct=5,
                                        n_playout=pure_mcts_playout_num)
            # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
            # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

            # human player, input your move in the format: 2,3
            human1 = Human()
            human2 = Human()
            human3 = Human()


            # set start_player=0 for human first
            # game.start_play(human, mcts_player, start_player=1, is_shown=1)
            # game.start_play(mcts_player2, mcts_player, start_player=1, is_shown=1)
            # game.start_play(human1, human2, human3, start_player=1, is_shown=1)
            if(number_player==3):
                # game.start_play(human1, pure_mcts_player1, pure_mcts_player2, start_player=start_player, is_shown=1)
                game.start_play(human1, mcts_player2, mcts_player3, start_player=start_player, is_shown=1, draw =opt.show_GUI)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights','-w', type=str, default='', help='initial weights path, init empty, empty lead to pure MCTS, weight lead to Alpha MCTS')
    parser.add_argument('--number_player','-np', type=int, default=3, help='number of players, init 3')
    parser.add_argument('--width', type=int, default=6, help='width of board, init 6')
    parser.add_argument('--number_in_row','-n', type=int, default=4, help='win condition, init 4')
    parser.add_argument('--start','-st', type=int, default=0, help='start number of players')
    parser.add_argument('--pure_num', type=int, default=2000, help='play out numbers of pure MCTS, default 2000')
    parser.add_argument('--alpha_num', type=int, default=1000, help='play out numbers of Alpha MCTS, default 1000')
    parser.add_argument('--show_GUI', nargs='?', const=True, default=False, help='draw GUI or not, default True')
    parser.add_argument('--res_num', type=int, default=0, help='res block num, init 0')
    parser.add_argument('--min_max', nargs='?', const=True, default=False, help='play with agent using minmax methods, default True')
    opt = parser.parse_args()

    run()
