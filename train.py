# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
import pickle
from game import Board, Game
import time
import re
import copy
import multiprocessing as mp
from multiprocessing import Pool
# from mcts_pure import MCTSPlayer as MCTS_Pure
# from mcts_alphaZero import MCTSPlayer
from MCTS import MCTSPlayer as MCTS_Pure
from MCTS_alpha import MCTSPlayerAlpha as MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from Net_util_pytorch import PolicyValueNet

import argparse
from pathlib import Path


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = opt.width  #default 6
        self.board_height = opt.width
        self.n_in_row = opt.number_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = opt.learn_rate
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = opt.n_playout  # default 800 num of simulations for each move
        self.c_puct = 5
        if opt.record:
            self.buffer_size = 30000
        else:
            self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        if(opt.multiprocessing):
            self.play_batch_size = 2
            self.process_num=2
        else:
            self.play_batch_size = 1
            self.process_num=1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = opt.kl_target
        self.check_freq = opt.check_freq  # default 50
        self.game_batch_num = opt.max_batch
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = opt.init_playout
        self.res_num = opt.res_num  #init 0
        self.atten_num = opt.atten_num  #init 0
        self.atten = opt.atten
        self.use_gpu = opt.use_gpu
        atten_cad_blk_num=opt.atten_cad_blk_num
        self.multiprocessing=opt.multiprocessing
        
        self.policy_value_net=[]
        self.mcts_player=[]
        if init_model:
            # start training from an initial policy-value net
            for i in range(self.process_num):
                self.policy_value_net.append (PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   res_num=self.res_num,
                                                   atten_num=self.atten_num,
                                                   use_gpu=self.use_gpu,
                                                   model_file=init_model,
                                                   atten=self.atten,
                                                   drop=opt.drop,
                                                   atten_cad_blk_num=atten_cad_blk_num))
            print("NEW policyvaluenet load from weight file")
        else:
            # start training from a new policy-value net
            for i in range(self.process_num):
                self.policy_value_net.append (PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   res_num=self.res_num,
                                                   atten_num=self.atten_num,
                                                   use_gpu=self.use_gpu,
                                                   atten=self.atten,
                                                   drop=opt.drop,
                                                   atten_cad_blk_num=atten_cad_blk_num))
            print("NEW policyvaluenet")
        
        for i in range(self.process_num):
                self.mcts_player.append ( MCTSPlayer(
            policy_value_fn=self.policy_value_net[i].policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1))
        print("NEW Alpha mcts player")

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(
                        mcts_porb.reshape(self.board_height,
                                          self.board_width)), i)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, mcts_player, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
    
    def multip_collect_selfplay_data(self, que, mcts_player, n_games=1):
        """collect self-play data for training"""
        print("start sub task")
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            # self.data_buffer.extend(play_data)
            que.put([episode_len,play_data])
        print("end sub task")

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        # print("shape of training data",np.shape(state_batch))

        # start_epoch = time.time()
        old_probs, old_v = self.policy_value_net[0].policy_value(state_batch)
        # print(old_probs,old_v)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net[0].train_step(
                state_batch, mcts_probs_batch, winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net[0].policy_value(state_batch)
            kl = np.mean(
                np.sum(old_probs *
                       (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                       axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # end_epoch = time.time()
        # elapsed = end_epoch - start_epoch
        # avg_time = elapsed* 1000
        # print("training timg {:.5f}ms".format(avg_time))
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}").format(kl, self.lr_multiplier, loss,
                                                  entropy, explained_var_old,
                                                  explained_var_new))
        ret_list = [
            kl, self.lr_multiplier, loss, entropy, explained_var_old,
            explained_var_new
        ]
        return ret_list

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(
            # policy_value_fn=self.policy_value_net.policy_value_fn,
            policy_value_fn=self.policy_value_net[0].policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout)
        pure_mcts_player1 = MCTS_Pure(c_puct=5,
                                      n_playout=self.pure_mcts_playout_num)
        pure_mcts_player2 = MCTS_Pure(c_puct=5,
                                      n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player1,
                                          pure_mcts_player2,
                                          start_player=i % 3,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[0] + 0.5 * win_cnt[-1]) / n_games
        print(
            "num_playouts:{}, win: {}, lose 1: {}, lose 2: {}, tie:{}".format(
                self.pure_mcts_playout_num, win_cnt[0], win_cnt[1], win_cnt[2],
                win_cnt[-1]))
        return win_ratio, win_cnt

    def run(self):
        """run the training pipeline"""
        try:
            # create dir
            desc = '_' + str(opt.width) + '_' + str(opt.width) + '_' + str(
                opt.number_in_row)
            dir = Path('models/' + opt.save_dir + desc)
            training_data = None
            evaluate_data = None
            if not dir.exists():
                dir.mkdir(parents=True, exist_ok=True)  # make directory
            init_batch = 0
            if opt.resume:
                init_batch = opt.init_batch
                print("resume from batch: ", init_batch)
            for i in range(init_batch, self.game_batch_num):
                start_epoch = time.time()
                # self.collect_selfplay_data(self.play_batch_size)
                if self.multiprocessing:
                    p = Pool(2)
                    manager = mp.Manager()
                    que = manager.Queue()
                    self.episode_len=0
                    print("start multi task")
                    for n in range(2):
                        p.apply_async(self.multip_collect_selfplay_data, args=(que,self.mcts_player[n], self.play_batch_size//2,))
                    p.close()
                    p.join()
                    for n in range(2):
                        episode_len, play_data= que.get()
                        # augment the data
                        self.episode_len+=episode_len
                        self.data_buffer.extend(play_data)
                else:
                    self.collect_selfplay_data(self.mcts_player[0],self.play_batch_size)
                print("len of deque",len(self.data_buffer))
                print("batch i:{}, episode_len:{}".format(
                    i*self.play_batch_size + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    for n in range(self.play_batch_size):
                        ret_list = self.policy_update()
                        ret_list.insert(0, i*self.play_batch_size+n)
                    if self.multiprocessing:
                        self.policy_value_net[1].policy_value_net.load_state_dict(copy.deepcopy(self.policy_value_net[0].policy_value_net.state_dict()))
                else:
                    ret_list = [i*self.play_batch_size, 0, 0, 0, 0, 0, 0]
                end_epoch = time.time()
                elapsed = end_epoch - start_epoch
                avg_time = elapsed * 1000
                print("training epoch time {:.5f}ms".format(avg_time))
                ret_list.insert(7, avg_time)
                if training_data is None:
                    training_data = np.array(ret_list).reshape(
                        -1, len(ret_list))
                else:
                    training_data = np.concatenate(
                        (training_data, np.array(ret_list).reshape(
                            -1, len(ret_list))),
                        axis=0)
                np.save(dir / 'training_data', training_data)
                # check the performance of the current model,
                # and save the model params

                # if (i + 1) % self.check_freq == 0:
                if ((i+1)*self.play_batch_size ) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio, win_cnt = self.policy_evaluate()
                    # save data
                    eva_list = np.array([
                        (i + 1)*self.play_batch_size, self.pure_mcts_playout_num, win_ratio,
                        win_cnt[0], win_cnt[1], win_cnt[2], win_cnt[-1]
                    ])
                    if evaluate_data is None:
                        evaluate_data = eva_list.reshape(-1, len(eva_list))
                    else:
                        evaluate_data = np.concatenate(
                            (evaluate_data, eva_list.reshape(
                                -1, len(eva_list))),
                            axis=0)
                    np.save(dir / 'evaluate_data', evaluate_data)
                    model_path = dir / ('current_policy_' + str(i + 1) +
                                        '.model')
                    self.policy_value_net[0].save_model(str(model_path))
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        model_path = dir / ('best_policy_' + str(i + 1) +
                                            '.model')
                        self.policy_value_net[0].save_model(str(model_path))
                        if (self.best_win_ratio >= 0.8
                                and self.pure_mcts_playout_num <
                                opt.max_playout):  # default 9000
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

    def record(self, opt):
        string_list = re.split(r'[.]', opt.weights)
        print("start recording: ", string_list[0])
        while True:
            # start_epoch = time.time()
            self.collect_selfplay_data(self.play_batch_size)
            total_len = len(self.data_buffer)
            if total_len >= self.buffer_size:
                f = open(string_list[0] + "_data_buffer.data", 'wb')
                pickle.dump(self.data_buffer, f)
                f.close()
                break
            print("episode_len:{}, total_len:{}".format(
                self.episode_len, total_len))

    def supervised(self, opt):
        try:
            data_path = opt.supervised_data
            self.data_buffer = pickle.load(open(data_path, 'rb'))
            # string_list=re.split('current|best',data_path)
            # path=string_list
            print("path of training data: ", data_path)

            # create dir
            desc = '_' + "supervised" + '_' + str(opt.width) + '_' + str(
                opt.width) + '_' + str(opt.number_in_row)
            dir = Path('models/' + opt.save_dir + desc)
            training_data = None
            evaluate_data = None
            if not dir.exists():
                dir.mkdir(parents=True, exist_ok=True)  # make directory

            for i in range(0, self.game_batch_num):
                start_epoch = time.time()
                # self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}".format(i + 1))
                # if len(self.data_buffer) > self.batch_size:
                ret_list = self.policy_update()
                ret_list.insert(0, i)
                # else:
                #     ret_list = [i, 0, 0, 0, 0, 0, 0]
                if training_data is None:
                    training_data = np.array(ret_list).reshape(
                        -1, len(ret_list))
                else:
                    training_data = np.concatenate(
                        (training_data, np.array(ret_list).reshape(
                            -1, len(ret_list))),
                        axis=0)
                np.save(dir / 'training_data', training_data)
                # check the performance of the current model,
                # and save the model params
                end_epoch = time.time()
                elapsed = end_epoch - start_epoch
                avg_time = elapsed * 1000
                print("training epoch time {:.5f}ms".format(avg_time))
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio, win_cnt = self.policy_evaluate(n_games=3)
                    # save data
                    eva_list = np.array([
                        i + 1, self.pure_mcts_playout_num, win_ratio,
                        win_cnt[0], win_cnt[1], win_cnt[2], win_cnt[-1]
                    ])
                    if evaluate_data is None:
                        evaluate_data = eva_list.reshape(-1, len(eva_list))
                    else:
                        evaluate_data = np.concatenate(
                            (evaluate_data, eva_list.reshape(
                                -1, len(eva_list))),
                            axis=0)
                    np.save(dir / 'evaluate_data', evaluate_data)
                    model_path = dir / ('current_policy_' + str(i + 1) +
                                        '.model')
                    self.policy_value_net.save_model(str(model_path))
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        model_path = dir / ('best_policy_' + str(i + 1) +
                                            '.model')
                        self.policy_value_net.save_model(str(model_path))
                        if (self.best_win_ratio >= 0.6
                                and self.pure_mcts_playout_num <
                                opt.max_playout):  # default 9000
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        '-w',
                        type=str,
                        default='',
                        help='initial weights path, init empty')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models',
        help=
        'save to models/save_dir+desc/xxxx, desc include width, num in row, init model'
    )
    # parser.add_argument('--name', type=str, default='', help='save to save_dir/xxxx+name')
    parser.add_argument('--number_player',
                        '-np',
                        type=int,
                        default=3,
                        help='number of players, init 3')
    parser.add_argument('--width',
                        type=int,
                        default=6,
                        help='width of board, init 6')
    parser.add_argument('--number_in_row',
                        '-n',
                        type=int,
                        default=4,
                        help='win condition, init 4')
    parser.add_argument('--n_playout',
                        type=int,
                        default=800,
                        help='Alpha MCTS playout num, init 800')
    parser.add_argument('--res_num',
                        type=int,
                        default=0,
                        help='res block num, init 0')
    parser.add_argument('--atten_num',
                        type=int,
                        default=0,
                        help='attention block num, init 0')
    parser.add_argument('--check_freq',
                        type=int,
                        default=50,
                        help='performance check freq, init 50')
    parser.add_argument('--init_playout',
                        type=int,
                        default=1000,
                        help='initial pure MCTS playout, init 1000')
    parser.add_argument('--max_playout',
                        type=int,
                        default=9000,
                        help='max pure MCTS playout, init 9000')

    parser.add_argument('--resume',
                        nargs='?',
                        const=True,
                        default=False,
                        help='resume most recent training, init False')
    parser.add_argument('--init_batch',
                        type=int,
                        default=0,
                        help='initial batch number, init 0')
    parser.add_argument('--max_batch',
                        type=int,
                        default=3000,
                        help='max batch number, init 3000')
    parser.add_argument('--learn_rate',
                        '-lr',
                        type=float,
                        default=2e-3,
                        help='base learning rate, init 2e-3')
    parser.add_argument('--kl_target',
                        type=float,
                        default=2e-2,
                        help='D_kl target, init 0.02')
    parser.add_argument('--use_gpu',
                        nargs='?',
                        const=True,
                        default=False,
                        help='using gpu, init False')
    parser.add_argument('--atten',
                        nargs='?',
                        const=True,
                        default=False,
                        help='using attention, init False')
    parser.add_argument('--drop',
                        type=float,
                        default=0.,
                        help='universal drop rate, init 0.')
    parser.add_argument('--atten_cad_blk_num',
                        '-blk_num',
                        type=int,
                        default=4,
                        help='attention cascad block num, init 4')
    parser.add_argument(
        '--record',
        nargs='?',
        const=True,
        default=False,
        help='record data from a model, init False, need weights input')
    parser.add_argument(
        '--supervised_data',
        '-sw',
        type=str,
        default='',
        help=
        'training data recorded for supervised learning, input to train a model, init empty'
    )
    parser.add_argument('--depths',
                        nargs='*',
                        type=int,
                        required=False,
                        default=[1, 1, 1, 1],
                        help='depths of attention blocks, default [1,1,1,1]')
    parser.add_argument('--multiprocessing',
                        '-mp',
                        nargs='?',
                        const=True,
                        default=False,
                        help='using 2 multiprocessing, init False')
    # parser.add_argument(
    #     '--test',
    #     nargs='?',
    #     const=True,
    #     default=False,
    #     help='record data from a model, init False, need weights input')
    opt = parser.parse_args()

    model_file = None
    if opt.weights != '':
        model_file = opt.weights
    training_pipeline = TrainPipeline(model_file)
    if opt.record:
        if model_file == '':
            raise ValueError("need model file to record data")
        training_pipeline.record(opt)
    elif opt.supervised_data != '':
        training_pipeline.supervised(opt)
    else:
        training_pipeline.run()
