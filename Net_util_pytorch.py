import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from mix_transformer import *
from mix_transformer_simple import MixVisionTransformer as MixSimple


class PolicyValueNet(object):
    def __init__(self,
                 board_width=8,
                 board_height=8,
                 res_num=0,
                 atten_num=0,
                 use_gpu=False,
                 model_file='',
                 player_num=3,
                 atten=False,
                 drop=0.,
                 atten_cad_blk_num=4,
                 depths=[1,1,1,1]) -> None:
        device = "cpu"
        if use_gpu and torch.cuda.is_available():
            print("using GPU!")
            device = "cuda:0"
        self.device = torch.device(device)
        if atten:
            print("pure attention")
        else:
            print("res block num: ",res_num)
            print("attention block num: ",atten_num)
        
        self.player_num = player_num
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty

        if atten:
            if board_width in [8, 11]:
                if board_width == 8:
                    self.policy_value_net = MixSimple(
                        drop_rate=drop,
                        attn_drop_rate=drop,
                        drop_path_rate=drop).to(self.device)
                else:
                    self.policy_value_net = MixSimple(
                        img_size=[11, 6, 2, 1],
                        drop_rate=drop,
                        attn_drop_rate=drop,
                        drop_path_rate=drop).to(self.device)
        else:
            self.policy_value_net = Net(board_width, board_height, player_num,
                                        res_num, atten_num).to(self.device)

        if model_file != '':
            self.policy_value_net.load_state_dict(torch.load(model_file))

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

    def policy_value(self, state_batch):

        input = torch.from_numpy(np.array(state_batch)).float().to(self.device)
        probs_torch, value_torch = self.policy_value_net(input)
        probs = np.exp(probs_torch.detach().cpu().numpy())
        value = value_torch.detach().cpu().numpy()

        return probs, value

    def policy_value_fn(self, board):
        available_move = board.availables
        # np.copy(), np.ascontiguousarray is also viable.
        # For numpy may not store array contiguoutly(ndim >= 1) in memory (C order)
        input = torch.from_numpy(board.current_state().copy()).float().to(
            self.device).view(-1, self.player_num * 2, self.board_width,
                              self.board_height)
        probs_torch, value_torch = self.policy_value_net(input)
        probs = np.exp(probs_torch.detach().cpu().numpy().flatten())
        value = value_torch.item()
        # print(np.shape(probs))
        # print(np.shape(available_move))
        act_probs = zip(available_move, probs[available_move])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        self.optimizer.zero_grad()

        for param in self.optimizer.param_groups:
            param['lr'] = lr

        state_batch = torch.from_numpy(np.array(state_batch)).float().to(
            self.device)
        mcts_probs = torch.from_numpy(np.array(mcts_probs)).float().to(
            self.device)
        winner_batch = torch.from_numpy(np.array(winner_batch)).float().to(
            self.device)

        log_prob_batch, val_batch = self.policy_value_net(state_batch)

        loss_val = F.mse_loss(val_batch.view(-1), winner_batch)
        loss_act = -torch.mean(torch.sum(mcts_probs * log_prob_batch, dim=1))
        loss = loss_val + loss_act
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(
            torch.sum(torch.exp(log_prob_batch) * log_prob_batch, dim=1))

        return loss.item(), entropy.item()

    # def get_policy_param(self):
    #     pass

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)


class Net(nn.Module):
    def __init__(self, board_width, board_height, player_num, res_num,
                 atten_num) -> None:
        super().__init__()
        self.res_num=res_num
        self.atten_num=atten_num
        self.board_width = board_width
        self.board_height = board_height
        self.conv1 = nn.Conv2d(player_num * 2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.attenblocks = nn.Sequential(
            *(atten_num *
              [Block(dim=64, num_heads=2, mlp_ratio=1, sr_ratio=4)]))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*(res_num * [ResBlock(128, 128)]))
        

        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)

        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        if self.atten_num:
            X = X.flatten(2).transpose(1, 2)
            # X = self.attenblocks(X)
            for i, blk in enumerate(self.attenblocks):
                X = blk(X, self.board_width, self.board_height)
            X = X.transpose(1, 2).reshape(-1, 64, self.board_width,
                                        self.board_height)
        
        X = F.relu(self.conv3(X))
        X = self.resblocks(X)
        

        act_x = F.relu(self.act_conv1(X))
        act_x = self.act_fc1(
            act_x.view(-1, 4 * self.board_width * self.board_height))
        act_x = F.log_softmax(act_x, dim=1)

        val_x = F.relu(self.val_conv1(X))
        val_x = self.val_fc1(
            val_x.view(-1, 2 * self.board_width * self.board_height))
        val_x = self.val_fc2(val_x)
        val_x = torch.tanh(val_x)

        return act_x, val_x


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,
                               output_channel,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel,
                               output_channel,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, X):
        output = self.conv1(X)
        output = F.relu(self.bn1(output))
        output = self.bn1(self.conv2(X))
        output += X
        return F.relu(output)