import numpy as np
from MCTS import MCTS
from MCTS import MCTSPlayer
from MCTS import TreeNode
import copy


def Nsoftmax(x, temp):
    x = x**(1.0 / temp)
    probs = x - np.max(x)
    probs /= np.sum(probs)
    return probs


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTSPlayerAlpha(MCTSPlayer):
    def __init__(self, **kwargs) -> None:
        who_play = kwargs.get('who_play', 'alpha_MCTS')
        self.mcts = None
        self.self_play = kwargs.get('is_selfplay', False)
        if who_play == 'alpha_MCTS':
            policy_value_fn = kwargs.get('policy_value_fn')
            c_puct = kwargs.get('c_puct', 5)
            n_playout = kwargs.get('n_playout', 800)
            num_player = kwargs.get('num_player', 3)
            self.mcts = MCTSAlpha(c_puct,
                                  n_playout,
                                  num_player,
                                  policy_value_fn=policy_value_fn)

    def get_action(self, board_state):
        available_move = board_state.availables
        if len(available_move) > 0:
            next_move, probs = self.mcts.get_move(board_state)
            if self.self_play:
                # dirichlet
                probs=0.75*probs+0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                next_move = np.random.choice(next_move, p=probs)
                self.mcts.update_with_move(next_move)
            else:
                # next_move, probs = self.mcts.get_move(board_state)
                next_move = np.random.choice(next_move, p=probs)
                self.mcts.update_with_move(-1)
            return next_move
        else:
            print("board is full")

    def __str__(self) -> str:
        return "alpha MCTS player"


class MCTSAlpha(MCTS):
    def __init__(self,
                 c_puct,
                 n_playout,
                 num_player,
                 policy_value_fn=None) -> None:
        super(MCTSAlpha, self).__init__(c_puct, n_playout, num_player,
                                        policy_value_fn)

    def _playout(self, board_state):
        cur_node = self._root
        # self.test
        # print("playout")
        while (True):
            #search leaf
            # print("node_child",cur_node.children,cur_node.is_leaf())
            if cur_node.is_leaf():
                break
            move, cur_node = cur_node.select()
            # print("move",move)
            board_state.do_move(move)
        end, winner = board_state.game_end()
        action_prior, leaf_val = self._policy_value_fn(board_state)
        cur_player = board_state.get_current_player()
        if not end:

            cur_node.expand(action_prior)
            winner = cur_player
        else:
            if winner == -1:
                leaf_val = 0
            else:
                leaf_val = 1
        # else:
        # print(winner)

        cur_node.update_recursive(
            winner, (cur_player + (self.num_player - 1)) % self.num_player,
            leaf_val=leaf_val)
        return cur_node

    def get_move(self, board_state, temp=1e-3):
        for i in range(self._n_playout):
            cur_board_state = copy.deepcopy(board_state)
            cur_node = self._playout(cur_board_state)
        # move, prob =self._root.select(print=True)
        act_prob = [(act, node._n_visits)
                    for act, node in self._root.children.items()]
        move, visits = zip(*act_prob)
        # prob = Nsoftmax(np.array(visits), temp)
        prob = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return move, prob