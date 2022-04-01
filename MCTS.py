import numpy as np
import copy
from operator import itemgetter
# from sympy import expand


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


class MCTSPlayer(object):
    def __init__(self, **kwargs) -> None:
        who_play = kwargs.get('who_play', 'pure_MCTS')
        self.mcts = None
        self.player = None
        if who_play == 'pure_MCTS':
            policy_value_fn = kwargs.get('policy_value_fn')
            c_puct = kwargs.get('c_puct', 5)
            n_playout = kwargs.get('n_playout', 2000)
            num_player = kwargs.get('num_player', 3)
            self.mcts = MCTS(c_puct, n_playout, num_player)
        # pass
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board_state):
        available_move = board_state.availables
        if len(available_move) > 0:
            next_move, _ = self.mcts.get_move(board_state)
            self.mcts.update_with_move(-1)
            return next_move
        else:
            print("board is full")

    def __str__(self) -> str:
        return "base MCTS player"


class MCTS(object):
    def __init__(self,
                 c_puct,
                 n_playout,
                 num_player,
                 policy_value_fn=None) -> None:
        self._root = TreeNode(None, 1, num_player, c_puct)
        self.num_player = num_player
        print("num_player", num_player)
        # self._policy_value_fn=None
        if policy_value_fn:
            self._policy_value_fn = policy_value_fn
        else:
            self._policy_value_fn = self.naive_policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def naive_policy_value_fn(self, board_state):
        available_move = board_state.availables
        len_avail = len(available_move)
        prior = list(np.ones(len_avail) / len_avail)
        return zip(available_move, prior), 0

    def get_move(self, board_state):
        for i in range(self._n_playout):
            cur_board_state = copy.deepcopy(board_state)
            self._playout(cur_board_state)
        move, prob = self._root.select(print=False)
        return move, prob

    def _playout(self, board_state):
        cur_node = self._root
        # cur_player=board_state.get_current_player()
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
            # np.random.choice(p=prob)
            # print(action_prior)
            cur_node.expand(action_prior)
            winner = self.simulation_rollout(board_state)
        # else:
        # print(winner)

        cur_node.update_recursive(
            winner, (cur_player + (self.num_player - 1)) % self.num_player)

        # cur_node.update_recursive(winner,cur_player)

    def simulation_rollout(self, board_state, limit=1000):

        for i in range(limit):
            available_move = board_state.availables
            if len(available_move) > 0:
                # max_action=np.random.choice(available_move)
                action = rollout_policy_fn(board_state)
                max_action = max(action, key=itemgetter(1))[0]
                board_state.do_move(max_action)
                end, winner = board_state.game_end()
                if end:
                    return winner
            else:
                return -1
        print("reach rollout limit")
        return -1

    def update_with_move(self, last_move):
        if last_move == -1:
            self._root = TreeNode(None, 1, self.num_player, self._c_puct)
        else:
            self._root = self._root.children[last_move]
            self._root.parent = None


class TreeNode(object):
    def __init__(self, parent, p, num_player, c_puct=5) -> None:
        self.parent = parent
        self.children = {}
        self._n_visits = 0
        self._Q = 0
        self._U = 0
        self._P = p
        self._num_player = num_player
        self._c_puct = c_puct

    def expand(self, action_prior):
        for action, prior in action_prior:
            if action not in self.children:
                self.children[action] = TreeNode(self, prior, self._num_player,
                                                 self._c_puct)

    def update_recursive(self, winner, cur_player, leaf_val=1):
        if self.parent != None:
            self.parent.update_recursive(
                winner,
                (cur_player + (self._num_player - 1)) % self._num_player)
        if winner == -1:
            self.update(0)
        else:
            if winner == cur_player:
                self.update((self._num_player - 1) * leaf_val)
            else:
                self.update(-1 * leaf_val)

    def is_leaf(self):
        return self.children == {}

    def select(self, print=False):
        return max(self.children.items(),
                   key=lambda act_dict: act_dict[1].get_value(toprint=print))

    def get_value(self, toprint=False):
        self._U = self._c_puct * self._P * np.sqrt(
            self.parent._n_visits) / (1 + self._n_visits)
        if (toprint):
            print("Q", self._Q, "U", self._U, self._n_visits)
        return self._U + self._Q

    def update(self, value):
        self._n_visits += 1
        # if self.parent:

        self._Q += 1.0 * (value - self._Q) / self._n_visits
