import numpy as np
import copy

from sympy import expand


class MCTSPlayer(object):
    def __init__(self,**kwargs) -> None:
        who_play = kwargs.get('who_play','human')
        if who_play=='pure_MCTS':
            policy_value_fn = kwargs.get('policy_value_fn')
            c_puct = kwargs.get('c_puct',5)
            n_playout = kwargs.get('n_playout',2000)
            self.mcts=MCTS(policy_value_fn, c_puct, n_playout)
        # pass
    def get_action(self,board_state):
        available_move=board_state.availables
        if len(available_move)>0:
            next_move=self.mcts.get_move(board_state)
            return next_move
        else:
            print("board is full")
        

    def __str__(self) -> str:
        return "base MCTS player"

class MCTS(object):
    def __init__(self, policy_value_fn, c_puct, n_playout) -> None:
        self._root=TreeNode(1)
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
    
    def get_move(self, board_state):
        for i in range(self.n_playout):
            cur_board_state=copy.deepcopy(board_state)
            self._playout(cur_board_state) 
        move=self._root.select(self._c_puct)
        return move
    
    def _playout(self,board_state):
        cur_node=self._root
        while(True):
            #search leaf
            if cur_node.is_leaf:
                break
            move = cur_node.select()
            board_state.do_move(move)
        end, winner = board_state.game_end()
        prob, leaf_val = self._policy_value_fn(board_state)
        if not end:
            np.random.choice(p=prob)
            self._root.expand()
            self.simulation_rollout()
        else:
            
        self._root.update_recursive()
        
    def simulation_rollout():
        pass


class TreeNode(object):
    def __init__(self, p) -> None:
        self._parent = None
        self._children = {}  
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = p
    
    def expand(self):
        pass

    def update_recursive(self):
        pass
    def is_leaf(self):
        return self._children=={}

