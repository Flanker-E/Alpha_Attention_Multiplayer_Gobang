
import MCTS
from MCTS import MCTSPlayer


class MCTSPlayerAlpha(MCTSPlayer):
    def get_action(self, board_state):
        return super().get_action(board_state)
        
    def __str__(self) -> str:
        return super().__str__()

class MCTSAlpha(MCTS):
    def _playout(self,board_state):
        pass

# class TreeNodeAlpha