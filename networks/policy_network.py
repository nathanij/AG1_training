from typing import List, Tuple
import numpy as np
import random
import keras
from mcts.board_state import BoardState

class PolicyNetwork:
    def __init__(self, weight_path: str):
        self.address_ = weight_path
        self.model_ = keras.models.load_model(self.address_)

    def refresh(self):
        self.model_ = keras.models.load_model(self.address_)

    def eval(self, state: BoardState) -> List[Tuple[float, int]]:
        position = state.get_state_array()
        position.reshape(-1,19,19,1)
        policy = self.model_.predict(position) # TODO: check if this needs to be indexed into
        pairings = []
        for move, strength in enumerate(policy):
            if (move // 19 < 9 and move % 19 < 9) or move == 361: # Filter moves early
                pairings.append((strength, move))
        pairings.sort(key = lambda x: (x[0], -x[1]), reverse = True)  # Descending strength, ascending move order
        return pairings
