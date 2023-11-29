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
        # policy = self.network_.eval(np.array(position))
        # start as random
        pairings = [(random.random(), i) for i in range(362)]
        # pairings = []
        #for move, strength in enumerate(policy):
            #pairings.append((strength, move))
        pairings.sort(key = lambda x: (x[0], -x[1]), reverse = True)  # Descending strength, ascending move order
        return pairings
