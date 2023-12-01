import numpy as np
import keras
from mcts.board_state import BoardState

class ValueNetwork:
    # Network evaluates the probability of a black win from the given situation
    def __init__(self, weight_path: str):
        self.address_ = weight_path
        self.model_ = keras.models.load_model(self.address_)

    def refresh(self):
        self.model_ = keras.models.load_model(self.address_)

    def eval(self, state: BoardState) -> float:
        position = state.get_state_array()
        return self.model_.predict(position)[0][0]
    