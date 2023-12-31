import copy
import numpy as np
import time

from mcts.board_state import BoardState
from mcts.search_driver import SearchDriver
from networks.value_network import ValueNetwork
from networks.policy_network import PolicyNetwork
from mcts.search_node import SearchNode


class SimDriver:
    def __init__(self, policy_network_path: str, value_network_path: str,
                 board_size: int, exploration_factor: int, search_limit: int,
                 expansion_limit: int):
        self.policy_network_ = PolicyNetwork(policy_network_path)
        self.value_network_ = ValueNetwork(value_network_path)
        self.board_size_ = board_size
        self.root_ = SearchNode(None, BoardState(board_size), 0)
        self.exploration_factor_ = exploration_factor
        self.search_limit_ = search_limit
        self.expansion_limit_ = expansion_limit
        self.training_states_ = []
        self.visit_counts_ = []

    def reset(self):
        new_state = BoardState(self.board_size_)
        self.root_ = SearchNode(None, new_state, 0)
        self.training_states_ = []
        self.visit_counts_ = []

    def training_states(self) -> np.ndarray:
        return np.vstack(self.training_states_)
    
    def result_tags(self) -> np.array:
        val = self.root_.result()  # 1 for white win, 0.5 for draw, 0 for black win  TODO: build
        return np.full(len(self.training_states_), val)
    
    def policy_tags(self) -> np.ndarray:
        return np.vstack(self.visit_counts_)
    
    # Tree structuring:
    # Root node passed into search_driver
    # After the end of search, root is updated to the most visited child
    # Root has a .finished() method, that is used instead of an exterior game state
    # THis is accessible via the simulation driver
    # It returns the new state object to be put into the generated leaf

    def simulate(self):
        self.reset()
        turn = 0
        start_time = time.time()
        while not self.root_.finished():
            pre_state = self.root_.get_state_array()  # TODO: change this to pad the 9x9
            # active_player = self.root_.get_active_player()
            # pre_state.append(active_player)
            self.training_states_.append(pre_state)
            search_driver = SearchDriver(self.policy_network_,
                                         self.value_network_, 
                                         self.board_size_, self.root_,
                                         self.exploration_factor_,
                                         self.search_limit_,
                                         self.expansion_limit_)
            # start = time.time()
            while not search_driver.finished():
                search_driver.expand()
            move = search_driver.most_visited()
            self.visit_counts_.append(search_driver.normalized_visit_count())  # TODO: pad this to 19x19 as well
            self.root_ = self.root_.new_root_from(move)
            for row in self.root_.state_.board_:
                print(row)
            print(f'Eval after move {turn}: {self.root_.value_}')
            turn += 1
        print(f'Average time per move: {(time.time() - start_time) / turn} seconds')
        input()
            # print(f'Score: {self.root_.state_.score_}')

