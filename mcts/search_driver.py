import math
from typing import List, Tuple
from mcts.search_node import SearchNode
from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork


class SearchDriver:
    # Minimize instead of maximize if black is moving
    # For search, illegal moves should not be considered
    # This will train the engine not to make them either (as training policy 
    # vector will not include any)
    def __init__(self, policy_network: PolicyNetwork, 
                 value_network: ValueNetwork, 
                 board_size: int, root: SearchNode,
                 exploration_factor: int, search_limit: int,
                 expansion_limit: int):
        self.policy_network_ = policy_network
        self.value_network_ = value_network
        self.board_size_ = board_size
        self.root_ = root
        self.exploration_factor_ = exploration_factor
        self.iterations_ = 0
        self.search_limit_ = search_limit
        self.expansion_limit_ = expansion_limit


    def finished(self):
        return self.iterations_ >= self.search_limit_
    
    def exploration_score(self, parent: SearchNode, child: SearchNode) -> float:
        scalar = self.exploration_factor_ * child.policy_score()
        visit_score = math.sqrt(parent.visits() + parent.num_children()) / (1 + child.visits())
        return scalar * visit_score
    
    def explore(self, cur: SearchNode) -> List[Tuple[float, int, SearchNode]]:
        move_strengths = self.policy_network_.eval(cur)  # TODO: 9x9 plz and thanks
        i = 0
        candidates = []
        while i < len(move_strengths) and len(candidates) < self.expansion_limit_:
            move = move_strengths[i][1]
            result = cur.state().next_move(move) # TODO: recalc columns and rows inside this for 9x9
            if result is not None:
                candidates.append((move_strengths[i][0], move, result))
            i += 1
        return candidates
    
    def evaluate(self, child: SearchNode):
        value = self.value_network_.eval(child.state())
        child.set_value(value)
    
    def expand(self):
        level = 0
        self.iterations_ += 1
        cur = self.root_
        while not cur.is_leaf():
            cur.add_visit()
            branches = cur.branches()
            if cur.get_active_player() == 1:  # white so maximize
                max_value = -float('inf')
                best_child = None
                for move in branches:
                    child = cur.child_at(move)
                    q = child.average_value()
                    u = self.exploration_score(cur, child)
                    if q + u > max_value:
                        max_value = q + u
                        best_child = child
                cur = best_child
            else:  # black so minimize
                min_value = float('inf')
                best_child = None
                for move in branches:
                    child = cur.child_at(move)
                    q = child.average_value()
                    u = -self.exploration_score(cur, child)
                    if q + u < min_value:
                        min_value = q + u
                        best_child = child
                cur = best_child
            level += 1
        cur.add_visit()
        for strength, move, state in self.explore(cur):
            child = cur.add_child(move, strength, state)
            self.evaluate(child)
        added_score, added_children = cur.reeval_leaf()
        if cur == self.root_:
            return
        cur = cur.ascend()
        while cur != self.root_ and cur is not None:
            cur.reevaluate(added_score, added_children)
            cur = cur.ascend()

    def most_visited(self) -> int:
        max_visits = -1
        max_move = -1
        for move in self.root_.branches():
            child = self.root_.child_at(move)
            if child.visits() > max_visits:
                max_visits = child.visits()
                max_move = move
        return max_move
    
    def normalized_visit_count(self) -> List[float]:
        visits = 0
        target = [0] * 362  # This should be fine to remain
        for move in self.root_.branches():
            child = self.root_.child_at(move)
            target[self.convert_move_num(move, 9, 19)] = child.visits()
            visits += child.visits()
        for i in range(len(target)):
            target[i] /= visits
        return target
    
    def convert_move_num(move: int, prev: int, new: int) -> int:
        old_row = move // prev
        old_col = move % prev
        return old_row * new + old_col

