from typing import List, Optional, Tuple, Type
from mcts.board_state import BoardState


class SearchNode:
    def __init__(self, parent: Optional[Type['SearchNode']], state: BoardState,
                 policy_score: float):
        self.parent_ = parent
        self.state_ = state
        self.visits_ = 0
        self.policy_score_ = policy_score
        self.value_ = 0  # determined by the value network, called when the node is made
        self.total_value_ = 0  # cumulative value score (not divided by visits)
        self.num_descendants_ = 0
        self.children_ = dict()

    def finished(self) -> bool:
        return self.state_.finished()
    
    def get_state_array(self):
        return self.state_.get_state_array()
    
    def get_active_player(self) -> bool:
        return self.state_.get_active_player()
    
    def is_leaf(self) -> bool:
        return len(self.children_) == 0
    
    def num_children(self) -> int:
        return len(self.children_)
    
    def policy_score(self) -> float:
        return self.policy_score_
    
    def visits(self) -> int:
        return self.visits_
    
    def state(self) -> BoardState:
        return self.state_
    
    def add_visit(self):
        self.visits_ += 1

    def branches(self) -> List[int]:
        return self.children_.keys()
    
    def average_value(self) -> float:
        return (self.total_value_ + self.value_) / (self.num_descendants_ + 1)

    def child_at(self, move: int) -> Type['SearchNode']:
        return self.children_[move]
    
    def add_child(self, move: int, strength: float, state: BoardState):
        child_state = SearchNode(self, state, strength)
        self.children_[move] = child_state
        return child_state
    
    def set_value(self, value: float):
        self.value_ = value

    # gathers and adds scores for generated leaves to their parent
    def reeval_leaf(self) -> Tuple[int, int]:
        score = children = 0
        for move in self.children_:
            children += 1
            score += self.children_[move].value_
        self.total_value_ = score
        self.num_descendants_ = children
        return (score, children)

    # adds scores for generated leaves into ancestors
    def reevaluate(self, added_score: float, added_descs: float):
        self.total_value_ += added_score
        self.num_descendants_ += added_descs

    def ascend(self):
        if self.parent_ is None:
            raise Exception("Non-root has no parent")
        return self.parent_
    
    def new_root_from(self, move: int) -> Type['SearchNode']:
        new_root = self.child_at(move)
        new_root.parent_ = None
        return new_root

    def result(self) -> float:
        return self.state_.get_winner()
    
    # TODO: EXPAND POST-LEAF SELECTION NOT PRE
