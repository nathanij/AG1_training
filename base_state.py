from copy import deepcopy
import numpy as np


class BaseState:
    # This assumes 9x9 board
    # Pads to 19x19 when passing into model
    def __init__(self, board, active, model):
        self.board_ = board
        self.active_ = active
        self.model_ = model
        self.size_ = 9

    def naive_suggest(self):
        moves = []
        for i in range(82):
            #print(f'Testing move {i}')
            res = self.make_move(i)
            if res is not None:
                moves.append((i, res))
        return moves
    
    def make_move(self, i):
        if i == 81:
            return self.eval(deepcopy(self.board_))
        row = i // 9
        col = i % 9
        new_board = self.validate(row, col)
        if new_board is None:
            return None
        return self.eval(new_board)
    
    def eval(self, board):
        # pad out board
        for row in board:
            row += ([0.5] * 10)
        for _ in range(10):
            board.append(([0.5] * 19))
        arr = np.asarray(board).astype('float32')
        position = arr.reshape(-1,19,19,1)
        return self.model_.predict(position)[0][0]
    
    def validate(self, row, col):
        board = deepcopy(self.board_)
        # check if space is already occupied
        if board[row][col] != 0.5:
            return None
        
        # set up for bfs
        visited = set()
        captured = []
        move_color = self.active_
        opp_color = self.opp(self.active_)
        board[row][col] = self.active_

        # bfs through adjacent points
        foci = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for y, x in foci:
            prevs = set()
            if not self.bfs_(y, x, visited, prevs, opp_color, board):
                for point in prevs:
                    captured.append(point)
            visited.union(prevs)

        # if no captures check if placement has liberties
        if len(captured) == 0:
            if not self.bfs_(row, col, set(), set(), move_color, board):
                return None

        # make captures
        for y, x in captured:
            board[y][x] = 0.5
        return board

    def opp(self, color: int) -> int:
        opp = 1 if color == 0 else 0
        return opp
    
    def bfs_(self, row, col, visited, prevs, match_color, board) -> bool:
        opp_color = self.opp(match_color)
        if row < 0 or col < 0 or row >= self.size_ or col >= self.size_:
            return False
        coord = (row, col)
        if coord in prevs or coord in visited:
            return False
        if board[row][col] == 0.5:
            return True # unrestricted edge
        if board[row][col] == opp_color:
            return False # restricted edge
        
        # implicitly board[row][col] = match color
        if row == 0 or col == 0 or row == self.size_ - 1 or col == self.size_ - 1:
            return True
        prevs.add(coord)

        adj = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        retval = False
        for y,x in adj:
            retval = retval or self.bfs_(y, x, visited, prevs, match_color, board)
        return retval
    