from base_state import BaseState


def predict(board, active, model):
    # Assumes 2d 9x9 input, converted to 0, 0.5, 1 form
    # pass in preloaded model to avoid reloading every time

    # create State    
    state = BaseState(board, active, model)
    moves = state.naive_suggest()
    moves.sort(key = lambda x: x[1])
    if active == 1:
        moves.reverse()
    if len(moves) < 5:
        return moves
    return moves[:5]