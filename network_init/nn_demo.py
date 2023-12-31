import keras
import numpy as np

path = f'/Users/nathanieljames/Desktop/AlphaGoOne/training/networks/weights/lossfinal'

model = keras.models.load_model(path)

board = [[0.5] * 19 for _ in range(19)]

def convert(board):
    b = np.asarray(board).astype("float32")
    print(b.shape)
    return b.reshape(-1,19,19,1)

def printb(board):
    for row in board:
        print(row)

def display_eval():
    print(max(0, min(model.predict(convert(board))[0][0], 1)))

display_eval()

board = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
]

for row in board:
    for i in range(len(row)):
        if row[i] == -1:
            row[i] = 0
        elif row[i] == 0:
            row[i] = 0.5

printb(board)


display_eval()

for row in board:
    for i in range(len(row)):
        if row[i] == 0:
            row[i] == 1
        elif row[i] == 1:
            row[i] == 0

printb(board)

display_eval()

board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0],
    [0, 0, -1, 1, 0, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, 1, -1, 0, 0],
    [0, 0, -1, 0, 0, 0, -1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, -1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, -1, -1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, -1, 1, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, -1, 0, -1, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0],
    [0, 0, 1, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 0],
    [0, 0, 1, -1, 0, -1, 0, 0, 1, 0, -1, 0, 0, -1, 0, 0, -1, 1, 0],
    [0, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

for row in board:
    for i in range(len(row)):
        if row[i] == -1:
            row[i] = 0
        elif row[i] == 0:
            row[i] = 0.5
printb(board)

display_eval()

board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 1, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, -1, -1, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, -1, 0, 0],
    [0, 1, 1, 1, 1, -1, 0, -1, 1, 1, -1, 0, -1, -1, 0, -1, 0, -1, 0],
    [0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, -1, 1, -1, 0, 0, 0, 1, 0],
    [0, 0, -1, 0, -1, 0, -1, -1, -1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0, 0, 0, 1, -1, -1, 0, -1, 0],
    [0, -1, 1, 1, -1, -1, 1, 1, 0, 0, 1, 0, -1, -1, 1, -1, -1, 0, 0],
    [0, 0, -1, -1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, -1, 0, 0],
    [0, 0, -1, 0, 1, 0, 0, -1, 1, 1, -1, -1, 1, -1, 0, 1, -1, 0, 0],
    [0, 1, 0, -1, 1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 1, -1, -1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, -1, -1, 0, 0, 1, -1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, 1, 0, 1, -1, 1, 0, 0],
    [0, 0, 1, 1, 1, -1, 1, 1, -1, 0, 0, -1, 1, 1, -1, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 1, 1, -1, 0]
]

for row in board:
    for i in range(len(row)):
        if row[i] == -1:
            row[i] = 0
        elif row[i] == 0:
            row[i] = 0.5
printb(board)

display_eval()