from predict import predict
import keras

weight_path = "/Users/nathanieljames/Desktop/AG1_place/lossfinal"
model = keras.models.load_model(weight_path)

print("loaded")
input()

board = [([0.5] * 9) for _ in range(9)]
print(predict(board, 0, model))
input()

board[0][1] = 0
board[1][0] = 0
board[1][1] = 1
board[1][2] = 0
print(predict(board, 0, model))
input()

board[8][8] = 1
board[7][7] = 1
print(predict(board, 1, model))