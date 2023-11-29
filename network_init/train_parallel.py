import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection  import train_test_split

input_shape = (19, 19, 1)  # Adjust the input shape as needed
output_dim = 1  # Replace with the desired output dimension

# Define the input layer
input_layer = layers.Input(shape=input_shape)
# Parallel Convolutional Tower 1
tower1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
tower1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(tower1)
tower1 = layers.Flatten()(tower1)
tower1 = layers.Dense(64, activation = 'relu')(tower1)
# Parallel Convolutional Tower 2
tower2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
tower2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(tower2)
tower2 = layers.Flatten()(tower2)
tower2 = layers.Dense(64, activation = 'relu')(tower2)
# Parallel Convolutional Tower 3
tower3 = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(input_layer)
tower3 = layers.Conv2D(64, (7, 7), activation='relu', padding='same')(tower3)
tower3 = layers.Flatten()(tower3)
tower3 = layers.Dense(64, activation = 'relu')(tower3)
# Merge the outputs from the three towers
merged = layers.concatenate([tower1, tower2, tower3], axis=-1)

# Common Layers
#merged = layers.MaxPooling2D((2, 2))(merged)
merged = layers.Dropout(0.2)(merged)
# merged = layers.Flatten()(merged)
merged = layers.Dense(128, activation='relu')(merged)

# Output Layer
output_layer = layers.Dense(output_dim, activation='sigmoid')(merged)

# Create the model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model

training_set = []
tag_set = []
load_path = f'/afs/ece/usr/nathanij/Private/18500/data'
for f in os.listdir(load_path):
    if f.endswith(".h5"):
        with h5py.File(load_path, "r") as data:
            training_set.append(data["states"][:])
            tag_set.append(data["tags"][:])
training_set = np.concatenate(training_set)
tag_set = np.concatenate(tag_set)
x_train, x_val, y_train, y_val = train_test_split(training_set, tag_set, test_size=0.2, random_state=19)
del training_set
del tag_set

x_train = x_train.reshape(-1, 19, 19, 1)
x_val = x_val.reshape(-1, 19, 19, 1)
training_iter = 2

acc_filepath = f'/afs/ece/usr/nathanij/Private/18500/weights/acc{training_iter}'
loss_filepath = f'/afs/ece/usr/nathanij/Private/18500/weights/loss{training_iter}'
acc_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=acc_filepath,
    save_weights_only=False,
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True)
loss_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=loss_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience = 20)

schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

# Print the model summary
model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size = 128, epochs = 1000, shuffle=True, callbacks = [acc_callback, loss_callback, early])