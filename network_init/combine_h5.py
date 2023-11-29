import os
import h5py
import numpy as np

training_set = []
tag_set = []
load_path = f'/afs/ece/usr/nathanij/Private/18500/data'
for f in os.listdir(load_path):
    if f.endswith(".h5"):
        with h5py.File(os.path.join(load_path, f), "r") as data:
            training_set.append(data["states"][:])
            tag_set.append(data["tags"][:])
training_set = np.concatenate(training_set)
tag_set = np.concatenate(tag_set)

store_path = os.path.join(load_path, "130k_sample.h5")
with h5py.File(store_path, "w") as file:
    # Create datasets and store the NumPy arrays
    file.create_dataset("states", data=training_set)
    file.create_dataset("tags", data=tag_set)