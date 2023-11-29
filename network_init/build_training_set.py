import os
import numpy as np
import h5py
import gc


tset = []
tags = []
start = 25000
end = 50000
for i in range(27): # 27
    print(i)
    load_path = f'/Users/nathanieljames/Desktop/AlphaGoOne/training/data/value_network_data_{i}.npz'
    data = np.load(load_path)
    seg1 = data['states'][start:end]
    seg2 = data['tags'][start:end]
    del data
    gc.collect()
    tset.append(seg1)
    tags.append(seg2)

tset1 = np.concatenate(tset)
del tset
gc.collect()
tags1 = np.concatenate(tags).astype('float32')
del tags
gc.collect()

with h5py.File("/Users/nathanieljames/Desktop/AlphaGoOne/training/data/25-50ktrain.h5", "w") as file:
    # Create datasets and store the NumPy arrays
    file.create_dataset("states", data=tset1)
    file.create_dataset("tags", data=tags1)