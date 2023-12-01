import time
import h5py
import os
import math
import numpy as np
from sim_driver import SimDriver

run_iter = 0  # increment with each run to distinguish data

# TODO: adjust these once on ECE machine
base_path = '/Users/nathanieljames/Desktop/AlphaGoOne/training'
weight_path = os.path.join(base_path, os.path.join('networks', 'weights'))
policy_network_path = os.path.join(base_path, os.path.join(weight_path, 'lossfinal'))
value_network_path = os.path.join(base_path, os.path.join(weight_path, 'lossfinal'))

input_arrays = []       # board state + turn
value_tag_arrays = []   # game result
policy_tag_arrays = []  # Normalized MCTS visit counts

exploration_factor = math.sqrt(2) / 8  # from paper
board_size = 9
search_limit = 10  # starting very low to test
expansion_limit = 41  # up to how many legal moves generated during MCTS, can decrease as policy network strength increases (using 2 for testing)
sim_driver = SimDriver(policy_network_path, value_network_path, board_size
                       exploration_factor, search_limit, expansion_limit)
sim_length = 1  # TODO: change to much larger once tested
for _ in range(sim_length):
    sim_driver.simulate()
    input_arrays.append(sim_driver.training_states())
    value_tag_arrays.append(sim_driver.result_tags())
    policy_tag_arrays.append(sim_driver.policy_tags())
input_data = np.concatenate(input_arrays).astype('float32') # I think concatenate is right, we can check dimensions when we get here
value_tags = np.concatenate(value_tag_arrays).astype('float32')
policy_tags = np.concatenate(policy_tag_arrays).astype('float32')
input()
# TODO: adjust pathing on ECE machine
save_dir = os.path.join(base_path, 'data/mcts_data')
save_path = os.path.join(save_dir, f'training_data_{run_iter}.h5')
with h5py.File(save_path, "w") as file:
    file.create_dataset("states", data = input_data)
    file.create_dataset("value_tags", data = value_tags)
    file.create_dataset("policy_tags", data = policy_tags)
