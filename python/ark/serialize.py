# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pickle

# Save the state_dict of a module to a file
def save(state_dict, state_dict_file_path):
    if not isinstance(state_dict, dict):
        print("Warning: Invalid state_dict saved to", state_dict_file_path)
    with open(state_dict_file_path, "wb") as f:
        pickle.dump(state_dict, f)

# load the state_dict of a module from a file
def load(state_dict_file_path):
    with open(state_dict_file_path, "rb") as f:
        state_dict = pickle.load(f)
        if not isinstance(state_dict, dict):
            print("Warning: Invalid state_dict file")
        return state_dict

def convert_state_dict_to_pytorch(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key] = torch.from_numpy(state_dict[key])
    return new_state_dict

def convert_state_dict_to_numpy(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key] = state_dict[key].numpy()
    return new_state_dict
