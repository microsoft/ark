# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle

# Save the state_dict of a module to a file
def save(state_dict, state_dict_file):
    if not isinstance(state_dict, dict):
        raise RuntimeError("Invalid state_dict")
    with open(state_dict_file, "wb") as f:
        pickle.dump(state_dict, f)

# load the state_dict of a module from a file
def load(state_dict, state_dict_file):
    with open(state_dict_file, "rb") as f:
        state_dict = pickle.load(f)
        if isinstance(state_dict, dict):
            return state_dict
        else:
            raise RuntimeError("Invalid state_dict file: " + state_dict_file)
