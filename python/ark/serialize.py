# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle

# Save the state_dict of a module to a file
def save_state_dict(module, state_dict_file):
    with open(state_dict_file, "wb") as f:
        pickle.dump(module.state_dict(), f)

# load the state_dict of a module from a file
def load_state_dict(module, state_dict_file):
    with open(state_dict_file, "rb") as f:
        state_dict = pickle.load(f)
        if isinstance(state_dict, dict):
            module.load_state_dict(state_dict)
        else:
            raise RuntimeError("Invalid state_dict file: " + state_dict_file)
