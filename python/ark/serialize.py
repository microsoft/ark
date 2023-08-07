# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pickle
import logging


def save(state_dict, state_dict_file_path: str):
    """
    Save the state_dict of a module to a file
    """
    if not isinstance(state_dict, dict):
        logging.warn(
            "Warning: Invalid state_dict saved to", state_dict_file_path
        )
    with open(state_dict_file_path, "wb") as f:
        pickle.dump(state_dict, f)


def load(state_dict_file_path: str):
    """
    Load the state_dict of a module from a file
    """
    with open(state_dict_file_path, "rb") as f:
        state_dict = pickle.load(f)
        if not isinstance(state_dict, dict):
            logging.warn("Warning: Invalid state_dict file")
        return state_dict


def convert_state_dict(state_dict: dict, type="numpy"):
    """
    Convert the state_dict of a module to np.ndarray or torch.Tensor type
    """
    new_state_dict = {}
    for key in state_dict:
        if type == "torch":
            new_state_dict[key] = torch.from_numpy(state_dict[key])
        elif type == "numpy":
            new_state_dict[key] = state_dict[key].numpy()
        else:
            logging.error(
                "Invalid type: " + type + " valid types are torch and numpy"
            )
            raise TypeError(
                "Invalid type: " + type + " valid types are torch and numpy"
            )
    return new_state_dict
