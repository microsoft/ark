# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
