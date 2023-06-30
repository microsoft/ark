# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import multiprocessing

import ark

d_model = 512  # Dimension of word embeddings
d_ff = 2048  # Dimension of the hidden layer in the feed-forward network
d_k = d_v = 64  # Dimensions of K(=Q) and V in the attention mechanism
n_layers = 2  # Number of encoder and decoder layers
n_heads = 8  # Number of heads in Multi-Head Attention set to 8

batch_size = 2
seq_len = 64
src_vocab_size = 128

num_gpu = 2
