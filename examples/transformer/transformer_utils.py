# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn

import ark

d_model = 512  # Dimension of word embeddings
d_ff = 2048  # Dimension of the hidden layer in the feed-forward network
d_k = d_v = 64  # Dimensions of K(=Q) and V in the attention mechanism
n_layers = 2  # Number of encoder and decoder layers
n_heads = 8  # Number of heads in Multi-Head Attention set to 8

batch_size = 1
seq_len = 64
src_vocab_size = 128

# The number of input tokens is 10
# Used for constructing the masks
input_seq_len = 10

# Megatron-LM on 2 GPU
num_gpu = 2
n_heads_per_gpu = n_heads // num_gpu
