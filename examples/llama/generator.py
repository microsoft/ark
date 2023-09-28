# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import numpy as np
from model import ModelArgs, ModelArgs7B, Transformer
from transformers import AutoTokenizer, LlamaTokenizer


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim)
    )
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs).astype(np.float32)
    freqs_cis = np.exp(1j * freqs)
    return freqs_cis


class Generator:
    def __init__(
        self,
        args: ModelArgs,
        batch_size: int = 1,
        dtype: np.dtype = np.float16,
        world_size: int = 1,
    ):
        self.args = args
        self.batch_size = batch_size
        self.dtype = dtype

        assert self.batch_size <= args.max_batch_size

        # TODO: support multi-GPU
        assert world_size == 1

        self.tokenizer = None

        self.tokens: ark.Tensor = None
        self.freqs_cis: ark.Tensor = None
        self.mask: ark.Tensor = None
        self.logits: ark.Tensor = None

        self.runtime: ark.Runtime = None

    def launch(self, pth_path: str, tok_path: str):
        # Load a pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, use_fast=True, revision="main"
        )

        # Initiate ARK
        ark.init()

        dtype_ark = ark.DataType.from_numpy(self.dtype)

        # Can make it smaller
        seq_len = self.args.max_seq_len
        assert seq_len <= self.args.max_seq_len

        # May need to change
        start_pos = 0

        # Pre-calculated freqs_cis
        freqs_cis_np = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
        )[0:seq_len]
        freqs_cis_np = freqs_cis_np.astype(np.complex64)
        freqs_cis_np = (
            np.stack([freqs_cis_np.real, freqs_cis_np.imag], axis=-1)
            .astype(self.dtype)
            .reshape(1, seq_len, 1, self.args.dim // self.args.n_heads)
        )
        self.freqs_cis = ark.tensor(list(freqs_cis_np.shape), dtype_ark)

        # Pre-calculated mask
        if seq_len > 1:
            mask_np = np.full(
                (1, 1, seq_len, seq_len), -np.inf, dtype=self.dtype
            )
            mask_np = np.triu(mask_np, k=start_pos + 1)
            self.mask = ark.tensor(list(mask_np.shape), dtype_ark)

        # User inputs
        self.tokens = ark.tensor([self.batch_size, seq_len], dtype_ark)

        # Transformer
        ark.set_rank(0)
        ark.set_world_size(1)
        module = Transformer(self.args, dtype_ark, local_rank=0, world_size=1)
        self.logits = module.forward(
            self.tokens, start_pos, self.freqs_cis, self.mask
        )

        # Make sure we can read state_dict before initiating runtime
        param_names = set(module.params_dict().keys())
        state_dict = torch.load(pth_path)
        state_dict = {
            k: v.float().numpy().astype(self.dtype)
            for k, v in state_dict.items()
            if k in param_names
        }

        # Initiate runtime
        self.runtime = ark.Runtime()
        self.runtime.launch()

        # Initiate model parameters & precalculated values
        module.load_state_dict(state_dict)
        self.freqs_cis.from_numpy(freqs_cis_np)
        if self.mask:
            self.mask.from_numpy(mask_np)

    def run(self, tokens: np.ndarray):
        pass


if __name__ == "__main__":
    pth_path = "/mnt/changhohwang/llama-2-7b/consolidated.00.pth"
    tok_path = 

    gen = Generator()
    gen.launch()

