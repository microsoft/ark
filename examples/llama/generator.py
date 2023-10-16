# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import json
import torch
import argparse
import numpy as np
from model import ModelArgs, ModelArgs7B, Transformer

from llama.tokenizer import Tokenizer


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
        dtype: np.dtype = np.float16,
        local_rank: int = 0,
        world_size: int = 1,
        seq_len: int = 2048,
    ):
        self.args = args
        self.dtype = dtype

        # TODO: support multi-GPU
        assert local_rank == 0
        assert world_size == 1
        self.local_rank = local_rank
        self.world_size = world_size
        self.seq_len = seq_len

        self.tokenizer: Tokenizer = None

        self.tokens: ark.Tensor = None
        self.freqs_cis: ark.Tensor = None
        self.mask: ark.Tensor = None
        self.logits: ark.Tensor = None

        self.runtime: ark.Runtime = None

    def launch(
        self,
        pth_path: str,
        tok_path: str,
    ):
        # Load a pretrained tokenizer
        self.tokenizer = Tokenizer(model_path=tok_path)
        self.args.vocab_size = self.tokenizer.n_words

        # Initiate ARK
        ark.init()

        dtype_ark = ark.DataType.from_numpy(self.dtype)

        # start_pos is always 0 since ARK doesn't have KV cache yet
        start_pos = 0

        # Pre-allocated user inputs, later assigned
        self.tokens = ark.tensor([1, self.seq_len], ark.int32)

        # Pre-allocated and calculated freqs_cis, later assigned
        freqs_cis_np = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len
        )[: self.seq_len]
        freqs_cis_np = freqs_cis_np.astype(np.complex64)
        freqs_cis_np = (
            np.stack([freqs_cis_np.real, freqs_cis_np.imag], axis=-1)
            .astype(self.dtype)
            .reshape(1, self.seq_len, 1, self.args.dim // self.args.n_heads)
        )
        self.freqs_cis = ark.tensor(list(freqs_cis_np.shape), dtype_ark)

        # Pre-allocated and calculated mask, later assigned
        mask_np = np.full(
            (1, 1, self.seq_len, self.seq_len), -np.inf, dtype=self.dtype
        )
        mask_np = np.triu(mask_np, k=start_pos + 1)
        self.mask = ark.tensor(list(mask_np.shape), dtype_ark)

        # Transformer
        ark.set_rank(self.local_rank)
        ark.set_world_size(self.world_size)
        module = Transformer(
            self.args,
            dtype_ark,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )
        self.module = module

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
        self.mask.from_numpy(mask_np)

    @torch.inference_mode()
    def run(self, prompt: str):
        prompt = f"[INST] {prompt} [/INST]"
        prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        input_ids = np.array(
            [
                prompt_ids
                + [self.tokenizer.pad_id] * (self.seq_len - len(prompt_ids))
            ]
        )

        output_ids = []
        for cur_pos in range(len(prompt_ids), self.seq_len):
            self.tokens.from_numpy(input_ids)
            self.runtime.run()
            logits = self.logits.to_numpy()
            next_token = np.argmax(logits[0, cur_pos - 1, :]).item()
            input_ids[0, cur_pos] = next_token
            if next_token == self.tokenizer.eos_id:
                break
            output_ids.append(next_token)

        output_text = self.tokenizer.decode(output_ids)
        return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--tok_path", type=str, required=True)

    args = parser.parse_args()

    with open(args.params_path, "r") as f:
        params = json.load(f)

    gen = Generator(ModelArgs7B(**params))
    print("gen.args", gen.args)

    gen.launch(args.pth_path, args.tok_path)

    prompt_list = [
        "Where is the captial of France?",
    ]
    for i, prompt in enumerate(prompt_list):
        output = gen.run(prompt)
        print(f"---\nPrompt[{i}]: {prompt}\nOutput[{i}]: {output}")
