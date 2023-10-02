# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import argparse
import numpy as np
from model import ModelArgs, ModelArgs7B, Transformer
from transformers import AutoTokenizer, PreTrainedTokenizer


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
        seq_len: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: list = None,
        world_size: int = 1,
    ):
        self.args = args
        self.dtype = dtype
        self.seq_len = seq_len
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.stop_token_ids = stop_token_ids

        if self.seq_len <= 0 or self.seq_len > self.args.max_seq_len:
            self.seq_len = self.args.max_seq_len
        if self.stop_token_ids is None:
            self.stop_token_ids = []

        # TODO: support multi-GPU
        assert world_size == 1

        self.tokenizer: PreTrainedTokenizer = None

        self.tokens: ark.Tensor = None
        self.freqs_cis: ark.Tensor = None
        self.mask: ark.Tensor = None
        self.logits: ark.Tensor = None

        self.runtime: ark.Runtime = None

    def launch(
        self,
        pth_path: str,
        tok_dir: str,
    ):
        # Load a pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_dir, use_fast=True, revision="main"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id not in self.stop_token_ids:
            self.stop_token_ids.append(self.tokenizer.eos_token_id)
        self.args.vocab_size = self.tokenizer.vocab_size

        # Initiate ARK
        ark.init()

        dtype_ark = ark.DataType.from_numpy(self.dtype)

        # May need to change
        start_pos = 0

        # Pre-calculated freqs_cis
        freqs_cis_np = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
        )[0:self.seq_len]
        freqs_cis_np = freqs_cis_np.astype(np.complex64)
        freqs_cis_np = (
            np.stack([freqs_cis_np.real, freqs_cis_np.imag], axis=-1)
            .astype(self.dtype)
            .reshape(1, self.seq_len, 1, self.args.dim // self.args.n_heads)
        )
        self.freqs_cis = ark.tensor(list(freqs_cis_np.shape), dtype_ark)

        # Pre-calculated mask
        if self.seq_len > 1:
            mask_np = np.full(
                (1, 1, self.seq_len, self.seq_len), -np.inf, dtype=self.dtype
            )
            mask_np = np.triu(mask_np, k=start_pos + 1)
            self.mask = ark.tensor(list(mask_np.shape), dtype_ark)

        # User inputs
        self.tokens = ark.tensor([1, self.seq_len], ark.int32)

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

    @torch.inference_mode()
    def run(self, prompt: str):
        enc = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=self.seq_len,
        )
        input_ids = enc.input_ids[0]
        output_ids = list(input_ids)

        self.tokens.from_numpy(input_ids)

        self.runtime.run()
        logits = self.logits.to_numpy()

        last_token_logits = logits[0, -1, :]

        probs = torch.softmax(
            torch.tensor(last_token_logits, dtype=torch.float32), dim=-1
        )
        indices = torch.multinomial(probs, num_samples=2)
        tokens = [int(token) for token in indices.tolist()]

        token = tokens[0]
        output_ids.append(token)

        output = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--tok_dir", type=str, required=True)

    args = parser.parse_args()

    gen = Generator(ModelArgs7B())
    gen.launch(args.pth_path, args.tok_dir)

    prompt_list = [
        "The capital of France is",
        "The square root of nine is",
        "King minus man plus woman is",
    ]
    for i, prompt in enumerate(prompt_list):
        output = gen.run(prompt)
        print(f"---\nPrompt[{i}]: {prompt}\nOutput[{i}]: {output}")
