# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import multiprocessing as mp
import json
import numpy as np
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import ark
from model import ModelArgs, ModelArgs13B, ModelArgs70B, Transformer

import torch
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import llama.model as model_pt
from llama.tokenizer import Tokenizer


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim)
    )
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs).astype(np.float32)
    freqs_cis = np.exp(1j * freqs)
    return freqs_cis


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: model_pt.ModelArgs = model_pt.ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = model_pt.Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: model_pt.Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full(
            (bsz, total_len), pad_id, dtype=torch.long, device="cuda"
        )
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(
                t, dtype=torch.long, device="cuda"
            )
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[
                    :, prev_pos + 1 : cur_pos + 1
                ] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][
                    start : len(prompt_tokens[i]) + max_gen_len
                ]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


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
        ckpt_dir: str,
        tok_path: str,
    ):
        # Load a pretrained tokenizer
        self.tokenizer = Tokenizer(model_path=tok_path)
        self.args.vocab_size = self.tokenizer.n_words

        # Initiate ARK
        ark.init()
        ark.set_rank(self.local_rank)
        ark.set_world_size(self.world_size)

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
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        ckpt_path = checkpoints[self.local_rank]
        print(f"Loading checkpoint from {ckpt_path} for rank {self.local_rank}")
        state_dict = torch.load(ckpt_path)
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

        elapsed = self.runtime.stop()
        print(f"elapsed {elapsed:.5f} ms, itr {len(output_ids)}")
        output_text = self.tokenizer.decode(output_ids)
        return output_text


def worker(args: argparse.Namespace, rank: int):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    def log(msg):
        print(f"[Rank {rank}] {msg}")

    with open(args.params_path, "r") as f:
        params = json.load(f)

    prompt_list = ["Where is the capital of France?"]
    if args.only_torch_model:
        generator = Llama.build(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=args.tok_path,
            max_seq_len=1024,
            max_batch_size=1,
        )
        generation_tokens = generator.generate(
            prompt_tokens=[
                generator.tokenizer.encode(
                    f"[INST] {prompt} [/INST]", bos=True, eos=False
                )
                for prompt in prompt_list
            ],
            max_gen_len=1024,
            temperature=0,
            top_p=0.9,
            logprobs=False,
            echo=False,
        )
        output_text = [{"generation": generator.tokenizer.decode(t)} for t in generation_tokens]
        if rank == 0:
            log(f"{output_text}")
        return

    gen = Generator(
        ModelArgs13B(**params),
        local_rank=rank,
        world_size=args.ngpus,
        seq_len=1024,
    )
    if rank == 0:
        log(f"gen.args {gen.args}")

    log("Launching generator...")
    gen.launch(args.ckpt_dir, args.tok_path)

    for i, prompt in enumerate(prompt_list):
        output = gen.run(prompt)
        if rank == 0:
            log(f"---\nPrompt[{i}]: {prompt}\nOutput[{i}]: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--tok_path", type=str, required=True)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--only_torch_model", type=bool, default=False)

    args = parser.parse_args()

    os.environ["ARK_IPC_LISTEN_PORT_BASE"] = "42500"

    # For torch.distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(args.ngpus)
    procs = []
    for i in range(args.ngpus):
        p = mp.Process(target=worker, args=(args, i))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
