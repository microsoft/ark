# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys
sys.path.append("llama")

from typing import List, Optional

import fire
import time

from llama import Llama
import torch


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    seq_len: int = 128,
    batch_size: int = 256,
    gen_len: int = 128,
    warmup: int = 3,
    iteration: int = 5,
):
    total_len = seq_len + gen_len

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=total_len,
        max_batch_size=batch_size,
    )

    tokens = torch.randint(
        low=0, high=generator.tokenizer.n_words - 1, size=(batch_size, total_len), dtype=torch.int32
    )

    print(f"Profiling... (seq_len={seq_len}, batch_size={batch_size}, gen_len={gen_len}, warmup={warmup}, iteration={iteration})")

    def gen():
        _ = generator.model.forward(tokens[:, :seq_len], 0)
        for pos in range(1, gen_len):
            _ = generator.model.forward(tokens[:, (seq_len + pos - 1):(seq_len + pos)], pos)

    for _ in range(warmup):
        gen()
    start = time.time()
    for _ in range(iteration):
        gen()
    end = time.time()
    print(f"Elapsed: {(end - start)/iteration:.5f} sec/iteration")



if __name__ == "__main__":
    fire.Fire(main)
