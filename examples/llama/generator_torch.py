import argparse
import multiprocessing as mp
import os

from llama import Llama


def worker(args: argparse.Namespace, rank: int):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    def log(msg):
        print(f"[Rank {rank}] {msg}")

    prompt_list = ["Where is the capital of France?"]
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tok_path,
        max_seq_len=512,
        max_batch_size=1,
    )
    generation_tokens = generator.generate(
        prompt_tokens=[
            generator.tokenizer.encode(
                f"[INST] {prompt} [/INST]", bos=True, eos=False
            )
            for prompt in prompt_list
        ],
        max_gen_len=512,
        temperature=0,
        top_p=0.9,
        logprobs=False,
        echo=False,
    )
    output_text = [
        {"generation": generator.tokenizer.decode(t)}
        for t in generation_tokens
    ]
    if rank == 0:
        log(f"{output_text}")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--tok_path", type=str, required=True)
    parser.add_argument("--ngpus", type=int, default=1)

    args = parser.parse_args()

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
