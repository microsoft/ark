# Llama2 over ARK

Llama2 examples over ARK.

## Quick Start

0. Install `gpudma` and ARK Python following the [ARK Install Instructions](../../docs/install.md).

1. Install Llama2 requirements.

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Update submodules.

    ```bash
    git submodule update --init --recursive
    ```

3. Download Llama2 model weights and tokenizer weights.
    * The model weights should be compatible with the [official PyTorch implementation](https://github.com/facebookresearch/llama/blob/main/llama/model.py).
    * The tokenizer weights should be compatible with the [HuggingFace implementation](https://huggingface.co/meta-llama).

4. Run the model accuracy test. `--pth_path` is the path to the model weights file (`*.pth`).

    ```bash
    python3 model_test.py --pth_path=/path/to/model/weights.pth
    ```

5. Test text generation. `--pth_path` is the path to the model weights file (`*.pth`) and `--tok_dir` is the path to the tokenizer weights directory.

    ```bash
    python3 generator.py --pth_path=/path/to/model/weights.pth --tok_dir=/path/to/tokenizer/weights
    ```

## Multi-GPU Inference

Multi-GPU version will be added in a future release.
