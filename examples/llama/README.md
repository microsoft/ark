# Llama-2 over ARK

Llama-2 examples over ARK.

## Quick Start

0. Install `gpudma` and ARK Python following the [ARK Install Instructions](../../docs/install.md).

1. Install Llama-2 requirements.

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Update submodules.

    ```bash
    git submodule update --init --recursive
    ```

3. Download Llama-2 model weights and tokenizer weights.
    * The model weights should be compatible with the [official PyTorch implementation](https://github.com/facebookresearch/llama/blob/main/llama/model.py).

4. Run the model accuracy test. `--pth_path` is the path to the model weights.

    ```bash
    python3 model_test.py --pth_path=/path/to/model/weights
    ```
