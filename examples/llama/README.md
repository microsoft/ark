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

3. Install `llama` submodule.
    
    ```bash
    cd llama
    python3 -m pip install -e .
    cd ..
    ```

4. Download Llama2 model weights and tokenizer weights.
    * The model and tokenizer should be compatible with the [official PyTorch implementation](https://github.com/facebookresearch/llama/blob/main/llama).

5. Run the model accuracy test. `--pth_path` is the path to the model weights file (`consolidated.00.pth`).

    ```bash
    python3 model_test.py --pth_path=/path/to/model/weights.pth
    ```

6. Test text generation. `--pth_path` is the path to the model weights file (`consolidated.00.pth`), `--tok_path` is the path to the tokenizer weights file (`tokenizer.model`), and `--params_path` is the path to the model parameters (`params.json`).

    ```bash
    python3 generator.py --pth_path=consolidated.00.pth --tok_path=tokenizer.model --params_path=params.json
    ```

## Multi-GPU Inference

Multi-GPU version will be added in a future release.
