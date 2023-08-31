# ARK LLaMa Implementation Example

This code provides an example of the LLaMa model implementation using ARK. We have successfully implemented all parts of the Transformer block, including the attention layer, RMSnorm layer, and the feed-forward layer.

You can run the example by executing the following command:

```bash
python llama_ark.py
```

# Cloning the Official LLaMa Implementation
To verify the correctness of the LLaMa code implementation, please clone the official implementation of LLaMa by executing the following command in the current directory:

```bash
cd examples/llama
git clone https://github.com/facebookresearch/llama
```

Currently their is a bug that will cause some errors in the PyTorch inference stage. Please change the following line in the file `llama/llama/model.py`:

```python
# At line 270
@torch.inference_mode()
# Change to
@torch.no_grad()
```

Next, please install the dependencies of LLaMa official implementations and run the PyTorch LLaMa model according to the instructions in the LLaMa official repository. 

# Verifying the Correctness of the ARK LLaMa Implementation

To verify the correctness of the ARK LLaMa implementation, please execute the following command:

```bash
cd examples/llama
python llama_test.py
```
