# ARK LLaMa Implementation Example

This code provides an example of the LLaMa model implementation using ARK. We have successfully implemented all parts of the Transformer block, including the attention layer, RMSnorm layer, and the feed-forward layer.

# Cloning the Official LLaMa Implementation
To verify the correctness of the LLaMa code implementation, please clone the official implementation of LLaMa by executing the following command in the current directory:

```bash
cd examples/llama
git clone https://github.com/facebookresearch/llama
```

Next, please install the dependencies of LLaMa official implementations and run the PyTorch LLaMa model according to the instructions in the LLaMa official repository. 

# Verifying the Correctness of the ARK LLaMa Implementation

To verify the correctness of the ARK LLaMa implementation, please execute the following command:

```bash
cd examples/llama
python llama_test.py
```
