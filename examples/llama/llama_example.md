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

If you want to test the correctness and the performance of the ARK LLaMa implementation on multiple GPUs, please execute the following command:

```bash
# The number of GPUs
world_size = 2
cd examples/llama
python -m torch.distributed.launch \
       --nproc_per_node $world_size \
       llama_test.py
```

Please note that the ARK has no dependency on PyTorch, here we use PyTorch distributed launch utility to launch multiple processes for multiple GPUs. Pytorch is also used to compare the results of the ARK LLaMa implementation with the official LLaMa implementation.
