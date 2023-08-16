# ARK LLaMa Implementation Example

This is an implementation of the LLaMa model using ARK. In this code, we have fully implemented all parts of the Transformer block, including the attention layer, RMSnorm layer and the feed forward layer.

To verify the correctness of the LLaMa code implementation, you can first clone the official implementation of LLaMa by running the following command:

```bash
git clone https://github.com/facebookresearch/llama
```

Then, please install the dependencies of LLaMa official implementations and run LLaMa model according to the instructions in the official repository. 

You can run the following command to verify the correctness of the ARK LLaMa implementation:

```bash
cd examples/llama
python llama_test.py
```
