# Parallel Training with Multi GPU Communication

In this document, we will discuss how to implement parallel training over different GPUs using multi-process programming. This is particularly useful when we want to utilize the full potential of multiple GPUs and speed up the training process.

## Getting Started

First, make sure you have completed the [installation](./install.md) process. Then, you can run the tutorial example at [tutorial.py](../examples/tutorial/tutorial.py) to see how to use Ark to communicate between different GPUs.

## Ping-Pong Transfer Example

In this example, we will demonstrate communication between two GPUs (GPU 0 and GPU 1) using a simple ping-pong transfer. We will send a tensor from GPU 0 to GPU 1, and then simultaneously send the tensor back from GPU 1 to GPU 0.