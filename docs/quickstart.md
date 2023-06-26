# Building a Simple Model with ARK using C++

In this tutorial, we will demonstrate how to use ARK to build a DNN application. The example source code is located at [examples/tutorial](../examples/tutorial).

After we finish the [install](./install.md) and set ARK_ROOT, we can build the example and run it.

```bash
export ARK_ROOT=${HOME}/.ark
cd examples/tutorial
make
./build/tutorial
```

There is some environment variables that can be used to configure ARK. Please refer to [Environment Variables](./env.md) for more details.

# Building a Simple Model with ARK in Python

In this tutorial, we will demonstrate how to use ARK to run a simple DNN appication in Python. We will be using a basic Python example to illustrate the process.

1. First, we import the `ark` module and initialize it:

   ```python
   import ark

   ark.init()
   ```

2. Next, we create a `Dims` object and print its contents:

   ```python
   a = ark.Dims([1, 2, 3, 4])
   print(ark.NO_DIM)
   print(a[2])
   ```

3. We then set the random seed and generate a random number:

   ```python
   ark.srand(42)

   random_number = ark.rand()

   print(random_number)
   ```

4. We create a `TensorBuf` object with a specified size and alignment:

   ```python
   buf = ark.TensorBuf(1024, 1)
   ```

5. We define the dimensions for the tensor using `Dims` objects:

   ```python
   shape = ark.Dims(1, 2, 3, 4)
   ldims = ark.Dims(4, 4, 4, 4)
   offs = ark.Dims(0, 0, 0, 0)
   pads = ark.Dims(1, 1, 1, 1)
   ```

6. We create a `Tensor` object using the dimensions and buffer:

   ```python
   tensor = ark.Tensor(
       shape,
       ark.TensorType.FP32,
       buf,
       ldims,
       offs,
       pads,
       False,
       False)

