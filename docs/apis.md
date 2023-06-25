### Namespace: ark

#### Functions

- `void srand(int seed = -1)`: Sets the random seed.
- `int rand()`: Generates a random integer.

#### Initialization

- `void init()`: Initializes an ark program. Call this function to clean up the shared memory directory.

#### Data Types

- `typedef long long int DimType`: Data type for dimensions.

#### Constants

- `DIMS_LEN`: Maximum number of dimensions (4).
- `NO_DIM`: Constant representing an invalid dimension (-1).

#### Dims

- `struct Dims`: Represents an up-to-`DIMS_LEN`-dimensional vector.

  - **Constructors**:
    - `Dims(DimType d0 = NO_DIM, DimType d1 = NO_DIM, DimType d2 = NO_DIM, DimType d3 = NO_DIM)`: Constructs a `Dims` object with given four dimensions.
    - `Dims(const Dims &dims_)`: Constructs a `Dims` object by copying another `Dims` object.
    - `Dims(const std::vector<DimType> &vec)`: Constructs a `Dims` object from a vector. If the vector is shorter than `DIMS_LEN`, appends `NO_DIM`s. Raises an error if the vector is longer than `DIMS_LEN`.
  
  - **Methods**:
    - `DimType size() const`: Returns the volume of dimensions. If the dimensions are invalid, returns -1.
    - `int ndims() const`: Returns the number of valid dimensions.
    - `Dims dims4() const`: Returns a new `Dims` object with 4 valid dimensions by prepending 1s.
    - `bool is_no_dim() const`: Returns true if the dimensions are empty.
    - `bool is_invalid() const`: Returns true if the dimensions are invalid.
  
  - **Operators**:
    - `DimType &operator[](DimType idx)`: Returns a reference to the dimension at the specified index.
    - `const DimType &operator[](DimType idx) const`: Returns a constant reference to the dimension at the specified index.
    - `constexpr Dims &operator=(const Dims &) = default`: Default assignment operator.
  
  - **Friend Functions**:
    - `bool operator==(const Dims &a, const Dims &b)`: Equality operator.
    - `bool operator!=(const Dims &a, const Dims &b)`: Inequality operator.
    - `std::ostream &operator<<(std::ostream &os, const Dims &dims)`: Stream insertion operator.
  
#### Tensor

class Tensor

 Tensor is a view of a TensorBuf. A TensorBuf can have multiple Tensor points to different area of the same TensorBuf.

 Illustration of a single axis of a tensor:

 0           off ldim
 |------------|-------------shape-------------|---------------------------|
               <----------------------------->
                  data range of this
                  tensor




Tensor is a basic unit in ARK's model. 

- **Constructors**:

Tensor(const Dims &shape, TensorType type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           bool imported, int id, const std::string &name);

- shape: shape of the tensor
example: 


    TensorBuf *buf = this->create_tensor_buf();
    Tensor *ret = new Tensor{shape,    type,     buf,
                             ldims,    offs,     pads,
                             exported, imported, (int)this->tns_storage.size(),
                             name};

This will create a tensor with shape {1, 16, 16} and half precision.


# Model Class: An Essential Tool for Tensor Manipulation and Neural Network Operations

The Model class is a powerful and versatile API designed to assist users in creating, manipulating, and performing operations on tensors in an efficient and user-friendly manner. This class offers a variety of functions that serve as essential building blocks for implementing complex deep learning models and other tensor-based algorithms.

The Model class provides a wide range of functions for tensor manipulation, including:

- **Tensor creation**: Create tensors with specified shapes and data types, with optional parameters such as buffer, layout dimensions, offsets, padding, and more.

- **Reshaping**: Change the shape of a tensor according to specified dimensions, with options to allow zero-sized dimensions or infer a dimension's size from the input tensor.

- **Identity**: Create an identical tensor with specified execution dependencies.

- **Sharding**: Divide a tensor along a specified axis into smaller shards with a given dimension per shard.

- **Reduction**: Perform reduction operations along a specified axis of a tensor, with support for ReLU activation.

- **Layer normalization**: Apply layer normalization to a tensor, resulting in an output tensor.

- **Softmax activation**: Apply the softmax activation function to a tensor, with the operation being performed on the last dimension of the input tensor.

- **Transpose**: Rearrange a tensor's dimensions according to a specified permutation.

- **Matrix multiplication**: Perform matrix multiplication between two tensors, with optional parameters to control transposition and ReLU activation.

- **Linear layer**: Implement a fully connected layer of a neural network model, with support for bias and ReLU activation.

- **2D convolution**: Implement a 2D convolution layer using the 'im2col' method, with options for bias and output tensor.

- **Max-pooling**: Apply max-pooling to a tensor using specified kernel size and stride, reducing its spatial dimensions.

- **Scalar multiplication**: Multiply a tensor by a scalar value, element-wise.

- **GELU activation**: Apply the Gaussian Error Linear Unit (GELU) activation function to a tensor, element-wise.

- **Element-wise addition**: Perform an element-wise addition operation between two tensors.

- **Element-wise multiplication**: Perform an element-wise multiplication operation between two tensors.

- **Tensor communication**: Send a tensor to a destination GPU and receive it on the other end, enabling multi-GPU operations.

