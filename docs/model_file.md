# Model File Format

## Hierarchy

    - Rank (Int)
    - WorldSize (Int)
    - Nodes (Array of Node)
        - Node (Object)
            - Id (Int)
            - ProducerNodeIds (Array of Int)
            - ConsumerNodeIds (Array of Int)
            - Ops (Array of Op)
                - Op (Object)
                    - Type (String)
                    - Name (String)
                    - IsVirtual (Boolean)
                    - ReadTensors (Array of Tensor)
                        - Tensor (Object, details below)
                    - WriteTensors (Array of Tensor)
                        - Tensor (Object, details below)
                    - ResultTensors (Array of Tensor)
                        - Tensor (Object, details below)
                    - Args (Object, structure depends on Op Type)

A `Tensor` object has the following structure:

    - Tensor (Object)
        - Id (Int)
        - DataType (String)
        - Buffer (Object)
            - Id (Int)
            - Rank (Int)
            - SendTags (Array of Int)
            - RecvTags (Array of Int)
        - Shape (Array of Int)
        - Strides (Array of Int)
        - Offsets (Array of Int)
        - Pads (Array of Int)

An `Args` object has a flexible structure depending on the type of `Op`, which would look like as follows.

    "Args": {
        "ArgName0": { "ArgType0": ArgValue0 },
        "ArgName1": { "ArgType1": ArgValue1 },
        ...
    }

## Node

A `Node` object describes a node in the computation graph.

A node consists of an array of one or more operators (`Op`s). The operators in a node are supposed to be executed in the order that appears in the array, but they may not have hard dependencies in between, i.e., a later operator's computation may depend only on a part of a previous operator's result. For example, if an element-wise operator is followed by another element-wise operator with the same shape of data, each element of the later operator depends only on a single element from the earlier operator. This poses possibility of operator fusion when we design an execution plan of this model.

Each node may produce or consume tensors. Produced tensors are those that appear in an operator's `ResultTensors` array, while consumed tensors are those that appear in an operator's `ReadTensors` or `WriteTensors` array. Each node has a unique ID, and declares `ProducerNodeIds` and `ConsumerNodeIds` to describe dependencies between nodes. `ProducerNodeIds` lists IDs of all nodes that produce tensors consumed by this node. Similarly, `ConsumerNodeIds` lists IDs of all nodes that consume tensors produced by this node.

## Op

An `Op` object describes computation that reads from `ReadTensors`, reads & writes on `WriteTensors`, and returns `ResultTensors`. `Type` is a string that describes the type of computation, and `Name` is a user-provided name just for readability of the model file. `IsVirtual` is a boolean value that is `true` only if this operator does not perform any real computation, so called a virtual operator. `Args` is a key-value object with a flexible structure that describes a few details of the operator computation. A key of `Args` is a name of an argument, and the value is another key-value object, of which key is the data type of the argument and value is the actual value of the argument. The following is an example of `Args` of a `Matmul` type operator (explanation of each field follows below).

    "Args": {
        "InputDimNC": {
            "DIMS": [1,1]
        },
        "TransposeInput": {
            "BOOL": false
        },
        "TransposeOther": {
            "BOOL": true
        },
        "OtherDimNC": {
            "DIMS": [1,1]
        },
        "ShapeMNK": {
            "DIMS": [512,4096,11008]
        },
        "StridesACDB": {
            "DIMS": [11008,4096,4096,11008]
        }
    }

The followings are possible data types of an argument.

- `INT`: 32-bit signed integer.
- `INT64`: 64-bit signed integer.
- `UINT64`: 64-bit unsigned integer.
- `BOOL`: boolean.
- `FLOAT`: 32-bit float.
- `DIMS`: integer array of which length is up to 4.
- `TENSOR`: a tensor object.
- `OFFSET`: an offset object that describes a relative memory address, which consists of `BufferId` (ID of a buffer) and `Value` (integer offset within the buffer) fields.

### Arguments Details

The followings describe arguments of different types of operators. Those that are not listed here would mean that they do not need any arguments.

- `Matmul`
    - `InputDimNC` (type: `DIMS`): considering 4-dimensional matrix multiplication between [N,C,H,W] format tensors, `InputDimNC` argument represents the [N,C] value of the first input tensor. If the tensor is 3-dimensional ([C,H,W]), N is set to 1. If the tensor is 2-dimensional ([H,W]), both N and C are set to 1.
    - `OtherDimNC` (type: `DIMS`): considering 4-dimensional matrix multiplication between [N,C,H,W] format tensors, `OtherDimNC` argument represents the [N,C] value of the second input tensor. If the tensor is 3-dimensional ([C,H,W]), N is set to 1. If the tensor is 2-dimensional ([H,W]), both N and C are set to 1.
    - `ShapeMNK` (type: `DIMS`): problem shape of matrix multiplication in the [M,N,K] format.
    - `StridesACDB` (type: `DIMS`): stride length of the lowest dimension of matrix A, C, D, and B, in order. Matrix A and B refer to the first and the second tensors, respectively, and both of matrix C and D refer to the output tensor.
    - `TransposeInput` (type: `BOOL`): true if the last two dimensions of the first input tensor should be transposed before computation, and vice versa.
    - `TransposeOther` (type: `BOOL`): true if the last two dimensions of the second input tensor should be transposed before computation, and vice versa.

- `ReduceSum`, `ReduceMax`, `ReduceMean`
    - `Axis` (type: `INT`): the axis to reduce.
    - `KeepDim` (type: `BOOL`): if true, the output tensor has the same shape as the input tensor but the reduced axis of which size is 1. Otherwise, the reduced axis will be squeezed, thus the output tensor will have one less dimension than the input.

- `ScalarAssign`, `ScalarAdd`, `ScalarMul`
    - `Value` (type: `FLOAT`): the scalar value used by the computation.

- `Transpose`
    - `Permutation` (type: `DIMS`): if the input tensor shape has N dimensions, `Permutation` is an arbitrary permutation of [0, 1, 2, ..., N-1], where the i-th value represents the index of the output tensor dimension that has the same size as the i-th dimension of the input tensor.

## Tensor

A `Tensor` object describes an N-dimensional (0 < N <= 4) data patch over the memory. A tensor points to the whole or a part of memory space of a `Buffer` object. Since a tensor itself is not a buffer, multiple tensors may point to the same or overlapped address space, while buffers represent an exclusive address space from each other. Also, while a buffer represents only a sequential address space, a tensor can refer to a strided address space. For example, if `Shape` is `[s_0, s_1, s_2, s_3]`, `Strides` is `[t_0, t_1, t_2, t_3]`, and `Offsets` is `[f_0, f_1, f_2, f_3]` (where `t_i` >= `s_i` + `f_i`), the tensor refers to data elements from index `f_i` to index `f_i + s_i - 1` in each dimension, among the entire dimension from `0` to `t_i - 1`. `Shape`, `Strides`, and `Offsets` should have the same number of dimensions.

`DataType` field of a tensor represents the data type of each element. Currently, the following data types are supported.

- `FP32`
- `FP16`
- `BF16`
- `INT32`
- `UINT32`
- `INT8`
- `UINT8`
- `BYTE`

`Pads` field is currently unused and will be used in the future. For now, `Pads` should be an array of ones with the same number of dimension as `Shape`.

## Buffer

As explained in the Tensor section, a `Buffer` object represents a unique and sequential memory space. Buffer objects have following fields.

- `Id`: a unique buffer ID.
- `Rank`: rank of the model that physically allocates this buffer. If -1, it refers to the local rank (the rank of this model file).
- `SendTags`: a list of (`RemoteRank`, `Tag`) tuples, which indicates that the data of this buffer may be copied to a remote buffer allocated by `RemoteRank`. `Tag` is an integer to identify the corresponding `RecvTags` entry of the remote buffer, i.e., the remote buffer should have a corresponding (`Rank`, `Tag`) entry in its `RecvTags`, where `Rank` is the local rank and the same `Tag` value.
- `RecvTags`: a list of (`RemoteRank`, `Tag`) tuples, which indicates that this buffer may be overwritten by data from a remote buffer allocated by `RemoteRank`. `Tag` is an integer to identify the corresponding `SendTags` entry of the remote buffer, i.e., the remote buffer should have a corresponding (`Rank`, `Tag`) entry in its `SendTags`, where `Rank` is the local rank and the same `Tag` value.
