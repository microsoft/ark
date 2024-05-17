# Plan File Format

See an example plan file: [Example 1](../examples/tutorial/default_plan.json)

## Hierarchy

    - Rank (Int)
    - WorldSize (Int)
    - NumProcessors (Int)
    - NumWarpsPerProcessor (Int)
    - TaskInfos (Array of TaskInfo)
        - TaskInfo (Object)
            - Id (Int)
            - NumWarps (Int)
            - SramBytes (Int)
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
                    - Config (Object, structure depends on Op Type)
    - ProcessorGroups (Array of ProcessorGroup)
        - ProcessorGroup (Object)
            - ProcessorRange (Array of Int)
            - ResourceGroups (Array of ResourceGroup)
                - ResourceGroup (Object)
                    - ProcessorRange (Array of Int)
                    - WarpRange (Array of Int)
                    - SramRange (Array of Int)
                    - TaskGroups (Array of TaskGroup)
                        - TaskGroup (Object)
                            - TaskId (Int)
                            - TaskRange (Array of Int)
                            - Granularity (Int)

## TaskInfo

A `TaskInfo` object describes a sequential set of operators. The followings describe each field of `TaskInfo`.

- `Id`: a unique ID.
- `NumWarps`: number of concurrent warps per processor needed.
- `SramBytes`: bytes of SRAM (e.g., CUDA shared memory) needed.
- `Ops`: array of `Op`s as described below.

## Op

Structure of an `Op` object in a plan file is the same as [the one in the model file](model_file.md#op), but in a plan file, it has the additional `Config` field that describes detailed implementation of the operator. `Config` has a flexible structure depending on the operator type.

### Config Details

The followings explain a few fields that many configs commonly consist of.

- `NumWarps`: number of concurrent warps needed to calculate a single output tile.
- `SramBytes`: bytes of SRAM needed to calculate a single output tile.
- `NumTasks`: total number of output tiles need to compute.

The followings describe `Config` structure of different types of operators.

- `Matmul`
    - `NumWarps`
    - `SramBytes`
    - `NumTasks`
    - `TileShapeMNK`: tile shape of matrix multiplication in the [M,N,K] format.
    - `TilePadMNK`: this field is not well defined and will be updated in the future. Currently, it should be the same as `TileShapeMNK`.

- `ReduceSum`, `ReduceMax`, `ReduceMean`
    - `NumWarps`
    - `SramBytes`
    - `NumTasks`
    - `ImplType`: type of reduction implementation, either `WarpWise` or `ElementWise`.

- `Send`, `SendDone`, `Recv`
    - `NumWarps`: should be always 1.
    - `SramBytes`: should be always 0.
    - `NumTasks`: should be always 1.

- `Embedding`
    - `NumWarps`
    - `SramBytes`
    - `NumTasks`

- `Noop`
    - `NumWarps`: should be always 1.
    - `SramBytes`: should be always 0.
    - `NumTasks`: should be always 0.

- `Default`: all other operators that are not listed above follow this structure.
    - `NumWarps`
    - `SramBytes`
    - `NumTasks`
    - `Tile`: 2-dimensional shape of a single output tile.

## ProcessorGroup

## ResourceGroup

## TaskGroup
