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

`ProcessorRange`, `WarpRange`, `SramRange`, and `TaskRange` are in the "range" format, i.e., `[Begin, End, Step]` that indicates an arithmetic integer sequence with a common difference of `Step`, starting from `Begin` and ends before `End` (does not include `End`). They alternatively can be in the format `[Begin, End]` that assumes `Step` is 1.

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

A `ProcessorGroup` object describes computing tasks of a group of processors (e.g., streaming multiprocessor of NVIDIA architectures). `ProcessorRange` describes the set of processor IDs used by this processor group. A processor group consists of a sequence of `ResourceGroup`s. Each resource group should use the whole or a part of the processors used by the processor group.

Multiple processor groups are executed in the order as appears in the plan file, but not necessarily in sequence. Specifically, if a later processor group uses a part of processors used by an earlier processor group, the executor will put a processor barrier to synchronize all processors in both processor groups. Otherwise, if a later processor group does not use any processors of earlier processor groups, the later processor group will be executed immediately without a processor barrier, which may run concurrently with earlier processor groups.

## ResourceGroup

A `ResourceGroup` object describes computing tasks that use the entire or a subset of computing resources of a processor group. The followings explain each field of a `ResourceGroup` object.

- `ProcessorRange`: range of processors to use, which should be a subset of `ProcessorRange` of the processor group.
- `WarpRange`: range of concurrent warps to use. The same range of warps is used across all processors of the resource group.
- `SramRange`: range of SRAM space to use in byte unit, where the step of the range should be 1. The same range of SRAM is used across all processors of the resource group.
- `TaskGroups`: list of `TaskGroup` that are fetched in order. Tasks in a `TaskGroup` are distributed across all concurrent warps in the resource group.

## TaskGroup

A `TaskGroup` object describes computing tasks. Each task can be typically considered as computing a single output tile of an operator. The `TaskId` field declares the type of task, of which details are found from `TaskInfos`. The `TaskRange` field declares tasks to run, which should be within the range `[0, NumTasks)` where `NumTasks` is found from `Config` of operators in the `TaskInfo`. If there are multiple operators in a `TaskInfo`, all operators should have the same `NumTasks`.

Tasks in the `TaskRange` are distributed across processors in the resource group. If `Granularity` is 1, the distribution is round-robin. Otherwise, the distribution assigns `Granularity` consequent tasks to each processor (as long as there are enough tasks), and then assign the following task to the next processor. `Granularity` should be always a positive integer.
