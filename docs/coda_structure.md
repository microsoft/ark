# Code Structure Overview  
  
This codebase is organized into several top-level files and directories, each serving a specific purpose. Here's a brief overview of the structure:  
  
## Top-level Files  
  
- `cpu_timer.cc` and `cpu_timer.h`: Implementation and header files for CPU timing functionality.  
- `dims.cc` and `dims_test.cc`: Implementation and tests for the dimensions functionality.  
- `env.cc` and `env.h`: Environment-related implementation and header files.  
- `executor.cc`: Execution-related functionality.  
- `file_io.cc` and `file_io.h`: File input/output implementation and header files.  
- `init.cc`: Initialization-related functionality.  
- `kahypar.h` and `kahypar_test.cc`: Header and test files for the Kahypar library.  
- `logging.cc` and `logging.h`: Logging implementation and header files.  
- `math.cc` and `math.h`: Mathematical functions implementation and header files.  
- `model.cc`, `model_io.cc`, and `model_io.h`: Model-related implementation and header files.  
- `model_test.cc`: Model-related test file.  
- `random.cc`: Random number generation functionality.  
- `tensor.cc`: Tensor-related functionality.  
- `threading.h`: Threading-related header file.  
- `utils.cc`: General utility functions.  
  
## Directories  
  
### gpu  
  
This documentation provides an overview of the contents and functionalities of the GPU folder within the codebase.    
The GPU folder contains all GPU-related files, including GPU buffer management, GPU memory management, GPU kernel implementation and tests, and GPU manager functionality. The following is a brief description of each file in the folder:  
  
#### Files  
  
- `gpu_buf.cc` and `gpu_buf.h`: Implementation and header files for GPU buffer management.  
- `gpu_comm_sw.cc` and `gpu_comm_sw.h`: Implementation and header files for GPU communication software.  
- `gpu_common.h`: Header file containing common GPU-related definitions and utilities.  
- `gpu_compile.cc` and `gpu_compile.h`: Implementation and header files for GPU kernel compilation.  
- `gpu_kernel.cc`, `gpu_kernel.h`, and `gpu_kernel_test.cc`: Implementation, header, and test files for GPU kernel functionality.  
- `gpu_logging.h`: Header file for GPU-related logging functionality.  
- `gpu_mem.cc`, `gpu_mem.h`, and `gpu_mem_test.cc`: Implementation, header, and test files for GPU memory management.  
- `gpu_mgr.cc`, `gpu_mgr.h`, and `gpu_mgr_test.cc`: Implementation, header, and test files for the GPU manager functionality.   
  
### include  
  
This directory contains the interface header files for the entire project.
`ark.h`, `ark_utils.h` are the interface header files for the ARK library.

# CUDA Kernels Folder Documentation  
  
This documentation provides an overview of the contents and functionalities of the CUDA Kernels folder within the codebase.  
  
The CUDA Kernels folder contains all CUDA kernel-related header files, including various mathematical operations, matrix manipulation, memory management, and communication operations. The following is a brief description of each file in the folder:  
  
### Files  
  
- `activation.h`: Header file for activation functions (e.g., ReLU, sigmoid, tanh) used in neural networks.  
- `arithmetic.h`: Header file for arithmetic operations (e.g., addition, subtraction, multiplication, division) on tensors.  
- `base_op.h`: Header file for basic operations on tensors, such as element-wise operations and reduction operations.  
- `comm.h`: Header file for communication operations between GPU devices.  
- `comm_mm.h`: Header file for communication operations related to matrix multiplication.  
- `common.h`: Header file containing common CUDA kernel-related definitions and utilities.  
- `gemm.h`: Header file for General Matrix Multiplication (GEMM) operations.  
- `im2col.h`: Header file for the im2col operation, which is used in convolutional neural networks.  
- `layernorm.h`: Header file for layer normalization operations in neural networks.  
- `matmul.h`: Header file for matrix multiplication operations on tensors.  
- `reduce.h`: Header file for reduction operations (e.g., sum, product, minimum, maximum) on tensors.  
- `smem.h`: Header file for shared memory management within CUDA kernels.  
- `softmax.h`: Header file for softmax operation, which is commonly used in neural network classification tasks.  
- `sync.h`: Header file for synchronization operations within and between CUDA kernels.  
- `static_math.h`: Header file for static mathematical functions (e.g., exponential, logarithm, power) on tensors.  
- `transform.h`: Header file for various data transformation operations on tensors.  
- `transpose.h`: Header file for tensor transpose operations.  
- `unit_op.h`: Header file for unit operations on tensors, such as scaling and shifting.  
- `vec.h`: Header file for vector operations on tensors.  
- `arch.h`: Header file for defining the architecture of the GPU devices.  
- `ark_kernels.h`: Header file for Automatic Runtime Kernels (ARK), which are used to optimize the performance of CUDA kernels.  
- `broadcast.h`: Header file for broadcasting operations on tensors.    
### ipc  
  
The ipc directory contains files related to inter-process communication, including the implementation of shared memory, socket communication, and IPC-related test files.  

# IPC Folder Documentation  
  
This documentation provides an overview of the contents and functionalities of the Inter-Process Communication (IPC) folder within the codebase.  
  
## Overview  
  
The IPC folder contains all the files related to inter-process communication, including communication using sockets, shared memory, locks, and more. The following is a brief description of each file in the folder:  
  
### Files  
  
- `ipc_coll.cc` and `ipc_coll.h`: Implementation and header files for IPC collective communication.  
- `ipc_coll_test.cc`: Test file for IPC collective communication.  
- `ipc_hosts.cc` and `ipc_hosts.h`: Implementation and header files for IPC host management.  
- `ipc_lock.cc` and `ipc_lock.h`: Implementation and header files for IPC locking mechanisms.  
- `ipc_mem.cc`, `ipc_mem.h`, and `ipc_mem_test.cc`: Implementation, header, and test files for IPC memory management.  
- `ipc_shm.cc` and `ipc_shm.h`: Implementation and header files for IPC shared memory management.  
- `ipc_socket.cc`, `ipc_socket.h`, and `ipc_socket_test.cc`: Implementation, header, and test files for IPC socket communication.  
- `ipc_table.cc` and `ipc_table.h`: Implementation and header files for IPC table management.  
  
### net  
  
This directory contains files for the InfiniBand network implementation and tests.  
  
The Net folder contains all the files related to network communication using InfiniBand (IB), a high-performance, low-latency networking technology. The following is a brief description of each file in the folder:  
  
### Files  
  
- `net_ib.cc` and `net_ib.h`: Implementation and header files for InfiniBand network communication.  
- `net_ib_test.cc`: Test file for InfiniBand network communication.

### ops  
  
This directory contains files for implementing and testing various operations, such as addition, multiplication, and convolution. It also includes a subdirectory named `kernels` that contains simple kernel implementations.  
  
### sched  
  
This directory contains files related to scheduling, including different scheduling algorithms, code generation, operation graph representation, and profiling functionality.  

# Scheduling Folder Documentation  
  
This documentation provides an overview of the contents and functionalities of the Scheduling folder within the codebase.  
  
## Overview  
  
The Scheduling folder contains all the files related to scheduling operations, including various scheduling algorithms, code generation, operation graph management, profiling, and testing. The following is a brief description of each file in the folder:  
  
### Files  
  
- `sched`: Directory containing the implementation of different scheduling algorithms.  
  - `sched_default.cc`: Implementation file for the default scheduling algorithm.  
  - `sched_kahypar.cc`: Implementation file for the KaHyPar scheduling algorithm.  
  - `sched_simple.cc`: Implementation file for the simple scheduling algorithm.  
- `sched.h`: Header file for scheduling operations.  
- `sched_codegen.cc` and `sched_codegen.h`: Implementation and header files for scheduling code generation.  
- `sched_op.cc` and `sched_op.h`: Implementation and header files for scheduling operations.  
- `sched_opgraph.cc` and `sched_opgraph.h`: Implementation and header files for scheduling operation graph management.  
- `sched_opseq.cc` and `sched_opseq.h`: Implementation and header files for scheduling operation sequence management.  
- `sched_profiler.cc` and `sched_profiler.h`: Implementation and header files for scheduling profiler.  
- `sched_test.cc`: Test file for scheduling operations.  
- `sched_tile.cc` and `sched_tile.h`: Implementation and header files for scheduling tiling operations.    

### unittest  
  
This directory contains utility files for unit testing.  
