# ARK Architecture






# Code Structure Overview  
  
This codebase is organized into several top-level files and directories, each serving a specific purpose. Here's a brief overview of the structure:  
  
## Directories  
  
### gpu  
  
The GPU folder contains all GPU-related files, including GPU buffer management, GPU memory management, GPU kernel implementation and tests, and GPU manager functionality.  
  
### include  
  
This directory contains the interface header files for the entire project.  
`ark.h`, `ark_utils.h` are the interface header files for the ARK library.  
  
#### kernel
  
The CUDA Kernels folder contains all CUDA kernel-related header files, including various mathematical operations, matrix manipulation, memory management, and communication operations.  
  
### ipc  
  
The IPC folder contains all the files related to inter-process communication, including communication using sockets, shared memory, locks, and more.  
  
### net  
  
This directory contains files for the InfiniBand network implementation and tests.  
  
### ops  
  
This directory contains files for implementing and testing various operations, such as addition, multiplication, and convolution. It also includes a subdirectory named `kernels` that contains simple kernel implementations.  
  
### sched  
  
The Scheduling folder contains all the files related to scheduling operations, including various scheduling algorithms, code generation, operation graph management, profiling, and testing.  
  
### unittest  
  
This directory contains utility files for unit testing.  


The next graph shows the 
