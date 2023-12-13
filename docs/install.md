# ARK Install Instructions

## Prerequisites

* Linux kernel >= 4.15.0

    - If you have a lower version, you can upgrade it via:
        ```bash
        sudo apt-get update
        sudo apt-get install -y linux-image-4.15.0-13-generic linux-header-4.15.0-13-generic
        ```

* CMake >= 3.25.0 and Python >= 3.8

* Supported GPUs
    - NVIDIA GPUs: Volta (CUDA >= 11.1) / Ampere (CUDA >= 11.1) / Hopper (CUDA >= 12.0)
        - Hopper support will be added in the future.
    - AMD GPUs: CDNA2 (ROCm >= 5.0) / CDNA3
        - Multi-GPU execution is not yet supported for AMD GPUs and will be supported by a future release.
        - CDNA3 support will be added in the future.

* Mellanox OFED

## Docker Images

We currently provide only *base images* for ARK, which contain all the dependencies for ARK but do not contain ARK itself. The ARK-installed images will be provided in the future.

You can pull a base image as follows.
```
# For NVIDIA GPUs
docker pull ghcr.io/microsoft/ark/ark:base-dev-cuda12.1
# For AMD GPUs
docker pull ghcr.io/microsoft/ark/ark:base-dev-rocm5.6
```

Check [ARK containers](https://github.com/microsoft/ark/pkgs/container/ark%2Fark) for all available Docker images.

*NOTE(Dec 2023): ROCm Docker images are not yet verified enough and may be updated in the future.*

The following is an example `docker run` command for NVIDIA GPUs.
```
# Run a container for NVIDIA GPUs
docker run \
    --privileged \
    --ulimit memlock=-1:-1 \
    --net=host \
    --ipc=host \
    --gpus all \
    -it --name [Container Name] [Image Name] bash
```

The following is an example `docker run` command for AMD GPUs.
```
# Run a container for AMD GPUs
docker run \
    --privileged \
    --ulimit memlock=-1:-1 \
    --net=host \
    --ipc=host \
    --security-opt seccomp=unconfined --group-add video \
    -it --name [Container Name] [Image Name] bash
```

## Install ARK Python

1. Go to the repo root directory and install Python dependencies.

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2. Install ARK Python.

    ```bash
    python3 -m pip install .
    ```

3. (Optional) Run the tutorial code to verify the installation.

    ```bash
    cd examples/tutorial
    python3 quickstart_tutorial.py
    ```

## (Optional) Install ARK C++ and Run Unit Tests

If you want to use only the core C++ interfaces, follow the instructions below.

1. Go to the repo root directory and configure CMake. Replace `CMAKE_INSTALL_PREFIX` with your desired installation directory.

    **NOTE:** if you install ARK C++ for debugging purposes, use `-DCMAKE_BUILD_TYPE=Debug` option.

    ```bash
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/usr/local ..
    ```

2. Build ARK.

    ```bash
    make -j build
    ```

3. (Optional) We offer CTest unit tests for ARK C++. To build the tests, run:

    ```bash
    make -j ut
    ```

    **NOTE:** currently unit tests require at least 4 GPUs in the system for communication tests. GPUs also need to be peer-to-peer accessible (e.g., on the same PCIe switch or using NVLink/xGMI).

    Lock GPU clock frequency for stable test results. For example, on NVIDIA GPUs:

    ```bash
    sudo nvidia-smi -pm 1
    for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
        sudo nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
    done
    ```

    Run the tests.

    ```bash
    ARK_ROOT=$PWD ctest --verbose
    ```

    **NOTE:** unit tests may take tens of minutes to finish.

4. Install ARK C++.

    ```bash
    sudo make install
    ```
