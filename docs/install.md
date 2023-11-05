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
        - CDNA3 support will be added in the future.

* Mellanox OFED

## Docker Images

We currently provide only *base images* for ARK, which contain all the dependencies for ARK but do not contain ARK itself (no [`gpudma`](https://github.com/microsoft/ark/blob/main/docs/install.md#install-gpudma) as well, which should be installed on the host side). The ARK-installed images will be provided in the future.

You can pull a base image as follows.
```
# For NVIDIA GPUs
docker pull ghcr.io/microsoft/ark/ark:base-dev-cuda12.1
# For AMD GPUs
docker pull ghcr.io/microsoft/ark/ark:base-dev-rocm5.7
```

Check [ARK containers](https://github.com/microsoft/ark/pkgs/container/ark%2Fark) for all available Docker images.

To run ARK in a Docker container with NVIDIA GPUs, we need to mount `/dev` and `/lib/modules` into the container so that the container can use `gpumem` driver. Specifically, add `--privileged -v /dev:/dev -v /lib/modules:/lib/modules` in the `docker run` command. The following is an example.
```
docker run \
    --privileged \
    --cap-add=ALL \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --net=host \
    --ipc=host \
    --gpus all \
    -v /dev:/dev \
    -v /lib/modules:/lib/modules \
    -it --name [Container Name] [Image Name] bash
```

AMD GPUs do not need `gpudma` driver, so you don't need to mount `/dev` and `/lib/modules` into the container.

## Install `gpudma` (NVIDIA GPUs only)

**NOTE: if you are using a CUDA Docker container, the steps in this section should be done on the host.**

1. Pull submodules.

    ```bash
    git submodule update --init --recursive
    ```

2. Compile `gpudma`.

    ```bash
    cd third_party
    make gpudma
    ```
    - This may fail if you don't have a proper `gcc` version, which will be notified by an error message. In that case, [install an alternative version of `gcc`](https://github.com/chhwang/devel-note/wiki/Building-GCC-from-source).

3. Load `gpumem` driver.

    ```bash
    sudo insmod gpudma/module/gpumem.ko
    sudo chmod 666 /dev/gpumem
    ```

4. Check if the `gpumem` driver is running.

    ```bash
    lsmod | grep gpumem
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

    Run the tests. If you do want to disable cross-node networking, pass `ARK_DISABLE_IB=1` environment variable to the test command.

    ```bash
    ARK_ROOT=$PWD ctest --verbose
    ```

    **NOTE:** unit tests may take tens of minutes to finish.

4. Install ARK C++.

    ```bash
    sudo make install
    ```
