# ARK Install Instructions

## Preliminaries

* Linux kernel >= 4.15.0

    - If you have a lower version, you can upgrade it via:
        ```
        apt-get update
        apt-get install -y linux-image-4.15.0-13-generic linux-header-4.15.0-13-generic
        ```

* CMake >= 3.25.0 and Python >= 3.7

* GPUs with CUDA compute capability >= 7.0 and CUDA version >= 11.1

* To run ARK in a Docker container, we need to mount `/dev` and `/lib/modules` into the container so that the container can use `gpumem` driver. Add the following options in the `docker run` command:
    ```
    -v /dev:/dev -v /lib/modules:/lib/modules
    ```

## Install `gpudma`

*NOTE: if you are using a Docker container, the following steps should be done on the host.*

1. Compile `gpudma`.

    ```
    cd third_party
    make gpudma
    ```
    - This may fail if you don't have a proper `gcc` version, which will be notified by an error message. In that case, [install an alternative version of `gcc`](https://github.com/chhwang/devel-note/wiki/Building-GCC-from-source).

2. Load `gpumem` driver.

    ```
    insmod third_party/gpudma/module/gpumem.ko
    chmod 666 /dev/gpumem
    ```

3. Check if the `gpumem` driver is running.

    ```
    lsmod | grep gpumem
    ```

## Install ARK Python

1. Go to the repo root directory and install Python dependencies.

    ```
    python3 -m pip install -r requirements.txt
    ```

2. Install ARK Python.

    ```
    python3 -m pip install .
    ```

3. (Optional) Run the tutorial code to verify the installation.

    ```
    cd examples/tutorial
    python3 tutorial.py
    ```

## (Optional) Install ARK C++ and Run Unit Tests

If you want to use only the core C++ interfaces, follow the instructions below.

1. Go to the repo root directory and configure CMake. Replace `CMAKE_INSTALL_PREFIX` with your desired installation directory.

    ```
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
    ```

2. Build ARK.

    ```
    make -j build
    ```

3. (Optional) We offer CTest unit tests for ARK C++. To build the tests, run:

    ```
    make -j ut
    ```

    **NOTE:** currently unit tests require at least 4 GPUs in the system for communication tests.

    Lock GPU clock frequency for stable test results:

    ```
    nvidia-smi -pm 1
    for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
        nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
    done
    ```

    Run the tests:

    ```
    ARK_ROOT=$PWD ctest --output-on-failure
    ```

    **NOTE:** unit tests may take tens of minutes to finish.

4. Install ARK C++.

    ```
    sudo make install
    ```
