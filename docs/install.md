# ARK Install Instructions

## Preliminaries

* Linux kernel >= 4.15.0

    - If you have a lower version, you can upgrade it via:
        ```
        apt-get update
        apt-get install -y linux-image-4.15.0-13-generic linux-header-4.15.0-13-generic
        ```

* To run ARK in a Docker container, we need to mount `/dev` and `/lib/modules` into the container so that the container can use `gpumem` driver. Add the following options in the `docker run` command:
    ```
    -v /dev:/dev -v /lib/modules:/lib/modules
    ```

## Install `gpudma`

*NOTE: if you are using a Docker container, the following step 1, 2, and 3 should be done on the host, and step 4 should be done on both the host and the container.*

1. Compile `gpudma`.

    ```
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

## Install ARK and Run Unit Tests

1. Download third-parties & compile ARK and unittest.

    ```
    make -j
    ```

2. Install ARK.

    ```
    make install
    ```

    Installation directory is specified as `ARK_ROOT` environment variable,
    which is `${HOME}/.ark` by default when it is unset.

3. (Optional) Run unittest.

    ```
    ./scripts/unittest.sh
    ```

    **NOTE:** currently unittest supports CUDA compute capapbility >= 7.0 only,
    and it requires at least 2 GPUs in the system for communication tests.


## Install ARK Python Bindings  
  
We offer Python bindings for ARK, allowing users to access and utilize ARK in their Python projects. These bindings are created using pybind11 and built on top of the C++ API.
  
### Building the Python Bindings  
1. Install pybind11 using pip:  

```bash
pip install -r requirements.txt
```

2. Run the following command to build the Python bindings:  

```bash
python setup.py build_ext
```

After running this script, the `ark.cpython-38-x86_64-linux-gnu.so` will be generated in the `ark/python/build/lib.linux-x86_64-3.8` directory. If --inplace is specified, a shared library will be generated in the current directory. Note that the name of the generated library may be different depending on the Python version and the operating system.
  
3. Add the generated library to your PYTHONPATH:  

```bash
export PYTHONPATH="$ARK_DIR/ark/python/build/lib.linux-x86_64-3.8:${PYTHONPATH}"
```
  
### Testing the Python Bindings  
  
Change to the `ark/python` directory and run the Python test script:  

```bash
cd ark/python
python unittest/api_test.py
```

This will test the Python bindings for ARK. If the tests pass, you have successfully built and tested the Python bindings.  

