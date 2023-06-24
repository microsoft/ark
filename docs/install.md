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

## Install `gpumem` Driver

*NOTE: if you are using a Docker container, the following step 1, 2, and 3 should be done on the host, and step 4 should be done on both the host and the container.*

1. Compile NVIDIA driver source code.

    - Run `make` in `/usr/src/nvidia-xxx-xxx.yy.zz` (replace `xxx-xxx.yy.zz` into your own version).
    - `make` will fail if you don't have a proper version of `gcc`, which will be notified by an error message. In that case, [install an alternative version of `gcc`](https://github.com/chhwang/devel-note/wiki/Building-GCC-from-source) or [download a different version of driver](https://www.nvidia.com/en-us/drivers/unix/).
    - If you download a different driver, run following commands (replace `xxx.yy` into your own version).
        ```
        chmod +x NVIDIA-Linux-x86_64-xxx.yy.run
        ./NVIDIA-Linux-x86_64-xxx.yy.run -x
        cd NVIDIA-Linux-x86_64-xxx.yy.run/kernel
        make
        ```

2. Compile `gpumem` driver.

    - Change directory into [`gpumem/`](gpumem).
    - If you compiled the NVIDIA driver from `/usr/src`, just run `make`. If you downloaded a different driver, run
    
        ```NVIDIA_SRC=/path/to/NVIDIA-Linux-x86_64-xxx.yy.run/kernel make```
    - If it succeed without errors or warnings, run
    
        ```sh drvload.sh```

3. Verify that `gpumem` driver is running.

    ```lsmod | grep gpumem```

4. Verify that `gpumem` driver is working correctly.

    - Change directory into [`tests/test_gpumem/`](tests/test_gpumem).
    - Run `make`.
    - Run `./gpudirect`.
    - Example of a desired result.
    
        ```
        GPU virtual address: 0x7f5ac7a00000
        Allocating 262144 bytes: locked 4 pages.
        page_count 4, page_size 65536
        page 0: physical address 0x387000660000
        page 1: physical address 0x387000670000
        page 2: physical address 0x387000680000
        page 3: physical address 0x387000690000
        Testing page 0: CPU virtual address 0x7f5ae6eb9000 ... done          
        Testing page 1: CPU virtual address 0x7f5ae6ec9000 ... done          
        Testing page 2: CPU virtual address 0x7f5ae6ed9000 ... done          
        Testing page 3: CPU virtual address 0x7f5ae6ee9000 ... done          
        Test succeed.
        ```

    **NOTE:** current `gpumem` driver seems to be unstable as it sometimes stops working during runtime. If an ARK loop gets stuck, we may need to double-check whether it is working using the `tests/test_gpumem/` test code.

## Install ARK and Run Unit Tests

1. Download third-parties & compile ARK and unittest.

    ```make -j```

2. Install ARK.

    ```make install```

    Installation directory is specified as `ARK_ROOT` environment variable,
    which is `${HOME}/.ark` by default when it is unset.

3. (Optional) Run unittest.

    ```./scripts/unittest.sh```

    **NOTE:** currently unittest supports CUDA compute capapbility >= 7.0 only,
    and it requires at least 2 GPUs in the system for communication tests.


## Install Ark Python Bindings  
  
We offer Python bindings for Ark, allowing users to access and utilize Ark in their Python projects. These bindings are created using pybind11 and built on top of the C++ API.
  
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

This will test the Python bindings for Ark. If the tests pass, you have successfully built and tested the Python bindings.  

