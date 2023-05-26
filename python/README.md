# Ark Python Bindings  
  
This README provides instructions on how to build and test the Python bindings for Ark.  
  
## Building the Python Bindings  
  
1. Run the following command to build the Python bindings:  
  
python3 setup.py build_ext

After running this script, the `ark.cpython-38-x86_64-linux-gnu.so` will be generated in the `ark/python/build/lib.linux-x86_64-3.8` directory.  
  
2. Add the generated library to your PYTHONPATH:  

export PYTHONPATH="$ARK_DIR/ark/python/build/lib.linux-x86_64-3.8:${PYTHONPATH}"

  
## Testing the Python Bindings  
  
1. Change to the `ark/python` directory:  

cd ark/python

2. Run the Python test script:  
python3 python_test.py

This will test the Python bindings for Ark. If the tests pass, you have successfully built and tested the Python bindings.  
