# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Executor

class Executor(_Executor):
    def tensor_memcpy_host_to_device(self, dst, src):
        print("new tensor memcpy")
        super().tensor_memcpy_host_to_device(dst, src)
    def tensor_memcpy_device_to_host(self, dst, src):
        super().tensor_memcpy_device_to_host(dst, src)

