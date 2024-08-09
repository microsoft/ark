# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class dtype: ...


class float32: ...


class float16: ...


class bfloat16: ...


class int32: ...


class int8: ...


class uint8: ...


class ubyte: ...


class Tensor: ...



class nn:


    class Module: ...
    

    class Parameter: ... 


class autograd:


    class Function: 


        def apply(self, *args, **kwargs): ...

