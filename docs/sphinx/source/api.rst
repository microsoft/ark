******************
API Document
******************

``ark``
=====================

.. automodule:: ark
    :members: init, srand, rand, NO_DIM, DIMS_LEN


``ark.Dims``
=====================

.. automodule:: ark.Dims
    :members: size, ndims, __getitem__, __setitem__, __repr__

``ark.Model``
=====================

.. automodule:: ark.Model
    :members: tensor, reshape, identity, sharding, reduce, layernorm, softmax,
            transpose, linear, im2col, conv2d, max_pool, scale, relu, gelu, add, mul, send,
            send_done, recv, send_mm, recv_mm, all_reduce

``ark.Executor``
=====================

.. automodule:: ark.Executor
    :members: compile, launch, run, wait, stop

