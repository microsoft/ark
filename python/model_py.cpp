// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ark.h"

namespace py = pybind11;

void register_model(py::module &m) {
    py::class_<ark::Model>(m, "_Model")
        .def(py::init<int>(), py::arg("rank") = 0)
        .def("tensor", &ark::Model::tensor,
             "construct a tensor with given shape and data type.",
             py::return_value_policy::reference_internal, py::arg("shape"),
             py::arg("ttype"), py::arg("buf") = nullptr,
             py::arg("ldims") = ark::Dims(), py::arg("offs") = ark::Dims(),
             py::arg("pads") = ark::Dims(),
             py::arg("deps") = std::vector<ark::Tensor *>(),
             py::arg("exported") = false, py::arg("imported_rank") = -1,
             py::arg("name") = "tensor")
        .def("reshape",
             (ark::Tensor *
              (ark::Model::*)(ark::Tensor *, const std::vector<ark::DimType> &,
                              bool, ark::Tensor *, const std::string &)) &
                 ark::Model::reshape,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("shape"), py::arg("allowzero") = false,
             py::arg("output") = nullptr, py::arg("name") = "reshape")
        .def("identity", &ark::Model::identity,
             "Returns an identical tensor of `input` with execution "
             "dependencies `deps`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("deps") = std::vector<ark::Tensor *>(),
             py::arg("name") = "identity")
        .def("sharding", &ark::Model::sharding,
             "Shard `input` along `axis` into `dim_per_shard`-dimensional "
             "shards.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("dim_per_shard"),
             py::arg("name") = "sharding")
        .def("reduce_sum", &ark::Model::reduce_sum,
             "Performs reduction along the `axis` of the `input` tensor and "
             "stores the result in `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("keepdims"), py::arg("output") = nullptr,
             py::arg("name") = "reduce_sum")
        .def("reduce_mean", &ark::Model::reduce_mean,
             "Performs reduction along the `axis` of the `input` tensor and "
             "stores the result in `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("keepdims"), py::arg("output") = nullptr,
             py::arg("name") = "reduce_mean")
        .def("reduce_max", &ark::Model::reduce_max,
             "Performs reduction along the `axis` of the `input` tensor and "
             "stores the result in `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("keepdims"), py::arg("output") = nullptr,
             py::arg("name") = "reduce_max")
        .def("layernorm", &ark::Model::layernorm,
             "Applies layer normalization to the `input` tensor and returns "
             "the normalized tensor as `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "layernorm")
        .def("rmsnorm", &ark::Model::rmsnorm,
             "Applies RMS (Root Mean Square Layer Normalization) normalization "
             "to the `input` tensor and returns "
             "the normalized tensor as `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "rmsnorm")
        .def("softmax", &ark::Model::softmax,
             "Applies softmax activation to the `input` tensor, with the "
             "softmax operator being performed on the last dimension of the "
             "input tensor.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "softmax")
        .def("transpose", &ark::Model::transpose,
             "Transposes the `input` tensor according to the given `perm` "
             "permutation. For example, transpose(input, {0, 1 ,3, 2}) will "
             "swap the last two dimensions of the input tensor. Currently, "
             "only 4D tensors are supported.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("perm"), py::arg("output") = nullptr,
             py::arg("name") = "transpose")
        .def("matmul", &ark::Model::matmul,
             "Performs matrix multiplication between the `input` tensor and "
             "`other` tensor, storing the result in `output`. Optional "
             "parameters allow controlling the behavior of the multiplication, "
             "such as transposing the input tensors and applying a ReLU "
             "activation.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("splitk") = 1, py::arg("trans_input") = false,
             py::arg("trans_other") = false, py::arg("name") = "matmul",
             py::arg("gran_lev") = -1)
        .def("im2col", &ark::Model::im2col,
             "Implements the 'im2col' method for 2D convolution layers, which "
             "takes an `input` tensor and reshapes it to a 2D matrix by "
             "extracting image patches from the input tensor based on the "
             "provided parameters.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("kernel_height"), py::arg("kernel_width"),
             py::arg("stride_height"), py::arg("stride_width"),
             py::arg("pad_height"), py::arg("pad_width"),
             py::arg("dilation_height"), py::arg("dilation_width"),
             py::arg("output") = nullptr, py::arg("name") = "im2col")
        .def("max_pool", &ark::Model::max_pool,
             "Applies max-pooling on the `input` tensor using `kernel_size` "
             "and `stride`, reducing its spatial size. The output shape is "
             "calculated based on the input tensor's shape and the stride "
             "value as follows: {is[0], (is[1] + stride - 1) / stride, (is[2] "
             "+ stride - 1) / stride, is[3]}, where 'is' represents the input "
             "tensor's shape.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("kernel_size"), py::arg("stride"),
             py::arg("output") = nullptr, py::arg("name") = "max_pool")
        .def("scale", &ark::Model::scale,
             "Multiplies the `input` tensor by a scalar `val`, element-wise.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("val"), py::arg("output") = nullptr,
             py::arg("name") = "scale")
        .def("exp", &ark::Model::exp,
             "Calculates the exponential of the `input` tensor, element-wise.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "exp")
        .def("sqrt", &ark::Model::sqrt,
             "Calculates the square root of the `input` tensor, element-wise.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "sqrt")
        .def("rope", &ark::Model::rope,
             "Performs rotary position embedding (RoPE) on the `input` "
             "tensor",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "rope")
        .def("relu", &ark::Model::relu, "ReLU activation",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "relu")
        .def("gelu", &ark::Model::gelu,
             "Applies the Gaussian Error Linear Unit (GELU) activation "
             "function to the `input` tensor, element-wise. GELU is a smooth "
             "approximation of the rectifier function and is widely used in "
             "deep learning models.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "gelu")
        .def("sigmoid", &ark::Model::sigmoid, "Sigmoid activation",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "sigmoid")
        .def("add", &ark::Model::add,
             "Performs an element-wise addition operator between the `input` "
             "tensor and the `other` tensor",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "add")
        .def("sub", &ark::Model::sub,
             "Performs an element-wise addition operator between the "
             "`input` "
             "tensor and the `other` tensor",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "sub")
        .def("mul", &ark::Model::mul,
             "Performs an element-wise multiplication operator between the "
             "`input` tensor and the `other` tensor,",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "mul")
        .def("div", &ark::Model::div,
             "Performs an element-wise division operator between the "
             "`input` tensor and the `other` tensor,",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "div")
        .def("send", &ark::Model::send,
             "Sends a tensor to a destination GPU (`dst_rank`). Multiple "
             "tensors can be sent to the same GPU,so an identifier `id` is "
             "required to distinguish the tensor. Each 'send' operator must "
             "have a corresponding 'recv' operator that have the same id in "
             "another GPU's model.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("dst_rank"), py::arg("bytes") = 0,
             py::arg("name") = "send")
        .def("send_done", &ark::Model::send_done,
             "Blocks the execution until the corresponding 'send' operator "
             "with the specified `id` is completed.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("dst_rank"), py::arg("name") = "send_done")
        .def("recv", &ark::Model::recv,
             "Receives a tensor from a source GPU (`src_rank`), identified "
             "by "
             "the `id` parameter. Blocks the execution until the "
             "corresponding "
             "'recv' operator is completed.",
             py::return_value_policy::reference_internal, py::arg("id"),
             py::arg("src_rank"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "recv")
        .def("send_mm", &ark::Model::send_mm,
             "Similar to the 'send_done' function, but implemented using "
             "CUDA "
             "in-stream RDMA copy and Low Latency (LL) protocol.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("gpu_dst"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "send_mm")
        .def("recv_mm", &ark::Model::recv_mm,
             "Similar to the 'recv' function, but implemented using CUDA "
             "in-stream RDMA copy and Low Latency (LL) protocol.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("gpu_src"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "recv_mm")
        .def("send_mscclpp", &ark::Model::send_mscclpp,
             "Sends a tensor to a destination GPU (`dst_rank`). Multiple "
             "tensors can be sent to the same GPU,so an identifier `id` is "
             "required to distinguish the tensor. Each 'send' operator must "
             "have a corresponding 'recv' operator that have the same id in "
             "another GPU's model.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("sid"), py::arg("dst_rank"), py::arg("bytes") = 0,
             py::arg("name") = "send_mscclpp")
        .def("send_done_mscclpp", &ark::Model::send_done_mscclpp, "",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("dst_rank"), py::arg("name") = "send_done_mscclpp")
        .def("recv_mscclpp", &ark::Model::recv_mscclpp,
             "Receives a tensor from a source GPU (`src_rank`), identified by "
             "the `id` parameter. Blocks the execution until the corresponding "
             "'recv' operator is completed.",
             py::return_value_policy::reference_internal, py::arg("sid"),
             py::arg("src_rank"), py::arg("bytes"), py::arg("output") = nullptr,
             py::arg("name") = "recv_mscclpp")
        .def("all_gather", &ark::Model::all_gather,
             "Performs an all-gather operator across all GPUs",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"),
             py::arg("output") = std::vector<ark::Tensor *>(),
             py::arg("name") = "all_gather")
        .def("local_all_gather_mscclpp", &ark::Model::local_all_gather_mscclpp,
             "Performs an all-gather operator across all GPUs",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("ngpus_per_node"), py::arg("axis") = 0,
             py::arg("name") = "local_all_gather_mscclpp")
        .def("all_reduce", &ark::Model::all_reduce,
             "Performs an all-reduce operator across all GPUs, aggregating "
             "the input tensors. Takes the `input` tensor, the current "
             "GPU's "
             "`gpu_id`, and the total number of GPUs `gpu_num`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"), py::arg("output") = nullptr,
             py::arg("name") = "all_reduce")
        .def("local_all_reduce_mscclpp", &ark::Model::local_all_reduce_mscclpp,
             "Performs an all-reduce operator across all GPUs, aggregating "
             "the input tensors. Takes the `input` tensor, the current "
             "GPU's "
             "`gpu_id`, and the total number of GPUs `gpu_num`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"),
             py::arg("name") = "local_all_reduce_mscclpp")
        .def("local_reduce_scatter_mscclpp",
             &ark::Model::local_reduce_scatter_mscclpp,
             "Performs a reduce-scatter operator across all GPUs in a node, "
             "aggregating "
             "the input tensors. Takes the `input` tensor, the current "
             "GPU's "
             "`gpu_id`, and the total number of GPUs `gpu_num`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"),
             py::arg("name") = "local_all_reduce_mscclpp")
        .def("local_all_reduce_packet_mscclpp",
             &ark::Model::local_all_reduce_packet_mscclpp,
             "Performs an all-reduce operator across all GPUs, aggregating "
             "the input tensors. Takes the `input` tensor, the current "
             "GPU's "
             "`gpu_id`, and the total number of GPUs `gpu_num`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"),
             py::arg("name") = "local_all_reduce_packet_mscclpp")
        .def("embedding", &ark::Model::embedding, "Embedding layer.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("weight"), py::arg("output") = nullptr,
             py::arg("name") = "embedding")
        .def("cast", &ark::Model::cast, "Tensor type casting.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("ttype"), py::arg("output") = nullptr,
             py::arg("name") = "cast");
}
