// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark.h"
#include <iostream>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

// TODO: add more checks for the host_buffer, currently users must make sure
// that the host_tensor's memory layout is contiguous
void tensor_memcpy_host_to_device(ark::Executor *executor, ark::Tensor *tns,
                                  py::buffer host_buffer)
{
    py::buffer_info info = host_buffer.request();
    size_t bytes = tns->shape_bytes();
    void *host_buffer_ptr = info.ptr;
    executor->tensor_memcpy(tns, (const void *)host_buffer_ptr, bytes);
}

void tensor_memcpy_device_to_host(ark::Executor *executor,
                                  py::buffer host_buffer, ark::Tensor *tns)
{
    py::buffer_info info = host_buffer.request();
    size_t bytes = tns->shape_bytes();
    void *host_buffer_ptr = info.ptr;
    executor->tensor_memcpy((void *)host_buffer_ptr, tns, bytes);
}

PYBIND11_MODULE(_ark_core, m)
{
    m.doc() = "ARK python module interface"; // optional module docstring
    m.def("_init", &ark::init,
          "Init an ark program. Call this function to clean up the shared "
          "memory directory. This is useful when the previous run crashed, as "
          "this forces to remove locks generated by previous runs. This may "
          "crash other ARK processes running on the same machine, if there are "
          "any.");

    m.def("srand", &ark::srand, py::arg("seed") = -1,
          "Sets the seed for the random number generator");
    m.def("rand", &ark::rand, "Generates a random integer");
    m.attr("DIMS_LEN") = py::int_(static_cast<int>(ark::DIMS_LEN));
    m.attr("NO_DIM") = py::int_(static_cast<int>(ark::NO_DIM));

    py::class_<ark::Dims>(m, "Dims", "Up-to-`DIMS_LEN`-dimensional vector.")
        .def(py::init([](ark::DimType d0, ark::DimType d1, ark::DimType d2,
                         ark::DimType d3) {
                 return std::make_unique<ark::Dims>(d0, d1, d2, d3);
             }),
             py::arg_v("d0", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"),
             py::arg_v("d1", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"),
             py::arg_v("d2", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"),
             py::arg_v("d3", static_cast<int>(ark::NO_DIM),
                       "default value: NO_DIM"))
        .def(py::init<const ark::Dims &>(), "Copy another Dims object.")
        .def(py::init<const std::vector<ark::DimType> &>(),
             "Construct from a vector. If the vector is shorter than DIMS_LEN, "
             "put following NO_DIMs. Raise an error if the vector is longer "
             "than DIMS_LEN.")
        .def("size", &ark::Dims::size,
             "Return the volume of dimensions. If the dimensions are invalid, "
             "return -1")
        .def("ndims", &ark::Dims::ndims,
             "Return the number of valid dimensions.")
        .def("__getitem__",
             [](const ark::Dims &d, ark::DimType idx) { return d[idx]; })
        .def("__setitem__", [](ark::Dims &d, ark::DimType idx,
                               ark::DimType value) { d[idx] = value; })
        .def("__repr__", [](const ark::Dims &d) {
            std::ostringstream os;
            os << d;
            return os.str();
        });

    py::enum_<ark::TensorType>(m, "TensorType",
                               "Type of tensor data, FP16, FP32, or INT32")
        .value("FP16", ark::TensorType::FP16)
        .value("FP32", ark::TensorType::FP32)
        .value("INT32", ark::TensorType::INT32)
        .export_values();

    py::class_<ark::TensorBuf>(m, "TensorBuf",
                               "TensorBuf refers to a data array that can be "
                               "shared by multiple tensors.")
        .def(py::init<const ark::DimType &, int>(), py::arg("bytes") = 0,
             py::arg("id") = -1)
        .def_readwrite("bytes", &ark::TensorBuf::bytes)
        .def_readwrite("id", &ark::TensorBuf::id)
        .def_readwrite("immutable", &ark::TensorBuf::immutable);

    py::class_<ark::Tensor>(m, "_Tensor")
        .def(py::init<const ark::Dims &, ark::TensorType, ark::TensorBuf *,
                      const ark::Dims &, const ark::Dims &, const ark::Dims &,
                      bool, bool, int, const std::string &>(),
             py::arg("shape"), py::arg("type"), py::arg("buf"),
             py::arg("ldims"), py::arg("offs"), py::arg("pads"),
             py::arg("exported"), py::arg("imported"), py::arg("id"),
             py::arg("name"))
        .def_property_readonly("shape",
                               [](const ark::Tensor &t) {
                                   py::list shape_list;
                                   for (int i = 0; i < t.ndims(); ++i) {
                                       shape_list.append((int)t.shape[i]);
                                   }
                                   return shape_list;
                               })
        .def("offset", &ark::Tensor::offset, py::arg("i0") = 0,
             py::arg("i1") = 0, py::arg("i2") = 0, py::arg("i3") = 0)
        .def("size", &ark::Tensor::size,
             "Number of elements in the tensor excluding padding.")
        .def("ndims", &ark::Tensor::ndims,
             "Number of dimensions in the tensor.")
        .def("padded_shape", &ark::Tensor::padded_shape,
             "Shape of the tensor including padding.")
        .def("type_bytes", &ark::Tensor::type_bytes,
             "Number of bytes of each element in the tensor.")
        .def("shape_bytes", &ark::Tensor::shape_bytes,
             "Number of bytes of the tensor.")
        .def("ldims_bytes", &ark::Tensor::ldims_bytes,
             "Should be the same as the number of bytes of the TensorBuf.")
        .def("offset_bytes", &ark::Tensor::offset_bytes, py::arg("i0") = 0,
             py::arg("i1") = 0, py::arg("i2") = 0, py::arg("i3") = 0)
        .def("tensor_type", &ark::Tensor::tensor_type,
             "The type of the tensor.");

    py::class_<ark::Model>(m, "_Model")
        .def(py::init<int>(), py::arg("rank") = 0)
        .def("tensor", &ark::Model::tensor,
             "construct a tensor with given shape and data type.",
             py::return_value_policy::reference_internal, py::arg("shape"),
             py::arg("dtype"), py::arg("buf") = nullptr,
             py::arg("ldims") = ark::Dims(), py::arg("offs") = ark::Dims(),
             py::arg("pads") = ark::Dims(),
             py::arg("deps") = std::vector<ark::Tensor *>(),
             py::arg("exported") = false, py::arg("imported") = false,
             py::arg("name") = "tensor")
        .def("reshape",
             (ark::Tensor * (ark::Model::*)(ark::Tensor *, const ark::Dims &,
                                            bool, ark::Tensor *,
                                            const std::string &)) &
                 ark::Model::reshape,
             "Reshape `input` to `shape`. If one dimension of `shape` is -1, "
             "it will be inferred from the `input`. If one dimension of "
             "`shape` is 0, by default (`allowzero` is false), that dimension "
             "is unchanged from the corresponding one of `input`. If "
             "`allowzero` is true, that dimension is set to 0, which means "
             "that the reshaped tensor is an empty tensor, i.e., `input` "
             "should also be an empty tensor. If `allowzero` is true, `shape` "
             "should not include both 0 and -1 at the same time. If `shape` is "
             "an empty vector, `input` will be converted to a scalar.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("shape"), py::arg("allowzero") = false,
             py::arg("output") = nullptr, py::arg("name") = "reshape")
        .def("reshape",
             (ark::Tensor * (ark::Model::*)(ark::Tensor *,
                                            std::initializer_list<ark::DimType>,
                                            bool, ark::Tensor *,
                                            const std::string &)) &
                 ark::Model::reshape,
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("shape"), py::arg("allowzero") = false,
             py::arg("output") = nullptr, py::arg("name") = "reshape")
        .def("identity", &ark::Model::identity,
             "Returns an identical tensor of `input` with execution "
             "dependencies `deps`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("deps") = std::vector<ark::Tensor *>(),
             py::arg("output") = nullptr, py::arg("name") = "identity")
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
             py::arg("axis"), py::arg("output") = nullptr,
             py::arg("name") = "reduce_sum")
        .def("reduce_mean", &ark::Model::reduce_mean,
             "Performs reduction along the `axis` of the `input` tensor and "
             "stores the result in `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("output") = nullptr,
             py::arg("name") = "reduce_mean")
        .def("reduce_max", &ark::Model::reduce_max,
             "Performs reduction along the `axis` of the `input` tensor and "
             "stores the result in `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("axis"), py::arg("output") = nullptr,
             py::arg("name") = "reduce_max")
        .def("layernorm", &ark::Model::layernorm,
             "Applies layer normalization to the `input` tensor and returns "
             "the normalized tensor as `output`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("output") = nullptr, py::arg("name") = "layernorm")
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
             py::arg("trans_other") = false, py::arg("is_relu") = false,
             py::arg("name") = "matmul", py::arg("gran_lev") = -1)
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
        .def("conv2d", &ark::Model::conv2d,
             "Implements a 2D convolution layer using the 'im2col' method.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride"), py::arg("padding"),
             py::arg("bias") = false, py::arg("output") = nullptr,
             py::arg("name") = "conv2d")
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
        .def("add", &ark::Model::add,
             "Performs an element-wise addition operator between the `input` "
             "tensor and the `other` tensor",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "add")
        .def("mul", &ark::Model::mul,
             "Performs an element-wise multiplication operator between the "
             "`input` tensor and the `other` tensor,",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("other"), py::arg("output") = nullptr,
             py::arg("name") = "mul")
        .def("send", &ark::Model::send,
             "Sends a tensor to a destination GPU (`dst_rank`). Multiple "
             "tensors can be sent to the same GPU,so an identifier `id` is "
             "required to distinguish the tensor. Each 'send' operator must "
             "have a corresponding 'recv' operator that have the same id in "
             "another GPU's model.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("dst_rank"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "send")
        .def("send_done", &ark::Model::send_done,
             "Blocks the execution until the corresponding 'send' operator "
             "with the specified `id` is completed.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("dst_rank"), py::arg("output") = nullptr,
             py::arg("name") = "send_done")
        .def("recv", &ark::Model::recv,
             "Receives a tensor from a source GPU (`src_rank`), identified by "
             "the `id` parameter. Blocks the execution until the corresponding "
             "'recv' operator is completed.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("id"), py::arg("src_rank"), py::arg("bytes") = 0,
             py::arg("output") = nullptr, py::arg("name") = "recv")
        .def("send_mm", &ark::Model::send_mm,
             "Similar to the 'send_done' function, but implemented using CUDA "
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
        .def("all_gather", &ark::Model::all_gather,
             "Performs an all-gather operator across all GPUs",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"),
             py::arg("output") = std::vector<ark::Tensor *>(),
             py::arg("name") = "all_gather")
        .def("all_reduce", &ark::Model::all_reduce,
             "Performs an all-reduce operator across all GPUs, aggregating "
             "the input tensors. Takes the `input` tensor, the current GPU's "
             "`gpu_id`, and the total number of GPUs `gpu_num`.",
             py::return_value_policy::reference_internal, py::arg("input"),
             py::arg("gpu_id"), py::arg("gpu_num"), py::arg("output") = nullptr,
             py::arg("name") = "all_reduce");
    // register class Executor
    py::class_<ark::Executor>(m, "_Executor",
                              "Convenience class for executing a model.")
        .def(py::init<const int, int, int, ark::Model &, const std::string &,
                      int>(),
             py::arg("gpu_id"), py::arg("rank"), py::arg("world_size"),
             py::arg("model"), py::arg("name"),
             py::arg("num_warps_per_sm") = 16)
        .def("compile", &ark::Executor::compile,
             "Compile the model. This must be called before `launch()`.")
        .def("launch", &ark::Executor::launch,
             "Launch the model (not running yet). This must be called after "
             "`compile()`.")
        .def("run", &ark::Executor::run, py::arg("iter"),
             "Run the model for `iter` iterations.")
        .def("wait", &ark::Executor::wait,
             "Wait for the previous run to finish.")
        .def("stop", &ark::Executor::stop,
             "Stop the model and return the elapsed time in milliseconds. Once "
             "this is called, we need to call `launch()` again to run the "
             "model again.")
        .def("tensor_memcpy_host_to_device", &tensor_memcpy_host_to_device,
             "Copy contiguous data from a host buffer to the given tensor's "
             "(possibly non-contiguous) data range on GPU.",
             py::arg("tns"), py::arg("src"))
        .def("tensor_memcpy_device_to_host", &tensor_memcpy_device_to_host,
             "Copy (possibly non-contiguous) data from a tensor on GPU to a "
             "contiguous host buffer. The given number of bytes is copied, in "
             "order of appearance on the memory. This function assumes that "
             "`dst` is large enough to hold the data.",
             py::arg("dst"), py::arg("tns"))
        .def("tensor_clear", &ark::Executor::tensor_clear,
             "Set all bytes of `tns` into zero.", py::arg("tns"));
}
