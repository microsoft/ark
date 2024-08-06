// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_reduce.hpp"

#include "logging.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpReduce::ModelOpReduce(const std::string &type_name, ModelTensorRef input,
                             int axis, bool keepdims, ModelTensorRef output)
    : ModelOp(type_name) {
    check_null(input);
    Dims reduced_shape{input->shape()};
    if (axis < 0) {
        axis += reduced_shape.ndims();
    }
    if (axis < 0 || axis >= reduced_shape.ndims()) {
        ERR(ModelError, "invalid reduction axis ", axis);
    }
    if (keepdims) {
        reduced_shape[axis] = 1;
    } else {
        reduced_shape.erase(axis);
    }
    if (output) {
        check_match_data_type(input, output);
        if (output->shape() != reduced_shape) {
            ERR(ModelError, "invalid output shape ", output->shape(),
                " with input shape ", input->shape(), ", reduction axis ", axis,
                ", and keepdims ", keepdims);
        }
        if (output == input) {
            ERR(ModelError,
                "output tensor cannot be the same as input tensor for "
                "reduce_sum op");
        }
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), reduced_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_ = {{"Axis", axis}, {"KeepDim", keepdims}};
    verify();
}

std::string ModelOpReduce::impl_name(const Json &config) const {
    check_fields_config(config, {"NumWarps", "SramBytes", "ImplType"});
    check_fields_args(args_, {"Axis", "KeepDim"});

    std::string red_type;
    if (type()->type_name() == "ReduceSum") {
        red_type = "sum";
    } else if (type()->type_name() == "ReduceMax") {
        red_type = "max";
    } else if (type()->type_name() == "ReduceMean") {
        red_type = "mean";
    } else {
        ERR(PlanError, "unsupported reduce type: ", type()->type_name());
    }

    int num_warps = config.at("NumWarps");
    int sram_bytes = config.at("SramBytes");
    std::string impl_type = config.at("ImplType");
    int axis = args_.at("Axis").value<int>();
    bool keep_dims = args_.at("KeepDim").value<bool>();

    // Translate the axis value into 4D representation.
    axis += 4 - read_tensors_[0]->shape().ndims();

    if (impl_type == "WarpWise") {
        impl_type = "w";
        if (axis != 3) {
            ERR(PlanError,
                "warp-wise reduction is supported only for "
                "the last axis");
        }
    } else if (impl_type == "ElementWise") {
        impl_type = "e";
    } else {
        ERR(PlanError, "unsupported implementation type: ", impl_type);
    }

    Dims output_strides = write_tensors_[0]->strides();
    Dims output_shape = write_tensors_[0]->shape();
    if (!keep_dims) {
        output_strides.insert(axis, 1);
        output_shape.insert(axis, 1);
    }

    return function_name_string(
        "reduce_" + impl_type + "_" + red_type,
        {
            vec_string(read_tensors_[0]->strides().dims4()),
            vec_string(read_tensors_[0]->shape().dims4()),
            vec_string(output_strides.dims4()),
            vec_string(output_shape.dims4()),
            vec_string(Dims(1, 1, 1, 1)),
            std::to_string(num_warps),
            std::to_string(sram_bytes),
            std::to_string(axis),
        });
}

std::vector<ModelOpArg> ModelOpReduce::impl_args([
    [maybe_unused]] const Json &config) const {
    return {result_tensors_[0], read_tensors_[0]};
}

Json ModelOpReduce::default_config([[maybe_unused]] const ArchRef arch) const {
    Json config;
    config["NumWarps"] = 1;
    int axis = args_.at("Axis").value<int>();
    if (axis == read_tensors_[0]->shape().ndims() - 1 || axis == -1) {
        config["ImplType"] = "WarpWise";
        config["SramBytes"] = 256;
    } else {
        config["ImplType"] = "ElementWise";
        config["SramBytes"] = 0;
    }
    config["NumTasks"] = result_tensors_[0]->shape().nelems();
    return config;
}

Tensor Model::reduce_max(Tensor input, int axis, bool keepdims, Tensor output,
                         const std::string &config, const std::string &name) {
    return impl_
        ->create_op<ModelOpReduceMax>(config, name, input.ref_, axis, keepdims,
                                      output.ref_)
        ->result_tensors()[0];
}

Tensor Model::reduce_mean(Tensor input, int axis, bool keepdims, Tensor output,
                          const std::string &config, const std::string &name) {
    return impl_
        ->create_op<ModelOpReduceMean>(config, name, input.ref_, axis, keepdims,
                                       output.ref_)
        ->result_tensors()[0];
}

Tensor Model::reduce_sum(Tensor input, int axis, bool keepdims, Tensor output,
                         const std::string &config, const std::string &name) {
    return impl_
        ->create_op<ModelOpReduceSum>(config, name, input.ref_, axis, keepdims,
                                      output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
