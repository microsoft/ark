// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_cache.hpp"

#include "ops_common.hpp"

namespace ark {
namespace {

void check_contiguous_fixed_layout(ModelTensorRef tensor,
                                   const std::string &name) {
    if (tensor->shape() != tensor->strides() ||
        tensor->shape() != tensor->padded_shape() ||
        !tensor->offsets().is_zeros()) {
        ERR(ModelError, name,
            " must use a fixed contiguous layout with zero offsets");
    }
}

Dims slot_shape_from_cache(ModelTensorRef cache) {
    Dims slot_shape = cache->shape();
    if (slot_shape.ndims() < 2) {
        ERR(ModelError,
            "KV-cache shape must be [max_seq, ...slot_shape]. Given: ",
            cache->shape());
    }
    slot_shape.erase(0);
    return slot_shape;
}

DimType max_seq_from_cache(ModelTensorRef cache) { return cache->shape()[0]; }

}  // namespace

ModelOpKvCacheSlot::ModelOpKvCacheSlot(ModelTensorRef cache,
                                       ModelTensorRef token,
                                       ModelTensorRef position,
                                       ModelTensorRef output)
    : ModelOp("KvCacheSlot") {
    check_null(cache);
    check_null(token);
    check_null(position);
    check_match_data_type(cache, token);

    if (!cache->is_external()) {
        ERR(ModelError, "KV-cache tensor must be an external placeholder");
    }
    if (!position->is_external()) {
        ERR(ModelError, "KV-cache position must be an external placeholder");
    }
    check_contiguous_fixed_layout(cache, "KV-cache tensor");
    check_contiguous_fixed_layout(token, "KV-cache token tensor");
    check_contiguous_fixed_layout(position, "KV-cache position tensor");

    Dims slot_shape = slot_shape_from_cache(cache);
    check_match_shape(token, slot_shape);
    check_match_shape(position, {1});
    if (position->data_type()->type_name() != "INT32") {
        ERR(ModelError, "KV-cache position tensor must have INT32 dtype");
    }

    if (output) {
        check_match_data_type(token, output);
        check_match_shape(output, slot_shape);
        check_contiguous_fixed_layout(output, "KV-cache output tensor");
    } else {
        output = std::make_shared<ModelTensor>(
            token->data_type(), std::make_shared<ModelBuffer>(), slot_shape);
    }

    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {token, position};
    write_tensors_ = {cache, position, output};
    result_tensors_ = {result};

    verify();
}

std::string ModelOpKvCacheSlot::impl_name(const Json &config) const {
    check_fields_config(config, {"NumWarps", "Tile"});
    if (config.contains("NumTasks") && config.at("NumTasks") != 1) {
        ERR(PlanError, "KvCacheSlot requires NumTasks=1");
    }
    int num_warps = config.at("NumWarps");
    Dims unit_slot_dims(config.at("Tile").get<std::vector<DimType>>());

    const auto slot = result_tensors_[0];
    return function_name_string(
        "kv_cache_slot",
        {std::to_string(max_seq_from_cache(write_tensors_[0])),
         vec_string(slot->strides().dims4()), vec_string(slot->shape().dims4()),
         vec_string(unit_slot_dims.dims4()), std::to_string(num_warps),
         std::to_string(0)});
}

std::vector<ModelOpArg> ModelOpKvCacheSlot::impl_args(
    [[maybe_unused]] const Json &config) const {
    return {result_tensors_[0], write_tensors_[0], read_tensors_[0],
            read_tensors_[1]};
}

Json ModelOpKvCacheSlot::default_config(
    [[maybe_unused]] const ArchRef arch) const {
    Json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    config["Tile"] = result_tensors_[0]->shape().vector();
    config["NumTasks"] = 1;
    return config;
}

Tensor Model::kv_cache_slot(Tensor cache, Tensor token, Tensor position,
                            Tensor output, const std::string &name) {
    return impl_
        ->create_op<ModelOpKvCacheSlot>(name, cache.ref_, token.ref_,
                                        position.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
