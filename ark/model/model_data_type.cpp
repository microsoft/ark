// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_data_type.hpp"

#include <map>

#include "bfloat16.h"
#include "half.h"
#include "logging.h"

namespace ark {

///
/// NOTE: how to add a new data type
///   1. Add an instance using `MODEL_DATA_TYPE_INSTANCE()` macro.
///   2. Add a registration using `MODEL_DATA_TYPE_REGISTER()` macro.
///   3. Expose the symbol in `include/ark/model.hpp`.
///

#define MODEL_DATA_TYPE_INSTANCE(_name, _type) \
    extern const ModelDataType _name =         \
        std::make_shared<ModelDataT>(#_name, #_type, sizeof(_type));

#define MODEL_DATA_TYPE_REGISTER(_name) instances[#_name] = _name;

extern const ModelDataType NONE =
    std::make_shared<ModelDataT>("NONE", "void", 0);
MODEL_DATA_TYPE_INSTANCE(FP32, float);
MODEL_DATA_TYPE_INSTANCE(FP16, fp16);
MODEL_DATA_TYPE_INSTANCE(BF16, bf16);
MODEL_DATA_TYPE_INSTANCE(INT32, int32_t);
MODEL_DATA_TYPE_INSTANCE(UINT32, uint32_t);
MODEL_DATA_TYPE_INSTANCE(INT8, int8_t);
MODEL_DATA_TYPE_INSTANCE(UINT8, uint8_t);
MODEL_DATA_TYPE_INSTANCE(BYTE, char);

const ModelDataType ModelDataT::from_name(const std::string &type_name) {
    static std::map<std::string, ModelDataType> instances;
    if (instances.empty()) {
        MODEL_DATA_TYPE_REGISTER(NONE);
        MODEL_DATA_TYPE_REGISTER(FP32);
        MODEL_DATA_TYPE_REGISTER(FP16);
        MODEL_DATA_TYPE_REGISTER(BF16);
        MODEL_DATA_TYPE_REGISTER(INT32);
        MODEL_DATA_TYPE_REGISTER(UINT32);
        MODEL_DATA_TYPE_REGISTER(INT8);
        MODEL_DATA_TYPE_REGISTER(UINT8);
        MODEL_DATA_TYPE_REGISTER(BYTE);
    }
    auto it = instances.find(type_name);
    if (it == instances.end()) {
        ERR(InvalidUsageError, "Unknown data type: ", type_name);
    }
    return it->second;
}

}  // namespace ark
