// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/data_type.hpp"

#include <map>

#include "bfloat16.h"
#include "float8.h"
#include "half.h"
#include "logging.h"
#include "model/model_data_type.hpp"

namespace ark {

///
/// NOTE: how to add a new data type
///   1. Add an instance using `DATA_TYPE_INSTANCE()` macro.
///   2. Add a registration using `DATA_TYPE_REGISTER()` macro.
///   3. Expose the symbol in `include/ark/data_type.hpp`.
///

#define DATA_TYPE_INSTANCE(_name, _type) \
    extern const DataType _name(         \
        std::make_shared<ModelDataT>(#_name, #_type, sizeof(_type)));

#define DATA_TYPE_REGISTER(_name) instances[#_name] = &_name;

extern const DataType NONE(std::make_shared<ModelDataT>("NONE", "void", 0));
DATA_TYPE_INSTANCE(FP32, float);
DATA_TYPE_INSTANCE(FP16, fp16);
DATA_TYPE_INSTANCE(BF16, bf16);
DATA_TYPE_INSTANCE(FP8_E4M3, fp8_e4m3);
DATA_TYPE_INSTANCE(FP8_E5M2, fp8_e5m2);
DATA_TYPE_INSTANCE(INT32, int32_t);
DATA_TYPE_INSTANCE(UINT32, uint32_t);
DATA_TYPE_INSTANCE(INT8, int8_t);
DATA_TYPE_INSTANCE(UINT8, uint8_t);
DATA_TYPE_INSTANCE(BYTE, char);

const DataType &DataType::from_name(const std::string &type_name) {
    static std::map<std::string, const DataType *> instances;
    if (instances.empty()) {
        DATA_TYPE_REGISTER(NONE);
        DATA_TYPE_REGISTER(FP32);
        DATA_TYPE_REGISTER(FP16);
        DATA_TYPE_REGISTER(BF16);
        //DATA_TYPE_REGISTER(FP8_E4M3);
        //DATA_TYPE_REGISTER(FP8_E5M2);
        DATA_TYPE_REGISTER(INT32);
        DATA_TYPE_REGISTER(UINT32);
        DATA_TYPE_REGISTER(INT8);
        DATA_TYPE_REGISTER(UINT8);
        DATA_TYPE_REGISTER(BYTE);
    }
    auto it = instances.find(type_name);
    if (it == instances.end()) {
        ERR(InvalidUsageError, "Unknown data type: ", type_name);
    }
    return *(it->second);
}

size_t DataType::bytes() const { return ref_->bytes(); }

const std::string &DataType::name() const { return ref_->type_name(); }

}  // namespace ark
