// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_DATA_TYPE_HPP
#define ARK_DATA_TYPE_HPP

#include <memory>
#include <string>

namespace ark {

class DataType;

extern const DataType NONE;
extern const DataType FP32;
extern const DataType FP16;
extern const DataType BF16;
extern const DataType INT32;
extern const DataType UINT32;
extern const DataType INT8;
extern const DataType UINT8;
extern const DataType BYTE;

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

class DataType {
   protected:
    friend class Model;
    ModelDataType ref_;

   public:
    DataType() = default;
    DataType(ModelDataType ref) : ref_(ref) {}
    DataType(const DataType &other) = default;
    DataType &operator=(const DataType &other) = default;

    bool operator==(const DataType &other) const { return ref_ == other.ref_; }
    bool operator!=(const DataType &other) const { return ref_ != other.ref_; }

    bool is_none() const { return !ref_; }

    ModelDataType ref() const { return ref_; }

    size_t bytes() const;

    const std::string &name() const;

    static const DataType &from_name(const std::string &type_name);
};

}  // namespace ark

#endif  // ARK_DATA_TYPE_HPP
