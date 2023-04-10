// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_TENSOR_H_
#define ARK_TENSOR_H_

#include "ark/dims.h"
#include "third_party/json/json.h"

namespace ark {

// TensorBuf refers to a data array that can be shared by multiple tensors.
struct TensorBuf
{
    TensorBuf(const DimType &bytes = 0, int id = -1);
    TensorBuf(const TensorBuf &) = default;

    DimType bytes;
    int id;
    bool immutable = false;
};

void to_json(nlohmann::json &j, const TensorBuf &tbuf);
void from_json(const nlohmann::json &j, TensorBuf &tbuf);

// Type of tensor data.
typedef enum
{
    FP16,
    FP32,
    INT32,
} TensorType;

// clang-format off
NLOHMANN_JSON_SERIALIZE_ENUM(TensorType, {
    {FP16, "f16"},
    {FP32, "f32"},
    {INT32, "i32"},
})
// clang-format on

// Tensor is a view of a TensorBuf.
//
// Illustration of a single axis of a tensor:
//
// 0           off                                                        ldim
// |------------|-------------shape-------------|---------------------------|
//               <----------------------------->
//                  data range of this tensor
//
struct Tensor
{
    Tensor(const Dims &shape, TensorType type, TensorBuf *buf,
           const Dims &ldims, const Dims &offs, const Dims &pads, bool exported,
           bool imported, int id, const std::string &name);
    Tensor(const Tensor &) = default;

    void update_pads(const std::vector<DimType> &pads);
    DimType offset(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                   DimType i3 = 0) const;
    DimType size() const;
    int ndims() const;
    Dims padded_shape() const;
    unsigned int type_bytes() const;
    DimType shape_bytes() const;
    DimType ldims_bytes() const;
    DimType offset_bytes(DimType i0 = 0, DimType i1 = 0, DimType i2 = 0,
                         DimType i3 = 0) const;

    bool is_sequential() const;

    // TensorBuf that this tensor is associated with
    TensorBuf *buf;
    // Data type of each element in the tensor
    TensorType type;
    // Shape of the tensor
    Dims shape;
    // Leading dimensions of the underlying data array
    Dims ldims;
    // Offset of the tensor in the underlying data array
    Dims offs;
    // Unit dimensions of the underlying data array. ldims[x] should be always
    // divided by udims[x].
    Dims pads;
    // Whether this tensor is accessed by remote devices
    bool exported;
    // if imported is true, the tensor is imported from another GPU and don't
    // need to allocate a TensorBuf for it.
    bool imported;
    // Unique id of this tensor
    int id;
    // Name of this tensor
    const std::string name;
};

void to_json(nlohmann::json &j, const Tensor &tns);
void from_json(const nlohmann::json &j, Tensor &tns);

} // namespace ark

#endif // ARK_TENSOR_H_
