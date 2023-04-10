// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_DIMS_H_
#define ARK_DIMS_H_

#include "third_party/json/json.h"
#include <vector>

namespace ark {

typedef long long int DimType;

enum
{
    DIMS_LEN = 4,
    NO_DIM = -1
};

const std::string shape_str(const std::vector<DimType> &shape);

// Up-to-4-dimensional vector.
struct Dims
{
    Dims(DimType d0 = NO_DIM, DimType d1 = NO_DIM, DimType d2 = NO_DIM,
         DimType d3 = NO_DIM);
    Dims(const Dims &dims_);
    Dims(const std::vector<DimType> &vec);

    DimType size() const;
    int ndims() const;
    Dims dims4() const;
    bool is_no_dim() const;
    bool is_invalid() const;

    DimType &operator[](DimType idx)
    {
        return data[idx];
    }
    const DimType &operator[](DimType idx) const
    {
        return data[idx];
    }

    friend bool operator==(const Dims &a, const Dims &b);
    friend bool operator!=(const Dims &a, const Dims &b);

    friend std::ostream &operator<<(std::ostream &os, const Dims &dims);

    DimType data[DIMS_LEN];
};

void to_json(nlohmann::json &j, const Dims &dims);
void from_json(const nlohmann::json &j, Dims &dims);

} // namespace ark

#endif // ARK_DIMS_H_
