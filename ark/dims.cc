// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "json.h"
#include "logging.h"
#include <vector>

namespace ark {

// Construct with given four dimensions.
Dims::Dims(DimType d0, DimType d1, DimType d2, DimType d3)
{
    this->data[0] = d0;
    this->data[1] = d1;
    this->data[2] = d2;
    this->data[3] = d3;
    if (this->is_invalid()) {
        LOGERR("invalid dims given: <", d0, ", ", d1, ", ", d2, ", ", d3, ">");
    }
}

// Copy another Dims object.
Dims::Dims(const Dims &dims_)
{
    if (dims_.is_invalid()) {
        LOGERR("invalid dims given");
    }
    for (int i = 0; i < DIMS_LEN; ++i) {
        this->data[i] = dims_.data[i];
    }
}

// Construct from a vector. If the vector is shorter than DIMS_LEN, put
// following NO_DIMs. Raise an error if the vector is longer than DIMS_LEN.
Dims::Dims(const std::vector<DimType> &vec)
{
    int ds = (int)vec.size();
    if (ds > DIMS_LEN) {
        LOGERR("only support dims with size <= ", DIMS_LEN, ". Given: ", *this);
    }
    int i = 0;
    for (; i < ds; ++i) {
        const DimType &v = vec[i];
        if (v < 0 && v != NO_DIM) {
            LOGERR("invalid dims given at index ", i, ": ", v);
        }
        this->data[i] = v;
    }
    for (; i < DIMS_LEN; ++i) {
        this->data[i] = NO_DIM;
    }
}

// Return the volume of dimensions. If the dimensions are invalid, return -1.
DimType Dims::size() const
{
    const DimType *v = this->data;
    if (v[0] == NO_DIM) {
        return -1;
    }
    DimType ret = v[0];
    for (int i = 1; i < DIMS_LEN; ++i) {
        if (v[i] == NO_DIM) {
            break;
        } else {
            ret *= v[i];
        }
    }
    return ret;
}

// Return the number of valid dimensions.
int Dims::ndims() const
{
    const DimType *v = this->data;
    int ret = 0;
    for (; ret < DIMS_LEN; ++ret) {
        if (v[ret] == NO_DIM) {
            break;
        }
    }
    return ret;
}

// Return a new Dims object with 4 valid dimensions by prepending 1s.
Dims Dims::dims4() const
{
    const DimType *v = this->data;
    int nd = this->ndims();
    if (nd > DIMS_LEN) {
        LOGERR("only support dims with size <= ", DIMS_LEN, ". Given: ", nd);
    }
    Dims ret;
    for (int i = 0; i < DIMS_LEN - nd; ++i) {
        ret.data[i] = 1;
    }
    for (int i = 0; i < nd; ++i) {
        ret.data[DIMS_LEN - nd + i] = v[i];
    }
    return ret;
}

// Return true if the dimensions are empty.
bool Dims::is_no_dim() const
{
    const DimType *v = this->data;
    for (int i = 0; i < DIMS_LEN; ++i) {
        if (v[i] != NO_DIM) {
            return false;
        }
    }
    return true;
}

// Return true if the dimensions are invalid.
bool Dims::is_invalid() const
{
    // NO_DIM should not appear before a valid dimension.
    bool invalid_seen = false;
    const DimType *v = this->data;
    for (int i = 0; i < DIMS_LEN; ++i) {
        if (invalid_seen) {
            if (v[i] != NO_DIM) {
                return true;
            }
        } else {
            if (v[i] == NO_DIM) {
                invalid_seen = true;
            } else if (v[i] < 0) {
                return true;
            }
        }
    }
    return false;
}

DimType Dims::erase(int idx)
{
    int nd = this->ndims();
    if (idx >= nd || -idx > nd) {
        LOGERR("invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd;
    }
    DimType ret = this->data[idx];
    for (int i = idx; i < nd - 1; ++i) {
        this->data[i] = this->data[i + 1];
    }
    this->data[nd - 1] = NO_DIM;
    return ret;
}

DimType &Dims::operator[](int idx)
{
    int nd = this->ndims();
    if (idx >= nd || -idx > nd) {
        LOGERR("invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd;
    }
    return this->data[idx];
}

const DimType &Dims::operator[](int idx) const
{
    int nd = this->ndims();
    if (idx >= nd || -idx > nd) {
        LOGERR("invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd;
    }
    return this->data[idx];
}

bool operator==(const Dims &a, const Dims &b)
{
    for (int i = 0; i < DIMS_LEN; ++i) {
        if (a.data[i] != b.data[i]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const Dims &a, const Dims &b)
{
    return !(a == b);
}

void to_json(nlohmann::json &j, const Dims &dims)
{
    j.clear();
    for (int i = 0; i < dims.ndims(); ++i) {
        j.push_back(dims.data[i]);
    }
}

void from_json(const nlohmann::json &j, Dims &dims)
{
    dims = Dims{j.get<std::vector<DimType>>()};
}

std::ostream &operator<<(std::ostream &os, const Dims &dims)
{
    if (dims.is_invalid()) {
        LOGERR("invalid dims given");
    }
    os << '<';
    if (dims.data[0] != NO_DIM) {
        os << dims.data[0];
        for (int i = 1; i < DIMS_LEN; ++i) {
            if (dims.data[i] == NO_DIM) {
                break;
            }
            os << ", " << dims.data[i];
        }
    }
    os << '>';
    return os;
}

} // namespace ark
