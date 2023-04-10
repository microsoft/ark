// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/dims.h"
#include "ark/logging.h"

using namespace std;

namespace ark {

// Return shape string from a vector of DimType
const string shape_str(const vector<DimType> &shape)
{
    stringstream ss;
    ss << "<";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i];
        if (i != shape.size() - 1) {
            ss << ", ";
        }
    }
    ss << ">";
    return ss.str();
}

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
Dims::Dims(const vector<DimType> &vec)
{
    int ds = (int)vec.size();
    if (ds > DIMS_LEN) {
        LOGERR("only support dims with size <= ", DIMS_LEN,
               ". Given: ", shape_str(vec));
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

bool operator==(const Dims &a, const Dims &b)
{
    return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}

bool operator!=(const Dims &a, const Dims &b)
{
    return (a[0] != b[0]) || (a[1] != b[1]) || (a[2] != b[2]) || (a[3] != b[3]);
}

void to_json(nlohmann::json &j, const Dims &dims)
{
    j.clear();
    for (int i = 0; i < dims.ndims(); ++i) {
        j.push_back(dims[i]);
    }
}

void from_json(const nlohmann::json &j, Dims &dims)
{
    dims = Dims{j.get<vector<DimType>>()};
}

ostream &operator<<(ostream &os, const Dims &dims)
{
    if (dims.is_invalid()) {
        LOGERR("invalid dims given");
    }
    os << '<';
    if (dims[0] != NO_DIM) {
        os << dims[0];
        for (int i = 1; i < DIMS_LEN; ++i) {
            if (dims[i] == NO_DIM) {
                break;
            }
            os << ", " << dims[i];
        }
    }
    os << '>';
    return os;
}

} // namespace ark
