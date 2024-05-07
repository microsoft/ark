// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/dims.hpp"

#include <vector>

#include "error.hpp"
#include "json.hpp"
#include "logging.h"

namespace ark {

Dims::Dims() {}

Dims::Dims(DimType d0) { data_ = {d0}; }

Dims::Dims(DimType d0, DimType d1) { data_ = {d0, d1}; }

Dims::Dims(DimType d0, DimType d1, DimType d2) { data_ = {d0, d1, d2}; }

// Construct with given four dimensions.
Dims::Dims(DimType d0, DimType d1, DimType d2, DimType d3) {
    data_ = {d0, d1, d2, d3};
}

// Copy another Dims object.
Dims::Dims(const Dims &dims_) {
    if (dims_.is_invalid()) {
        ERR(InvalidUsageError, "invalid dims given");
    }
    data_ = dims_.data_;
}

// Construct from a vector. Raise an error if the vector is longer than
// DIMS_LEN.
Dims::Dims(const std::vector<DimType> &vec) {
    int ds = (int)vec.size();
    if (ds > DIMS_LEN) {
        ERR(InvalidUsageError, "only support dims with size <= ", DIMS_LEN,
            ". Given: ", *this);
    }
    data_ = vec;
}

// Return the volume of dimensions. If there is a negative dimension, return -1.
DimType Dims::nelems() const {
    if (this->has_negative()) return -1;
    if (data_.empty()) return 0;
    DimType ret = 1;
    for (auto d : data_) {
        ret *= d;
    }
    return ret;
}

// Return the number of dimensions.
int Dims::ndims() const { return (int)data_.size(); }

// Return a new Dims object with 4 valid dimensions by prepending 1s.
Dims Dims::dims4() const {
    std::vector<DimType> vec;
    for (auto i = data_.size(); i < DIMS_LEN; ++i) {
        vec.emplace_back(1);
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        vec.emplace_back(data_[i]);
    }
    return vec;
}

// Return true if all valid dimensions are zero.
bool Dims::is_zeros() const {
    if (this->is_invalid()) return false;
    for (auto d : data_) {
        if (d != 0) return false;
    }
    return true;
}

// Return true if the dimensions are empty.
bool Dims::is_no_dim() const { return data_.size() == 0; }

bool Dims::has_negative() const {
    for (auto d : data_) {
        if (d < 0) return true;
    }
    return false;
}

// Return true if the dimensions are invalid.
bool Dims::is_invalid() const { return data_.size() > DIMS_LEN; }

const std::vector<DimType> &Dims::vector() const { return data_; }

void Dims::insert(int idx, DimType dim) {
    int nd = data_.size();
    if (nd >= DIMS_LEN) {
        ERR(InvalidUsageError, "too many dimensions: ", *this);
    }
    if (idx > nd || -idx > nd + 1) {
        ERR(InvalidUsageError, "invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd + 1;
    }
    data_.emplace_back(0);
    for (int i = nd; i > idx; --i) {
        data_[i] = data_[i - 1];
    }
    data_[idx] = dim;
}

DimType Dims::erase(int idx) {
    int nd = data_.size();
    if (idx >= nd || -idx > nd) {
        ERR(InvalidUsageError, "invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd;
    }
    DimType ret = data_[idx];
    for (int i = idx; i < nd - 1; ++i) {
        data_[i] = data_[i + 1];
    }
    data_.pop_back();
    return ret;
}

DimType &Dims::operator[](int idx) {
    int nd = data_.size();
    if (idx >= nd || -idx > nd) {
        ERR(InvalidUsageError, "invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd;
    }
    return data_[idx];
}

const DimType &Dims::operator[](int idx) const {
    int nd = data_.size();
    if (idx >= nd || -idx > nd) {
        ERR(InvalidUsageError, "invalid index given: ", idx, " for ", *this);
    }
    if (idx < 0) {
        idx += nd;
    }
    return data_[idx];
}

bool operator==(const Dims &a, const Dims &b) {
    if (a.ndims() != b.ndims()) {
        return false;
    }
    for (auto i = 0; i < a.ndims(); ++i) {
        if (a.data_[i] != b.data_[i]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const Dims &a, const Dims &b) { return !(a == b); }

std::ostream &operator<<(std::ostream &os, const Dims &dims) {
    if (dims.is_invalid()) {
        ERR(InvalidUsageError, "invalid dims given");
    }
    int ndims = dims.ndims();
    os << "<";
    if (ndims > 0) {
        os << dims[0];
        for (int i = 1; i < ndims; ++i) {
            os << ", " << dims[i];
        }
    }
    os << '>';
    return os;
}

}  // namespace ark
