// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_DIMS_HPP
#define ARK_DIMS_HPP

#include <ostream>
#include <string>
#include <vector>

namespace ark {

// Data type for dimension.
typedef int64_t DimType;

// DIMS_LEN is the maximum number of dimensions of a tensor.
constexpr DimType DIMS_LEN = 4;

// Up-to-`DIMS_LEN`-dimensional vector.
class Dims {
   private:
    std::vector<DimType> data_;

   public:
    Dims();

    Dims(DimType d0);

    Dims(DimType d0, DimType d1);

    Dims(DimType d0, DimType d1, DimType d2);

    Dims(DimType d0, DimType d1, DimType d2, DimType d3);

    // Copy another Dims object.
    Dims(const Dims &dims_);
    // Construct from a vector. If the vector is shorter than DIMS_LEN, put
    // following NO_DIMs. Raise an error if the vector is longer than DIMS_LEN.
    Dims(const std::vector<DimType> &vec);

    // Return the number of elements. If the dimensions are invalid, return
    // -1.
    DimType nelems() const;
    // Return the number of valid dimensions.
    int ndims() const;
    // Return a new Dims object with 4 valid dimensions by prepending 1s.
    Dims dims4() const;
    // Return true if all valid dimensions are zero.
    bool is_zeros() const;
    // Return true if the dimensions are empty.
    bool is_no_dim() const;
    //
    bool has_negative() const;
    // Return true if the dimensions are invalid.
    bool is_invalid() const;
    // Return a vector of valid dimensions.
    const std::vector<DimType> &vector() const;
    // Insert a dimension at the given index.
    void insert(int idx, DimType dim);
    // Erase the dimension at the given index and return the erased dimension.
    DimType erase(int idx);

    std::string serialize(int indent = -1) const;

    static Dims deserialize(const std::string &serialized);

    DimType &operator[](int idx);

    const DimType &operator[](int idx) const;

    Dims &operator=(const Dims &) = default;

    friend bool operator==(const Dims &a, const Dims &b);
    friend bool operator!=(const Dims &a, const Dims &b);
};

std::ostream &operator<<(std::ostream &os, const Dims &dims);

}  // namespace ark

#endif  // ARK_DIMS_HPP
