// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_DIMS_H
#define ARK_DIMS_H

#include <string>
#include <vector>
#include <ostream>

namespace ark {

// Data type for dimension.
typedef long long int DimType;

// DIMS_LEN is the maximum number of dimensions of a tensor. If a tensor
// has less than DIMS_LEN dimensions, the remaining dimensions will be NO_DIM.
enum { DIMS_LEN = 4, NO_DIM = -1 };

// Up-to-`DIMS_LEN`-dimensional vector.
class Dims {
  private:
    DimType data[DIMS_LEN];

  public:
    // Construct with given four dimensions.
    Dims(DimType d0 = NO_DIM, DimType d1 = NO_DIM, DimType d2 = NO_DIM,
         DimType d3 = NO_DIM);
    // Copy another Dims object.
    Dims(const Dims &dims_);
    // Construct from a vector. If the vector is shorter than DIMS_LEN, put
    // following NO_DIMs. Raise an error if the vector is longer than DIMS_LEN.
    Dims(const std::vector<DimType> &vec);

    // Return the volume of dimensions. If the dimensions are invalid, return
    // -1.
    DimType size() const;
    // Return the number of valid dimensions.
    int ndims() const;
    // Return a new Dims object with 4 valid dimensions by prepending 1s.
    Dims dims4() const;
    // Return true if the dimensions are empty.
    bool is_no_dim() const;
    // Return true if the dimensions are invalid.
    bool is_invalid() const;
    // Insert a dimension at the given index.
    void insert(int idx, DimType dim);
    // Erase the dimension at the given index and return the erased dimension.
    DimType erase(int idx);

    std::string serialize() const;

    DimType &operator[](int idx);

    const DimType &operator[](int idx) const;

    Dims &operator=(const Dims &) = default;

    friend bool operator==(const Dims &a, const Dims &b);
    friend bool operator!=(const Dims &a, const Dims &b);

    friend std::ostream &operator<<(std::ostream &os, const Dims &dims);
};

}  // namespace ark

#endif  // ARK_DIMS_H
