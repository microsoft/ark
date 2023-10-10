// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"

using namespace std;

namespace ark {
namespace utils {

/// Return a random @ref half_t array.
/// @param num Number of elements
/// @param max_val Maximum value
/// @return std::unique_ptr<half_t[]>
unique_ptr<half_t[]> rand_halfs(size_t num, float max_val) {
    return rand_array<half_t>(num, max_val);
}

/// Return a random float array.
/// @param num Number of elements
/// @param max_val Maximum value
/// @return std::unique_ptr<float[]>
unique_ptr<float[]> rand_floats(size_t num, float max_val) {
    return rand_array<float>(num, max_val);
}

/// Return a random bytes array.
/// @param num Number of elements
/// @return std::unique_ptr<uint8_t[]>
unique_ptr<uint8_t[]> rand_bytes(size_t num) {
    return rand_array<uint8_t>(num, 255);
}

/// Return an array of values starting from `begin` with difference `diff`.
/// @tparam T Type of the array
/// @param num Number of elements
/// @param begin First value
/// @param diff Difference between two values
/// @return std::unique_ptr<T[]>
template <typename T>
unique_ptr<T[]> range_array(size_t num, float begin, float diff) {
    T *ret = new T[num];
    for (size_t i = 0; i < num; ++i) {
        ret[i] = T(begin);
        begin += diff;
    }
    return unique_ptr<T[]>(ret);
}

/// Return a @ref half_t range array.
/// @param num Number of elements
/// @param begin First value
/// @param diff Difference between two values
/// @return std::unique_ptr<half_t[]>
unique_ptr<half_t[]> range_halfs(size_t num, float begin, float diff) {
    return range_array<half_t>(num, begin, diff);
}

/// Return a float range array.
/// @param num Number of elements
/// @param begin First value
/// @param diff Difference between two values
/// @return std::unique_ptr<float[]>
unique_ptr<float[]> range_floats(size_t num, float begin, float diff) {
    return range_array<float>(num, begin, diff);
}

}  // namespace utils
}  // namespace ark
