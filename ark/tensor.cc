// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cassert>
#include <string>

#include "ark.h"
#include "gpu/gpu_mgr.h"
#include "logging.h"
#include "math.h"

#define DEBUG_PADDING 0
#define PADDING_DEBUG(...)           \
    do {                             \
        if (DEBUG_PADDING) {         \
            LOG(DEBUG, __VA_ARGS__); \
        }                            \
    } while (0);

namespace ark {

TensorBuf::TensorBuf(const DimType &bytes_, int id_) : bytes{bytes_}, id{id_} {}

size_t TensorBuf::get_buf_offset() const {
    if (this->buf == nullptr) {
        LOG(ERROR, "TensorBuf is not configured yet");
    }
    return static_cast<GpuBuf *>(this->buf)->get_offset();
}

// Tensor constructor
Tensor::Tensor(const Dims &shape_, const TensorType &type_, TensorBuf *buf_,
               const Dims &ldims_, const Dims &offs_, const Dims &pads_,
               bool exported_, int imported_rank_, int id_,
               const std::string &name_)
    : buf{buf_},
      type{type_},
      exported{exported_},
      imported_rank{imported_rank_},
      id{id_},
      name{name_} {
    if (shape_.size() == 0) {
        LOG(ERROR,
            "Tensor shape should consist of positive numbers. Given: ", shape_);
    } else if (shape_.is_no_dim()) {
        // Assume a single-element constant
        this->shape = {1};
    } else {
        this->shape = shape_;
    }
    int ndims = this->shape.ndims();
    if (ldims_.is_no_dim()) {
        this->ldims = this->shape;
    } else {
        if (ndims != ldims_.ndims()) {
            LOG(ERROR,
                "Tensor shape and ldims should have the same number of "
                "dimensions. Given: shape ",
                this->shape, " ldims ", ldims_);
        }
        this->ldims = ldims_;
    }
    if (offs_.is_no_dim()) {
        std::vector<DimType> dims_vec;
        for (int i = 0; i < ndims; ++i) {
            dims_vec.push_back(0);
        }
        this->offs = Dims{dims_vec};
    } else {
        if (ndims != offs_.ndims()) {
            LOG(ERROR,
                "Tensor shape and offs should have the same number of "
                "dimensions. Given: shape ",
                this->shape, " offs ", offs_);
        }
        this->offs = offs_;
    }
    if (pads_.is_no_dim()) {
        std::vector<DimType> dims_vec;
        for (int i = 0; i < ndims; ++i) {
            dims_vec.push_back(1);
        }
        this->pads = Dims{dims_vec};
    } else {
        if (ndims != pads_.ndims()) {
            LOG(ERROR,
                "Tensor shape and pads should have the same number of "
                "dimensions. Given: shape ",
                this->shape, " pads ", pads_);
        }
        this->pads = pads_;
    }
    for (int i = 0; i < ndims; ++i) {
        if (this->ldims[i] % this->pads[i] != 0) {
            LOG(ERROR, "Tensor ldims should be a multiple of pads. ldims ",
                this->ldims, " pads ", this->pads);
        }
    }
    for (int i = 0; i < ndims; ++i) {
        if (this->offs[i] + this->shape[i] > this->ldims[i]) {
            LOG(ERROR, "Tensor exceeds the memory boundary. offs ", this->offs,
                " shape ", this->shape, " ldims ", this->ldims);
        }
    }
}

bool tensor_reshape_helper(const Dims &shape, const Dims &ldims,
                           const Dims &offs, const Dims &new_shape,
                           Dims &new_ldims, Dims &new_offs) {
    // Infer the new ldims and offs
    std::vector<DimType> reverse_ldims;
    std::vector<DimType> reverse_offs;

    int orig_idx = shape.ndims() - 1;
    int new_idx = new_shape.ndims() - 1;
    DimType orig_shape_stack = shape[orig_idx];
    DimType new_shape_stack = new_shape[new_idx];
    DimType orig_ldim_stack = ldims[orig_idx];
    DimType div_stack = 1;
    while (orig_idx >= 0 && new_idx >= 0) {
        if (orig_shape_stack == new_shape_stack) {
            if (orig_ldim_stack % div_stack != 0) {
                return false;
            }
            DimType new_off = offs[orig_idx];
            for (auto i = orig_idx + 1; i < ldims.ndims(); i++) {
                new_off *= ldims[i];
            }
            std::for_each(reverse_ldims.begin(), reverse_ldims.end(),
                          [&new_off](DimType d) { new_off /= d; });
            reverse_ldims.push_back(orig_ldim_stack / div_stack);
            reverse_offs.push_back(new_off);
            div_stack = 1;
            new_idx--;
            orig_idx--;
            if (new_idx >= 0) {
                new_shape_stack = new_shape[new_idx];
            }
            if (orig_idx >= 0) {
                orig_shape_stack = shape[orig_idx];
                orig_ldim_stack = ldims[orig_idx];
            }
        } else if (orig_shape_stack > new_shape_stack) {
            div_stack *= new_shape[new_idx];
            reverse_ldims.push_back(new_shape[new_idx]);
            reverse_offs.push_back(0);
            new_idx--;
            if (new_idx >= 0) {
                new_shape_stack *= new_shape[new_idx];
            }
        } else {
            if (ldims[orig_idx] != shape[orig_idx] || offs[orig_idx] != 0) {
                return false;
            }
            orig_idx--;
            if (orig_idx >= 0) {
                orig_shape_stack *= shape[orig_idx];
                orig_ldim_stack *= ldims[orig_idx];
            }
        }
    }
    while (new_idx >= 0 && new_shape[new_idx] == 1) {
        reverse_ldims.push_back(1);
        reverse_offs.push_back(0);
        new_idx--;
    }
    while (orig_idx >= 0 && shape[orig_idx] == 1) {
        if (ldims[orig_idx] != shape[orig_idx] || offs[orig_idx] != 0) {
            return false;
        }
        orig_idx--;
    }
    if (orig_idx >= 0 || new_idx >= 0) {
        return false;
    }
    std::reverse(reverse_ldims.begin(), reverse_ldims.end());
    std::reverse(reverse_offs.begin(), reverse_offs.end());
    new_ldims = reverse_ldims;
    new_offs = reverse_offs;
    return true;
}

// Helper for `Tensor::update_pads()`.
static Dims calc_pads(const Dims &tile, const Dims &ldims) {
    // 1. Match the number of dimensions. If `tile` has more dimensions than
    // `ldims`, remove the leading 1s. If `tile` has less dimensions than
    // `ldims`, add 1s to the leading dimensions.
    //
    // 2. Replace -1 in `tile` with 1.
    //

    int ndims = ldims.ndims();
    Dims tile_copy = tile;
    while (tile_copy.ndims() > ndims) {
        if (tile_copy[0] == 1) {
            tile_copy.erase(0);
        } else {
            LOG(ERROR, "invalid tile ", tile, " for ldims ", ldims);
        }
    }
    std::vector<DimType> tmp;
    for (int i = 0; i < ndims - tile_copy.ndims(); ++i) {
        tmp.emplace_back(1);
    }
    for (int i = 0; i < tile_copy.ndims(); ++i) {
        tmp.emplace_back((tile_copy[i] == -1) ? 1 : tile_copy[i]);
    }
    return Dims(tmp);
}

static Dims elem_to_bytes(const Dims &elem, const TensorType &type) {
    Dims tmp(elem);
    tmp[-1] *= type.bytes();
    return tmp;
}

//
bool Tensor::update_pads(const Dims &tile, const Tensor *ref_tensor,
                         const Dims &ref_orig_ldims) {
    Dims orig_ldims = this->ldims;
    Dims orig_bldims = elem_to_bytes(orig_ldims, this->type);
    Dims ref_orig_bldims;
    if (ref_tensor != nullptr) {
        ref_orig_bldims = elem_to_bytes(ref_orig_ldims, ref_tensor->type);
    }
    if ((ref_tensor == nullptr) || (orig_bldims == ref_orig_bldims)) {
        // `tile` is supposed to be directly applied for ldims of this tensor.
        Dims tile_copy = tile;
        if ((ref_tensor != nullptr) && (this->type != ref_tensor->type)) {
            int this_bytes = this->type.bytes();
            int ref_bytes = ref_tensor->type.bytes();
            if (this_bytes > ref_bytes) {
                if (this_bytes % ref_bytes != 0) {
                    LOG(WARN, "unexpected error");
                    return false;
                }
                tile_copy[-1] *= this_bytes / ref_bytes;
            } else {
                if (ref_bytes % this_bytes != 0) {
                    LOG(WARN, "unexpected error");
                    return false;
                }
                if (tile_copy[-1] % (ref_bytes / this_bytes) != 0) {
                    LOG(WARN, "unexpected error");
                    return false;
                }
                tile_copy[-1] /= (ref_bytes / this_bytes);
            }
        }
        Dims new_pads = calc_pads(tile_copy, this->ldims);
        for (int i = 0; i < this->ldims.ndims(); ++i) {
            DimType new_udim = math::lcm(this->pads[i], new_pads[i]);
            this->pads[i] = new_udim;
            this->ldims[i] = math::pad(this->ldims[i], new_udim);
        }
        PADDING_DEBUG("updated pads: tile ", tile_copy, " orig_ldims ",
                      orig_ldims, " new_pads ", new_pads, " new_ldims ",
                      this->ldims);
        return true;
    }

    // `tile` is supposed to be applied for `ref_tensor`, not this tensor.

    if ((orig_ldims.size() != ref_orig_ldims.size()) ||
        (ref_tensor->ldims.size() < orig_ldims.size())) {
        LOG(WARN, "unexpected error.");
        return false;
    } else if (ref_tensor->ldims == ref_orig_ldims) {
        // ldims of `ref_tensor` is not changed; nothing to do here
        return true;
    }

    // calculate what the ldims of this tensor would be if we reshape a tensor
    // with shape `ref_orig_ldims` and ldims `ref_tensor->ldims` into another
    // shape `this->ldims`.
    Dims ref_bldims = elem_to_bytes(ref_tensor->ldims, ref_tensor->type);
    Dims ref_boffs = elem_to_bytes(ref_tensor->offs, ref_tensor->type);
    Dims target_bldims;
    Dims target_boffs;
    std::stringstream ss;
    ss << "padding conflict detected. ref_orig_bldims=" << ref_orig_bldims
       << " ref_bldims=" << ref_bldims << " ref_boffs=" << ref_boffs
       << " orig_bldims=" << orig_bldims;
    if (!tensor_reshape_helper(ref_orig_bldims, ref_bldims, ref_boffs,
                               orig_bldims, target_bldims, target_boffs)) {
        LOG(WARN, ss.str());
        return false;
    } else if (target_bldims[-1] % this->type.bytes() != 0) {
        LOG(WARN, ss.str());
        return false;
    }
    Dims target_ldims = target_bldims;
    target_ldims[-1] /= this->type_bytes();
    if (target_ldims.ndims() != this->ldims.ndims()) {
        LOG(WARN, "unexpected error");
        return false;
    }
    // check if `target_ldims` is feasible for this tensor.
    for (int i = 0; i < this->ldims.ndims(); ++i) {
        if (target_ldims[i] % this->pads[i] != 0) {
            LOG(WARN, "the current padding ", this->pads,
                " is not feasible for ldims ", this->ldims,
                " and target_ldims ", target_ldims);
            return false;
        }
    }
    this->ldims = target_ldims;
    // no need to update `pads`.
    PADDING_DEBUG("updated pads: tile ", tile, " ref_shape ", ref_tensor->shape,
                  " ref_ldims ", ref_tensor->ldims, " ref_offs ",
                  ref_tensor->offs, " orig_shape ", this->shape, " orig_ldims ",
                  orig_ldims, " new_ldims ", target_ldims);
    return true;
}

// Offset to the element [i0][i1][i2][i3] of this tensor in the TensorBuf.
DimType Tensor::offset(DimType i0, DimType i1, DimType i2, DimType i3) const {
    auto &l = this->ldims;
    auto &o = this->offs;
    int ndims = this->shape.ndims();
    if (ndims == 0) {
        return 0;
    } else if (ndims == 1) {
        return o[0] + i0;
    } else if (ndims == 2) {
        return ((o[0] + i0) * l[1]) + o[1] + i1;
    } else if (ndims == 3) {
        return ((o[0] + i0) * l[1] * l[2]) + ((o[1] + i1) * l[2]) + o[2] + i2;
    }
    return ((o[0] + i0) * l[1] * l[2] * l[3]) + ((o[1] + i1) * l[2] * l[3]) +
           ((o[2] + i2) * l[3]) + o[3] + i3;
}

// Number of elements in the tensor excluding padding.
DimType Tensor::size() const { return this->shape.size(); }

// Number of dimensions in the tensor.
int Tensor::ndims() const { return this->shape.ndims(); }

// Number of bytes of each element in the tensor.
int Tensor::type_bytes() const { return this->type.bytes(); }

// Number of bytes of the tensor.
DimType Tensor::shape_bytes() const {
    return this->shape.size() * this->type_bytes();
}

// Should be the same as the number of bytes of the TensorBuf.
DimType Tensor::ldims_bytes() const {
    return this->ldims.size() * this->type_bytes();
}

// Offset in bytes.
DimType Tensor::offset_bytes(DimType i0, DimType i1, DimType i2,
                             DimType i3) const {
    return this->offset(i0, i1, i2, i3) * this->type_bytes();
}

bool Tensor::is_alloced() const {
    if (this->buf == nullptr) {
        return false;
    }
    return this->buf->buf != nullptr;
}

bool Tensor::is_sequential() const {
    // Shape and ldims should be the same except for the first dimension.
    for (int i = 1; i < this->shape.ndims(); ++i) {
        if (this->shape[i] != this->ldims[i]) {
            return false;
        }
    }
    return true;
}

void Tensor::write(const void *buf) {
    if (buf == nullptr) {
        LOG(ERROR, "the given host buffer is null");
    }
    GpuBuf *gbuf = static_cast<GpuBuf *>(this->buf->buf);
    if (gbuf == nullptr) {
        LOG(ERROR, "failed to get GPU buffer for tensor ", this->name);
    }
    size_t bytes = this->shape_bytes();
    int ndims = this->ndims();
    char *ptr = (char *)buf;
    if (ndims == 1) {
        gpu_memcpy(gbuf, this->offset_bytes(0), ptr, 0, bytes);
        return;
    }
    size_t done = 0;
    size_t rem = bytes;
    for (DimType i = 0; i < this->shape[0]; ++i) {
        if (ndims == 2) {
            size_t cb =
                std::min(rem, (size_t)this->shape[1] * this->type_bytes());
            gpu_memcpy(gbuf, this->offset_bytes(i, 0), &ptr[done], 0, cb);
            rem -= cb;
            done += cb;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < this->shape[1]; ++j) {
            if (ndims == 3) {
                size_t cb =
                    std::min(rem, (size_t)this->shape[2] * this->type_bytes());
                gpu_memcpy(gbuf, this->offset_bytes(i, j, 0), &ptr[done], 0,
                           cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < this->shape[2]; ++k) {
                size_t cb =
                    std::min(rem, (size_t)this->shape[3] * this->type_bytes());
                gpu_memcpy(gbuf, this->offset_bytes(i, j, k, 0), &ptr[done], 0,
                           cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
            }
        }
    }
    assert(rem == 0);
    assert(done == bytes);
}

void *Tensor::read(void *buf) {
    GpuBuf *gbuf = static_cast<GpuBuf *>(this->buf->buf);
    if (gbuf == nullptr) {
        LOG(ERROR, "failed to get GPU buffer for tensor ", this->id);
    }
    size_t bytes = this->shape_bytes();
    int ndims = this->ndims();
    if (buf == nullptr) {
        buf = ::malloc(bytes);
        if (buf == nullptr) {
            LOG(ERROR, "failed to allocate host buffer");
        }
    }
    char *ptr = (char *)buf;
    if (ndims == 1) {
        gpu_memcpy(ptr, 0, gbuf, this->offset_bytes(0), bytes);
        return ptr;
    }
    size_t done = 0;
    size_t rem = bytes;
    for (DimType i = 0; i < this->shape[0]; ++i) {
        if (ndims == 2) {
            size_t cb =
                std::min(rem, (size_t)this->shape[1] * this->type_bytes());
            gpu_memcpy(&ptr[done], 0, gbuf, this->offset_bytes(i, 0), cb);
            rem -= cb;
            done += cb;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < this->shape[1]; ++j) {
            if (ndims == 3) {
                size_t cb =
                    std::min(rem, (size_t)this->shape[2] * this->type_bytes());
                gpu_memcpy(&ptr[done], 0, gbuf, this->offset_bytes(i, j, 0),
                           cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < this->shape[2]; ++k) {
                size_t cb =
                    std::min(rem, (size_t)this->shape[3] * this->type_bytes());
                gpu_memcpy(&ptr[done], 0, gbuf, this->offset_bytes(i, j, k, 0),
                           cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
            }
        }
    }
    assert(rem == 0);
    assert(done == bytes);
    return buf;
}

void *Tensor::read_raw(void *buf) {
    GpuBuf *gbuf = static_cast<GpuBuf *>(this->buf->buf);
    if (gbuf == nullptr) {
        LOG(ERROR, "failed to get GPU buffer for tensor ", this->id);
    }
    size_t bytes = this->ldims_bytes();
    if (buf == nullptr) {
        buf = ::malloc(bytes);
        if (buf == nullptr) {
            LOG(ERROR, "failed to allocate host buffer");
        }
    }
    gpu_memcpy(buf, 0, gbuf, 0, bytes);
    return buf;
}

void Tensor::clear() {
    GpuBuf *buf = static_cast<GpuBuf *>(this->buf->buf);
    if (buf == nullptr) {
        LOG(ERROR, "failed to get GPU buffer for tensor ", this->name);
    }
    int ndims = this->ndims();
    size_t bytes = this->shape_bytes();
    assert(bytes % 4 == 0);
    size_t num = bytes >> 2;
    if (ndims == 1) {
        gpu_memset(buf, this->offset_bytes(0), 0, num);
        return;
    }
    size_t done = 0;
    size_t rem = num;
    for (DimType i = 0; i < this->shape[0]; ++i) {
        if (ndims == 2) {
            bytes = (size_t)this->shape[1] * this->type_bytes();
            assert(bytes % 4 == 0);
            size_t cn = std::min(rem, bytes >> 2);
            gpu_memset(buf, this->offset_bytes(i, 0), 0, cn);
            rem -= cn;
            done += cn;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < this->shape[1]; ++j) {
            if (ndims == 3) {
                bytes = (size_t)this->shape[2] * this->type_bytes();
                assert(bytes % 4 == 0);
                size_t cn = std::min(rem, bytes >> 2);
                gpu_memset(buf, this->offset_bytes(i, j, 0), 0, cn);
                rem -= cn;
                done += cn;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < this->shape[2]; ++k) {
                bytes = (size_t)this->shape[3] * this->type_bytes();
                assert(bytes % 4 == 0);
                size_t cn = std::min(rem, bytes >> 2);
                gpu_memset(buf, this->offset_bytes(i, j, k, 0), 0, cn);
                rem -= cn;
                done += cn;
                if (rem == 0) {
                    break;
                }
            }
        }
    }
    assert(rem == 0);
    assert(done == num);
}

}  // namespace ark
