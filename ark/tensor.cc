// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <string>

#include "ark.h"
#include "gpu/gpu_mgr.h"
#include "logging.h"
#include "math.h"

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

//
void Tensor::update_pads(const std::vector<DimType> &pads_) {
    int ndims = this->ldims.ndims();
    std::vector<DimType> tmp;
    for (int i = 0; i < ndims - (int)pads_.size(); ++i) {
        tmp.emplace_back(1);
    }
    for (int i = 0; i < (int)pads_.size(); ++i) {
        tmp.emplace_back(pads_[i] == -1 ? 1 : pads_[i]);
    }
    Dims new_pads{tmp};
    for (int i = 0; i < ndims; ++i) {
        DimType new_udim = math::lcm(this->pads[i], new_pads[i]);
        this->pads[i] = new_udim;
        this->ldims[i] = math::pad(this->ldims[i], new_udim);
    }
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
        gpu_memcpy(gbuf->ref(this->offset_bytes(0)), ptr, bytes);
        return;
    }
    size_t done = 0;
    size_t rem = bytes;
    for (DimType i = 0; i < this->shape[0]; ++i) {
        if (ndims == 2) {
            size_t cb =
                std::min(rem, (size_t)this->shape[1] * this->type_bytes());
            gpu_memcpy(gbuf->ref(this->offset_bytes(i, 0)), &ptr[done], cb);
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
                gpu_memcpy(gbuf->ref(this->offset_bytes(i, j, 0)), &ptr[done],
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
                gpu_memcpy(gbuf->ref(this->offset_bytes(i, j, k, 0)),
                           &ptr[done], cb);
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
        gpu_memcpy(ptr, gbuf->ref(this->offset_bytes(0)), bytes);
        return ptr;
    }
    size_t done = 0;
    size_t rem = bytes;
    for (DimType i = 0; i < this->shape[0]; ++i) {
        if (ndims == 2) {
            size_t cb =
                std::min(rem, (size_t)this->shape[1] * this->type_bytes());
            gpu_memcpy(&ptr[done], gbuf->ref(this->offset_bytes(i, 0)), cb);
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
                gpu_memcpy(&ptr[done], gbuf->ref(this->offset_bytes(i, j, 0)),
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
                gpu_memcpy(&ptr[done],
                           gbuf->ref(this->offset_bytes(i, j, k, 0)), cb);
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
    gpu_memcpy(buf, gbuf->ref(this->offset_bytes(0)), bytes);
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
        gpu_memset(buf->ref(this->offset_bytes(0)), 0, num);
        return;
    }
    size_t done = 0;
    size_t rem = num;
    for (DimType i = 0; i < this->shape[0]; ++i) {
        if (ndims == 2) {
            bytes = (size_t)this->shape[1] * this->type_bytes();
            assert(bytes % 4 == 0);
            size_t cn = std::min(rem, bytes >> 2);
            gpu_memset(buf->ref(this->offset_bytes(i, 0)), 0, cn);
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
                gpu_memset(buf->ref(this->offset_bytes(i, j, 0)), 0, cn);
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
                gpu_memset(buf->ref(this->offset_bytes(i, j, k, 0)), 0, cn);
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
