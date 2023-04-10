// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_H_
#define ARK_MODEL_H_

#include <array>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "ark/ops/ops_common.h"
#include "ark/tensor.h"

namespace ark {

//
class Model
{
  public:
    // Constructors.
    Model()
    {
    }
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    Tensor *tensor(const Dims &shape, TensorType dtype,
                   TensorBuf *buf = nullptr, const Dims &ldims = {},
                   const Dims &offs = {}, const Dims &pads = {},
                   const std::vector<Tensor *> &deps = {},
                   bool exported = false, bool imported = false,
                   const std::string &name = "tensor");

    Tensor *reshape(Tensor *input, const Dims &shape, bool allowzero,
                    Tensor *output, const std::string &name);
    Tensor *reshape(Tensor *input, std::initializer_list<DimType> shape,
                    bool allowzero = false, Tensor *output = nullptr,
                    const std::string &name = "reshape");

    Tensor *identity(Tensor *input, const std::vector<Tensor *> &deps = {},
                     Tensor *output = nullptr,
                     const std::string &name = "identity");
    std::vector<Tensor *> sharding(Tensor *input, DimType axis,
                                   DimType dim_per_shard,
                                   const std::string &name = "sharding");

    Tensor *reduce(Tensor *input, DimType axis, Tensor *output = nullptr,
                   bool is_relu = false, const std::string &name = "reduce");
    Tensor *transpose(Tensor *input, Dims perm, Tensor *output = nullptr,
                      const std::string &name = "transpose");
    Tensor *matmul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                   DimType splitk = 1, bool trans_input = false,
                   bool trans_other = false, bool is_relu = false,
                   const std::string &name = "matmul", int gran_lev = -1);
    Tensor *linear(Tensor *input, DimType out_features, bool bias = true,
                   Tensor *output = nullptr, DimType splitk = 1,
                   bool is_relu = false, const std::string &name = "linear",
                   int gran_lev = -1);
    Tensor *im2col(Tensor *input, int kernel_height, int kernel_width,
                   int stride_height, int stride_width, int pad_height,
                   int pad_width, int dilation_height, int dilation_width,
                   Tensor *output = nullptr,
                   const std::string &name = "im2col");
    Tensor *conv2d(Tensor *input, DimType in_channels, DimType out_channels,
                   DimType kernel_size, DimType stride, DimType padding,
                   bool bias = false, Tensor *output = nullptr,
                   const std::string &name = "conv2d");
    Tensor *max_pool(Tensor *input, DimType kernel_size, DimType stride,
                     Tensor *output = nullptr,
                     const std::string &name = "max_pool");
    Tensor *scale(Tensor *input, float val, Tensor *output = nullptr,
                  const std::string &name = "scale");
    Tensor *add(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "add");
    Tensor *mul(Tensor *input, Tensor *other, Tensor *output = nullptr,
                const std::string &name = "mul");
    Tensor *send(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "send");
    Tensor *send_done(Tensor *input, int id, Tensor *output = nullptr,
                      const std::string &name = "send_done");
    Tensor *recv(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                 Tensor *output = nullptr, const std::string &name = "recv");

    Tensor *send_mm(Tensor *input, int id, int gpu_dst, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "send_mm");
    Tensor *recv_mm(Tensor *input, int id, int gpu_src, std::size_t bytes = 0,
                    Tensor *output = nullptr,
                    const std::string &name = "recv_mm");
    Tensor *all_reduce(Tensor *input, int gpu_id, int gpu_num,
                       Tensor *output = nullptr,
                       const std::string &name = "all_reduce");

    TensorBuf *create_tensor_buf(const DimType bytes = 0);
    void destroy_tensor_buf(const TensorBuf *buf);
    Op *create_op(const OpType &type, const OpPrecType &prec_type,
                  const std::vector<Tensor *> &in_deps,
                  const std::vector<Tensor *> &out_deps,
                  const std::vector<OpArg> &args, const std::string &name,
                  int gran_lev = -1);

    const std::list<std::unique_ptr<TensorBuf>> &get_tensor_bufs() const
    {
        return tns_bufs_storage;
    };
    const std::list<std::unique_ptr<Tensor>> &get_tensors() const
    {
        return tns_storage;
    };
    const std::list<std::unique_ptr<Op>> &get_ops() const
    {
        return ops_storage;
    };
    const Op *get_gen_op(Tensor *tns) const;
    const std::set<Op *> &get_ref_ops(Tensor *tns) const;
    bool is_no_ref(Tensor *tns) const;

  private:
    std::list<std::unique_ptr<TensorBuf>> tns_bufs_storage;
    std::list<std::unique_ptr<Tensor>> tns_storage;
    std::list<std::unique_ptr<Op>> ops_storage;

    std::map<Tensor *, Op *> gen_op;
    std::map<Tensor *, std::set<Op *>> ref_ops;
    int next_eid = 0;
    std::map<std::string, int> name_cnts;
};

void to_json(nlohmann::json &j, const Model &model);
void from_json(const nlohmann::json &j, Model &model);

} // namespace ark

#endif // ARK_MODEL_H_