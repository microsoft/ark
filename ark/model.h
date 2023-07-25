// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_H_
#define ARK_MODEL_H_

#include "include/ark.h"
#include "ops/ops_common.h"

namespace ark {

class Model::Impl
{
  public:
    Impl() = default;
    ~Impl() = default;

    // Create a new TensorBuf object with `bytes` bytes.
    // A common usage is setting `bytes` to 0 during declaring a model and let
    // the scheduler determine the value after the model is completely defined.
    TensorBuf *create_tensor_buf(const DimType bytes = 0);

    void destroy_tensor_buf(const TensorBuf *buf);

    // Creates and returns an operator of the specified 'type'. This function
    // serves as a base function for other model operator functions.
    Op *add_op(const OpType type, const OpPrecType prec_type,
               const std::vector<Tensor *> &in_deps,
               const std::vector<Tensor *> &out_deps, const OpArgs &args,
               const std::string &name, const OpConfigMap *cfg_map, int gran_lev = -1);
    Op *add_op(Op &op);

    /// Delete an existing operator from the model.
    /// @param op the existing op to be deleted.
    void delete_op(Op *op);

    std::list<TensorBuf *> get_tensor_bufs() const;
    std::list<Tensor *> get_tensors() const;
    std::list<Op *> get_ops() const;
    const Op *get_gen_op(Tensor *tns) const;
    const std::set<Op *> &get_ref_ops(Tensor *tns) const;
    bool is_no_ref(Tensor *tns) const;

  protected:
    /// Rank of this model.
    int rank;
    /// Number of assigned EIDs.
    int next_eid = 0;

    friend class Model;

  private:
    /// Stores all tensor buffers.
    std::list<std::unique_ptr<TensorBuf>> tns_bufs_storage;
    /// Stores all tensors.
    std::list<std::unique_ptr<Tensor>> tns_storage;
    /// Stores all operators.
    std::list<std::unique_ptr<Op>> ops_storage;
    /// Maps a tensor to its generating operator.
    std::map<Tensor *, Op *> gen_op;
    /// Maps a tensor to its referencing operators.
    std::map<Tensor *, std::set<Op *>> ref_ops;
    /// Count the number of tensors requested the same name.
    std::map<std::string, int> name_cnts;
};

} // namespace ark

#endif // ARK_MODEL_H_
