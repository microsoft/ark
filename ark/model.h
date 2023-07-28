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

    /// Create a new @ref TensorBuf object with @p bytes bytes.
    ///
    /// A common usage is setting @p bytes to 0 during declaring a model and let
    /// the scheduler determine the value after the model is completely defined.
    ///
    /// @param bytes the number of bytes of the @ref TensorBuf
    /// @return the created @ref TensorBuf
    TensorBuf *create_tensor_buf(const DimType bytes = 0);

    /// Remove a @ref TensorBuf object from the model.
    void destroy_tensor_buf(const TensorBuf *buf);

    /// Add a new @ref Op to the model.
    /// @param type the type of the @ref Op.
    /// @param prec_type the precision type of the @ref Op.
    /// @param inputs the input tensors of the @ref Op, including execution
    /// dependencies.
    /// @param output_refs the output reference tensors of the @ref Op. Output
    /// tensors are created based on these references.
    /// @param args the arguments of the @ref Op.
    /// @param name the name of the @ref Op.
    /// @param cfg_map the configuration map of the @ref Op
    /// @param gran_lev the granularity level of the @ref Op. Larger values
    /// should indicate finer-grained Ops. If it is -1, the granularity level
    /// will be automatically determined by the scheduler.
    /// @return the output tensors of the @ref Op.
    std::vector<Tensor *> add_op(const OpType type, const OpPrecType prec_type,
                                 const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &output_refs,
                                 const OpArgs &args, const std::string &name,
                                 const OpConfigMap *cfg_map, int gran_lev = -1);

    /// Add a new @ref Op to the model.
    /// @param type the type of the @ref Op.
    /// @param prec_type the precision type of the @ref Op.
    /// @param inputs the input tensors of the @ref Op, including execution
    /// dependencies.
    /// @param output_refs the output reference tensors of the @ref Op. Output
    /// tensors are created based on these references.
    /// @param args the arguments of the @ref Op.
    /// @param name the name of the @ref Op.
    /// @param cfg_map the configuration map of the @ref Op
    /// @param gran_lev the granularity level of the @ref Op. Larger values
    /// should indicate finer-grained Ops. If it is -1, the granularity level
    /// will be automatically determined by the scheduler.
    /// @return the output tensors of the @ref Op.
    std::vector<Tensor *> add_op(Op &op);

    /// Delete an existing @ref Op from the model.
    /// @param op the existing @ref Op to be deleted.
    void delete_op(Op *op);

    /// Get references to all @ref TensorBuf objects.
    /// @return a list of @ref TensorBuf pointers.
    std::list<TensorBuf *> get_tensor_bufs() const;

    /// Get references to all @ref Tensor objects.
    /// @return a list of @ref Tensor pointers.
    std::list<Tensor *> get_tensors() const;

    /// Get references to all @ref Op objects.
    /// @return a list of @ref Op pointers.
    std::list<Op *> get_ops() const;

    /// Get the producer @ref Op of @p tns.
    /// @param tns the @ref Tensor to query.
    const Op *get_producer(Tensor *tns) const;

    /// Get the user @ref Op of @p tns.
    /// @param tns the @ref Tensor to query.
    const std::set<Op *> &get_users(Tensor *tns) const;

    /// True if @p tns has no user.
    /// @param tns the @ref Tensor to query.
    bool is_no_user(Tensor *tns) const;

    /// Model graph analysis

    /// Get a list of all operators that have no user.
    /// @return a list of @ref Op pointers.
    std::list<const Op *> get_leaf_ops() const;

    /// Get a list of all operators that produce inputs or output references
    /// of @p op.
    /// @param op the @ref Op to query.
    /// @return a list of @ref Op pointers.
    std::list<const Op *> get_producer_ops(const Op *op) const;

    /// Get a list of all operators that consume any output tensors of @p op.
    /// @param op the @ref Op to query.
    /// @return a list of @ref Op pointers.
    std::list<const Op *> get_user_ops(const Op *op) const;

    /// Check if there is any cyclic dependency in the model. If so, return
    /// the first cyclic @ref Op.
    /// @return the first cyclic @ref Op if there is any, otherwise nullptr.
    const Op *get_cyclic_op() const;

  protected:
    /// Rank of this model.
    int rank;
    /// Number of assigned EIDs.
    int next_eid = 0;

    friend class Model;

  private:
    /// Append a postfix to a name to make it unique.
    /// @param name the name to append postfix.
    /// @return the name with postfix.
    std::string append_name_postfix(const std::string &name);

    /// Stores all tensor buffers.
    std::list<std::unique_ptr<TensorBuf>> tns_bufs_storage;
    /// Stores all tensors.
    std::list<std::unique_ptr<Tensor>> tns_storage;
    /// Stores all Ops.
    std::list<std::unique_ptr<Op>> ops_storage;
    /// Maps a tensor to its producer Op.
    std::map<Tensor *, Op *> tns_to_producer;
    /// Maps a tensor to its user Ops.
    std::map<Tensor *, std::set<Op *>> tns_to_users;
    /// Count the number of tensors requested the same name.
    std::map<std::string, int> name_cnts;
};

} // namespace ark

#endif // ARK_MODEL_H_
