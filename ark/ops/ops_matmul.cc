// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "math.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

MatmulOp::MatmulOp(OpPrecType prec_type, Tensor *mat_a, Tensor *mat_b,
                   Tensor *mat_y, Dims nca, Dims ncb, Dims problem_size,
                   Dims leading_dims, bool is_column_a, bool is_column_b,
                   bool is_relu, const string &name, int gran_lev)
    : Op{OP_MATMUL,
         prec_type,
         {mat_a, mat_b},
         {mat_y},
         {{nca, ncb, problem_size, leading_dims, is_column_a, is_column_b,
           is_relu}},
         name,
         gran_lev}
{
}

std::string MatmulOp::function_name(const OpConfig &cfg) const
{
    Tensor *mat_y = this->out_deps[0];

    int ndims_y = mat_y->shape.ndims();
    const OpTile &tile_out = cfg.out_deps_tiles[0];
    CHECK(mat_y->ldims[ndims_y - 1] % tile_out.y == 0);
    if (ndims_y > 1) {
        CHECK(mat_y->ldims[ndims_y - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    Dims nca;
    Dims ncb;
    Dims problem_size;
    Dims leading_dims;
    bool is_column_a;
    bool is_column_b;
    bool is_relu;
    this->args.get(&nca, 0);
    this->args.get(&ncb, 1);
    this->args.get(&problem_size, 2);
    this->args.get(&leading_dims, 3);
    this->args.get(&is_column_a, 4);
    this->args.get(&is_column_b, 5);
    this->args.get(&is_relu, 6);

    const OpTile &tile_in0 = cfg.in_deps_tiles[0];
    const OpTile &tile_in1 = cfg.in_deps_tiles[1];
    CHECK(tile_in0.y == tile_in1.x);
    Dims shape{tile_out.x, tile_out.y, tile_in0.y};

    return Op::function_name("ark::matmul",
                             {{
                                 nca,                // NCA
                                 ncb,                // NCB
                                 shape,              // Shape
                                 problem_size,       // ProblemSize
                                 leading_dims,       // LeadingDims
                                 is_column_a,        // IsColumnA
                                 is_column_b,        // IsColumnB
                                 is_relu,            // IsRelu
                                 cfg.num_warps * 32, // ThreadsNum
                                 cfg.smem_bytes,     // SmemBytes
                             }});
}

Tensor *Model::matmul(Tensor *mat_a, Tensor *mat_b, Tensor *mat_y,
                      DimType split_k, bool trans_a, bool trans_b, bool is_relu,
                      const string &name, int gran_lev)
{
    CHECK(mat_a != nullptr);
    CHECK(mat_b != nullptr);
    CHECK(split_k >= 1);
    LOG(DEBUG, "matmul ", mat_a->shape, " ", mat_b->shape, " ", mat_a->ldims,
        " ", mat_b->ldims, " ", split_k);

    // Shape verification.
    const Dims &shp_a = mat_a->shape;
    const Dims &shp_b = mat_b->shape;
    int ndims_a = shp_a.ndims();
    int ndims_b = shp_b.ndims();

    if (ndims_a < 1) {
        LOGERR("mat_a has an empty shape: ", shp_a);
    }
    if (ndims_b < 1) {
        LOGERR("mat_b has an empty shape: ", shp_b);
    }

    // m: the number of rows of output matrix (row-major)
    // n: the number of columns of output matrix (row-major)
    // k: the inner dimension of matrix multiplication
    DimType m;
    DimType n;
    DimType k;
    DimType k2;

    m = (ndims_a == 1) ? 1 : shp_a[ndims_a - 2];
    k = shp_a[ndims_a - 1];
    if (trans_a) {
        DimType tmp = m;
        m = k;
        k = tmp;
    }
    n = (ndims_b == 1) ? 1 : shp_b[ndims_b - 1];
    k2 = (ndims_b == 1) ? shp_b[0] : shp_b[ndims_b - 2];
    if (trans_b) {
        DimType tmp = n;
        n = k2;
        k2 = tmp;
    }
    if (k != k2) {
        LOGERR("inner dimensions mismatch: ", k, " and ", k2);
    }

    OpPrecType pt;
    if (mat_a->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (mat_a->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(mat_a->type));
    }
    if (mat_a->type != mat_b->type) {
        LOGERR("input data types mismatch: ", type_str(mat_a->type), ", ",
               type_str(mat_b->type));
    }
    if (mat_y != nullptr && mat_a->type != mat_b->type) {
        LOGERR("invalid output data type: ", type_str(mat_y->type));
    }

    // N and C dimensions of matrix A
    Dims nca{1, 1};
    if (ndims_a == 4) {
        nca[0] = shp_a[0];
        nca[1] = shp_a[1];
    } else if (ndims_a == 3) {
        nca[1] = shp_a[0];
    }

    // N and C dimensions of matrix B
    Dims ncb{1, 1};
    if (ndims_b == 4) {
        ncb[0] = shp_b[0];
        ncb[1] = shp_b[1];
    } else if (ndims_b == 3) {
        ncb[1] = shp_b[0];
    }

    // Verify broadcasting
    if (nca[0] != ncb[0] && nca[0] != 1 && ncb[0] != 1) {
        LOGERR("N dimension mismatch: ", nca[0], " and ", ncb[0]);
    }
    if (nca[1] != ncb[1] && nca[1] != 1 && ncb[1] != 1) {
        LOGERR("C dimension mismatch: ", nca[1], " and ", ncb[1]);
    }

    // N and C dimension of output matrix
    Dims ncc{max(nca[0], ncb[0]), max(nca[1], ncb[1])};

    Dims output_shape;
    if (max(ndims_a, ndims_b) == 4) {
        output_shape = Dims{ncc[0], ncc[1], m, n};
    } else if (max(ndims_a, ndims_b) == 3) {
        output_shape = Dims{ncc[1], m, n};
    } else {
        output_shape = Dims{m, n};
    }

    // Create an output Tensor.
    if (mat_y == nullptr) {
        mat_y = this->tensor(output_shape, mat_a->type);
    } else {
        if (mat_y->type != mat_a->type) {
            LOGERR("output data type mismatch: ", type_str(mat_y->type),
                   " and ", type_str(mat_a->type));
        }
        if (mat_y->shape != output_shape) {
            LOGERR("output shape mismatch: ", mat_y->shape, " and ",
                   output_shape);
        }
    }

    // TODO: change matmul interface to receive `spu` value instead of
    // `split_k`.
    DimType spu = math::pad(math::div_up(k, split_k), 32);
    split_k = math::div_up(k, spu);
    if (split_k == 1) {
        const Dims &ldims_a = mat_a->ldims;
        const Dims &ldims_b = mat_b->ldims;
        const Dims &ldims_y = mat_y->ldims;
        Dims problem_size{m, n, k};
        Dims leading_dims{
            trans_a ? ldims_a[ndims_a - 2] : ldims_a[ndims_a - 1],
            ldims_y[ldims_y.ndims() - 1], ldims_y[ldims_y.ndims() - 1],
            trans_b ? ldims_b[ndims_b - 2] : ldims_b[ndims_b - 1]};
        MatmulOp op{pt,      mat_a,        mat_b,        mat_y,   nca,
                    ncb,     problem_size, leading_dims, trans_a, trans_b,
                    is_relu, name,         gran_lev};
        this->impl->add_op(op);
        return mat_y;
    } else if (split_k > k) {
        LOGERR("Split-K given larger than the K dimension size.");
    }

    // Split the inner dimension.
    Tensor *output_buffer;
    vector<Tensor *> mat_y_shards;
    if (mat_y->shape.ndims() == 4) {
        output_buffer =
            this->tensor({ncc[0] * split_k, ncc[1], m, n}, mat_y->type);
        mat_y_shards =
            this->sharding(output_buffer, 0, ncc[0], name + "/sharding_mat_y");
    } else {
        output_buffer = this->tensor({ncc[1] * split_k, m, n}, mat_y->type);
        mat_y_shards =
            this->sharding(output_buffer, 0, ncc[1], name + "/sharding_mat_y");
    }
    for (size_t i = 0; i < mat_y_shards.size(); ++i) {
        Tensor *t = mat_y_shards[i];
        // If the output dimension is not matching, drop the leading 1s.
        if (t->shape.ndims() != output_shape.ndims()) {
            Dims new_shape = t->shape;
            while (new_shape.ndims() != output_shape.ndims()) {
                if (new_shape[0] != 1) {
                    LOGERR("invalid shard shape: ", t->shape);
                }
                new_shape.erase(0);
            }
            mat_y_shards[i] = this->reshape(t, new_shape);
        }
    }

    int axis_a;
    int axis_b;
    if (trans_a) {
        axis_a = (ndims_a == 1) ? (ndims_a - 1) : (ndims_a - 2);
    } else {
        axis_a = ndims_a - 1;
    }
    if (trans_b) {
        axis_b = ndims_b - 1;
    } else {
        axis_b = (ndims_b == 1) ? (ndims_b - 1) : (ndims_b - 2);
    }
    vector<Tensor *> mat_a_shards =
        this->sharding(mat_a, axis_a, spu, name + "/sharding_mat_a");
    vector<Tensor *> mat_b_shards =
        this->sharding(mat_b, axis_b, spu, name + "/sharding_mat_b");

    CHECK(mat_y_shards.size() == (size_t)split_k);
    CHECK(mat_a_shards.size() == (size_t)split_k);
    CHECK(mat_b_shards.size() == (size_t)split_k);

    for (DimType i = 0; i < split_k; ++i) {
        this->matmul(mat_a_shards[i], mat_b_shards[i], mat_y_shards[i], 1,
                     trans_a, trans_b, false,
                     name + "/matmul_shard_" + to_string(i), gran_lev);
    }
    // Reduce after all outputs are ready.
    Tensor *ref = this->identity(output_buffer, mat_y_shards, nullptr,
                                 name + "/identity");
    Tensor *reduced = this->reduce_sum(ref, 0, mat_y, name + "/reduce_sum");
    if (is_relu) {
        // TODO: overwrite
        reduced = this->relu(reduced, nullptr, name + "/relu");
    }
    return reduced;
}

} // namespace ark
