// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "math.h"
#include "model.h"

using namespace std;

namespace ark {

extern const OpConfigMap MatmulConfigMap;

MatmulOp::MatmulOp(const std::string &prec_type, Tensor *mat_a, Tensor *mat_b,
                   Tensor *mat_y, Dims nca, Dims ncb, Dims problem_size,
                   Dims leading_dims, bool is_column_a, bool is_column_b,
                   const string &name, int gran_lev)
    : Op{OP_MATMUL,
         prec_type,
         {mat_a, mat_b},
         {mat_y},
         {{nca, ncb, problem_size, leading_dims, is_column_a, is_column_b}},
         name,
         &MatmulConfigMap,
         gran_lev} {}

std::string MatmulOp::function_name(const OpConfig &cfg) const {
    Tensor *mat_a = this->inputs[0];
    Tensor *mat_b = this->inputs[1];
    Tensor *mat_y = this->outputs[0];

    int ndims_y = mat_y->shape.ndims();
    OpTile tile_out = cfg.output_tiles[0];
    if (tile_out.x < 0) tile_out.x = mat_y->ldims.dims4()[2];
    if (tile_out.y < 0) tile_out.y = mat_y->ldims.dims4()[3];
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
    this->args.get(&nca, 0);
    this->args.get(&ncb, 1);
    this->args.get(&problem_size, 2);
    this->args.get(&leading_dims, 3);
    this->args.get(&is_column_a, 4);
    this->args.get(&is_column_b, 5);

    /// Re-calculate the exact leading dimensions. Assume this function is
    /// called after scheduling is done.

    const Dims &ldims_a = mat_a->ldims;
    const Dims &ldims_b = mat_b->ldims;
    const Dims &ldims_y = mat_y->ldims;
    int ndims_a = ldims_a.ndims();
    int ndims_b = ldims_b.ndims();
    leading_dims[0] = ldims_a[ndims_a - 1];
    leading_dims[1] = ldims_y[ldims_y.ndims() - 1];
    leading_dims[2] = ldims_y[ldims_y.ndims() - 1];
    leading_dims[3] = ldims_b[ndims_b - 1];

    DimType in_ldim_a = ldims_a[ndims_a - 1];
    DimType in_ldim_b = ldims_b[ndims_b - 2];

    // TODO: verify `leading_dims`

    const OpTile &tile_in0 = cfg.input_tiles[0];
    const OpTile &tile_in1 = cfg.input_tiles[1];
    CHECK(tile_in0.y == tile_in1.x);
    Dims shape{tile_out.x, tile_out.y, tile_in0.y};

    return Op::function_name("ark::matmul",
                             {{
                                 mat_y->ldims.dims4(),  // OutDims
                                 nca,                   // NCA
                                 ncb,                   // NCB
                                 shape,                 // Shape
                                 problem_size,          // ProblemSize
                                 leading_dims,          // LeadingDims
                                 in_ldim_a,             // InnerLdimA
                                 in_ldim_b,             // InnerLdimB
                                 is_column_a,           // IsColumnA
                                 is_column_b,           // IsColumnB
                                 cfg.num_warps * 32,    // NumThreads
                                 cfg.smem_bytes,        // SmemBytes
                             }});
}

Tensor *Model::matmul(Tensor *mat_a, Tensor *mat_b, Tensor *mat_y,
                      DimType split_k, bool trans_a, bool trans_b,
                      const string &name, int gran_lev) {
    CHECK(mat_a != nullptr);
    CHECK(mat_b != nullptr);
    CHECK(split_k >= 1);

    // Shape verification.
    const Dims &shp_a = mat_a->shape;
    const Dims &shp_b = mat_b->shape;
    int ndims_a = shp_a.ndims();
    int ndims_b = shp_b.ndims();

    if (ndims_a < 1) {
        LOG(ERROR, "mat_a has an empty shape: ", shp_a);
    }
    if (ndims_b < 1) {
        LOG(ERROR, "mat_b has an empty shape: ", shp_b);
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
        LOG(ERROR, "inner dimensions mismatch: ", k, " and ", k2);
    }

    if (mat_a->type != mat_b->type) {
        LOG(ERROR, "input data types mismatch: ", mat_a->type, ", ",
            mat_b->type);
    }
    if (mat_y != nullptr && mat_a->type != mat_b->type) {
        LOG(ERROR, "invalid output data type: ", mat_y->type);
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
        LOG(ERROR, "N dimension mismatch: ", nca[0], " and ", ncb[0]);
    }
    if (nca[1] != ncb[1] && nca[1] != 1 && ncb[1] != 1) {
        LOG(ERROR, "C dimension mismatch: ", nca[1], " and ", ncb[1]);
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
            LOG(ERROR, "output data type mismatch: ", mat_y->type, " and ",
                mat_a->type);
        }
        if (mat_y->shape != output_shape) {
            LOG(ERROR, "output shape mismatch: ", mat_y->shape, " and ",
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
        // NOTE: `leading_dims` here is just an expected value. We can
        // calculate the exact value after the OpConfig is given in
        // `MatmulOp::function_name()`.
        Dims leading_dims{
            trans_a ? ldims_a[ndims_a - 2] : ldims_a[ndims_a - 1],
            ldims_y[ldims_y.ndims() - 1], ldims_y[ldims_y.ndims() - 1],
            trans_b ? ldims_b[ndims_b - 2] : ldims_b[ndims_b - 1]};
        Dims problem_size{m, n, k};
        MatmulOp op{
            mat_y->type.name(), mat_a,        mat_b,   mat_y,   nca,  ncb,
            problem_size,       leading_dims, trans_a, trans_b, name, gran_lev};
        return this->impl->add_op(op)[0];
    } else if (split_k > k) {
        LOG(ERROR, "Split-K given larger than the K dimension size.");
    }

    // Split the inner dimension.
    Dims split_output_shape = output_shape;
    split_output_shape[0] *= split_k;
    Tensor *output_buffer = this->tensor(split_output_shape, mat_y->type);
    vector<Tensor *> mat_y_shards = this->sharding(
        output_buffer, 0, output_shape[0], name + "/sharding_mat_y");

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

    std::vector<Tensor *> shard_outputs;
    for (DimType i = 0; i < split_k; ++i) {
        Tensor *shard_output = this->matmul(
            mat_a_shards[i], mat_b_shards[i], mat_y_shards[i], 1, trans_a,
            trans_b, name + "/matmul_shard_" + to_string(i), gran_lev);
        shard_outputs.push_back(shard_output);
    }
    // Reduce after all outputs are ready.
    Dims reduce_input_shape{split_k, ncc[0] * ncc[1], m, n};
    Dims reduce_output_shape{1, ncc[0] * ncc[1], m, n};
    Tensor *ref =
        this->identity(output_buffer, shard_outputs, name + "/identity");
    ref = this->reshape(ref, reduce_input_shape);
    mat_y = this->reshape(mat_y, reduce_output_shape);
    Tensor *red = this->reduce_sum(ref, 0, mat_y, name + "/reduce_sum");
    if (red->shape != output_shape) {
        return this->reshape(red, output_shape);
    }
    return red;
}

const OpConfigMap MatmulConfigMap = {
    {{OP_ARCH_CUDA_70, "fp16"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 49152, {{128, 32}, {32, 256}}, {{128, 256}}, true, false},
     }},
    {{OP_ARCH_CUDA_80, "bf16"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 147456, {{128, 64}, {64, 256}}, {{128, 256}}, true, false},
         {4, 98304, {{128, 64}, {64, 128}}, {{128, 128}}, true, false},
         {4, 98304, {{64, 64}, {64, 64}}, {{64, 64}}, true, false},
     }},
    {{OP_ARCH_CUDA_80, "fp16"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 147456, {{128, 64}, {64, 256}}, {{128, 256}}, true, false},
         {4, 98304, {{128, 64}, {64, 128}}, {{128, 128}}, true, false},
         {4, 98304, {{64, 64}, {64, 64}}, {{64, 64}}, true, false},
     }},
    {{OP_ARCH_CUDA_80, "fp32"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 147456, {{128, 32}, {32, 256}}, {{128, 256}}, true, false},
         {4, 98304, {{128, 32}, {32, 128}}, {{128, 128}}, true, false},
         {4, 49152, {{64, 32}, {32, 64}}, {{64, 64}}, true, false},
     }},
};

}  // namespace ark
