#include "ark/logging.h"
#include "ark/math.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::matmul(Tensor *mat_a, Tensor *mat_b, Tensor *mat_y,
                      DimType splitk, bool trans_a, bool trans_b, bool is_relu,
                      const string &name, int gran_lev)
{
    assert(mat_a != nullptr);
    assert(mat_b != nullptr);
    LOG(DEBUG, "matmul ", mat_a->shape, " ", mat_b->shape, " ", mat_a->ldims,
        " ", mat_b->ldims, " ", splitk);
    assert(splitk >= 1);
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
    // TODO: support 4 or larger dimensions.
    if (ndims_a > 3 || ndims_b > 3) {
        LOGERR("unsupported input dimensions: ", ndims_a, " and ", ndims_b);
    }
    DimType bs;
    if (ndims_a == 3 && ndims_b == 3) {
        if ((shp_a[0] != shp_b[0]) && (shp_a[0] != 1) && (shp_b[0] != 1)) {
            LOGERR("batch size mismatch: ", shp_a[0], " and ", shp_b[0]);
        }
        bs = max(shp_a[0], shp_b[0]);
    } else if (ndims_a == 3) {
        bs = shp_a[0];
    } else if (ndims_b == 3) {
        bs = shp_b[0];
    } else {
        bs = 1;
    }
    // Create an output Tensor.
    if (mat_y == nullptr) {
        if (max(ndims_a, ndims_b) == 3) {
            mat_y = this->tensor({bs, m, n}, mat_a->type);
        } else {
            mat_y = this->tensor({m, n}, mat_a->type);
        }
    }
    // TODO: change matmul interface to receive `spu` value instead of `splitk`.
    DimType spu = math::pad(math::div_up(k, splitk), 32);
    splitk = math::div_up(k, spu);
    if (splitk == 1) {
        this->create_op(OP_MATMUL, pt, {mat_a, mat_b}, {mat_y},
                        {trans_a, trans_b, is_relu}, name, gran_lev);
        return mat_y;
    } else if (splitk > k) {
        LOGERR("Split-K given larger than the K dimension size.");
    }
    // Split the inner dimension.
    Tensor *output_buffer = this->tensor({bs * splitk, m, n}, mat_a->type);
    vector<Tensor *> mat_y_shards =
        this->sharding(output_buffer, 0, bs, name + "/sharding_mat_y");
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

    assert(mat_y_shards.size() == (size_t)splitk);
    assert(mat_a_shards.size() == (size_t)splitk);
    assert(mat_b_shards.size() == (size_t)splitk);

    for (DimType i = 0; i < splitk; ++i) {
        this->matmul(mat_a_shards[i], mat_b_shards[i], mat_y_shards[i], 1,
                     trans_a, trans_b, false,
                     name + "/matmul_shard_" + to_string(i), gran_lev);
    }
    // Reduce after all outputs are ready.
    Tensor *ref = this->identity(output_buffer, mat_y_shards, nullptr,
                                 name + "/identity");
    return this->reduce(ref, 0, mat_y, is_relu, name + "/reduce");
}

} // namespace ark
