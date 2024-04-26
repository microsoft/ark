// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_matmul.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpMatmul::ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                             ModelTensorRef output, bool trans_input,
                             bool trans_other)
    : ModelOp("Matmul") {
    // Shape verification.
    const Dims &shp_a = input->shape();
    const Dims &shp_b = other->shape();
    int ndims_a = shp_a.ndims();
    int ndims_b = shp_b.ndims();

    if (ndims_a < 1) {
        ERR(InvalidUsageError, "input has an empty shape: ", shp_a);
    }
    if (ndims_b < 1) {
        ERR(InvalidUsageError, "other has an empty shape: ", shp_b);
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
    if (trans_input) {
        DimType tmp = m;
        m = k;
        k = tmp;
    }
    n = (ndims_b == 1) ? 1 : shp_b[ndims_b - 1];
    k2 = (ndims_b == 1) ? shp_b[0] : shp_b[ndims_b - 2];
    if (trans_other) {
        DimType tmp = n;
        n = k2;
        k2 = tmp;
    }
    if (k != k2) {
        ERR(InvalidUsageError, "inner dimensions mismatch: ", k, " and ", k2);
    }

    check_match_data_type(input, other);
    if (output) {
        check_match_data_type(input, output);
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
        ERR(InvalidUsageError, "N dimension mismatch: ", nca[0], " and ",
            ncb[0]);
    }
    if (nca[1] != ncb[1] && nca[1] != 1 && ncb[1] != 1) {
        ERR(InvalidUsageError, "C dimension mismatch: ", nca[1], " and ",
            ncb[1]);
    }

    // N and C dimension of output matrix
    Dims ncc{std::max(nca[0], ncb[0]), std::max(nca[1], ncb[1])};

    Dims output_shape;
    if (std::max(ndims_a, ndims_b) == 4) {
        output_shape = Dims{ncc[0], ncc[1], m, n};
    } else if (std::max(ndims_a, ndims_b) == 3) {
        output_shape = Dims{ncc[1], m, n};
    } else {
        output_shape = Dims{m, n};
    }

    // Create an output Tensor.
    if (output) {
        check_shape(output, output_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    const Dims &strides_a = input->strides();
    const Dims &strides_b = other->strides();
    const Dims &strides_y = output->strides();
    // NOTE: `strides_mnk` here is just an expected value. We can
    // calculate the exact value only after a specific implementation is
    // determined.
    Dims strides_mnk{
        trans_input ? strides_a[ndims_a - 2] : strides_a[ndims_a - 1],
        strides_y[strides_y.ndims() - 1], strides_y[strides_y.ndims() - 1],
        trans_other ? strides_b[ndims_b - 2] : strides_b[ndims_b - 1]};

    // a.k.a. problem size
    Dims shapes_mnk{m, n, k};

    read_tensors_ = {input, other};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["InputDimNC"] = nca;
    args_["OtherDimNC"] = ncb;
    args_["ShapesMNK"] = shapes_mnk;
    args_["StridesMNK"] = strides_mnk;
    args_["IsInputColumnMajor"] = trans_input;
    args_["IsOtherColumnMajor"] = trans_other;

    verify();
}

ModelTensorRef Model::matmul(ModelTensorRef input, ModelTensorRef other,
                             ModelTensorRef output, bool trans_input,
                             bool trans_other, const std::string &name) {
    return impl_
        ->create_op<ModelOpMatmul>(name, input, other, output, trans_input,
                                   trans_other)
        ->result_tensors()[0];
}

}  // namespace ark
