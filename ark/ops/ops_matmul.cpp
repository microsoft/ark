// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_matmul.hpp"
#include "ops_copy.hpp"
#include "../model/model_tensor.hpp"

#include <utility>

#include "ops_common.hpp"
#include "utils/utils_math.hpp"

namespace ark {

static Dims calc_problem_size(const Dims &input_shape, const Dims &other_shape,
                              bool trans_input, bool trans_other) {
    int input_ndims = input_shape.ndims();
    int other_ndims = other_shape.ndims();

    if (input_ndims < 1 || other_ndims < 1) ERR(InternalError, "unexpected");

    DimType m;
    DimType n;
    DimType k;
    DimType k2;

    m = (input_ndims == 1) ? 1 : input_shape[-2];
    n = (other_ndims == 1) ? 1 : other_shape[-1];
    k = input_shape[-1];
    k2 = (other_ndims == 1) ? other_shape[0] : other_shape[-2];

    if (trans_input) {
        // Input is column-major
        std::swap(m, k);
    }
    if (trans_other) {
        // Other is column-major
        std::swap(n, k2);
    }
    if (k != k2) {
        ERR(ModelError, "padded inner dimensions mismatch: ", k, " and ", k2);
    }
    return {m, n, k};
}

static Dims calc_output_shape(const Dims &input_shape, const Dims &other_shape,
                              bool trans_input, bool trans_other) {
    // For m, n
    Dims mnk =
        calc_problem_size(input_shape, other_shape, trans_input, trans_other);
    int max_ndims = std::max(input_shape.ndims(), other_shape.ndims());
    if (max_ndims < 3) {
        return {mnk[0], mnk[1]};
    }
    // Considering 4-dimensional matrix multiplication between [N,C,H,W] format
    // tensors, `*_dim_nc` represents the [N,C] value according to the tensor
    // shape. If the tensor is 3-dimensional ([C,H,W]), N is set to 1.
    // If the tensor is 2-dimensional ([H,W]), both N and C are set to 1.
    Dims input_shape_dims4 = input_shape.dims4();
    Dims other_shape_dims4 = other_shape.dims4();
    Dims input_dim_nc{input_shape_dims4[0], input_shape_dims4[1]};
    Dims other_dim_nc{other_shape_dims4[0], other_shape_dims4[1]};
    // Broadcasted output
    Dims output_dim_nc = broadcast_shape(input_dim_nc, other_dim_nc);
    Dims output_shape;
    if (max_ndims == 4) {
        output_shape = {output_dim_nc[0], output_dim_nc[1], mnk[0], mnk[1]};
    } else {  // max_ndims == 3
        output_shape = {output_dim_nc[1], mnk[0], mnk[1]};
    }
    return output_shape;
}

ModelOpMatmul::ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                             ModelTensorRef output, bool trans_input,
                             bool trans_other)
    : ModelOp("Matmul") {
    Dims output_shape = calc_output_shape(input->shape(), other->shape(),
                                          trans_input, trans_other);
    Dims padded_output_shape = calc_output_shape(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);
    // Create an output Tensor.
    if (output) {
        check_match_shape(output, output_shape);
        check_match_padded_shape(output, padded_output_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape,
            Dims{}, Dims{}, padded_output_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input, other};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["TransposeInput"] = trans_input;
    args_["TransposeOther"] = trans_other;

    verify();
}

std::string ModelOpMatmul::impl_name(const Json &config) const {
    check_fields_config(config, {"NumWarps", "SramBytes", "Tile"});
    check_fields_args(args_, {"TransposeInput", "TransposeOther"});

    bool trans_input = args_.at("TransposeInput").value<bool>();
    bool trans_other = args_.at("TransposeOther").value<bool>();

    const auto &input = read_tensors_[0];
    const auto &other = read_tensors_[1];
    const auto &output = result_tensors_[0];

    check_match_data_type(input, other);
    check_match_data_type(input, output);

    Dims padded_problem_size = calc_problem_size(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);

    Dims output_shape = calc_output_shape(input->shape(), other->shape(),
                                          trans_input, trans_other);
    Dims padded_output_shape = calc_output_shape(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);
    check_match_shape(output, output_shape);
    check_match_padded_shape(output, padded_output_shape);

    Dims input_shape_dims4 = input->shape().dims4();
    Dims other_shape_dims4 = other->shape().dims4();
    Dims input_dim_nc{input_shape_dims4[0], input_shape_dims4[1]};
    Dims other_dim_nc{other_shape_dims4[0], other_shape_dims4[1]};
    Dims output_dim_nc = broadcast_shape(input_dim_nc, other_dim_nc);

    Dims strides_acdb{
        input->strides().dims4()[-1], output->strides().dims4()[-1],
        output->strides().dims4()[-1], other->strides().dims4()[-1]};

    int num_warps = config["NumWarps"];
    int smem_bytes = config["SramBytes"];
    Dims tile_shape = config["Tile"].get<std::vector<DimType>>();
    if (tile_shape.ndims() != 2) {
        ERR(PlanError, "Tile should have 2 elements");
    }
    // CUTLASS handles non-aligned shapes via boundary masking.
    // No divisibility check needed — the planner uses ceil-div for NumTasks.

    DimType inner_stride_a;
    DimType inner_stride_b;
    if (trans_input) {
        inner_stride_a = input->strides().dims4()[-2];
    } else {
        inner_stride_a = input->strides().dims4()[-1];
    }
    if (trans_other) {
        inner_stride_b = other->strides().dims4()[-1];
    } else {
        inner_stride_b = other->strides().dims4()[-2];
    }

    DimType size_a = inner_stride_a * output->strides()[-2];
    DimType size_b = inner_stride_b * output->strides()[-1];
    DimType size_c = output->strides()[-2] * output->strides()[-1];
    DimType batch_stride_c_a = input_dim_nc[1] == 1 ? 0 : size_a;
    DimType batch_stride_n_a =
        input_dim_nc[0] == 1 ? 0 : size_a * input_dim_nc[1];
    DimType batch_stride_c_b = other_dim_nc[1] == 1 ? 0 : size_b;
    DimType batch_stride_n_b =
        other_dim_nc[0] == 1 ? 0 : size_b * other_dim_nc[1];
    DimType batch_stride_c_c = output_dim_nc[1] == 1 ? 0 : size_c;
    DimType batch_stride_n_c =
        output_dim_nc[0] == 1 ? 0 : size_c * output_dim_nc[1];
    if (config.contains("BatchStrideNA")) {
        batch_stride_n_a = config["BatchStrideNA"].get<DimType>();
    }
    if (config.contains("BatchStrideNB")) {
        batch_stride_n_b = config["BatchStrideNB"].get<DimType>();
    }
    if (config.contains("BatchStrideNC")) {
        batch_stride_n_c = config["BatchStrideNC"].get<DimType>();
    }
    if (config.contains("BatchStrideCA")) {
        batch_stride_c_a = config["BatchStrideCA"].get<DimType>();
    }
    if (config.contains("BatchStrideCB")) {
        batch_stride_c_b = config["BatchStrideCB"].get<DimType>();
    }
    if (config.contains("BatchStrideCC")) {
        batch_stride_c_c = config["BatchStrideCC"].get<DimType>();
    }

    return function_name_string("matmul",
                                {
                                    vec_string(output->strides().dims4()),
                                    vec_string(input_dim_nc),
                                    vec_string(other_dim_nc),
                                    vec_string(tile_shape),
                                    vec_string(padded_problem_size),
                                    vec_string(strides_acdb),
                                    std::to_string(batch_stride_n_a),
                                    std::to_string(batch_stride_c_a),
                                    std::to_string(batch_stride_n_b),
                                    std::to_string(batch_stride_c_b),
                                    std::to_string(batch_stride_n_c),
                                    std::to_string(batch_stride_c_c),
                                    std::to_string(trans_input),
                                    std::to_string(trans_other),
                                    std::to_string(num_warps),
                                    std::to_string(smem_bytes),
                                });
}

std::vector<ModelOpArg> ModelOpMatmul::impl_args(
    [[maybe_unused]] const Json &config) const {
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1]};
}

// Compute CUTLASS shared memory requirement for a given tile.
// For CUTLASS 2.x MmaMultistage with 3 pipeline stages:
//   SramBytes = 3 * (TileM * TileK + TileK * TileN) * sizeof(dtype)
// where TileK = 64 for fp16/bf16, 32 for fp32.
static size_t compute_sram_bytes(DimType tm, DimType tn,
                                 const ModelDataType &data_type) {
    size_t tile_k = (data_type == FP32.ref()) ? 32 : 64;
    size_t dtype_bytes = (data_type == FP32.ref()) ? 4 : 2;
    return 3 * (tm * tile_k + tile_k * tn) * dtype_bytes;
}

// Select matmul tile that fits the problem dimensions and maximizes tasks.
// The tile must divide M and N evenly. We try from smallest to largest
// to maximize the number of tiles (= tasks = SMs used).
static const Json select_tile_config(const ArchRef arch,
                                     const ModelDataType &data_type,
                                     const Dims &mnk) {
    DimType M = mnk[0], N = mnk[1];
    // Candidate tiles: {TileM, TileN, NumWarps}
    // Ordered from smallest to largest. For each, M%TileM==0 and N%TileN==0 required.
    struct TileConfig { DimType tm; DimType tn; int nw; };
    // Only tiles validated to compile with CUTLASS 2.x epilogue.
    // [32,*] tiles fail due to epilogue OutputTileOptimalThreadMap zero-size.
    static const TileConfig candidates[] = {
        {64, 64, 4},
        {64, 128, 4},
        {128, 64, 4},
        {128, 128, 8},
        {128, 256, 8},
        {256, 128, 8},
    };
    // Find the best tile: prefer larger tiles (more compute per tile, better
    // pipeline amortization) but fall back to smaller tiles when there aren't
    // enough tasks to keep at least 4 SMs busy.
    int best = -1;
    size_t best_tasks = 0;
    size_t best_tile_area = 0;
    for (int i = 0; i < (int)(sizeof(candidates) / sizeof(candidates[0])); i++) {
        auto &c = candidates[i];
        if (M % c.tm == 0 && N % c.tn == 0) {
            size_t tasks = (M / c.tm) * (N / c.tn);
            size_t tile_area = c.tm * c.tn;
            bool pick = (best == -1);
            if (!pick && best_tasks < 4 && tasks > best_tasks) pick = true;
            if (!pick && tasks >= 4 && tile_area > best_tile_area) pick = true;
            if (pick) { best = i; best_tasks = tasks; best_tile_area = tile_area; }
        }
    }
    if (best == -1) {
        // No tile divides evenly. Pick the smallest tile; CUTLASS handles
        // boundary tiles via predicated loads/stores (ceil-div in planner).
        auto &c = candidates[0];  // [64, 64, 4]
        return {{"NumWarps", c.nw},
                {"SramBytes", (int)compute_sram_bytes(c.tm, c.tn, data_type)},
                {"Tile", {c.tm, c.tn}}};
    }
    auto &c = candidates[best];
    return {{"NumWarps", c.nw},
            {"SramBytes", (int)compute_sram_bytes(c.tm, c.tn, data_type)},
            {"Tile", {c.tm, c.tn}}};
}

static const Json get_default_config(const ArchRef arch,
                                     const ModelDataType &data_type,
                                     const Dims &mnk) {
    if (data_type != FP32.ref() && data_type != FP16.ref() &&
        data_type != BF16.ref()) {
        ERR(PlanError, "Unsupported data type: ", data_type->type_name());
    }
    if (!arch->belongs_to(ARCH_CUDA) && !arch->belongs_to(ARCH_ROCM)) {
        ERR(PlanError, "Unsupported architecture: ", arch->name());
    }
    if (arch->belongs_to(ARCH_CUDA)) {
        return select_tile_config(arch, data_type, mnk);
    }
    // ROCm: keep original behavior
    DimType tm = (mnk[0] > mnk[1]) ? 256 : 128;
    DimType tn = (mnk[0] > mnk[1]) ? 128 : 256;
    if (arch->belongs_to(ARCH_ROCM_942) && data_type == FP32.ref()) {
        return {{"NumWarps", 4}, {"SramBytes", 24672}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == FP16.ref()) {
        return {{"NumWarps", 4}, {"SramBytes", 24672}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == BF16.ref()) {
        return {{"NumWarps", 4}, {"SramBytes", 24624}, {"Tile", {tm, tn}}};
    }
    ERR(InternalError, "Unexpected error");
    return {};
}

Json ModelOpMatmul::default_config(const ArchRef arch) const {
    auto result = result_tensors_[0];
    check_fields_args(args_, {"TransposeInput", "TransposeOther"});
    Dims mnk = calc_problem_size(read_tensors_[0]->padded_shape(),
                                 read_tensors_[1]->padded_shape(),
                                 args_.at("TransposeInput").value<bool>(),
                                 args_.at("TransposeOther").value<bool>());
    return get_default_config(arch, result->data_type(), mnk);
}

Tensor Model::matmul(Tensor input, Tensor other, Tensor output,
                     bool trans_input, bool trans_other,
                     const std::string &name) {
    return impl_
        ->create_op<ModelOpMatmul>(name, input.ref(), other.ref(), output.ref(),
                                   trans_input, trans_other)
        ->result_tensors()[0];
}

ModelOpMatmulGelu::ModelOpMatmulGelu(ModelTensorRef input, ModelTensorRef other,
                                     ModelTensorRef output, bool trans_input,
                                     bool trans_other)
    : ModelOpMatmul(input, other, output, trans_input, trans_other) {
    type_ = ModelOpT::from_name("MatmulGelu");
}

std::string ModelOpMatmulGelu::impl_name(const Json &config) const {
    // Reuse the parent impl_name but replace "matmul" with "matmul_gelu"
    std::string name = ModelOpMatmul::impl_name(config);
    // The name starts with "matmul<" — replace prefix
    if (name.substr(0, 7) == "matmul<") {
        name = "matmul_gelu<" + name.substr(7);
    }
    return name;
}

Tensor Model::matmul_gelu(Tensor input, Tensor other, Tensor output,
                          bool trans_input, bool trans_other,
                          const std::string &name) {
    return impl_
        ->create_op<ModelOpMatmulGelu>(name, input.ref(), other.ref(),
                                       output.ref(), trans_input, trans_other)
        ->result_tensors()[0];
}

// ---- MatmulScale: matmul with register-level scale fusion ----

ModelOpMatmulScale::ModelOpMatmulScale(ModelTensorRef input, ModelTensorRef other,
                                       ModelTensorRef output, bool trans_input,
                                       bool trans_other, float scale)
    : ModelOpMatmul(input, other, output, trans_input, trans_other) {
    type_ = ModelOpT::from_name("MatmulScale");
    args_["Scale"] = scale;
}

std::string ModelOpMatmulScale::impl_name(const Json &config) const {
    // Reuse parent impl_name but replace "matmul" with "matmul_scale"
    // and append scale factor as a template parameter
    std::string name = ModelOpMatmul::impl_name(config);
    float scale = args_.at("Scale").value<float>();
    if (name.substr(0, 7) == "matmul<") {
        // Insert scale parameter: matmul_scale<..., ScaleBits>
        // Encode scale as integer bits for template parameter
        union { float f; uint32_t u; } conv;
        conv.f = scale;
        name = "matmul_scale<" + name.substr(7);
        // Remove trailing ">" and add scale bits
        name = name.substr(0, name.size() - 1) + ", " + std::to_string(conv.u) + ">";
    }
    return name;
}

Tensor Model::matmul_scale(Tensor input, Tensor other, float scale,
                           Tensor output, bool trans_input, bool trans_other,
                           const std::string &name) {
    return impl_
        ->create_op<ModelOpMatmulScale>(name, input.ref(), other.ref(),
                                        output.ref(), trans_input, trans_other,
                                        scale)
        ->result_tensors()[0];
}

ModelOpMatmulAdd::ModelOpMatmulAdd(ModelTensorRef input, ModelTensorRef other,
                                   ModelTensorRef residual,
                                   ModelTensorRef output, bool trans_input,
                                   bool trans_other)
    : ModelOp("MatmulAdd") {
    Dims output_shape = calc_output_shape(input->shape(), other->shape(),
                                          trans_input, trans_other);
    Dims padded_output_shape = calc_output_shape(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);
    if (output) {
        check_match_shape(output, output_shape);
        check_match_padded_shape(output, padded_output_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape,
            Dims{}, Dims{}, padded_output_shape);
    }
    // Residual must match output shape
    check_match_shape(residual, output_shape);
    check_match_padded_shape(residual, padded_output_shape);

    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input, other, residual};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["TransposeInput"] = trans_input;
    args_["TransposeOther"] = trans_other;

    verify();
}

std::string ModelOpMatmulAdd::impl_name(const Json &config) const {
    check_fields_config(config, {"NumWarps", "SramBytes", "Tile"});
    check_fields_args(args_, {"TransposeInput", "TransposeOther"});

    bool trans_input = args_.at("TransposeInput").value<bool>();
    bool trans_other = args_.at("TransposeOther").value<bool>();

    const auto &input = read_tensors_[0];
    const auto &other = read_tensors_[1];
    const auto &output = result_tensors_[0];

    Dims padded_problem_size = calc_problem_size(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);
    Dims output_shape = calc_output_shape(input->shape(), other->shape(),
                                          trans_input, trans_other);
    Dims padded_output_shape = calc_output_shape(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);

    Dims input_shape_dims4 = input->shape().dims4();
    Dims other_shape_dims4 = other->shape().dims4();
    Dims input_dim_nc{input_shape_dims4[0], input_shape_dims4[1]};
    Dims other_dim_nc{other_shape_dims4[0], other_shape_dims4[1]};
    Dims output_dim_nc = broadcast_shape(input_dim_nc, other_dim_nc);

    Dims strides_acdb{
        input->strides().dims4()[-1], output->strides().dims4()[-1],
        output->strides().dims4()[-1], other->strides().dims4()[-1]};

    int num_warps = config["NumWarps"];
    int smem_bytes = config["SramBytes"];
    Dims tile_shape = config["Tile"].get<std::vector<DimType>>();

    DimType inner_stride_a;
    DimType inner_stride_b;
    if (trans_input) {
        inner_stride_a = input->strides().dims4()[-2];
    } else {
        inner_stride_a = input->strides().dims4()[-1];
    }
    if (trans_other) {
        inner_stride_b = other->strides().dims4()[-1];
    } else {
        inner_stride_b = other->strides().dims4()[-2];
    }

    DimType size_a = inner_stride_a * output->strides()[-2];
    DimType size_b = inner_stride_b * output->strides()[-1];
    DimType size_c = output->strides()[-2] * output->strides()[-1];
    DimType batch_stride_c_a = input_dim_nc[1] == 1 ? 0 : size_a;
    DimType batch_stride_n_a =
        input_dim_nc[0] == 1 ? 0 : size_a * input_dim_nc[1];
    DimType batch_stride_c_b = other_dim_nc[1] == 1 ? 0 : size_b;
    DimType batch_stride_n_b =
        other_dim_nc[0] == 1 ? 0 : size_b * other_dim_nc[1];
    DimType batch_stride_c_c = output_dim_nc[1] == 1 ? 0 : size_c;
    DimType batch_stride_n_c =
        output_dim_nc[0] == 1 ? 0 : size_c * output_dim_nc[1];

    return function_name_string("matmul_add",
                                {
                                    vec_string(output->strides().dims4()),
                                    vec_string(input_dim_nc),
                                    vec_string(other_dim_nc),
                                    vec_string(tile_shape),
                                    vec_string(padded_problem_size),
                                    vec_string(strides_acdb),
                                    std::to_string(batch_stride_n_a),
                                    std::to_string(batch_stride_c_a),
                                    std::to_string(batch_stride_n_b),
                                    std::to_string(batch_stride_c_b),
                                    std::to_string(batch_stride_n_c),
                                    std::to_string(batch_stride_c_c),
                                    std::to_string(trans_input),
                                    std::to_string(trans_other),
                                    std::to_string(num_warps),
                                    std::to_string(smem_bytes),
                                });
}

std::vector<ModelOpArg> ModelOpMatmulAdd::impl_args(
    [[maybe_unused]] const Json &config) const {
    // Args: output, input A, input B, residual
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1],
            read_tensors_[2]};
}

Json ModelOpMatmulAdd::default_config(const ArchRef arch) const {
    check_fields_args(args_, {"TransposeInput", "TransposeOther"});
    Dims mnk = calc_problem_size(read_tensors_[0]->padded_shape(),
                                 read_tensors_[1]->padded_shape(),
                                 args_.at("TransposeInput").value<bool>(),
                                 args_.at("TransposeOther").value<bool>());
    return get_default_config(arch, result_tensors_[0]->data_type(), mnk);
}

Tensor Model::matmul_add(Tensor input, Tensor other, Tensor residual,
                         Tensor output, bool trans_input, bool trans_other,
                         const std::string &name) {
    return impl_
        ->create_op<ModelOpMatmulAdd>(name, input.ref(), other.ref(),
                                      residual.ref(), output.ref(),
                                      trans_input, trans_other)
        ->result_tensors()[0];
}

Tensor Model::mma(Tensor input, Tensor other, Tensor output,
                     bool trans_input, bool trans_other,
                     const std::string &name) {
    return impl_
        ->create_op<ModelOpMma>(name, input.ref(), other.ref(), output.ref(),
                                trans_input, trans_other)
        ->result_tensors()[0];
}

// ---- Mma: matmul with REGISTER output tensor ----

ModelOpMma::ModelOpMma(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output, bool trans_input,
                       bool trans_other)
    : ModelOpMatmul(input, other, output, trans_input, trans_other) {
    type_ = ModelOpT::from_name("Mma");
    // Tag the output tensor as REGISTER location
    for (auto &t : result_tensors_) {
        t->set_location(TensorLocation::REGISTER);
    }
    for (auto &t : write_tensors_) {
        t->set_location(TensorLocation::REGISTER);
    }
}

std::string ModelOpMma::impl_name(const Json &config) const {
    // Currently uses the same kernel as matmul.
    // When codegen supports REGISTER tensors, this will generate
    // MMA-only code (no epilogue store).
    return ModelOpMatmul::impl_name(config);
}


Tensor Model::store(Tensor output, Tensor input, const std::string &name) {
    return impl_
        ->create_op<ModelOpStore>(name, input.ref(), output.ref())
        ->result_tensors()[0];
}

// ---- Store: write register tensor to global memory ----

ModelOpStore::ModelOpStore(ModelTensorRef input, ModelTensorRef output)
    : ModelOpCopy(input, output) {
    type_ = ModelOpT::from_name("Store");
    // Ensure output is GLOBAL location
    for (auto &t : result_tensors_) {
        t->set_location(TensorLocation::GLOBAL);
    }
    for (auto &t : write_tensors_) {
        t->set_location(TensorLocation::GLOBAL);
    }
}

std::string ModelOpStore::impl_name(const Json &config) const {
    // Use "copy" kernel, not "store" (clashes with ark::store in load_store.h).
    // When codegen handles REGISTER tensors, this will be replaced with
    // epilogue-only code generation.
    check_fields_config(config, {"NumWarps", "Tile"});
    int num_warps = config.at("NumWarps");
    Dims unit_out_dims(config.at("Tile").get<std::vector<DimType>>());
    return function_name_string(
        "copy",
        {vec_string(read_tensors_[0]->strides().dims4()),
         vec_string(read_tensors_[0]->shape().dims4()),
         vec_string(write_tensors_[0]->strides().dims4()),
         vec_string(write_tensors_[0]->shape().dims4()),
         vec_string(unit_out_dims.dims4()),
         std::to_string(num_warps),
         "0"});
}



}  // namespace ark
