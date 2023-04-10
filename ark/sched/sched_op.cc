// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/sched/sched_op.h"
#include "ark/logging.h"
#include "ark/math.h"

using namespace std;
#define COM ", "

namespace ark {

const OpConfig *sched_op_config(const Op *op, const GpuInfo &gpu_info)
{
    assert(op != nullptr);
    assert(op->out_deps.size() > 0);
    Tensor *output = op->out_deps[0];
    if (output == nullptr) {
        return &ARK_OP_CONFIG_VIRT;
    }
    OpArchType arch_type;
    if (gpu_info.arch == GPU_ARCH_CUDA_70) {
        arch_type = OP_ARCH_CUDA_70;
    } else if (gpu_info.arch == GPU_ARCH_CUDA_80) {
        arch_type = OP_ARCH_CUDA_80;
    } else {
        LOGERR("unsupported GPU architecture: ", gpu_info.arch);
    }
    auto search = ARK_OP_CONFIG_MAP.find({op->type, arch_type, op->prec_type});
    if (search == ARK_OP_CONFIG_MAP.end()) {
        return &ARK_OP_CONFIG_VIRT;
    } else if (op->gran_lev >= 0) {
        if (search->second.size() > (unsigned int)op->gran_lev) {
            return &search->second[op->gran_lev];
        }
        LOGERR("invalid granularity level: ", op->gran_lev);
    }
    // Heuristic auto-selection of granularity level
    int gran_lev = 0;
    int ndims = output->shape.ndims();
    unsigned int min_wps =
        gpu_info.min_threads_per_block / gpu_info.threads_per_warp;
    for (auto &cfg : search->second) {
        assert(cfg.out_deps_tiles.size() > 0);
        const OpTile &ot = cfg.out_deps_tiles[0];
        DimType num_tiles;
        DimType dim_0;
        DimType dim_1;
        if (ndims == 1) {
            if (ot.x != 1) {
                ++gran_lev;
                continue;
            }
            dim_0 = output->shape[0];
            dim_1 = 1;
            num_tiles = math::div_up(dim_0, ot.y);
        } else {
            num_tiles = 1;
            for (int i = 0; i < ndims - 2; ++i) {
                num_tiles *= output->shape[i];
            }
            dim_0 = output->shape[ndims - 1];
            dim_1 = output->shape[ndims - 2];
            num_tiles *= math::div_up(dim_0, ot.y);
            num_tiles *= math::div_up(dim_1, ot.x);
        }
        if (gran_lev == (int)search->second.size() - 1) {
            // no more option, just use the finest-grained config
            break;
        }
        // magic condition
        if ((dim_0 * 2 > ot.y) && (dim_1 * 2 > ot.x) &&
            ((num_tiles * cfg.num_warps) >= (min_wps * gpu_info.num_sm / 2))) {
            break;
        }
        ++gran_lev;
    }
    if (gran_lev == (int)search->second.size()) {
        stringstream configs_str;
        if (search->second.size() > 0) {
            const OpTile &ot = search->second[0].out_deps_tiles[0];
            configs_str << "{ " << ot.x << ", " << ot.y << " }";
        }
        for (int i = 1; i < (int)search->second.size(); ++i) {
            const OpTile &ot = search->second[i].out_deps_tiles[0];
            configs_str << ", { " << ot.x << ", " << ot.y << " }";
        }
        configs_str << ".";
        LOGERR("no valid tile configuration found. Output shape ",
               output->shape, ", available tiles: ", configs_str.str());
    }
    return &search->second[gran_lev];
}

SchedOp::SchedOp(const Op *op_, const OpConfig *cfg_, const string name)
    : op{op_}, cfg{cfg_}, name{name}, tnums{}
{
    if (op_ == nullptr) {
        return;
    }
    if (cfg_ == &ARK_OP_CONFIG_VIRT) {
        LOG(DEBUG, "virtual op: ", op_->name);
        return;
    }
    LOG(DEBUG, "op: ", op_->name, ", cfg: num_warps ", cfg_->num_warps,
        " smem_bytes ", cfg_->smem_bytes, " #in_deps ",
        cfg_->in_deps_tiles.size(), " #out_deps ", cfg_->out_deps_tiles.size(),
        " sync_pre ", cfg_->sync_pre, " sync_post ", cfg_->sync_post);
    // pad the tensor of the SchedOp
    for (unsigned int i = 0; i < this->op->in_deps.size(); ++i) {
        if (i >= this->cfg->in_deps_tiles.size()) {
            LOG(DEBUG, "input tensor can not be all padded");
            break;
        }
        // Update pads based on the tile shape. The tiling is applied to the
        // last two dimensions of the tensor. If the tensor is 1D, the first
        // dimension of the tile shape should be 1.
        auto &tile = this->cfg->in_deps_tiles[i];
        int ndims = this->op->in_deps[i]->ndims();
        vector<DimType> pads;
        if (ndims == 1) {
            if (tile.x != 1) {
                LOGERR("invalid tile shape for 1D tensor: {", tile.x, ", ",
                       tile.y, "}");
            }
            pads.emplace_back(tile.y);
        } else {
            for (int j = 0; j < ndims - 2; ++j) {
                pads.emplace_back(1);
            }
            pads.emplace_back(tile.x);
            pads.emplace_back(tile.y);
        }
        this->op->in_deps[i]->update_pads(pads);
    }
    for (unsigned int i = 0; i < this->op->out_deps.size(); ++i) {
        auto &tile = this->cfg->out_deps_tiles[i];
        int ndims = this->op->out_deps[i]->ndims();
        vector<DimType> pads;
        if (ndims == 1) {
            if (tile.x != 1) {
                LOGERR("invalid tile shape for 1D tensor: {", tile.x, ", ",
                       tile.y, "}");
            }
            pads.emplace_back(tile.y);
        } else {
            for (int j = 0; j < ndims - 2; ++j) {
                pads.emplace_back(1);
            }
            pads.emplace_back(tile.x);
            pads.emplace_back(tile.y);
        }
        this->op->out_deps[i]->update_pads(pads);
    }
    // claculate the tile size for the SchedOp
    if ((this->op->out_deps.size() == 1) &&
        (this->cfg != &ARK_OP_CONFIG_VIRT)) {
        const OpTile &tile = this->cfg->out_deps_tiles[0];
        const Dims &s = this->op->out_deps[0]->shape;
        int ndims = s.ndims();
        vector<DimType> vec;
        if (ndims == 1) {
            vec.emplace_back((DimType)math::div_up(s[0], tile.y));
        } else {
            int i = 0;
            for (; i < ndims - 2; ++i) {
                vec.emplace_back(s[i]);
            }
            vec.emplace_back((DimType)math::div_up(s[i], tile.x));
            vec.emplace_back((DimType)math::div_up(s[i + 1], tile.y));
        }
        this->tnums = Dims{vec};
        LOG(DEBUG, "SchedOp: ", name, " tile num: ", this->tnums,
            " tile size: {", tile.x, ", ", tile.y, "}");
    }
}

const string SchedOp::func_string() const
{
    if (this->cfg == &ARK_OP_CONFIG_VIRT) {
        return "";
    } else if (this->op->type == OP_MATMUL) {
        return this->func_string_matmul();
    } else if (this->op->type == OP_SEND) {
        return this->func_string_send();
    } else if (this->op->type == OP_RECV) {
        return this->func_string_recv();
    } else if (this->op->type == OP_SEND_DONE) {
        return this->func_string_send_done();
    } else if (this->op->type == OP_SEND_MM) {
        return this->func_string_send_mm();
    } else if (this->op->type == OP_RECV_MM) {
        return this->func_string_recv_mm();
    } else if (this->op->type == OP_REDUCE) {
        return this->func_string_reduce();
    } else if (this->op->type == OP_SCALE) {
        return this->func_string_scale();
    } else if (this->op->type == OP_ADD) {
        return this->func_string_add();
    } else if (this->op->type == OP_MUL) {
        return this->func_string_mul();
    } else if (this->op->type == OP_IM2COL) {
        return this->func_string_im2col();
    } else if (this->op->type == OP_TRANSPOSE) {
        return this->func_string_transpose();
    }
    return "";
}

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            LOGERR("failed condition: " #cond);                                \
        }                                                                      \
    } while (0)

const string SchedOp::func_string_matmul() const
{
    CHECK(this->op->in_deps.size() == 2);
    CHECK(this->op->args.size() == 3);
    CHECK(this->op->args[0].type == OP_ARG_BOOL);
    CHECK(this->op->args[1].type == OP_ARG_BOOL);
    CHECK(this->op->args[2].type == OP_ARG_BOOL);

    const Tensor *tns_in_0 = this->op->in_deps[0];
    const Tensor *tns_in_1 = this->op->in_deps[1];
    const Tensor *tns_out = this->op->out_deps[0];

    Dims shp_in_0 = tns_in_0->shape;
    Dims shp_in_1 = tns_in_1->shape;
    Dims shp_out = tns_out->shape;

    bool trans_in_0 = *(bool *)this->op->args[0].val;
    bool trans_in_1 = *(bool *)this->op->args[1].val;
    bool is_relu = *(bool *)this->op->args[2].val;

    int ndims_in_0 = shp_in_0.ndims();
    int ndims_in_1 = shp_in_1.ndims();
    int ndims_out = shp_out.ndims();

    CHECK(ndims_in_0 >= 1);
    CHECK(ndims_in_1 >= 1);
    CHECK(ndims_out >= 2);

    unsigned int m = shp_out[ndims_out - 2];
    unsigned int n = shp_out[ndims_out - 1];
    unsigned int k;

    // Verify core dimensions
    if (trans_in_0) {
        k = (ndims_in_0 == 1) ? 1 : shp_in_0[ndims_in_0 - 2];
        CHECK(m == shp_in_0[ndims_in_0 - 1]);
    } else {
        k = shp_in_0[ndims_in_0 - 1];
        CHECK(m == ((ndims_in_0 == 1) ? 1 : shp_in_0[ndims_in_0 - 2]));
    }
    if (trans_in_1) {
        CHECK(n ==
              ((ndims_in_1 == 1) ? shp_in_1[0] : shp_in_1[ndims_in_1 - 2]));
        CHECK(k == ((ndims_in_1 == 1) ? 1 : shp_in_1[ndims_in_1 - 1]));
    } else {
        CHECK(n == ((ndims_in_1 == 1) ? 1 : shp_in_1[ndims_in_1 - 1]));
        CHECK(k ==
              ((ndims_in_1 == 1) ? shp_in_1[0] : shp_in_1[ndims_in_1 - 2]));
    }

    // Verify broadcasting
    const Dims &shp_in_l = ndims_in_0 > ndims_in_1 ? shp_in_0 : shp_in_1;
    const Dims &shp_in_s = ndims_in_0 <= ndims_in_1 ? shp_in_0 : shp_in_1;
    int ndims_in_l = shp_in_l.ndims();
    int ndims_in_s = shp_in_s.ndims();
    if (ndims_in_l > 2) {
        for (int i = 2; i < ndims_in_s; ++i) {
            DimType dim_l = shp_in_l[ndims_in_l - i];
            DimType dim_s = shp_in_s[ndims_in_s - i];
            CHECK((dim_l == dim_s) || (dim_l == 1) || (dim_s == 1));
        }
    }
    // TODO: support 4 or larger dimensions.
    CHECK(ndims_in_l <= 3);
    unsigned int bs;
    unsigned int bcast;
    if (ndims_in_l <= 2) {
        bs = 1;
        bcast = 0;
    } else {
        if ((ndims_in_0 > 2) && (ndims_in_1 > 2)) {
            if (shp_in_0[0] == shp_in_1[0]) {
                bs = shp_in_0[0];
                bcast = 0;
            } else if (shp_in_0[0] > shp_in_1[0]) {
                bs = shp_in_0[0];
                bcast = 1;
            } else {
                bs = shp_in_1[0];
                bcast = 2;
            }
        } else if (ndims_in_0 > 2) {
            bs = shp_in_0[0];
            bcast = 1;
        } else {
            bs = shp_in_1[0];
            bcast = 2;
        }
    }
    CHECK((m != 0) && (n != 0) && (k != 0) && (bs != 0));

    const OpTile &tile_in_0 = this->cfg->in_deps_tiles[0];
    const OpTile &tile_in_1 = this->cfg->in_deps_tiles[1];
    const OpTile &tile_out = this->cfg->out_deps_tiles[0];

    // Verify tile shapes
    // TODO: more verification
    CHECK(tile_in_0.x == tile_out.x);
    CHECK(tile_in_0.y == tile_in_1.x);
    CHECK(tile_in_1.y == tile_out.y);

    stringstream ss;
    ss << "ark::matmul<" << math::pad(m, tile_out.x) << COM
       << math::pad(n, tile_out.y) << COM << math::pad(k, tile_in_0.y) << COM
       << (trans_in_0 ? "true" : "false") << COM
       << (trans_in_1 ? "true" : "false") << COM << bcast << COM
       << (is_relu ? "true" : "false") << COM << this->cfg->num_warps * 32
       << COM << this->cfg->smem_bytes << COM << tile_out.x << COM << tile_out.y
       << COM << tile_in_0.y << ">";
    return ss.str();
}

const string SchedOp::func_string_im2col() const
{
    CHECK(this->op->in_deps.size() == 1);
    CHECK(this->op->args.size() == 8);
    for (int i = 0; i < 8; ++i) {
        CHECK(this->op->args[i].type == OP_ARG_INT);
    }

    const Tensor *tns_in = this->op->in_deps[0];
    const OpTile &tile_out = this->cfg->out_deps_tiles[0];

    stringstream ss;
    ss << "ark::im2col<"
       << "ark::Vec" << tns_in->shape << COM << "ark::Vec" << tns_in->ldims
       << COM << *(int *)this->op->args[0].val << COM
       << *(int *)this->op->args[1].val << COM << *(int *)this->op->args[2].val
       << COM << *(int *)this->op->args[3].val << COM
       << *(int *)this->op->args[4].val << COM << *(int *)this->op->args[5].val
       << COM << *(int *)this->op->args[6].val << COM
       << *(int *)this->op->args[7].val << COM << this->cfg->num_warps * 32
       << COM << this->cfg->smem_bytes << COM << tile_out.y << COM << tile_out.x
       << COM << 0 << ">";
    return ss.str();
}

const string SchedOp::func_string_transpose() const
{
    CHECK(this->op->in_deps.size() == 1);
    CHECK(this->op->args.size() == 1);
    CHECK(this->op->args[0].type == OP_ARG_INT);

    int tp_type = *(int *)this->op->args[0].val;
    string tp_type_str = to_string(tp_type);
    if (tp_type_str.size() == DIMS_LEN - 1) {
        tp_type_str = "0" + tp_type_str;
    }
    if (tp_type_str.size() != DIMS_LEN) {
        LOGERR("Unexpected error");
    }

    const Tensor *tns_in = this->op->in_deps[0];
    const Tensor *tns_out = this->op->out_deps[0];
    const OpTile &tile_out = this->cfg->out_deps_tiles[0];
    Dims unit_out_shape{1, 1, tile_out.x, tile_out.y};

    stringstream ss;
    ss << "ark::transpose" << tp_type_str << "<"
       << "ark::Vec" << tns_in->ldims << COM << "ark::Vec" << tns_out->ldims
       << COM << "ark::Vec" << tns_out->shape << COM << "ark::Vec"
       << unit_out_shape << COM << this->cfg->num_warps * 32 << COM
       << this->cfg->smem_bytes << ">";
    return ss.str();
}

const string SchedOp::func_string_send() const
{
    const Tensor *in = this->op->in_deps[0];
    CHECK(in->is_sequential());
    int eid = *(int *)this->op->args[0].val;
    int gpu_dst = *(int *)this->op->args[1].val;
    size_t bytes = *(size_t *)this->op->args[2].val;
    stringstream ss;
    ss << "if (threadIdx.x == 0) { ark::comm::send<" << gpu_dst << ", " << eid
       << ", " << eid << ", " << bytes << ">(); }\n";
    return ss.str();
}

const string SchedOp::func_string_recv() const
{
    int eid = *(int *)this->op->args[0].val;
    stringstream ss;
    ss << "if (threadIdx.x == 0) { ark::comm::recv<" << eid << ">(); }\n";
    return ss.str();
}

const string SchedOp::func_string_signal() const
{
    int eid = *(int *)this->op->args[0].val;
    int gpu_dst = *(int *)this->op->args[1].val;
    stringstream ss;
    ss << "if (threadIdx.x == 0) { ark::comm::send<" << gpu_dst << ", " << eid
       << ", " << eid << ", 4>(); }\n";
    return ss.str();
}

const string SchedOp::func_string_wait() const
{
    int eid = *(int *)this->op->args[0].val;
    stringstream ss;
    ss << "if (threadIdx.x == 0) { ark::comm::recv<" << eid << ">(); }\n";
    return ss.str();
}

const string SchedOp::func_string_send_done() const
{
    int eid = *(int *)this->op->args[0].val;
    stringstream ss;
    ss << "if (threadIdx.x == 0) { ark::comm::send_done<" << eid << ">(); }\n";
    return ss.str();
}

const string SchedOp::func_string_reduce() const
{
    CHECK(this->op->in_deps.size() == 1);

    const Tensor *tns_in = this->op->in_deps[0];
    const Tensor *tns_out = this->op->out_deps[0];
    bool is_relu = *(bool *)this->op->args[0].val;

    Dims shp_in = tns_in->shape;
    Dims shp_out = tns_out->shape;

    int ndims = shp_out.ndims();
    CHECK(ndims < 4);

    unsigned int ldm = tns_out->ldims[ndims - 1];
    unsigned int ldn = (ndims == 3) ? tns_out->ldims[ndims - 2] : 1;
    unsigned int k = shp_in[0] / shp_out[0];

    CHECK(shp_in[ndims - 1] == shp_out[ndims - 1]);
    if (ndims == 3) {
        CHECK(shp_in[ndims - 2] == shp_out[ndims - 2]);
    }

    const OpTile &tile_out = this->cfg->out_deps_tiles[0];
    CHECK(shp_out[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(shp_out[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    stringstream ss;
    ss << "ark::reduce_batch<" << ldm << COM << ldn << COM << k << COM
       << is_relu << COM << this->cfg->num_warps * 32 << COM
       << this->cfg->smem_bytes << COM << tile_out.y << COM << tile_out.x << COM
       << 1 << '>';
    return ss.str();
}

const string SchedOp::func_string_send_mm() const
{
    CHECK(this->op->in_deps.size() == 3);
    const Tensor *tns_in = this->op->in_deps[0];
    Dims shp_in = tns_in->shape;
    int ndims = tns_in->ndims();
    stringstream ss;

    CHECK(ndims == 2);
    unsigned int m = shp_in[ndims - 1];
    unsigned int n = shp_in[ndims - 2];
    // unsigned int k = shp_in[0] / shp_out[0];
    Dims pad_in = tns_in->pads;
    const OpTile &tile_in = this->cfg->in_deps_tiles[0];
    // Verify paddings
    CHECK((shp_in[ndims - 2] % tile_in.x == 0) ||
          (pad_in[ndims - 2] >= tile_in.x));
    CHECK((shp_in[ndims - 1] % tile_in.y == 0) ||
          (pad_in[ndims - 1] >= tile_in.y));
    ss << "ark::comm::sendLL<" << m << COM << n << COM
       << this->cfg->num_warps * 32 << COM << this->cfg->smem_bytes << COM
       << tile_in.y << COM << tile_in.x << COM << 1 << '>';
    return ss.str();
}

const string SchedOp::func_string_recv_mm() const
{
    CHECK(this->op->in_deps.size() == 3);
    const Tensor *data = this->op->in_deps[0];
    Dims shp_in = data->shape;
    int ndims = data->ndims();
    stringstream ss;

    CHECK(ndims == 2);
    unsigned int m = shp_in[ndims - 1];
    unsigned int n = shp_in[ndims - 2];
    // unsigned int k = shp_in[0] / shp_out[0];
    Dims pad_in = data->pads;
    const OpTile &tile_in = this->cfg->in_deps_tiles[0];
    // Verify paddings
    CHECK((shp_in[ndims - 2] % tile_in.x == 0) ||
          (pad_in[ndims - 2] >= tile_in.x));
    CHECK((shp_in[ndims - 1] % tile_in.y == 0) ||
          (pad_in[ndims - 1] >= tile_in.y));
    ss << "ark::comm::recvLL<" << m << COM << n << COM
       << this->cfg->num_warps * 32 << COM << this->cfg->smem_bytes << COM
       << tile_in.y << COM << tile_in.x << COM << 1 << '>';
    return ss.str();
}

const string SchedOp::func_string_scale() const
{
    CHECK(this->op->in_deps.size() == 1);

    const Tensor *tns_in = this->op->in_deps[0];
    const Tensor *tns_out = this->op->out_deps[0];

    Dims shp_in = tns_in->shape;
    Dims shp_out = tns_out->shape;

    CHECK(shp_in[1] == shp_out[1]);
    CHECK(shp_in[2] == shp_out[2]);
    CHECK(shp_in[3] == shp_out[3]);

    int ndims = shp_out.ndims();
    unsigned int ldm = tns_out->ldims[ndims - 1];
    unsigned int ldn = (ndims > 1) ? tns_out->ldims[ndims - 2] : 1;

    const OpTile &tile_out = this->cfg->out_deps_tiles[0];
    CHECK(shp_out[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(shp_out[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    stringstream ss;
    ss << "ark::scale<" << ldm << COM << ldn << COM << this->cfg->num_warps * 32
       << COM << this->cfg->smem_bytes << COM << tile_out.y << COM << tile_out.x
       << COM << 1 << '>';
    return ss.str();
}

const string SchedOp::func_string_add() const
{
    CHECK(this->op->in_deps.size() == 2);

    const Tensor *tns_in_0 = this->op->in_deps[0];
    const Tensor *tns_in_1 = this->op->in_deps[1];
    const Tensor *tns_out = this->op->out_deps[0];

    LOG(DEBUG, "func_string_add: ", tns_out->shape, " ", tns_out->ldims);

    int ndims = tns_out->shape.ndims();
    const OpTile &tile_out = this->cfg->out_deps_tiles[0];
    CHECK(tns_out->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(tns_out->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    Dims unit_out_shape{1, 1, tile_out.x, tile_out.y};

    stringstream ss;
    ss << "ark::add<"
       << "ark::Vec" << tns_in_0->ldims.dims4() << COM << "ark::Vec"
       << tns_in_0->shape.dims4() << COM << "ark::Vec"
       << tns_in_1->ldims.dims4() << COM << "ark::Vec"
       << tns_in_1->shape.dims4() << COM << "ark::Vec" << tns_out->ldims.dims4()
       << COM << "ark::Vec" << tns_out->shape.dims4() << COM << "ark::Vec"
       << unit_out_shape << COM << this->cfg->num_warps * 32 << COM
       << this->cfg->smem_bytes << '>';
    return ss.str();
}

const string SchedOp::func_string_mul() const
{
    CHECK(this->op->in_deps.size() == 2);

    const Tensor *tns_in_0 = this->op->in_deps[0];
    const Tensor *tns_in_1 = this->op->in_deps[1];
    const Tensor *tns_out = this->op->out_deps[0];

    LOG(DEBUG, "func_string_mul: ", tns_out->shape, " ", tns_out->ldims);

    int ndims = tns_out->shape.ndims();
    const OpTile &tile_out = this->cfg->out_deps_tiles[0];
    CHECK(tns_out->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(tns_out->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    Dims unit_out_shape{1, 1, tile_out.x, tile_out.y};

    stringstream ss;
    ss << "ark::mul<"
       << "ark::Vec" << tns_in_0->ldims.dims4() << COM << "ark::Vec"
       << tns_in_0->shape.dims4() << COM << "ark::Vec"
       << tns_in_1->ldims.dims4() << COM << "ark::Vec"
       << tns_in_1->shape.dims4() << COM << "ark::Vec" << tns_out->ldims.dims4()
       << COM << "ark::Vec" << tns_out->shape.dims4() << COM << "ark::Vec"
       << unit_out_shape << COM << this->cfg->num_warps * 32 << COM
       << this->cfg->smem_bytes << '>';
    return ss.str();
}

void to_json(nlohmann::json &j, const SchedOp &sop)
{
    // j = nlohmann::json{
    //     {"op", sop.get_op()->name},
    //     {"cfg", sop.get_cfg()},
    // };
}
void from_json(const nlohmann::json &j, SchedOp &sop)
{
}

} // namespace ark
