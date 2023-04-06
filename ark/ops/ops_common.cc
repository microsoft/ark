#include "ark/ops/ops_common.h"

using namespace std;

namespace ark {

OpArg::OpArg(int arg) : type{OP_ARG_INT}, val{new int{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(uint64_t arg) : type{OP_ARG_UINT64}, val{new uint64_t{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(bool arg) : type{OP_ARG_BOOL}, val{new bool{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(float arg) : type{OP_ARG_FLOAT}, val{new float{arg}}
{
    assert(this->val != nullptr);
}
OpArg::OpArg(const OpArg &arg) : type{arg.type}
{
    if (this->type == OP_ARG_INT) {
        this->val = new int{*(int *)arg.val};
    } else if (this->type == OP_ARG_UINT64) {
        this->val = new uint64_t{*(uint64_t *)arg.val};
    } else if (this->type == OP_ARG_BOOL) {
        this->val = new bool{*(bool *)arg.val};
    } else if (this->type == OP_ARG_FLOAT) {
        this->val = new float{*(float *)arg.val};
    }
}
OpArg::~OpArg()
{
    if (this->type == OP_ARG_INT) {
        delete static_cast<int *>(this->val);
    } else if (this->type == OP_ARG_UINT64) {
        delete static_cast<uint64_t *>(this->val);
    } else if (this->type == OP_ARG_BOOL) {
        delete static_cast<bool *>(this->val);
    } else if (this->type == OP_ARG_FLOAT) {
        delete static_cast<float *>(this->val);
    }
}
bool operator<(const OpArg &oa1, const OpArg &oa2)
{
    if (oa1.type != oa2.type) {
        return oa1.type < oa2.type;
    }
    assert(oa1.val != nullptr);
    assert(oa2.val != nullptr);
    switch (oa1.type) {
    case OP_ARG_INT:
        return *(int *)oa1.val < *(int *)oa2.val;
    case OP_ARG_UINT64:
        return *(uint64_t *)oa1.val < *(uint64_t *)oa2.val;
    case OP_ARG_BOOL:
        return *(bool *)oa1.val < *(bool *)oa2.val;
    case OP_ARG_FLOAT:
        return *(float *)oa1.val < *(float *)oa2.val;
    }
    assert(false);
    return false;
}
bool operator==(const OpArg &oa1, const OpArg &oa2)
{
    if (oa1.type != oa2.type) {
        return false;
    }
    assert(oa1.val != nullptr);
    assert(oa2.val != nullptr);
    switch (oa1.type) {
    case OP_ARG_INT:
        return *(int *)oa1.val == *(int *)oa2.val;
    case OP_ARG_UINT64:
        return *(uint64_t *)oa1.val == *(uint64_t *)oa2.val;
    case OP_ARG_BOOL:
        return *(bool *)oa1.val == *(bool *)oa2.val;
    case OP_ARG_FLOAT:
        return *(float *)oa1.val == *(float *)oa2.val;
    }
    assert(false);
    return false;
}

void to_json(nlohmann::json &j, const OpArg &oparg)
{
    j = nlohmann::json{
        {"type", oparg.type},
    };
    if (oparg.type == OP_ARG_INT) {
        j.emplace("val", *static_cast<int *>(oparg.val));
    } else if (oparg.type == OP_ARG_UINT64) {
        j.emplace("val", *static_cast<uint64_t *>(oparg.val));
    } else if (oparg.type == OP_ARG_BOOL) {
        j.emplace("val", *static_cast<bool *>(oparg.val));
    } else if (oparg.type == OP_ARG_FLOAT) {
        j.emplace("val", *static_cast<float *>(oparg.val));
    }
}

void from_json(const nlohmann::json &j, OpArg &oparg)
{
    j.at("type").get_to(oparg.type);
    if (oparg.type == OP_ARG_INT) {
        oparg.val = new int{j.at("val").get<int>()};
    } else if (oparg.type == OP_ARG_UINT64) {
        oparg.val = new uint64_t{j.at("val").get<uint64_t>()};
    } else if (oparg.type == OP_ARG_BOOL) {
        oparg.val = new bool{j.at("val").get<bool>()};
    } else if (oparg.type == OP_ARG_FLOAT) {
        oparg.val = new float{j.at("val").get<float>()};
    }
}

Op::Op(const OpType &type_, const OpPrecType &prec_type_,
       const vector<Tensor *> &in_deps_, const vector<Tensor *> &out_deps_,
       const vector<OpArg> &args_, const string &name_, int gran_lev_)
    : type{type_}, prec_type{prec_type_}, in_deps{in_deps_},
      out_deps{out_deps_}, args{args_}, name{name_}, gran_lev{gran_lev_}
{
    for (auto &tns : in_deps_) {
        assert(tns != nullptr);
    }
    for (auto &tns : out_deps_) {
        assert(tns != nullptr);
    }
}
bool operator<(const Op &op1, const Op &op2)
{
    if (op1.type != op2.type) {
        return op1.type < op2.type;
    }
    if (op1.prec_type != op2.prec_type) {
        return op1.prec_type < op2.prec_type;
    }
    for (size_t i = 0; i < op1.args.size(); ++i) {
        if (op1.args[i] == op2.args[i]) {
            continue;
        }
        return op1.args[i] < op2.args[i];
    }
    return false;
}
bool operator==(const Op &op1, const Op &op2)
{
    if (op1.type != op2.type) {
        return false;
    }
    if (op1.prec_type != op2.prec_type) {
        return false;
    }
    for (size_t i = 0; i < op1.args.size(); ++i) {
        if (op1.args[i] == op2.args[i]) {
            continue;
        }
        return false;
    }
    return true;
}
void to_json(nlohmann::json &j, const Op &op)
{
    j = nlohmann::json{
        {"type", op.type},          {"prec_type", op.prec_type},
        {"in_deps", vector<int>{}}, {"out_deps", vector<int>{}},
        {"args", op.args},          {"name", op.name},
    };
    for (Tensor *pt : op.in_deps) {
        j.at("in_deps").emplace_back(pt->id);
    }
    for (Tensor *pt : op.out_deps) {
        j.at("out_deps").emplace_back(pt->id);
    }
}
void from_json(const nlohmann::json &j, Op &op)
{
}

} // namespace ark
