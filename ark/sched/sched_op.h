// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHED_OP_H_
#define ARK_SCHED_OP_H_

namespace ark {

class SchedOp {
   public:
    SchedOp(const Op *op_, const OpConfig *cfg_, const std::string name);
    const Op *get_op() const { return op; }
    int get_num_warps() const { return const_cast<OpConfig *>(cfg)->num_warps; }
    const std::string &get_name() const { return name; }
    const OpConfig *get_cfg() const { return cfg; }
    const std::string function_name() const;
    const std::string serialize() const;
    bool is_virtual() const;

   private:
    const Op *op;
    const OpConfig *cfg;
    std::string name;
};

}  // namespace ark

#endif  // ARK_SCHED_OP_H_
