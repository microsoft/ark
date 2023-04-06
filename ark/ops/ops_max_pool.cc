#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::max_pool(Tensor *input, DimType kernel_size, DimType stride,
                        Tensor *output, const string &name)
{
    assert(input != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    if (output != nullptr && input->type != output->type) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    const Dims &is = input->shape;
    Dims os{{is[0], (is[1] + stride - 1) / stride,
             (is[2] + stride - 1) / stride, is[3]}};
    if (output == nullptr) {
        output = this->tensor(os, input->type);
    }
    this->create_op(OP_MAX_POOL, pt, {input}, {output}, {}, name);
    return output;
}

} // namespace ark
