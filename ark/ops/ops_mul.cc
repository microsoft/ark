#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::mul(Tensor *input, Tensor *other, Tensor *output,
                   const string &name)
{
    LOG(DEBUG, "mul ", input->shape, " ", other->shape);
    assert(input != nullptr);
    assert(other != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    if (input->type != other->type) {
        LOGERR("input data types mismatch: ", type_str(input->type), ", ",
               type_str(other->type));
    }
    if (output != nullptr && input->type != output->type) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    //
    for (int i = 0; i < DIMS_LEN; ++i) {
        if ((input->shape[i] != other->shape[i]) && (input->shape[i] != 1) &&
            (other->shape[i] != 1)) {
            LOGERR("invalid input shapes: ", input->shape, ", ", other->shape);
        }
    }
    Dims output_shape{{(input->shape[0] > other->shape[0]) ? input->shape[0]
                                                           : other->shape[0],
                       (input->shape[1] > other->shape[1]) ? input->shape[1]
                                                           : other->shape[1],
                       (input->shape[2] > other->shape[2]) ? input->shape[2]
                                                           : other->shape[2],
                       (input->shape[3] > other->shape[3]) ? input->shape[3]
                                                           : other->shape[3]}};
    if (output == nullptr) {
        output = this->tensor(output_shape, input->type);
    } else if (output->shape != output_shape) {
        LOGERR("invalid output shape: ", output->shape);
    }
    this->create_op(OP_MUL, pt, {input, other}, {output}, {}, name);
    return output;
}

} // namespace ark
