#include "ark/init.h"
#include "ark/model.h"
#include "ark/unittest/unittest_utils.h"

#define ITERATION 1

using namespace std;

ark::unittest::State test_simple_mm()
{
    // Hidden dimension of the dense layer.
    unsigned int units = 1024;
    // Input dimension of the dense layer.
    unsigned int in_dim = 1024;
    // Extra dimension of the input. CHANNEL=1 for 2D inputs.
    unsigned int channel = 128;
    // Batch size of the input.
    unsigned int batch_size = 1;

    ark::Model m;
    ark::Tensor *input = m.tensor({batch_size, channel, 1, in_dim}, ark::FP16);
    ark::Tensor *weight = m.tensor({1, in_dim, 1, units}, ark::FP16);
    ark::Tensor *output = m.matmul(input, weight);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    for (int i = 0; i < ITERATION; ++i) {
        UNITTEST(test_simple_mm);
    }
    return 0;
}