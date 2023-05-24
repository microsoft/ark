// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TEST_COMMON_H_
#define ARK_OPS_TEST_COMMON_H_

#include "ark/include/ark.h"
#include <string>

void test_bcast_fp32(std::string op_name, ark::DimType bs, ark::DimType n,
                     ark::DimType m, bool overwrite = false);
void test_bcast_fp16(std::string op_name, ark::DimType bs, ark::DimType n,
                     ark::DimType m, bool overwrite = false);

#endif // ARK_OPS_TEST_COMMON_H_
