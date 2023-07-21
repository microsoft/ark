// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_CONFIG_H_
#define ARK_OPS_CONFIG_H_

#include "include/ark.h"
#include "json.h"
#include "ops/ops_common.h"
#include <map>
#include <vector>

namespace ark {

// Key to find a list of OpConfigs from OpConfigMap.
struct OpConfigKey
{
    OpType op_type;
    OpArchType arch_type;
    OpPrecType prec_type;
};

bool operator<(const OpConfigKey &ops1, const OpConfigKey &ops2);
bool operator==(const OpConfigKey &ops1, const OpConfigKey &ops2);

void to_json(nlohmann::json &j, const OpConfig &cfg);
void from_json(const nlohmann::json &j, OpConfig &cfg);

// OpConfig for virtual ops.
extern const OpConfig ARK_OP_CONFIG_VIRT;
// Map from OpConfigKey to a list of OpConfigs.
extern const std::map<OpConfigKey, std::vector<OpConfig>> ARK_OP_CONFIG_MAP;

} // namespace ark

#endif // ARK_OPS_CONFIG_H_
