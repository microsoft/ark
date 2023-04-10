// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_CONFIGS_H_
#define ARK_OPS_CONFIGS_H_

#include "ops_common.h"
#include "third_party/json/json.h"
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

// 2-dimensional op tile
struct OpTile
{
    DimType x;
    DimType y;
};

// Configurations for execution of an operation.
struct OpConfig
{
    unsigned int num_warps = 0;
    unsigned int smem_bytes = 0;
    std::vector<OpTile> in_deps_tiles;
    std::vector<OpTile> out_deps_tiles;
    bool sync_pre = false;
    bool sync_post = false;
};

void to_json(nlohmann::json &j, const OpConfig &cfg);
void from_json(const nlohmann::json &j, OpConfig &cfg);

// OpConfig for virtual ops.
extern const OpConfig ARK_OP_CONFIG_VIRT;
// Map from OpConfigKey to a list of OpConfigs.
extern const std::map<OpConfigKey, std::vector<OpConfig>> ARK_OP_CONFIG_MAP;

} // namespace ark

#endif // ARK_OPS_CONFIGS_H_
