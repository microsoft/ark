// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "schedule.h"

#include "nlohmann/json.hpp"

namespace ark {

nlohmann::json to_json(const Range<int> &range) {
    nlohmann::json j;
    j["Begin"] = *range.begin();
    j["End"] = *range.end();
    if (range.step() != 1) {
        j["Step"] = range.step();
    }
    return j;
}

nlohmann::json to_json(const TaskInfo &task_info) {
    nlohmann::json j;
    j["Id"] = task_info.id;
    j["NumWarps"] = task_info.num_warps;
    j["SramBytes"] = task_info.sram_bytes;
    j["Detail"] = task_info.detail;
    return j;
}

nlohmann::json to_json(const TaskGroup &task_group) {
    nlohmann::json j;
    j["TaskId"] = task_group.task_id;
    j["TaskRange"] = to_json(task_group.task_range);
    j["TaskStride"] = task_group.task_stride;
    return j;
}

nlohmann::json to_json(const ResourceGroup &resource_group) {
    nlohmann::json j;
    j["ProcessorRange"] = to_json(resource_group.processor_range);
    j["WarpRange"] = to_json(resource_group.warp_range);
    j["SramRange"] = to_json(resource_group.sram_range);
    j["TaskGroups"] = nlohmann::json();
    for (const auto &tg : resource_group.task_groups) {
        j["TaskGroups"].emplace_back(to_json(tg));
    }
    return j;
}

nlohmann::json to_json(const ProcessorGroup &sched) {
    nlohmann::json j;
    j["ProcessorRange"] = to_json(sched.processor_range);
    j["ResourceGroups"] = nlohmann::json();
    for (const auto &rg : sched.resource_groups) {
        j["ResourceGroups"].emplace_back(to_json(rg));
    }
    return j;
}

nlohmann::json to_json(const Schedule &sched) {
    nlohmann::json j;
    j["NumProcessors"] = sched.num_processors;
    j["NumWarpsPerProcessor"] = sched.num_warps_per_processor;
    j["TaskInfos"] = nlohmann::json();
    for (const auto &ti : sched.task_infos) {
        j["TaskInfos"].emplace_back(to_json(ti));
    }
    j["ProcessorGroups"] = nlohmann::json();
    for (const auto &pg : sched.processor_groups) {
        j["ProcessorGroups"].emplace_back(to_json(pg));
    }
    return j;
}

std::string Schedule::serialize(int indent) const {
    auto j = to_json(*this);
    return j.dump(indent);
}

}  // namespace ark
