// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_SCHEDULE_H_
#define ARK_SCHEDULE_H_

#include <memory>
#include <string>
#include <vector>

#include "range.h"

namespace ark {

class TaskInfo {
   public:
    TaskInfo() {}
    ~TaskInfo() = default;

    int id;
    int num_warps;
    int sram_bytes;
    std::string detail;
};

class TaskGroup {
   public:
    TaskGroup() {}
    ~TaskGroup() = default;

    int task_id;
    int task_stride;
    Range<int> task_range;
};

class ResourceGroup {
   public:
    ResourceGroup() {}
    ~ResourceGroup() = default;

    Range<int> processor_range;
    Range<int> warp_range;
    Range<int> sram_range;
    std::vector<TaskGroup> task_groups;
};

class ProcessorGroup {
   public:
    ProcessorGroup() {}
    ~ProcessorGroup() = default;

    Range<int> processor_range;
    std::vector<ResourceGroup> resource_groups;
};

class Schedule {
   public:
    Schedule() {}
    ~Schedule() = default;

    std::string serialize(int indent = -1) const;

    int num_processors;
    int num_warps_per_processor;
    std::vector<TaskInfo> task_infos;
    std::vector<ProcessorGroup> processor_groups;
};

}  // namespace ark

#endif  // ARK_SCHEDULE_H_
