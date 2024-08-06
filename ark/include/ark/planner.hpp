// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_PLANNER_HPP
#define ARK_PLANNER_HPP

#include <ark/context.hpp>
#include <functional>
#include <memory>
#include <string>

namespace ark {

template <typename T>
class Range;

class PlannerContext : public Context {
   public:
    PlannerContext(Model& model);

    void processor_range(int start, int end, int step = 1);

    void warp_range(int start, int end, int step = 1);

    void sram_range(int start, int end, int step = 1);

    void sync(bool sync);

    void config(const std::string& config);

   private:
    void check_range(const std::string& key, const Range<int>& range);
};

class Planner {
   public:
    Planner(const Model& model, int device_id);

    ~Planner();

    using ConfigRule = std::function<std::string(const std::string &op,
                                                 const std::string &arch)>;

    void install_config_rule(ConfigRule rule);

    std::string plan(bool pretty = true) const;

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_PLANNER_HPP
