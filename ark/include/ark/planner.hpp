// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_PLANNER_HPP_
#define ARK_PLANNER_HPP_

#include <memory>
#include <string>

namespace ark {

class Model;

class DefaultPlanner {
   public:
    DefaultPlanner(const Model &model, int gpu_id);

    ~DefaultPlanner();

    std::string plan(int indent = -1) const;

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_PLANNER_HPP_
