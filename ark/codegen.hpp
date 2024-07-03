// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_CODEGEN_HPP_
#define ARK_CODEGEN_HPP_

#include <map>
#include <memory>
#include <string>

#include "model_buffer_manager.hpp"
#include "model/model_json.hpp"

namespace ark {

class CodeGenerator {
   public:
    CodeGenerator(const PlanJson &plan,
                  const std::map<size_t, size_t> &buffer_id_to_offset,
                  const std::string &name = "ark_kernel",
                  ModelBufferManager *buffer_manager = nullptr);

    ~CodeGenerator() = default;

    std::string code() const;

    size_t num_procs() const;

    size_t num_warps_per_proc() const;

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_CODEGEN_HPP_
