// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_CODEGEN_HPP_
#define ARK_CODEGEN_HPP_

#include <memory>
#include <string>

namespace ark {

class CodeGenerator {
   public:
    CodeGenerator(const std::string &plan,
                  const std::string &name = "ark_kernel");

    ~CodeGenerator() = default;

    std::string code() const;

    size_t num_procs() const;

    size_t num_warps_per_proc() const;

    size_t total_memory_bytes() const;

    struct TensorInfo {
        size_t id;
        size_t bytes;
        size_t offset;
    };

    const TensorInfo &tensor_info(size_t tensor_id) const;

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_CODEGEN_HPP_
