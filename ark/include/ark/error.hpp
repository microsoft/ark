// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_ERROR_HPP
#define ARK_ERROR_HPP

#include <stdexcept>
#include <string>

namespace ark {

/// Base class for all ARK errors.
class BaseError : public std::exception {
   private:
    std::string msg_;

   public:
    BaseError(const std::string &msg) : msg_(msg) {}
    const char *what() const noexcept override { return msg_.c_str(); }
};

#define REGISTER_ERROR_TYPE(_name)                        \
    class _name : public BaseError {                      \
       public:                                            \
        _name(const std::string &msg) : BaseError(msg) {} \
    };

/// Internal error in ARK, likely a bug.
REGISTER_ERROR_TYPE(InternalError)
/// Invalid usage of ARK API.
REGISTER_ERROR_TYPE(InvalidUsageError)
/// Invalid ARK model definition or usage.
REGISTER_ERROR_TYPE(ModelError)
/// Invalid ARK plan definition or usage.
REGISTER_ERROR_TYPE(PlanError)
/// Unsupported feature triggered.
REGISTER_ERROR_TYPE(UnsupportedError)
/// Error from invalid system state such as a system call failure.
REGISTER_ERROR_TYPE(SystemError)
/// Error from a CUDA/HIP API call.
REGISTER_ERROR_TYPE(GpuError)
/// Error from a unit test.
REGISTER_ERROR_TYPE(UnitTestError)

}  // namespace ark

#endif  // ARK_ERROR_HPP
