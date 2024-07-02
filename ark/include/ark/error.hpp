// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_ERROR_HPP
#define ARK_ERROR_HPP

#include <stdexcept>
#include <string>

namespace ark {

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

REGISTER_ERROR_TYPE(InternalError)
REGISTER_ERROR_TYPE(InvalidUsageError)
REGISTER_ERROR_TYPE(NotFoundError)
REGISTER_ERROR_TYPE(ModelError)
REGISTER_ERROR_TYPE(SchedulerError)
REGISTER_ERROR_TYPE(ExecutorError)
REGISTER_ERROR_TYPE(SystemError)
REGISTER_ERROR_TYPE(GpuError)
REGISTER_ERROR_TYPE(RuntimeError)
REGISTER_ERROR_TYPE(UnitTestError)

}  // namespace ark

#endif  // ARK_ERROR_HPP
