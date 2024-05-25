// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_ERROR_HPP_
#define ARK_ERROR_HPP_

#include <stdexcept>
#include <string>

namespace ark {

#define REGISTER_ERROR_TYPE(_name)                                 \
    class _name : public std::runtime_error {                      \
       public:                                                     \
        _name(const std::string &msg) : std::runtime_error(msg) {} \
    };

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

#endif  // ARK_ERROR_HPP_
