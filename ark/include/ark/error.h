// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_ERROR_H
#define ARK_ERROR_H

#include <stdexcept>
#include <string>

namespace ark {

class InvalidUsageError : public std::runtime_error {
   public:
    InvalidUsageError(const std::string &msg) : std::runtime_error(msg) {}
};

class ModelError : public std::runtime_error {
   public:
    ModelError(const std::string &msg) : std::runtime_error(msg) {}
};

class SchedulerError : public std::runtime_error {
   public:
    SchedulerError(const std::string &msg) : std::runtime_error(msg) {}
};

class ExecutorError : public std::runtime_error {
   public:
    ExecutorError(const std::string &msg) : std::runtime_error(msg) {}
};

class SystemError : public std::runtime_error {
   public:
    SystemError(const std::string &msg) : std::runtime_error(msg) {}
};

class GpuError : public std::runtime_error {
   public:
    GpuError(const std::string &msg) : std::runtime_error(msg) {}
};

class RuntimeError : public std::runtime_error {
   public:
    RuntimeError(const std::string &msg) : std::runtime_error(msg) {}
};

class UnitTestError : public std::runtime_error {
   public:
    UnitTestError(const std::string &msg) : std::runtime_error(msg) {}
};

}  // namespace ark

#endif  // ARK_ERROR_H
