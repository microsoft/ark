// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/log.hpp"

#include "logging.hpp"

namespace ark {

void log(LogLevel level, const std::string &file, int line,
         const std::string &msg) {
    _log(level, file, line, msg);
}

}  // namespace ark
