// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_LOG_HPP
#define ARK_LOG_HPP

#include <string>

namespace ark {

typedef enum { DEBUG, INFO, WARN, ERROR } LogLevel;

void log(LogLevel level, const std::string &file, int line,
         const std::string &msg);

}  // namespace ark

#endif  // ARK_LOG_HPP
