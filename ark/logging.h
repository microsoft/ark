// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_LOGGING_H_
#define ARK_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string>

namespace ark {

typedef enum { DEBUG, INFO, WARN, ERROR } LogLevel;

class Logging {
   public:
    Logging(const std::string &lv);

    const LogLevel &get_level() const;
    void set_level(LogLevel lv);

   private:
    const pid_t pid_;
    LogLevel level_;
};

// Get the global Logging.
Logging &get_logging();

void set_log_level(LogLevel lv);
const LogLevel &get_log_level();

void _log_header(std::ostream &os, const ark::LogLevel ll,
                 const std::string &file, const int line);

template <typename T>
void _log_helper(std::stringstream &ss, T value) {
    ss << value;
}

template <typename T, typename... Args>
void _log_helper(std::stringstream &ss, T value, Args... args) {
    ss << value;
    _log_helper(ss, args...);
}

template <LogLevel level, typename T, typename... Args>
inline void _log(const std::string &file, int line, T value, Args... args) {
    if (level >= get_logging().get_level()) {
        std::stringstream ss;
        _log_header(ss, level, file, line);
        _log_helper(ss, value, args...);
        ss << '\n';
        std::clog << ss.str();
    }
    if constexpr (level == ERROR) {
        throw std::runtime_error("ARK runtime error");
    }
}

// Logging.
#define LOG(level, ...)                                    \
    do {                                                   \
        ark::_log<level>(__FILE__, __LINE__, __VA_ARGS__); \
        break;                                             \
    } while (0)

#define CHECK(cond)                                 \
    do {                                            \
        if (!(cond)) {                              \
            LOG(ERROR, "failed condition: " #cond); \
        }                                           \
    } while (0)

}  // namespace ark

#endif  // ARK_LOGGING_H_
