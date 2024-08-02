// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_LOGGING_HPP_
#define ARK_LOGGING_HPP_

#include <iostream>
#include <sstream>
#include <string>

#include "ark/error.hpp"

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

template <LogLevel Level, bool AppendNewLine, typename T, typename... Args>
inline std::string _log_msg(const std::string &file, int line, T value,
                            Args... args) {
    std::stringstream ss;
    _log_header(ss, Level, file, line);
    _log_helper(ss, value, args...);
    if constexpr (AppendNewLine) ss << std::endl;
    return ss.str();
}

template <LogLevel Level, typename T, typename... Args>
inline void _log(const std::string &file, int line, T value, Args... args) {
    if (Level >= get_logging().get_level()) {
        std::clog << _log_msg<Level, true>(file, line, value, args...);
    }
    if constexpr (Level == ERROR) {
        throw std::runtime_error("ARK runtime error");
    }
}

template <typename Exception, typename T, typename... Args>
inline void _err(const std::string &file, int line, T value, Args... args) {
    throw Exception(_log_msg<ERROR, false>(file, line, value, args...));
}

// Logging.
#define LOG(level, ...)                                    \
    do {                                                   \
        ark::_log<level>(__FILE__, __LINE__, __VA_ARGS__); \
        break;                                             \
    } while (0)

#define ERR(exception, ...)                                             \
    do {                                                                \
        std::string exc_str = " (" #exception ")";                      \
        ark::_err<exception>(__FILE__, __LINE__, __VA_ARGS__, exc_str); \
        break;                                                          \
    } while (0)

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            ERR(ark::InvalidUsageError, "failed condition: " #cond); \
        }                                                            \
    } while (0)

}  // namespace ark

#endif  // ARK_LOGGING_HPP_
