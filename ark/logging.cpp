// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.hpp"

#include <unistd.h>

#include <iomanip>
#include <memory>
#include <sstream>

#include "cpu_timer.h"
#include "env.h"

namespace ark {

Logging::Logging(const std::string &lv) : pid_{::getpid()} {
    if (lv.size() == 0) {
        level_ = INFO;
    } else if (lv == "DEBUG") {
        level_ = DEBUG;
    } else if (lv == "WARN") {
        level_ = WARN;
    } else if (lv == "ERROR") {
        level_ = ERROR;
    } else {
        level_ = INFO;
    }
}

const LogLevel &Logging::get_level() const { return level_; }

void Logging::set_level(LogLevel lv) { level_ = lv; };

////////////////////////////////////////////////////////////////////////////////

// Get the global Logging.
Logging &get_logging() {
    static std::unique_ptr<Logging> ark_logging = nullptr;
    if (ark_logging.get() == nullptr) {
        ark_logging = std::make_unique<Logging>(get_env().log_level);
    }
    return *ark_logging;
}

void _log_header(std::ostream &os, const LogLevel ll, const std::string &file,
                 const int line) {
    os << "ARK " << std::setfill(' ') << std::setw(5) << ::getpid() << ' ';
    switch (ll) {
        case INFO:
            os << "INFO ";
            break;
        case DEBUG:
            os << "DEBUG ";
            break;
        case WARN:
            os << "WARN ";
            break;
        case ERROR:
            os << "ERROR ";
            break;
    }
    std::string file_name;
    size_t pos = file.rfind("ark/");
    if (pos == std::string::npos) {
        file_name = file;
    } else {
        file_name = file.substr(pos + 4);
    }
    os << file_name << ':' << line << ' ';
}

void set_log_level(LogLevel lv) { get_logging().set_level(lv); }

const LogLevel &get_log_level() { return get_logging().get_level(); }

}  // namespace ark
