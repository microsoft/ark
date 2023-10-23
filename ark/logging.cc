// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"

#include <unistd.h>

#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "cpu_timer.h"
#include "env.h"

namespace ark {

Logging::Logging(const char *lv) : pid{::getpid()} {
    if (lv == nullptr) {
        this->level = INFO;
    } else if (::strncmp(lv, "DEBUG", 6) == 0) {
        this->level = DEBUG;
    } else if (::strncmp(lv, "WARN", 5) == 0) {
        this->level = WARN;
    } else if (::strncmp(lv, "ERROR", 6) == 0) {
        this->level = ERROR;
    } else {
        this->level = INFO;
    }
}

const LogLevel &Logging::get_level() const { return this->level; }

void Logging::set_level(LogLevel lv) { this->level = lv; };

////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Logging> _ARK_LOGGING_GLOBAL = nullptr;

// Get the global Logging.
Logging &get_logging() {
    if (_ARK_LOGGING_GLOBAL.get() == nullptr) {
        _ARK_LOGGING_GLOBAL.reset(new Logging{get_env().log_level});
        assert(_ARK_LOGGING_GLOBAL.get() != nullptr);
    }
    return *_ARK_LOGGING_GLOBAL;
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
