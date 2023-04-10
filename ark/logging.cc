#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <unistd.h>

#include "ark/cpu_timer.h"
#include "ark/env.h"
#include "ark/logging.h"

using namespace std;

namespace ark {

Logging::Logging(const char *lv) : pid{getpid()}
{
    if (lv == nullptr) {
        this->level = INFO;
    } else if (strncmp(lv, "DEBUG", 6) == 0) {
        this->level = DEBUG;
    } else if (strncmp(lv, "WARN", 5) == 0) {
        this->level = WARN;
    } else if (strncmp(lv, "ERROR", 6) == 0) {
        this->level = ERROR;
    } else {
        this->level = INFO;
    }
}

const LogLevel &Logging::get_level() const
{
    return this->level;
}

void Logging::set_level(LogLevel lv)
{
    this->level = lv;
};

////////////////////////////////////////////////////////////////////////////////

unique_ptr<Logging> _ARK_LOGGING_GLOBAL = nullptr;

// Get the global Logging.
Logging &get_logging()
{
    if (_ARK_LOGGING_GLOBAL.get() == nullptr) {
        _ARK_LOGGING_GLOBAL.reset(new Logging{get_env().log_level});
        assert(_ARK_LOGGING_GLOBAL.get() != nullptr);
    }
    return *_ARK_LOGGING_GLOBAL;
}

void log_header(ostream &os, const LogLevel ll, const string &file,
                const int line)
{
    long usec = cpu_ntimer() / 1000;
    os << dec << setfill('0') << setw(6) << usec << " ARK " << setfill(' ')
       << setw(5) << getpid() << ' ';
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
    os << file << ':' << line << ' ';
}

ostream &log(ostream &os, const LogLevel ll, const string &file, const int line)
{
    log_header(os, ll, file, line);
    return os;
}

void set_log_level(LogLevel lv)
{
    get_logging().set_level(lv);
}

const LogLevel &get_log_level()
{
    return get_logging().get_level();
}

} // namespace ark