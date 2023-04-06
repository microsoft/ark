#ifndef ARK_LOGGING_H_
#define ARK_LOGGING_H_

#include <iostream>
#include <sstream>
#include <string>

#define ARK_LOG_SMEM_NAME "ark.log"

namespace ark {

typedef enum
{
    DEBUG,
    INFO,
    WARN,
    ERROR
} LogLevel;

class Logging
{
  public:
    Logging(const char *lv);

    const LogLevel &get_level() const;
    void set_level(LogLevel lv);

  private:
    const pid_t pid;
    LogLevel level;
};

// Get the global Logging.
Logging &get_logging();

void log_header(std::ostream &os, const ark::LogLevel ll,
                const std::string &file, const int line);

std::ostream &log(std::ostream &os, const ark::LogLevel ll,
                  const std::string &file, const int line);

void set_log_level(LogLevel lv);
const LogLevel &get_log_level();

// Logging macros.
#define SSTREAM_1(_1) _ss << (_1)
#define SSTREAM_2(_1, _2) _ss << (_1) << (_2)
#define SSTREAM_3(_1, _2, _3) _ss << (_1) << (_2) << (_3)
#define SSTREAM_4(_1, _2, _3, _4) _ss << (_1) << (_2) << (_3) << (_4)
#define SSTREAM_5(_1, _2, _3, _4, _5)                                          \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5)
#define SSTREAM_6(_1, _2, _3, _4, _5, _6)                                      \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6)
#define SSTREAM_7(_1, _2, _3, _4, _5, _6, _7)                                  \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7)
#define SSTREAM_8(_1, _2, _3, _4, _5, _6, _7, _8)                              \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)
#define SSTREAM_9(_1, _2, _3, _4, _5, _6, _7, _8, _9)                          \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8) << (_9)
#define SSTREAM_10(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10)                    \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10)
#define SSTREAM_11(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11)               \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11)
#define SSTREAM_12(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12)          \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12)
#define SSTREAM_13(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13)     \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13)
#define SSTREAM_14(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                   _14)                                                        \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13) << (_14)
#define SSTREAM_15(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                   _14, _15)                                                   \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13) << (_14) << (_15)
#define SSTREAM_16(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                   _14, _15, _16)                                              \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13) << (_14) << (_15) << (_16)
#define SSTREAM_17(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                   _14, _15, _16, _17)                                         \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13) << (_14) << (_15) << (_16) \
        << (_17)
#define SSTREAM_18(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                   _14, _15, _16, _17, _18)                                    \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13) << (_14) << (_15) << (_16) \
        << (_17) << (_18)
#define SSTREAM_19(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                   _14, _15, _16, _17, _18, _19)                               \
    _ss << (_1) << (_2) << (_3) << (_4) << (_5) << (_6) << (_7) << (_8)        \
        << (_9) << (_10) << (_11) << (_12) << (_13) << (_14) << (_15) << (_16) \
        << (_17) << (_18) << (_19)
#define NARGS_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14,    \
               _15, _16, _17, _18, _19, N, ...)                                \
    N
#define NARGS(...)                                                             \
    NARGS_(__VA_ARGS__, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, \
           4, 3, 2, 1, 0)
#define CONCATENATE(x, y) x##y
#define SSTREAM_(N, ...) CONCATENATE(SSTREAM_, N)(__VA_ARGS__)
#define SSTREAM(...) SSTREAM_(NARGS(__VA_ARGS__), __VA_ARGS__)

// Logging.
#define LOG_(level, ...)                                                       \
    do {                                                                       \
        if (level < ark::get_logging().get_level()) {                          \
            break;                                                             \
        }                                                                      \
        std::stringstream _ss;                                                 \
        ark::log_header(_ss, level, __FILE__, __LINE__);                       \
        SSTREAM(__VA_ARGS__);                                                  \
        _ss << '\n';                                                           \
        std::clog << _ss.str();                                                \
    } while (0)

//
#ifdef NDEBUG
#define LOG(level, ...)                                                        \
    do {                                                                       \
        if (level != ark::DEBUG) {                                             \
            LOG_(level, __VA_ARGS__);                                          \
        }                                                                      \
    } while (0)
#else
#define LOG(level, ...) LOG_(level, __VA_ARGS__)
#endif

// Logging of an error message and exit.
#define LOGERR(...)                                                            \
    do {                                                                       \
        std::stringstream _ss;                                                 \
        ark::log_header(_ss, ark::ERROR, __FILE__, __LINE__);                  \
        SSTREAM(__VA_ARGS__);                                                  \
        _ss << '\n';                                                           \
        std::clog << _ss.str();                                                \
        throw std::runtime_error("ARK runtime error");                         \
    } while (0)

} // namespace ark

#endif // ARK_LOGGING_H_
