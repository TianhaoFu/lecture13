/*!
 * \file logging.h
 * \brief defines a minimum set of logging macros
 *        that mimics the glog behavior.
 *
 *
 *  It allows you to write code like CHECK(condition) << message;
 */
#ifndef NEEDLE_LOGGING_H_
#define NEEDLE_LOGGING_H_

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NEEDLE_LOG_FATAL_THROW
#define NEEDLE_LOG_FATAL_THROW 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define NEEDLE_ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define NEEDLE_ALWAYS_INLINE __forceinline
#else
#define NEEDLE_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
#define NEEDLE_NO_INLINE __declspec(noinline)
#else
#define NEEDLE_NO_INLINE __attribute__((noinline))
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4722)
#pragma warning(disable : 4068)
#endif

namespace needle {
/*!
 * \brief exception class that will be thrown by
 *  default logger if NEEDLE_LOG_FATAL_THROW == 1
 */
struct Error : public std::runtime_error {
  /*!
   * \brief constructor
   * \param s the error message
   */
  explicit Error(const std::string &s) : std::runtime_error(s) {}
};

// This function allows us to ignore sign comparison in the right scope.
#define DEFINE_CHECK_FUNC(name, op)                                            \
  template <typename X, typename Y>                                            \
  NEEDLE_ALWAYS_INLINE bool LogCheck##name(const X &x, const Y &y) {           \
    return (x op y);                                                           \
  }                                                                            \
  NEEDLE_ALWAYS_INLINE bool LogCheck##name(int x, int y) {                     \
    return LogCheck##name<int, int>(x, y);                                     \
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
DEFINE_CHECK_FUNC(_LT, <)
DEFINE_CHECK_FUNC(_GT, >)
DEFINE_CHECK_FUNC(_LE, <=)
DEFINE_CHECK_FUNC(_GE, >=)
DEFINE_CHECK_FUNC(_EQ, ==)
DEFINE_CHECK_FUNC(_NE, !=)
#pragma GCC diagnostic pop

#define CHECK_BINARY_OP(name, op, x, y)                                        \
  if (!(needle::LogCheck##name((x), (y))))                                     \
  needle::LogMessageFatal(__FILE__, __LINE__).stream()                         \
      << "Check failed: " << #x " " #op " " #y << " (" << (x) << " vs. "       \
      << (y) << ") "                                                           \
      << ": " /* CHECK_XX(x, y) requires x and y can be serialized to string.  \
                 Use CHECK(x OP y) otherwise. NOLINT(*) */

// Always-on checking
#define CHECK(x)                                                               \
  if (!(x))                                                                    \
  needle::LogMessageFatal(__FILE__, __LINE__).stream()                         \
      << "Check failed: " #x << ": "
#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)
#define CHECK_NOTNULL(x)                                                       \
  ((x) == NULL ? needle::LogMessageFatal(__FILE__, __LINE__).stream()          \
                     << "Check  notnull: " #x << ' ',                          \
   (x) : (x)) // NOLINT(*)

#define LOG_INFO needle::LogMessage(__FILE__, __LINE__)
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LOG_INFO
#define LOG_FATAL needle::LogMessageFatal(__FILE__, __LINE__)
#define LOG_QFATAL LOG_FATAL

#define LOG(severity) LOG_##severity.stream()

class DateLogger {
public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif
  }
  const char *HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else
    time_t time_value = time(NULL);
    struct tm *pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else
    pnow = localtime(&time_value); // NOLINT(*)
#endif
    snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d", pnow->tm_hour,
             pnow->tm_min, pnow->tm_sec);
#endif
    return buffer_;
  }

private:
  char buffer_[9];
};

class LogMessage {
public:
  LogMessage(const char *file, int line)
      :
#ifdef __ANDROID__
        log_stream_(std::cout)
#else
        log_stream_(std::cerr)
#endif
  {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~LogMessage() { log_stream_ << '\n'; }
  std::ostream &stream() { return log_stream_; }

protected:
  std::ostream &log_stream_;

private:
  DateLogger pretty_date_;
  LogMessage(const LogMessage &);
  void operator=(const LogMessage &);
};

#if NEEDLE_LOG_FATAL_THROW == 0
class LogMessageFatal : public LogMessage {
public:
  LogMessageFatal(const char *file, int line) : LogMessage(file, line) {}
  ~LogMessageFatal() {
    log_stream_ << "\n" << StackTrace(1, LogStackTraceLevel()) << "\n";
    abort();
  }

private:
  LogMessageFatal(const LogMessageFatal &);
  void operator=(const LogMessageFatal &);
};
#else
class LogMessageFatal {
public:
  LogMessageFatal(const char *file, int line) {
    Entry::ThreadLocal()->Init(file, line);
  }
  std::ostringstream &stream() { return Entry::ThreadLocal()->log_stream; }
  NEEDLE_NO_INLINE ~LogMessageFatal() noexcept(false) {
    throw Entry::ThreadLocal()->Finalize();
  }

private:
  struct Entry {
    std::ostringstream log_stream;
    NEEDLE_NO_INLINE void Init(const char *file, int line) {
      DateLogger date;
      log_stream.str("");
      log_stream.clear();
      log_stream << "[" << date.HumanDate() << "] " << file << ":" << line
                 << ": ";
    }
    needle::Error Finalize() { return needle::Error(log_stream.str()); }
    NEEDLE_NO_INLINE static Entry *ThreadLocal() {
      static thread_local Entry *result = new Entry();
      return result;
    }
  };
  LogMessageFatal(const LogMessageFatal &);
  void operator=(const LogMessageFatal &);
};
#endif
} // namespace needle
#endif // NEEDLE_LOGGING_H_
