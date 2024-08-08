// suppress_warnings.h

#ifndef SUPPRESS_WARNINGS_H
#define SUPPRESS_WARNINGS_H

#if defined(__GNUC__) || defined(__clang__)
#define SUPPRESS_SIGN_CONVERSION_WARNINGS _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wsign-conversion\"")
#define RESTORE_WARNINGS _Pragma("GCC diagnostic pop")
#else
#define SUPPRESS_SIGN_CONVERSION_WARNINGS
#define RESTORE_WARNINGS
#endif

#endif // SUPPRESS_WARNINGS_H