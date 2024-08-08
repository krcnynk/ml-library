// config.h
#ifndef CONFIG_H
#define CONFIG_H

#ifdef __GNUC__
#define ML_CACHE_ALIGN __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define ML_CACHE_ALIGN __declspec(align(64))
#else
#define ML_CACHE_ALIGN
#endif

#endif // CONFIG_H
