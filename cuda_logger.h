/**
 * @file cuda_logger.h
 * Logger for CUDA adapted from header file wb.h for Heterogeneous Parallel Programming course (Coursera)
 * Also include the legacy cutil cudaCutilSafeCall that has been removed from the CUDA SDK.
 *   Created on: Dec 16, 2012
 *      Author: jH@CKtheRipper
 */

#pragma once

////
// Headers
////

// C++
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <sstream>
#include <string>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

////
// Logging
////

enum cudaUtilsLogLevel
{
    OFF,
    FATAL,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
    cudaUtilsLogLevelNum, // Keep this at the end
};

const char* _cudaUtilsLogLevelStr[] =
{
    "Off",
    "Fatal",
    "Error",
    "Warn",
    "Info",
    "Debug",
    "Trace",
    "***InvalidLogLevel***", // Keep this at the end
};

const char* _cudaUtilsLogLevelToStr(wbLogLevel level)
{
    assert(level >= OFF && level <= TRACE);
    return _cudaUtilsLogLevelStr[level];
}

//-----------------------------------------------------------------------------
// Begin: Ugly C++03 hack
// NVCC does not support C++11 variadic template yet

template<typename T1>
inline void _cudaUtilsLog(T1 const& p1)
{
    std::cout << p1;
}

template<typename T1, typename T2>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2)
{
    std::cout << p1 << p2;
}

template<typename T1, typename T2, typename T3>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3)
{
    std::cout << p1 << p2 << p3;
}

template<typename T1, typename T2, typename T3, typename T4>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4)
{
    std::cout << p1 << p2 << p3 << p4;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5)
{
    std::cout << p1 << p2 << p3 << p4 << p5;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6)
{
    std::cout << p1 << p2 << p3 << p4 << p5 << p6;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7)
{
    std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7, T8 const& p8)
{
    std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7, T8 const& p8, T9 const& p9)
{
    std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8 << p9;
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
inline void _cudaUtilsLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7, T8 const& p8, T9 const& p9, T10 const& p10)
{
    std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8 << p9 << p10;
}

// End: Ugly C++03 hack
//-----------------------------------------------------------------------------

#define cudaUtilsLog(level, ...)                                     \
    do                                                        \
    {                                                         \
        std::cout << _cudaUtilsLogLevelToStr(level) << " ";          \
        std::cout << __FUNCTION__ << "::" << __LINE__ << " "; \
        _cudaUtilsLog(__VA_ARGS__);                                  \
        std::cout << std::endl;                               \
    } while (0)


////
// Timer
////

// Namespace because windows.h causes errors
namespace CudaTimerNS
{
#if defined (_WIN32)
    #include <Windows.h>

    // CudaTimer class from: https://bitbucket.org/ashwin/cudatimer
    class CudaTimer
    {
    private:
        double        _freq;
        LARGE_INTEGER _time1;
        LARGE_INTEGER _time2;

    public:
        CudaTimer::CudaTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            _freq = 1.0 / freq.QuadPart;
            return;
        }

        void start()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&_time1);
            return;
        }

        void stop()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&_time2);
            return;
        }

        double value()
        {
            return (_time2.QuadPart - _time1.QuadPart) * _freq * 1000;
        }
    };
#elif defined (__APPLE__)
    #include <mach/mach_time.h>

    class CudaTimer
    {
    private:
        uint64_t _start;
        uint64_t _end;

    public:
        void start()
        {
            cudaDeviceSynchronize();
            _start = mach_absolute_time();
        }

        void stop()
        {
            cudaDeviceSynchronize();
            _end = mach_absolute_time();
        }

        double value()
        {
            static mach_timebase_info_data_t tb;

            if (0 == tb.denom)
                (void) mach_timebase_info(&tb); // Calculate ratio of mach_absolute_time ticks to nanoseconds

            return ((double) _end - (double) _start) * (tb.numer / tb.denom) / 1000000000ULL;
        }
    };
#else
    #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
        #include<time.h>
    #else
        #include<sys/time.h>
    #endif

    class CudaTimer
    {
    private:
        long long _start;
        long long _end;

        long long getTime()
        {
            long long time;
        #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0

            struct timespec ts;

            if ( 0 == clock_gettime(CLOCK_REALTIME, &ts) )
            {
                time  = 1000000000LL; // seconds->nanonseconds
                time *= ts.tv_sec;
                time += ts.tv_nsec;
            }
        #else
            struct timeval tv;

            if ( 0 == gettimeofday(&tv, NULL) )
            {
                time  = 1000000000LL; // seconds->nanonseconds
                time *= tv.tv_sec;
                time += tv.tv_usec * 1000; // ms->ns
            }
        #endif

            return time;
        }

    public:
        void start()
        {
            _start = getTime();
        }

        void stop()
        {
            _end = getTime();
        }

        double value()
        {
            return ((double) _end - (double) _start) / 1000000000LL;
        }
    };
#endif
}

enum cudaUtilsTimeType
{
    Generic,
    GPU,
    Compute,
    Copy,
    cudaUtilsTimeTypeNum, // Keep this at the end
};

const char* cudaUtilsTimeTypeStr[] =
{
    "Generic",
    "GPU    ",
    "Compute",
    "Copy   ",
    "***InvalidTimeType***", // Keep this at the end
};

const char* cudaUtilsTimeTypeToStr(cudaUtilsTimeType t)
{
    assert(t >= Generic && t < cudaUtilsTimeTypeNum);
    return cudaUtilsTimeTypeStr[t];
}

struct cudaUtilsTimerInfo
{
    cudaUtilsTimeType             type;
    std::string            name;
    CudaTimerNS::CudaTimer timer;

    bool operator == (const cudaUtilsTimerInfo& t2) const
    {
        return (type == t2.type && (0 == name.compare(t2.name)));
    }
};

typedef std::list< cudaUtilsTimerInfo> cudaUtilsTimerInfoList;
cudaUtilsTimerInfoList gTimerInfoList;

void cudaUtilsTime_start(wbTimeType timeType, const std::string timeStar)
{
    CudaTimerNS::CudaTimer timer;
    timer.start();

    cudaUtilsTimerInfo tInfo = { timeType, timeStar, timer };

    gTimerInfoList.push_front(tInfo);

    return;
}

void cudaUtilsTime_stop(wbTimeType timeType, const std::string timeStar)
{
    // Find timer

    const cudaUtilsTimerInfo searchInfo         = { timeType, timeStar };
    const cudaUtilsTimerInfoList::iterator iter = std::find( gTimerInfoList.begin(), gTimerInfoList.end(), searchInfo );

    // Stop timer and print time

    cudaUtilsTimerInfo& timerInfo = *iter;

    timerInfo.timer.stop();

    std::cout << "[" << cudaUtilsTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(10) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.name << std::endl;

    // Delete timer from list
    gTimerInfoList.erase(iter);

    return;
}

////
// Error checking
////
#define cudaUtilsCheckCall(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            cudaUtilsLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


