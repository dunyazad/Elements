#include "pch.h"

#include <Helium/HeliumCommon.h>

#include <sstream>

namespace Time
{
    std::chrono::steady_clock::time_point Now()
    {
        return std::chrono::high_resolution_clock::now();
    }

    uint64_t Microseconds(std::chrono::steady_clock::time_point& from, std::chrono::steady_clock::time_point& now)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(now - from).count();
    }

    std::chrono::steady_clock::time_point End(std::chrono::steady_clock::time_point& from, const std::string& message, int number)
    {
        auto now = std::chrono::high_resolution_clock::now();
        if (-1 == number)
        {
            printf("[%s] %.4f ms from start\n", message.c_str(), (float)(Microseconds(from, now)) / 1000.0f);
        }
        else
        {
            printf("[%6d - %s] %.4f ms from start\n", number, message.c_str(), (float)(Microseconds(from, now)) / 1000.0f);
        }
        return now;
    }

    std::string DateTime()
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        std::stringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S"); // Format: YYYYMMDD_HHMMSS
        return oss.str();
    }
}

std::string Miliseconds(const std::chrono::steady_clock::time_point beginTime, const char* tag)
{
    auto now = std::chrono::high_resolution_clock::now();
    auto timeSpan = std::chrono::duration_cast<std::chrono::nanoseconds>(now - beginTime).count();
    std::stringstream ss;
    ss << "[[[ ";
    if (nullptr != tag)
    {
        ss << tag << " - ";
    }
    ss << (float)timeSpan / 1000000.0 << " ms ]]]";
    return ss.str();
}
