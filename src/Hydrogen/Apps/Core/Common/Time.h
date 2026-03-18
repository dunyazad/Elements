#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace Huvitz
{
	namespace Time
	{
		inline std::chrono::steady_clock::time_point Now()
		{
			return std::chrono::high_resolution_clock::now();
		}

		inline uint64_t Microseconds(const std::chrono::steady_clock::time_point& from, const std::chrono::steady_clock::time_point& now)
		{
			return std::chrono::duration_cast<std::chrono::microseconds>(now - from).count();
		}

		inline std::chrono::steady_clock::time_point End(const std::chrono::steady_clock::time_point& from, const std::string& message, int number)
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

		inline std::string DateTime()
		{
			auto t = std::time(nullptr);
			struct tm timeDetails = {0};
			errno_t error = localtime_s(&timeDetails, &t);
			auto tmd = timeDetails;

			std::stringstream oss;
			oss << std::put_time(&tmd, "%Y%m%d_%H%M%S"); // Format: YYYYMMDD_HHMMSS
			return oss.str();
		}
	}

	inline std::string Miliseconds(const std::chrono::steady_clock::time_point beginTime, const char* tag)
	{
		auto now = Time::Now();
		auto timeSpan = Time::Microseconds(beginTime, now);
		std::stringstream ss;
		ss << "[[[ ";
		if (nullptr != tag)
		{
			ss << tag << " - ";
		}
		ss << (float)timeSpan / 1000000.0 << " ms ]]]";
		return ss.str();
	}
}

#ifndef TS
#define TS(name) auto time_##name = Huvitz::Time::Now();
#define TE(name) std::cout << Huvitz::Miliseconds(time_##name, #name) << std::endl;
#endif
