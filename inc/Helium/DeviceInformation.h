#pragma once

#include <Helium/HeliumCommon.h>

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <windows.h>
#include <dxgi.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

class HELIUM_API CPUInformation
{
public:
    static std::string GetCpuID();
    static std::string GetVendorString();
};

struct HELIUM_API GpuData
{
    std::wstring description;
    unsigned int vendorId;
    unsigned int deviceId;
    size_t videoMemory;
};

class HELIUM_API GPUInformation
{
public:
    static std::vector<GpuData> GetGPUList();
};

//int main() {
//    std::string cpuId = CPUInformation::GetCpuID();
//    std::string vendor = CPUInformation::GetVendorString();
//
//    std::cout << "CPU Vendor: " << vendor << std::endl;
//    std::cout << "CPU ID (Signature): " << cpuId << std::endl;
//
//    return 0;
//}

//int main() {
//    std::wcout.imbue(std::locale(""));
//
//    // GPUInformation 클래스 사용
//    std::vector<GpuData> gpus = GPUInformation::GetGpuList();
//
//    if (gpus.empty()) {
//        std::cout << "No GPU found or error occurred." << std::endl;
//        return 0;
//    }
//
//    std::cout << "=== Detected GPUs ===" << std::endl;
//    for (size_t i = 0; i < gpus.size(); ++i) {
//        const auto& gpu = gpus[i];
//
//        std::wcout << "GPU #" << i << ": " << gpu.description << std::endl;
//        std::cout << "  Vendor ID: 0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << gpu.vendorId << std::endl;
//        std::cout << "  Device ID: 0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << gpu.deviceId << std::endl;
//
//        std::cout << std::dec;
//        std::cout << "  VRAM: " << (gpu.videoMemory / 1024 / 1024) << " MB" << std::endl;
//        std::cout << "---------------------" << std::endl;
//    }
//
//    return 0;
//}