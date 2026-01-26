#include "pch.h"

#include <Helium/DeviceInformation.h>

#pragma comment(lib, "dxgi.lib")

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

std::string CPUInformation::GetCpuID()
{
	unsigned int regs[4] = { 0 }; // eax, ebx, ecx, edx

	// Get CPU signature (EAX=1)
#ifdef _WIN32
	__cpuid((int*)regs, 1);
#else
	__get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

	// The processor signature is stored in EAX.
	// EDX often contains feature flags which can also be part of the ID.
	// We will combine EAX and EDX to form a signature string.

	std::stringstream ss;
	ss << std::hex << std::uppercase << std::setfill('0');
	// Format: EAX value followed by EDX value
	ss << std::setw(8) << regs[0] << std::setw(8) << regs[3];

	return ss.str();
}

std::string CPUInformation::GetVendorString()
{
	unsigned int regs[4] = { 0 };
	char vendor[13];

#ifdef _WIN32
	__cpuid((int*)regs, 0);
#else
	__get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif

	// Vendor string is stored in EBX, EDX, ECX
	*reinterpret_cast<int*>(vendor) = regs[1];     // EBX
	*reinterpret_cast<int*>(vendor + 4) = regs[3]; // EDX
	*reinterpret_cast<int*>(vendor + 8) = regs[2]; // ECX
	vendor[12] = '\0';

	return std::string(vendor);
}

std::vector<GpuData> GPUInformation::GetGPUList()
{
	std::vector<GpuData> gpuList;

	IDXGIFactory* factory = nullptr;

	HRESULT result = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
	if (FAILED(result))
	{
		return gpuList;
	}

	IDXGIAdapter* adapter = nullptr;
	unsigned int index = 0;

	while (factory->EnumAdapters(index, &adapter) != DXGI_ERROR_NOT_FOUND)
	{
		DXGI_ADAPTER_DESC adapterDesc;
		adapter->GetDesc(&adapterDesc);

		GpuData data;
		data.description = adapterDesc.Description;
		data.vendorId = adapterDesc.VendorId;
		data.deviceId = adapterDesc.DeviceId;
		data.videoMemory = adapterDesc.DedicatedVideoMemory;

		gpuList.push_back(data);

		adapter->Release();
		index++;
	}

	factory->Release();
	return gpuList;
}

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