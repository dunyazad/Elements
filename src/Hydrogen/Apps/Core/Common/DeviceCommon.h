#pragma once

#include <Core/CodingSugar.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>

namespace Huvitz
{
#ifdef __CUDACC__
	template <typename T>
	__device__ __forceinline__ T fetch_val(const T* ptr, int idx)
	{
		return __ldg(ptr + idx);
	}

	template <>
	__device__ __forceinline__ float3 fetch_val(const float3* ptr, int idx)
	{
		float3 ret;
		const float* base = reinterpret_cast<const float*>(ptr + idx);
		ret.x = __ldg(base + 0);
		ret.y = __ldg(base + 1);
		ret.z = __ldg(base + 2);

		return ret;
	}

	__device__ __forceinline__ unsigned short atomicAddUShort(unsigned short* address, unsigned short val)
	{
		unsigned int* baseAddress = (unsigned int*)((size_t)address & ~3);
		unsigned int shift = ((size_t)address & 2) << 3;
		unsigned int old, assumed, newVal;
		old = *baseAddress;

		do
		{
			assumed = old;
			unsigned short oldVal = (unsigned short)((assumed >> shift) & 0xFFFF);
			unsigned int sum = (unsigned int)oldVal + (unsigned int)val;
			unsigned short finalSum = (sum > 0xFFFF) ? 0xFFFF : (unsigned short)sum;
			newVal = (assumed & ~(0xFFFF << shift)) | (static_cast<unsigned int>(finalSum) << shift);
			old = atomicCAS(baseAddress, assumed, newVal);

		} while (assumed != old);

		return (unsigned short)((old >> shift) & 0xFFFF);
	}

	__device__ inline float atomicMinFloat(float* address, float val)
	{
		int* addressAsInt = (int*)address;
		int old = *addressAsInt;
		int assumed;

		while (val < __int_as_float(old))
		{
			assumed = old;
			old = atomicCAS(addressAsInt, assumed, __float_as_int(val));

			if (assumed == old)
			{
				break;
			}
		}

		return __int_as_float(old);
	}

    __device__ inline void atomicMinFloatCAS(float* addr, float val)
    {
        int* addr_i = (int*)addr;
        int  old_i = *addr_i;
        int  val_i = __float_as_int(val);

        while (__int_as_float(old_i) > val)
        {
            int prev = atomicCAS(addr_i, old_i, val_i);
            if (prev == old_i) break;
            old_i = prev;
        }
    }

    __device__ inline void atomicMaxFloatCAS(float* addr, float val)
    {
        int* addr_i = (int*)addr;
        int  old_i = *addr_i;
        int  val_i = __float_as_int(val);

        while (__int_as_float(old_i) < val)
        {
            int prev = atomicCAS(addr_i, old_i, val_i);
            if (prev == old_i) break;
            old_i = prev;
        }
    }

	__device__ inline uint32_t StrongHash(uint64_t key, uint32_t maxBlocks)
	{
		key ^= key >> 33;
		key *= 0xff51afd7ed558ccdULL;
		key ^= key >> 33;
		key *= 0xc4ceb9fe1a85ec53ULL;
		key ^= key >> 33;
		return static_cast<uint32_t>(key % maxBlocks);
	}

	struct MinFloat3
	{
		__host__ __device__ float3 operator()(const float3& a, const float3& b) const
		{
			return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
		}
	};

	struct MaxFloat3
	{
		__host__ __device__ float3 operator()(const float3& a, const float3& b) const
		{
			return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
		}
	};
#endif
}
	
#define FETCH(ptr, idx) fetch_val(ptr, idx)

#include <nvapi.h>
#include <NvApiDriverSettings.h>

#ifdef __NVOPTIMUSENABLEMENT__
#define __NVOPTIMUSENABLEMENT__
extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#define PREFERRED_PSTATE_ID 0x0000001B
#define PREFERRED_PSTATE_PREFER_MAX 0x00000000
#define PREFERRED_PSTATE_PREFER_MIN 0x00000001
#endif

inline bool ForceGPUPerformance()
{
    NvAPI_Status status;

    status = NvAPI_Initialize();
    if (status != NVAPI_OK)
    {
        return false;
    }

    NvDRSSessionHandle hSession = 0;
    status = NvAPI_DRS_CreateSession(&hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    // (2) load all the system settings into the session
    status = NvAPI_DRS_LoadSettings(hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    NvDRSProfileHandle hProfile = 0;
    status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (status != NVAPI_OK)
    {
        return false;
    }

    NVDRS_SETTING drsGet = { 0, };
    drsGet.version = NVDRS_SETTING_VER;
    status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
    if (status != NVAPI_OK)
    {
        return false;
    }
    //auto m_gpu_performance = drsGet.u32CurrentValue;

    NVDRS_SETTING drsSetting = { 0, };
    drsSetting.version = NVDRS_SETTING_VER;
    drsSetting.settingId = PREFERRED_PSTATE_ID;
    drsSetting.settingType = NVDRS_DWORD_TYPE;
    drsSetting.u32CurrentValue = PREFERRED_PSTATE_PREFER_MAX;

    status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
    if (status != NVAPI_OK)
    {
        return false;
    }

    status = NvAPI_DRS_SaveSettings(hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    // (6) We clean up. This is analogous to doing a free()
    NvAPI_DRS_DestroySession(hSession);
    hSession = 0;

    return true;
}

inline std::tuple<double, double> CheckDeviceMemory(const char* tag = nullptr)
{
    size_t free_byte = 0;
    size_t total_byte = 0;

    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
    if (err != cudaSuccess)
    {
        printf("CUDA Error in MemInfo: %s\n", cudaGetErrorString(err));
        return { 0.0, 0.0 };
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    if (nullptr != tag)
    {
        printf("[%s] using: %.4f GB / Total %.4f GB (%.2f%%)\n",
            tag,
            used_db / (1024.0 * 1024.0 * 1024.0),
            total_db / (1024.0 * 1024.0 * 1024.0),
            (used_db / total_db) * 100.0);

        //printf("[%s] using: %.4f MB / Total %.4f MB (%.2f%%)\n",
        //    tag,
        //    used_db / (1024.0 * 1024.0),
        //    total_db / (1024.0 * 1024.0),
        //    (used_db / total_db) * 100.0);
    }
    return { used_db, total_db };
}