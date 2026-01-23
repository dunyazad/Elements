#pragma once

#pragma warning(disable: 4251)

#include <iostream>
#include <tuple>

#include <cuda_runtime.h>

#ifdef COPPER_EXPORTS
#define COPPER_API __declspec(dllexport)
#else
#define COPPER_API __declspec(dllimport)
#endif

#ifndef __CUSTOM_DEFINITIONS_FOR_CUDA__
#define __CUSTOM_DEFINITIONS_FOR_CUDA__
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA error %s (%d): %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            assert(false); \
        } \
    }

#ifndef LaunchKernel
#define LaunchKernel_256(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel_512(KERNEL, NOE, ...) { nvtxRangePushA(#KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; KERNEL<<<NOB, NOT>>>(__VA_ARGS__); nvtxRangePop(); }
#define LaunchKernel(KERNEL, NOE, ...) LaunchKernel_512(KERNEL, NOE, __VA_ARGS__)

#define LaunchKernel2D_16x16(KERNEL, NX, NY, ...) { \
    nvtxRangePushA(#KERNEL); \
    dim3 NOT(16, 16); \
    dim3 NOB((NX + NOT.x - 1) / NOT.x, (NY + NOT.y - 1) / NOT.y); \
    KERNEL<<<NOB, NOT>>>(__VA_ARGS__); \
    nvtxRangePop(); \
}
#define LaunchKernel2D(KERNEL, NX, NY, ...) LaunchKernel2D_16x16(KERNEL, NX, NY, __VA_ARGS__)

#define LaunchLambdaKernel_256(LAMBDA_KERNEL, NOE, ...) { nvtxRangePushA(#LAMBDA_KERNEL); auto NOT = 256; auto NOB = (NOE + NOT - 1) / NOT; LaunchLambdaKernelTemplate<<<NOB, NOT>>>(LAMBDA_KERNEL, NOE, __VA_ARGS__); nvtxRangePop(); }
#define LaunchLambdaKernel_512(LAMBDA_KERNEL, NOE, ...) { nvtxRangePushA(#LAMBDA_KERNEL); auto NOT = 512; auto NOB = (NOE + NOT - 1) / NOT; LaunchLambdaKernelTemplate<<<NOB, NOT>>>(LAMBDA_KERNEL, NOE, __VA_ARGS__); nvtxRangePop(); }
#define LaunchLambdaKernel(LAMBDA_KERNEL, NOE, ...) LaunchLambdaKernel_512(LAMBDA_KERNEL, NOE, __VA_ARGS__)

#define LaunchLambdaKernel2D_16x16(LAMBDA_KERNEL, NX, NY, ...) { \
    nvtxRangePushA(#LAMBDA_KERNEL); \
    dim3 NOT(16, 16); \
    dim3 NOB((NX + NOT.x - 1) / NOT.x, (NY + NOT.y - 1) / NOT.y); \
    LaunchLambdaKernelTemplate2D<<<NOB, NOT>>>(LAMBDA_KERNEL, NX, NY, __VA_ARGS__); \
    nvtxRangePop(); \
}
#define LaunchLambdaKernel2D(LAMBDA_KERNEL, NX, NY, ...) LaunchLambdaKernel2D_16x16(LAMBDA_KERNEL, NX, NY, __VA_ARGS__)
#endif

#ifndef CUDA_TS
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#endif

#ifndef CUDA_TE
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[%s] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#endif

#ifndef CUDA_MALLOC
#define CUDA_MALLOC(ptr, size) cudaMalloc(ptr, size);
#endif

#ifndef CUDA_FREE
#define CUDA_FREE(ptr) cudaFree(ptr);
#endif

#ifndef CUDA_SAFE_FREE
#define CUDA_SAFE_FREE(ptr) { if(ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; } }
//#define CUDA_SAFE_FREE(ptr) \
//    do { \
//        if (ptr) { \
//            cudaFree(ptr); \
//            ptr = nullptr; \
//        } \
//    } while(0)
#endif

#ifndef CUDA_MEMSET
#define CUDA_MEMSET(ptr, value, size) cudaMemset(ptr, value, size);
#endif

#ifndef CUDA_COPY_D2D
#define CUDA_COPY_D2D(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice);
#endif

#ifndef CUDA_COPY_D2H
#define CUDA_COPY_D2H(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
#endif

#ifndef CUDA_COPY_H2D
#define CUDA_COPY_H2D(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
#endif

#ifndef CUDA_COPY_H2H
#define CUDA_COPY_H2H(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyHostToHost);
#endif

#ifndef CUDA_SYNC
#define CUDA_SYNC() cudaDeviceSynchronize();
#endif

#ifndef RAW_PTR
#define RAW_PTR(x) (thrust::raw_pointer_cast((x).data()))
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef DEG2RAD
#define DEG2RAD (PI/180)
#endif

#ifndef RAD2DEG
#define RAD2DEG (180/PI)
#endif

#ifndef XYZ
#define XYZ(v) (v).x, (v).y, (v).z
#endif
#ifndef XYZW
#define XYZW(v) (v).x, (v).y, (v).z, (v).w
#endif
#endif

#include <nvapi510/include/nvapi.h>
#include <nvapi510/include/NvApiDriverSettings.h>

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
