#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <Copper/CuTransferFunction.h>

#define LDG(ptr, idx) __ldg(&(ptr)[idx])

template <typename T>
__device__ __forceinline__ T fetch_val(const T* ptr, int idx)
{
    return __ldg(&ptr[idx]);
}

template <>
__device__ __forceinline__ float3 fetch_val(const float3* ptr, int idx)
{
    float3 ret;
    ret.x = __ldg(&ptr[idx].x);
    ret.y = __ldg(&ptr[idx].y);
    ret.z = __ldg(&ptr[idx].z);
    return ret;
}

#define FETCH(ptr, idx) fetch_val(ptr, idx)

__global__ void computeDensityKernel(
    const float3* __restrict__ positions,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    float* __restrict__ outValues,
    int numParticles,
    float radius,
    int mode,
    float cellSize,
    int3 gridSize,
    float3 worldOrigin);

__device__ __forceinline__ float getDistSq(float3 a, float3 b)
{
    float3 d = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

__global__ void applyTransferFunctionKernel(
    uchar3* colors,
    const float* values,
    int numParticles,
    CuTransferFunction tf);

#endif // __CUDACC__
