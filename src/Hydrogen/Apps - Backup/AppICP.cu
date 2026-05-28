#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdio.h>
#include <vector>

#include <robin_hood/robin_hood.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <Helium/IVisualDebugging.h>
using VD = IVisualDebugging;

#include <Helium/Serialization.hpp>
#include <Helium/Color.hpp>

#include <Core/Core.h>
#include <Core/DataStructures/SparseCells.h>
#include <Core/DataStructures/VoxelDataBase.h>
#include <Core/DataStructures/CudaHashMap.cuh>
using namespace Huvitz;

#include "Apps.h"

struct DevicePointCloud
{
    float3* d_positions;
    float3* d_normals;
    float4* d_colors;
    size_t numberOfPoints;
	DevicePointCloud() : d_positions(nullptr), d_normals(nullptr), d_colors(nullptr), numberOfPoints(0) {}
    ~DevicePointCloud()
    {
        if (d_positions) cudaFree(d_positions);
        if (d_normals) cudaFree(d_normals);
        if (d_colors) cudaFree(d_colors);
    }

    void FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const float4* h_colors,
        size_t numberOfPoints)
    {
        this->numberOfPoints = numberOfPoints;
        cudaMalloc(&d_positions, sizeof(float3) * numberOfPoints);
        cudaMemcpy(d_positions, h_positions, sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
        cudaMalloc(&d_normals, sizeof(float3) * numberOfPoints);
        cudaMemcpy(d_normals, h_normals, sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
        cudaMalloc(&d_colors, sizeof(float4) * numberOfPoints);
        cudaMemcpy(d_colors, h_colors, sizeof(float4) * numberOfPoints, cudaMemcpyHostToDevice);
    }
};

__global__ void Kernel_InsertToHashMap(
    CudaHashMap<Morton64, unsigned int>::DeviceView hashMapView,
    const float3* positions,
    uint64_t numberOfPoints,
    float blockSize)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfPoints)
        return;
    float3 position = positions[idx];
    Morton64 key = Morton64::FromPosition({position.x, position.y, position.z}, blockSize);
    hashMapView.InsertAndGet(key);
}

class AppICP : public App
{
public:
    virtual void Execute() override
    {
        CUDAWarmUp();

        PLYFormat plyA;
		plyA.Deserialize("D:\\Debug\\MergingTest\\pointclouds\\MergedPointCloud1_5.ply");

        DevicePointCloud pcdA;
        pcdA.FromHostPointers(
            (float3*)plyA.GetPoints().data(),
            (float3*)plyA.GetNormals().data(),
            (float4*)plyA.GetColors().data(),
            plyA.GetPoints().size());

        PLYFormat plyB;
        plyB.Deserialize("D:\\Debug\\MergingTest\\pointclouds\\SubMap5.ply");

        DevicePointCloud pcdB;
        pcdB.FromHostPointers(
            (float3*)plyB.GetPoints().data(),
            (float3*)plyB.GetNormals().data(),
            (float4*)plyB.GetColors().data(),
            plyB.GetPoints().size());

		CudaHashMap<Morton64, unsigned int> hashMapA(pcdA.numberOfPoints * 4);
        CudaHashMap<Morton64, unsigned int> hashMapB(pcdB.numberOfPoints * 4);

        CUDA_TS(Kernel_InsertToHashMapA);
        {
			unsigned int blockSize = 256;
            unsigned int numBlocks = (unsigned int)((pcdA.numberOfPoints + blockSize - 1) / blockSize);

            Kernel_InsertToHashMap<<<numBlocks, blockSize>>>(
                hashMapA.GetDeviceView(),
                pcdA.d_positions,
                pcdA.numberOfPoints,
				0.05f);
        }
        cudaDeviceSynchronize();
        CUDA_TE(Kernel_InsertToHashMapA);

        CUDA_TS(Kernel_InsertToHashMapB);
        {
            unsigned int blockSize = 256;
            unsigned int numBlocks = (unsigned int)((pcdB.numberOfPoints + blockSize - 1) / blockSize);

            Kernel_InsertToHashMap << <numBlocks, blockSize >> > (
                hashMapB.GetDeviceView(),
                pcdB.d_positions,
                pcdB.numberOfPoints,
                0.05f);
        }
        cudaDeviceSynchronize();
        CUDA_TE(Kernel_InsertToHashMapB);
    }

    void CUDAWarmUp()
    {
        nvDriverSetting.forceGPUPerformance();

        {
            thrust::device_vector<int> device_array(1 << 20);
            thrust::sequence(device_array.begin(), device_array.end());
            thrust::sort(device_array.begin(), device_array.end(), thrust::greater<int>());
            cudaDeviceSynchronize();
        }

        cudaStream_t stream = nullptr;
        cached_allocator* alloc = nullptr;

        cudaFree(0);
    }
};

REGISTER_APP(AppICP, "AppICP");
