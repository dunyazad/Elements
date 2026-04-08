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
using namespace Huvitz;

#include <Core/DataStructures/CudaHashMap.cuh>

#include <Core/Operations/OperationPointCloudMerge.h>

namespace fs = std::filesystem;

using Morton64 = Huvitz::Morton64;

#include "Apps.h"

struct PointCloudData
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float4* colors = nullptr;
	unsigned int numberOfPoints = 0;
};

struct DVoxel
{
    float sdf[6];
    float weight[6];
    float nx[6], ny[6], nz[6]; /* 각 방향 슬롯별 독립 법선 */

    void initialize()
    {
        for (int i = 0; i < 6; ++i)
        {
            sdf[i] = 1.0f;
            weight[i] = 0.0f;
            nx[i] = ny[i] = nz[i] = 0.0f;
        }
    }
};

class AppVoxelCache : public App
{
public:
    virtual void Execute() override
    {
        CUDAWarmUp();

        printf("sizeof(DVoxel) = %zu\n", sizeof(DVoxel));
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

REGISTER_APP(AppVoxelCache, "AppVoxelCache");
