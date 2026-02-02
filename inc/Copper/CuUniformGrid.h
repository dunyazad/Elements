#pragma once

#include <Copper/CopperCommon.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

struct CuPointCloud;

struct COPPER_API CuUniformGrid
{
    float voxelSize = 0.1f;
    float cellSize = 0.3f;
    float3 minBound = { FLT_MAX, FLT_MAX, FLT_MAX };
    float3 maxBound = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    int3 gridResolution = make_int3(1, 1, 1);
    int numberOfPoints = 0;

    // 각 셀의 시작과 끝 인덱스 (크기: gridRes.x * gridRes.y * gridRes.z)
    thrust::device_vector<int> cellStart;
    thrust::device_vector<int> cellEnd;

    thrust::device_vector<int> sortedPointIndices;

    CuUniformGrid() : cellSize(0.0f), numberOfPoints(0)
    {
    }
};
