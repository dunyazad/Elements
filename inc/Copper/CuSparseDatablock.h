#pragma once

#include <Copper/CopperCommon.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

struct CuPointCloud;

struct COPPER_API CuSparseDataBlock
{
    int3 gridSize;
    int numberOfCells = 0;
    float cellSize = 0.0f;
    float3 worldOrigin;

    thrust::device_vector<int> hashCodes;
    thrust::device_vector<int> cellStartIndices;
    thrust::device_vector<int> cellEndIndices;

    CuSparseDataBlock();

    void Build(CuPointCloud* cloud);

    void ApplySOR(CuPointCloud* cloud, int k = 30, float stdDevMult = 1.0f);

private:
    float computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier);
};
