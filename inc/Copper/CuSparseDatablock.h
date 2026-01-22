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
    void Build(CuPointCloud* cloud, float cellSize);

    thrust::device_vector<float> ApplySOR(CuPointCloud* cloud, int k = 30, float stdDevMult = 1.0f);

    thrust::device_vector<float> ApplyPFOR(CuPointCloud* cloud, int k = 30, float distanceThreshold = 0.085f);

    thrust::device_vector<float> ApplyNND(CuPointCloud* cloud, int k = 30);

    thrust::device_vector<float> ApplyLDE(CuPointCloud* cloud, float radius);

    //thrust::device_vector<float> ApplyKDE(CuPointCloud* cloud, float bandwidth);

private:
    float computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier);
};
