#pragma once

#include <Copper/CopperCommon.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

struct CuPointCloud;

struct CuCellStats
{
    float3 cellMin;
    float3 cellMax;
    int pointCount;
    float3 pointCentroid;
    float3 avgNormal;
    float3 pcaNormal;
};

struct COPPER_API CuSparseCells
{
    int3 gridSize;
    int numberOfCells = 0;
    float cellSize = 0.0f;
    float3 worldOrigin;

    thrust::device_vector<int> hashCodes;
    thrust::device_vector<int> cellStartIndices;
    thrust::device_vector<int> cellEndIndices;

    CuSparseCells();

    void Build(CuPointCloud* cloud);
    void Build(CuPointCloud* cloud, float cellSize);

    thrust::device_vector<float> ApplySOR(CuPointCloud* cloud, int k = 30, float stdDevMult = 1.0f);

    thrust::device_vector<float> ApplyPFOR(CuPointCloud* cloud, int k = 30, float distanceThreshold = 0.085f);

    thrust::device_vector<float> ApplyNND(CuPointCloud* cloud, int k = 30);

    thrust::device_vector<float> ApplyLDE(CuPointCloud* cloud, float radius);

    //thrust::device_vector<float> ApplyKDE(CuPointCloud* cloud, float bandwidth);

    std::vector<std::pair<float3, float3>> GetActiveCellBounds();

    void ColorizePointsByCell(CuPointCloud* cloud);

    std::vector<CuCellStats> GetActiveCellStats(CuPointCloud* cloud);
    
private:
    float computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier);
};
