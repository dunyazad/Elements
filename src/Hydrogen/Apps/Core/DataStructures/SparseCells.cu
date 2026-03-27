#include <Core/DataStructures/SparseCells.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/DataStructures/PCD.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>

namespace Huvitz
{
    __constant__ int offsetX[14] = { 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    __constant__ int offsetY[14] = { 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    __constant__ int offsetZ[14] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    __device__ __forceinline__ int SpatialHash(int gridX, int gridY, int gridZ, int mask)
    {
        unsigned int hashValue = ((unsigned int)gridX * 92837111u)
            ^ ((unsigned int)gridY * 689287499u)
            ^ ((unsigned int)gridZ * 283923481u);
        return (int)(hashValue & (unsigned int)mask);
    }

    __global__ void Kernel_InsertPoints(
        const float3* __restrict__ points,
        int* hashTable,
        int* nextPoint,
        int numberOfPoints,
        float inverseCellSize,
        float3 origin,
        int tableMask)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numberOfPoints)
        {
            return;
        }

        float3 position = points[index];
        int gridX = __float2int_rd((position.x - origin.x) * inverseCellSize);
        int gridY = __float2int_rd((position.y - origin.y) * inverseCellSize);
        int gridZ = __float2int_rd((position.z - origin.z) * inverseCellSize);

        int slot = SpatialHash(gridX, gridY, gridZ, tableMask);
        nextPoint[index] = atomicExch(&hashTable[slot], index);
    }

    __global__ void Kernel_InitLabels(unsigned int* labels, int numberOfPoints)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < numberOfPoints)
        {
            labels[index] = (unsigned int)index;
        }
    }

    __global__ void Kernel_UnionFind_Link_SameCell(
        const int* __restrict__ hashTable,
        const int* __restrict__ nextPoint,
        const float3* __restrict__ positions,
        const float3* __restrict__ normals,
        unsigned int* labels,
        int numberOfPoints,
        float squaredDistance,
        float inverseCellSize,
        float3 origin,
        int tableMask)
    {
        int indexI = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexI >= numberOfPoints)
        {
            return;
        }

        float3 positionI = positions[indexI];
        int gridX = __float2int_rd((positionI.x - origin.x) * inverseCellSize);
        int gridY = __float2int_rd((positionI.y - origin.y) * inverseCellSize);
        int gridZ = __float2int_rd((positionI.z - origin.z) * inverseCellSize);

        int slot = SpatialHash(gridX, gridY, gridZ, tableMask);
        int indexJ = hashTable[slot];

        while (indexJ != -1)
        {
            if (indexI < indexJ)
            {
                unsigned int labelI = labels[indexI];
                unsigned int labelJ = labels[indexJ];
                if (labelI != labelJ)
                {
                    float3 positionJ = positions[indexJ];
                    float deltaX = positionI.x - positionJ.x;
                    float deltaY = positionI.y - positionJ.y;
                    float deltaZ = positionI.z - positionJ.z;

                    if (deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ <= squaredDistance)
                    {
                        unsigned int rootI = labelI;
                        unsigned int rootJ = labelJ;
                        while (rootI != rootJ)
                        {
                            unsigned int lowerBound = rootI < rootJ ? rootI : rootJ;
                            unsigned int upperBound = rootI < rootJ ? rootJ : rootI;
                            unsigned int oldBound = atomicMin(&labels[upperBound], lowerBound);
                            if (oldBound == upperBound)
                            {
                                break;
                            }
                            rootI = oldBound;
                            rootJ = lowerBound;
                        }
                    }
                }
            }
            indexJ = nextPoint[indexJ];
        }
    }

    __global__ void Kernel_UnionFind_Link_SameCell_Angle(
        const int* __restrict__ hashTable,
        const int* __restrict__ nextPoint,
        const float3* __restrict__ positions,
        const float3* __restrict__ normals,
        unsigned int* labels,
        int numberOfPoints,
        float squaredDistance,
        float cosAngleThreshold,
        float inverseCellSize,
        float3 origin,
        int tableMask)
    {
        int indexI = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexI >= numberOfPoints)
        {
            return;
        }

        float3 positionI = positions[indexI];
        int gridX = __float2int_rd((positionI.x - origin.x) * inverseCellSize);
        int gridY = __float2int_rd((positionI.y - origin.y) * inverseCellSize);
        int gridZ = __float2int_rd((positionI.z - origin.z) * inverseCellSize);

        int slot = SpatialHash(gridX, gridY, gridZ, tableMask);
        int indexJ = hashTable[slot];

        while (indexJ != -1)
        {
            if (indexI < indexJ)
            {
                unsigned int labelI = labels[indexI];
                unsigned int labelJ = labels[indexJ];
                if (labelI != labelJ)
                {
                    float3 positionJ = positions[indexJ];
                    float deltaX = positionI.x - positionJ.x;
                    float deltaY = positionI.y - positionJ.y;
                    float deltaZ = positionI.z - positionJ.z;

                    if (deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ <= squaredDistance)
                    {
                        float3 normalI = normals[indexI];
                        float3 normalJ = normals[indexJ];
                        float dotProduct = normalI.x * normalJ.x + normalI.y * normalJ.y + normalI.z * normalJ.z;

                        if (dotProduct >= cosAngleThreshold)
                        {
                            unsigned int rootI = labelI;
                            unsigned int rootJ = labelJ;
                            while (rootI != rootJ)
                            {
                                unsigned int lowerBound = rootI < rootJ ? rootI : rootJ;
                                unsigned int upperBound = rootI < rootJ ? rootJ : rootI;
                                unsigned int oldBound = atomicMin(&labels[upperBound], lowerBound);
                                if (oldBound == upperBound)
                                {
                                    break;
                                }
                                rootI = oldBound;
                                rootJ = lowerBound;
                            }
                        }
                    }
                }
            }
            indexJ = nextPoint[indexJ];
        }
    }

    __global__ void Kernel_UnionFind_Link_Neighbors(
        const int* __restrict__ hashTable,
        const int* __restrict__ nextPoint,
        const float3* __restrict__ positions,
        const float3* __restrict__ normals,
        unsigned int* labels,
        int numberOfPoints,
        float squaredDistance,
        float inverseCellSize,
        float3 origin,
        int tableMask)
    {
        int indexI = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexI >= numberOfPoints)
        {
            return;
        }

        float3 positionI = positions[indexI];
        int gridX = __float2int_rd((positionI.x - origin.x) * inverseCellSize);
        int gridY = __float2int_rd((positionI.y - origin.y) * inverseCellSize);
        int gridZ = __float2int_rd((positionI.z - origin.z) * inverseCellSize);

#pragma unroll
        for (int neighborIndex = 1; neighborIndex < 14; ++neighborIndex)
        {
            int slot = SpatialHash(gridX + offsetX[neighborIndex], gridY + offsetY[neighborIndex], gridZ + offsetZ[neighborIndex], tableMask);
            int indexJ = hashTable[slot];

            while (indexJ != -1)
            {
                unsigned int labelI = labels[indexI];
                unsigned int labelJ = labels[indexJ];
                if (labelI != labelJ)
                {
                    float3 positionJ = positions[indexJ];
                    float deltaX = positionI.x - positionJ.x;
                    float deltaY = positionI.y - positionJ.y;
                    float deltaZ = positionI.z - positionJ.z;

                    if (deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ <= squaredDistance)
                    {
                        unsigned int rootI = labelI;
                        unsigned int rootJ = labelJ;
                        while (rootI != rootJ)
                        {
                            unsigned int lowerBound = rootI < rootJ ? rootI : rootJ;
                            unsigned int upperBound = rootI < rootJ ? rootJ : rootI;
                            unsigned int oldBound = atomicMin(&labels[upperBound], lowerBound);
                            if (oldBound == upperBound)
                            {
                                break;
                            }
                            rootI = oldBound;
                            rootJ = lowerBound;
                        }
                    }
                }
                indexJ = nextPoint[indexJ];
            }
        }
    }

    __global__ void Kernel_UnionFind_Link_Neighbors_Angle(
        const int* __restrict__ hashTable,
        const int* __restrict__ nextPoint,
        const float3* __restrict__ positions,
        const float3* __restrict__ normals,
        unsigned int* labels,
        int numberOfPoints,
        float squaredDistance,
        float cosAngleThreshold,
        float inverseCellSize,
        float3 origin,
        int tableMask)
    {
        int indexI = blockIdx.x * blockDim.x + threadIdx.x;
        if (indexI >= numberOfPoints)
        {
            return;
        }

        float3 positionI = positions[indexI];
        int gridX = __float2int_rd((positionI.x - origin.x) * inverseCellSize);
        int gridY = __float2int_rd((positionI.y - origin.y) * inverseCellSize);
        int gridZ = __float2int_rd((positionI.z - origin.z) * inverseCellSize);

#pragma unroll
        for (int neighborIndex = 1; neighborIndex < 14; ++neighborIndex)
        {
            int slot = SpatialHash(gridX + offsetX[neighborIndex], gridY + offsetY[neighborIndex], gridZ + offsetZ[neighborIndex], tableMask);
            int indexJ = hashTable[slot];

            while (indexJ != -1)
            {
                unsigned int labelI = labels[indexI];
                unsigned int labelJ = labels[indexJ];
                if (labelI != labelJ)
                {
                    float3 positionJ = positions[indexJ];
                    float deltaX = positionI.x - positionJ.x;
                    float deltaY = positionI.y - positionJ.y;
                    float deltaZ = positionI.z - positionJ.z;

                    if (deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ <= squaredDistance)
                    {
                        float3 normalI = normals[indexI];
                        float3 normalJ = normals[indexJ];
                        float dotProduct = normalI.x * normalJ.x + normalI.y * normalJ.y + normalI.z * normalJ.z;

                        if (dotProduct >= cosAngleThreshold)
                        {
                            unsigned int rootI = labelI;
                            unsigned int rootJ = labelJ;
                            while (rootI != rootJ)
                            {
                                unsigned int lowerBound = rootI < rootJ ? rootI : rootJ;
                                unsigned int upperBound = rootI < rootJ ? rootJ : rootI;
                                unsigned int oldBound = atomicMin(&labels[upperBound], lowerBound);
                                if (oldBound == upperBound)
                                {
                                    break;
                                }
                                rootI = oldBound;
                                rootJ = lowerBound;
                            }
                        }
                    }
                }
                indexJ = nextPoint[indexJ];
            }
        }
    }

    __global__ void Kernel_UnionFind_Compress(unsigned int* labels, int numberOfPoints)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numberOfPoints)
        {
            return;
        }

        unsigned int root = labels[index];
        while (root != labels[root])
        {
            root = labels[root];
        }
        labels[index] = root;
    }

    SparseCells::SparseCells()
        : hashTable(nullptr), hashTableCapacity(0), tableMask(0)
        , nextPoint(nullptr), nextPointCapacity(0)
    {
    }

    SparseCells::~SparseCells()
    {
        if (hashTable)
        {
            cudaFree(hashTable);
        }
        if (nextPoint)
        {
            cudaFree(nextPoint);
        }
    }

    void SparseCells::Build(PCD* cloud, float cellSize, CUstream_st* stream)
    {
        if (!cloud || cloud->size() == 0)
        {
            return;
        }

        float3 minPoint = cloud->GetAABB().min;
        float3 maxPoint = cloud->GetAABB().max;

        gridSize = {
            (int)ceilf((maxPoint.x - minPoint.x) / cellSize) + 1,
            (int)ceilf((maxPoint.y - minPoint.y) / cellSize) + 1,
            (int)ceilf((maxPoint.z - minPoint.z) / cellSize) + 1
        };

        BuildInternal(cloud->GetPositions(), cloud->size(), cellSize, minPoint, stream);
    }

    void SparseCells::Build(float3* points, size_t numberOfPoints, float cellSize, CUstream_st* stream)
    {
        if (numberOfPoints == 0 || !points)
        {
            return;
        }

        gridSize = { 0, 0, 0 };
        BuildInternal(points, numberOfPoints, cellSize, { 0.0f, 0.0f, 0.0f }, stream);
    }

    void SparseCells::BuildInternal(float3* points, size_t numberOfPoints, float cellSize, float3 origin, CUstream_st* stream)
    {
        this->cellSize = cellSize;
        this->worldOrigin = origin;

        size_t tableSize = 1u << 16;
        while (tableSize < numberOfPoints * 2)
        {
            tableSize <<= 1;
        }
        tableMask = (int)(tableSize - 1);

        if (tableSize > hashTableCapacity)
        {
            if (hashTable)
            {
                cudaFree(hashTable);
            }
            hashTableCapacity = tableSize;
            cudaMalloc(&hashTable, sizeof(int) * hashTableCapacity);
        }

        if (numberOfPoints > nextPointCapacity)
        {
            if (nextPoint)
            {
                cudaFree(nextPoint);
            }
            nextPointCapacity = (size_t)(numberOfPoints * 1.5);
            cudaMalloc(&nextPoint, sizeof(int) * nextPointCapacity);
        }

        cudaMemsetAsync(hashTable, 0xFF, sizeof(int) * tableSize, stream);

        int blockSize = 256;
        int gridSizeCuda = (int)((numberOfPoints + blockSize - 1) / blockSize);

        Kernel_InsertPoints << <gridSizeCuda, blockSize, 0, stream >> > (
            points, hashTable, nextPoint,
            (int)numberOfPoints, 1.0f / cellSize, worldOrigin, tableMask);
    }

    void SparseCells::ApplyClustering(float3* points, float3* normals, size_t numberOfPoints, unsigned int* labels, float clusterDistance, CUstream_st* stream)
    {
        if (numberOfPoints == 0 || !labels || !hashTable || !normals)
        {
            return;
        }

        int blockSize = 256;
        int gridSizeCuda = (int)((numberOfPoints + blockSize - 1) / blockSize);
        float squaredDistance = clusterDistance * clusterDistance;
        float inverseCellSize = 1.0f / cellSize;

        Kernel_InitLabels << <gridSizeCuda, blockSize, 0, stream >> > (labels, (int)numberOfPoints);

        Kernel_UnionFind_Link_SameCell << <gridSizeCuda, blockSize, 0, stream >> > (
            hashTable, nextPoint, points, normals,
            labels, (int)numberOfPoints, squaredDistance, inverseCellSize, worldOrigin, tableMask);

        Kernel_UnionFind_Compress << <gridSizeCuda, blockSize, 0, stream >> > (labels, (int)numberOfPoints);

        Kernel_UnionFind_Link_Neighbors << <gridSizeCuda, blockSize, 0, stream >> > (
            hashTable, nextPoint, points, normals,
            labels, (int)numberOfPoints, squaredDistance, inverseCellSize, worldOrigin, tableMask);

        Kernel_UnionFind_Compress << <gridSizeCuda, blockSize, 0, stream >> > (labels, (int)numberOfPoints);
    }

    void SparseCells::ApplyClustering(PCD* cloud, unsigned int* labels, float clusterDistance, CUstream_st* stream)
    {
        if (!cloud)
        {
            return;
        }
        ApplyClustering(cloud->GetPositions(), cloud->GetNormals(), cloud->size(), labels, clusterDistance, stream);
    }

    void SparseCells::ApplyClustering(float3* points, float3* normals, size_t numberOfPoints, unsigned int* labels, float clusterDistance, float angleThreshold, CUstream_st* stream)
    {
        if (numberOfPoints == 0 || !labels || !hashTable || !normals)
        {
            return;
        }

        int blockSize = 256;
        int gridSizeCuda = (int)((numberOfPoints + blockSize - 1) / blockSize);
        float squaredDistance = clusterDistance * clusterDistance;
        float inverseCellSize = 1.0f / cellSize;
        float cosAngleThreshold = cosf(angleThreshold);

        Kernel_InitLabels << <gridSizeCuda, blockSize, 0, stream >> > (labels, (int)numberOfPoints);

        Kernel_UnionFind_Link_SameCell_Angle << <gridSizeCuda, blockSize, 0, stream >> > (
            hashTable, nextPoint, points, normals,
            labels, (int)numberOfPoints, squaredDistance, cosAngleThreshold, inverseCellSize, worldOrigin, tableMask);

        Kernel_UnionFind_Compress << <gridSizeCuda, blockSize, 0, stream >> > (labels, (int)numberOfPoints);

        Kernel_UnionFind_Link_Neighbors_Angle << <gridSizeCuda, blockSize, 0, stream >> > (
            hashTable, nextPoint, points, normals,
            labels, (int)numberOfPoints, squaredDistance, cosAngleThreshold, inverseCellSize, worldOrigin, tableMask);

        Kernel_UnionFind_Compress << <gridSizeCuda, blockSize, 0, stream >> > (labels, (int)numberOfPoints);
    }

    void SparseCells::ApplyClustering(PCD* cloud, unsigned int* labels, float clusterDistance, float angleThreshold, CUstream_st* stream)
    {
        if (!cloud)
        {
            return;
        }
        ApplyClustering(cloud->GetPositions(), cloud->GetNormals(), cloud->size(), labels, clusterDistance, angleThreshold, stream);
    }
}