#include <Core/DataStructures/SparseCells.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/DataStructures/PCD.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

namespace Huvitz
{
    __constant__ int offsetX[14] = { 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    __constant__ int offsetY[14] = { 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    __constant__ int offsetZ[14] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    __device__ __forceinline__ int SpatialHash(int gx, int gy, int gz, int mask)
    {
        unsigned int h = ((unsigned int)gx * 92837111u)
            ^ ((unsigned int)gy * 689287499u)
            ^ ((unsigned int)gz * 283923481u);
        return (int)(h & (unsigned int)mask);
    }

    __global__ void Kernel_InsertPoints(
        const float3* __restrict__ points,
        int* hashTable,
        int* nextPoint,
        int n,
        float invCell,
        float3 origin,
        int tableMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
        {
            return;
        }

        float3 p = points[i];
        int gx = __float2int_rd((p.x - origin.x) * invCell);
        int gy = __float2int_rd((p.y - origin.y) * invCell);
        int gz = __float2int_rd((p.z - origin.z) * invCell);

        int slot = SpatialHash(gx, gy, gz, tableMask);
        nextPoint[i] = atomicExch(&hashTable[slot], i);
    }

    __global__ void Kernel_InitLabels(unsigned int* labels, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            labels[i] = (unsigned int)i;
        }
    }

    __global__ void Kernel_UnionFind_Link(
        const int* __restrict__ hashTable,
        const int* __restrict__ nextPoint,
        const float3* __restrict__ pos,
        unsigned int* labels,
        int n,
        float d2,
        float invCell,
        float3 origin,
        int tableMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
        {
            return;
        }

        float3 p = pos[i];
        int gx = __float2int_rd((p.x - origin.x) * invCell);
        int gy = __float2int_rd((p.y - origin.y) * invCell);
        int gz = __float2int_rd((p.z - origin.z) * invCell);

        #pragma unroll
        for (int k = 0; k < 14; ++k)
        {
            int slot = SpatialHash(gx + offsetX[k], gy + offsetY[k], gz + offsetZ[k], tableMask);
            int j = hashTable[slot];

            bool isSameCell = (k == 0);

            while (j != -1)
            {
                if (!isSameCell || i < j)
                {
                    float3 pj = pos[j];
                    float dx = p.x - pj.x;
                    float dy = p.y - pj.y;
                    float dz = p.z - pj.z;
                    float distSq = dx * dx + dy * dy + dz * dz;

                    if (distSq <= d2)
                    {
                        if (labels[i] != labels[j])
                        {
                            unsigned int ri = (unsigned int)i;
                            unsigned int rj = (unsigned int)j;
                            while (ri != rj)
                            {
                                unsigned int lo = ri < rj ? ri : rj;
                                unsigned int hi = ri < rj ? rj : ri;
                                unsigned int old = atomicMin(&labels[hi], lo);
                                if (old == hi)
                                {
                                    break;
                                }
                                ri = old;
                                rj = lo;
                            }
                        }
                    }
                }
                j = nextPoint[j];
            }
        }
    }

    __global__ void Kernel_UnionFind_Compress(unsigned int* labels, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
        {
            return;
        }

        unsigned int root = labels[i];
        while (root != labels[root])
        {
            root = labels[root];
        }
        labels[i] = root;
    }

    SparseCells::SparseCells()
        : hashTable(nullptr), hashTableCapacity(0), tableMask(0)
        , nextPoint(nullptr), nextPointCapacity(0)
    {
        InitialAllocate();
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

    void SparseCells::InitialAllocate()
    {
        if (!nextPoint)
        {
            nextPointCapacity = 2000000;
            cudaMalloc(&nextPoint, sizeof(int) * nextPointCapacity);
        }
    }

    void SparseCells::Build(PCD* cloud, float cellSize, CUstream_st* stream)
    {
        if (!cloud || cloud->size() == 0)
        {
            return;
        }

        float3 minP = cloud->GetAABB().min;
        float3 maxP = cloud->GetAABB().max;

        gridSize = {
            (int)ceilf((maxP.x - minP.x) / cellSize) + 1,
            (int)ceilf((maxP.y - minP.y) / cellSize) + 1,
            (int)ceilf((maxP.z - minP.z) / cellSize) + 1
        };

        BuildInternal(cloud->GetPositions(), cloud->size(), cellSize, minP, stream);
    }

    void SparseCells::Build(float3* points, size_t n, float cellSize, CUstream_st* stream)
    {
        if (n == 0 || !points)
        {
            return;
        }

        gridSize = { 0, 0, 0 };
        BuildInternal(points, n, cellSize, { 0.0f, 0.0f, 0.0f }, stream);
    }

    void SparseCells::BuildInternal(float3* points, size_t n, float cellSize, float3 origin, CUstream_st* stream)
    {
        this->cellSize = cellSize;
        this->worldOrigin = origin;

        size_t tableSize = 1u << 16;
        while (tableSize < n * 2)
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

        if (n > nextPointCapacity)
        {
            if (nextPoint)
            {
                cudaFree(nextPoint);
            }
            nextPointCapacity = (size_t)(n * 1.5);
            cudaMalloc(&nextPoint, sizeof(int) * nextPointCapacity);
        }

        cudaMemsetAsync(hashTable, 0xFF, sizeof(int) * tableSize, stream);

        int bs = 256;
        int gs = (int)((n + bs - 1) / bs);

        Kernel_InsertPoints<<<gs, bs, 0, stream>>>(
            points, hashTable, nextPoint,
            (int)n, 1.0f / cellSize, worldOrigin, tableMask);
    }

    void SparseCells::ApplyClustering(float3* points, size_t n, unsigned int* outLabels, float clusterDistance, CUstream_st* stream)
    {
        if (n == 0 || !outLabels || !hashTable)
        {
            return;
        }

        int bs = 256;
        int gs = (int)((n + bs - 1) / bs);
        float d2 = clusterDistance * clusterDistance;
        float invCell = 1.0f / cellSize;

        Kernel_InitLabels<<<gs, bs, 0, stream>>>(outLabels, (int)n);

        Kernel_UnionFind_Link<<<gs, bs, 0, stream>>>(
            hashTable, nextPoint, points,
            outLabels, (int)n, d2, invCell, worldOrigin, tableMask);

        Kernel_UnionFind_Compress<<<gs, bs, 0, stream>>>(outLabels, (int)n);
    }

    void SparseCells::ApplyClustering(PCD* cloud, unsigned int* outLabels, float clusterDistance, CUstream_st* stream)
    {
        if (!cloud)
        {
            return;
        }
        ApplyClustering(cloud->GetPositions(), cloud->size(), outLabels, clusterDistance, stream);
    }
}