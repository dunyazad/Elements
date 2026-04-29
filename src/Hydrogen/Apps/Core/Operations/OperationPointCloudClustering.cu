#include <Core/Operations/OperationPointCloudClustering.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <stdio.h>

__device__ unsigned int FindRoot(unsigned int* labels, unsigned int idx)
{
    while (labels[idx] != idx)
    {
        labels[idx] = labels[labels[idx]];
        idx = labels[idx];
    }
    return idx;
}

__device__ void UnionNodes(unsigned int* labels, unsigned int i, unsigned int j)
{
    unsigned int rootI = FindRoot(labels, i);
    unsigned int rootJ = FindRoot(labels, j);

    while (rootI != rootJ)
    {
        unsigned int high = max(rootI, rootJ);
        unsigned int low = min(rootI, rootJ);
        unsigned int oldRoot = atomicMin(&labels[high], low);

        if (oldRoot == low) break;

        rootI = FindRoot(labels, rootI);
        rootJ = FindRoot(labels, rootJ);
    }
}

__global__ void MapGridKernel(
    float3* positions,
    unsigned int* gridIndices,
    unsigned int* pIdx,
    unsigned int numberOfPoints,
    float cellSize,
    float3 minBound,
    int3 gridRes)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfPoints) return;

    float3 p = positions[idx];
    int gx = (int)((p.x - minBound.x) / cellSize);
    int gy = (int)((p.y - minBound.y) / cellSize);
    int gz = (int)((p.z - minBound.z) / cellSize);

    gx = max(0, min(gx, gridRes.x - 1));
    gy = max(0, min(gy, gridRes.y - 1));
    gz = max(0, min(gz, gridRes.z - 1));

    gridIndices[idx] = (unsigned int)(gx + gy * gridRes.x + gz * gridRes.x * gridRes.y);
    pIdx[idx] = idx;
}

__global__ void BuildCellIndicesKernel(
    unsigned int* gridIndices,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int numberOfPoints)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfPoints) return;

    unsigned int currentHash = gridIndices[idx];

    if (idx == 0)
    {
        cellStart[currentHash] = 0;
    }
    else
    {
        unsigned int prevHash = gridIndices[idx - 1];
        if (currentHash != prevHash)
        {
            cellEnd[prevHash] = idx;
            cellStart[currentHash] = idx;
        }
    }

    if (idx == numberOfPoints - 1)
    {
        cellEnd[currentHash] = numberOfPoints;
    }
}

__global__ void ClusterKernel(
    float3* positions,
    float3* normals,
    unsigned int* pIdx,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int* labels,
    unsigned int numberOfPoints,
    float epsilon,
    float3 minBound,
    float cellSize,
    int3 gridRes)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfPoints) return;

    unsigned int originalIdx = pIdx[idx];
    float3 p = positions[originalIdx];
    float epsSq = epsilon * epsilon;

    int gx = (int)((p.x - minBound.x) / cellSize);
    int gy = (int)((p.y - minBound.y) / cellSize);
    int gz = (int)((p.z - minBound.z) / cellSize);

    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int nx = gx + dx;
                int ny = gy + dy;
                int nz = gz + dz;

                if (nx < 0 || nx >= gridRes.x || ny < 0 || ny >= gridRes.y || nz < 0 || nz >= gridRes.z) continue;

                unsigned int hash = (unsigned int)(nx + ny * gridRes.x + nz * gridRes.x * gridRes.y);
                unsigned int start = cellStart[hash];
                unsigned int end = cellEnd[hash];

                if (start == 0xFFFFFFFF) continue;

                for (unsigned int i = start; i < end; ++i)
                {
                    unsigned int neighborOriginalIdx = pIdx[i];
                    if (neighborOriginalIdx == originalIdx) continue;

                    float3 neighborP = positions[neighborOriginalIdx];
                    float distSq = (p.x - neighborP.x) * (p.x - neighborP.x) +
                        (p.y - neighborP.y) * (p.y - neighborP.y) +
                        (p.z - neighborP.z) * (p.z - neighborP.z);

                    if (distSq <= epsSq)
                    {
                        UnionNodes(labels, originalIdx, neighborOriginalIdx);
                    }
                }
            }
        }
    }
}

__global__ void FlattenLabelsKernel(unsigned int* labels, unsigned int numberOfPoints)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numberOfPoints)
    {
        labels[idx] = FindRoot(labels, idx);
    }
}

__global__ void InitLabelsKernel(unsigned int* labels, unsigned int numberOfPoints)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numberOfPoints)
    {
        labels[idx] = idx;
    }
}

class OperationPointCloudClustering
{
private:
    unsigned int maxPoints;
    unsigned int totalCells;
    int3 gridRes;

    unsigned int* d_gridIndices;
    unsigned int* d_pIdx;
    unsigned int* d_cellStart;
    unsigned int* d_cellEnd;

    void Allocate(unsigned int count)
    {
        maxPoints = count;
        cudaMalloc(&d_gridIndices, maxPoints * sizeof(unsigned int));
        cudaMalloc(&d_pIdx, maxPoints * sizeof(unsigned int));
    }

    void Deallocate()
    {
        if (d_gridIndices) cudaFree(d_gridIndices);
        if (d_pIdx) cudaFree(d_pIdx);
    }

    void Resize(unsigned int newCount)
    {
        Deallocate();
        Allocate(newCount);
    }

public:
    OperationPointCloudClustering(unsigned int maxCount, int3 res)
        : maxPoints(0), gridRes(res), d_gridIndices(nullptr), d_pIdx(nullptr)
    {
        totalCells = (unsigned int)gridRes.x * gridRes.y * gridRes.z;
        Allocate(maxCount);
        cudaMalloc(&d_cellStart, totalCells * sizeof(unsigned int));
        cudaMalloc(&d_cellEnd, totalCells * sizeof(unsigned int));
    }

    ~OperationPointCloudClustering()
    {
        Deallocate();
        if (d_cellStart) cudaFree(d_cellStart);
        if (d_cellEnd) cudaFree(d_cellEnd);
    }

    void Process(float3* d_positions, float3* d_normals, unsigned int* d_labels,
        unsigned int count, float epsilon, float3 minBound)
    {
        if (count > maxPoints)
        {
            Resize((unsigned int)(count * 1.2f));
        }

        float cellSize = epsilon;
        int blockSize = 256;
        int numBlocks = (count + blockSize - 1) / blockSize;

        cudaMemset(d_cellStart, 0xFF, totalCells * sizeof(unsigned int));
        cudaMemset(d_cellEnd, 0x00, totalCells * sizeof(unsigned int));

        InitLabelsKernel << <numBlocks, blockSize >> > (d_labels, count);

        MapGridKernel << <numBlocks, blockSize >> > (d_positions, d_gridIndices, d_pIdx, count, cellSize, minBound, gridRes);

        thrust::device_ptr<unsigned int> t_gridIndices(d_gridIndices);
        thrust::device_ptr<unsigned int> t_pIdx(d_pIdx);
        thrust::sort_by_key(t_gridIndices, t_gridIndices + count, t_pIdx);

        BuildCellIndicesKernel << <numBlocks, blockSize >> > (d_gridIndices, d_cellStart, d_cellEnd, count);

        ClusterKernel << <numBlocks, blockSize >> > (
            d_positions, d_normals, d_pIdx,
            d_cellStart, d_cellEnd, d_labels,
            count, epsilon, minBound, cellSize, gridRes
            );

        FlattenLabelsKernel << <numBlocks, blockSize >> > (d_labels, count);
        cudaDeviceSynchronize();
    }
};
