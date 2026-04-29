#pragma once
#include <Core/Common/DeviceCommon.h>

namespace Huvitz
{
    struct NoiseFilterCachingAllocator
    {
        typedef char value_type;
        std::unordered_multimap<size_t, void*> cache;
        cudaStream_t stream = nullptr;

        ~NoiseFilterCachingAllocator()
        {
            for (auto& pair : cache)
                cudaFree(pair.second);
        }

        char* allocate(std::ptrdiff_t n)
        {
            auto it = cache.find(n);
            if (it != cache.end())
            {
                void* ptr = it->second;
                cache.erase(it);
                return (char*)ptr;
            }
            void* ptr;
            cudaMallocAsync(&ptr, n, stream);
            return (char*)ptr;
        }

        void deallocate(char* ptr, size_t n)
        {
            cache.emplace(n, ptr);
        }
    };

    class PCD;
    class SparseCells
    {
    public:
        SparseCells();
        ~SparseCells();

        void Reserve(size_t maxCells, size_t maxPoints);

        void Build(PCD* cloud, float cellSize, CUstream_st* stream = nullptr);
        void Build(float3* points, size_t numberOfPoints, float cellSize, CUstream_st* stream = nullptr);

        void ApplyClustering(PCD* cloud, unsigned int* labels, float clusterDistance, CUstream_st* stream = nullptr);
        void ApplyClustering(float3* points, float3* normals, size_t numberOfPoints, unsigned int* labels, float clusterDistance, CUstream_st* stream = nullptr);
        void ApplyClustering(PCD* cloud, unsigned int* labels, float clusterDistance, float angleThreshold, CUstream_st* stream = nullptr);
        void ApplyClustering(float3* points, float3* normals, size_t numberOfPoints, unsigned int* labels, float clusterDistance, float angleThreshold, CUstream_st* stream = nullptr);

        inline float3 GetWorldOrigin() const { return worldOrigin; }
        inline int3 GetGridSize() const { return gridSize; }
        inline float GetCellSize() const { return cellSize; }

    private:
        int3 gridSize = { 0, 0, 0 };
        float cellSize = 0.0f;
        float3 worldOrigin = {};

        int* hashTable = nullptr;
        size_t hashTableCapacity = 0;
        int tableMask = 0;
        int* nextPoint = nullptr;
        size_t nextPointCapacity = 0;

        void BuildInternal(float3* points, size_t numberOfPoints, float cellSize, float3 origin, CUstream_st* stream);
    };
}
