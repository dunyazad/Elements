#pragma once
#include <Core/Common/DeviceCommon.h>

namespace Huvitz
{
    class PCD;
    class SparseCells
    {
    public:
        SparseCells();
        ~SparseCells();

        void Build(PCD* cloud, float cellSize, CUstream_st* stream = nullptr);
        void Build(float3* points, size_t numberOfPoints, float cellSize, CUstream_st* stream = nullptr);

        void ApplyClustering(PCD* cloud, unsigned int* labels, float clusterDistance, CUstream_st* stream = nullptr);
        void ApplyClustering(float3* points, float3* normals, size_t numberOfPoints, unsigned int* labels, float clusterDistance, CUstream_st* stream = nullptr);
        void ApplyClustering(PCD* cloud, unsigned int* labels, float clusterDistance, float angleThreshold, CUstream_st* stream = nullptr);
        void ApplyClustering(float3* points, float3* normals, size_t numberOfPoints, unsigned int* labels, float clusterDistance, float angleThreshold, CUstream_st* stream = nullptr);

        inline float3 GetWorldOrigin() const { return worldOrigin; }
        inline int3 GetGridSize() const { return gridSize; }
        inline float GetCellSize() const { return cellSize; }

        inline const int* GetHashTable() const { return hashTable; }
        inline const int* GetNextPoint() const { return nextPoint; }
        inline int GetTableMask() const { return tableMask; }

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