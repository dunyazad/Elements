#pragma once

#if defined(__CUDACC__)
#define COPPER_HOST_DEVICE __host__ __device__
#else
#define COPPER_HOST_DEVICE
#endif

namespace Copper
{
    struct CuSparseVoxel
    {
        static constexpr int kResolution = 8;
        static constexpr int kVoxelSize = kResolution * kResolution * kResolution;
        static constexpr int kInvalidIndex = -1;

        COPPER_HOST_DEVICE CuSparseVoxel()
        {
            Clear();
        }

        COPPER_HOST_DEVICE int GetPointIndex(int x, int y, int z) const
        {
            if (!IsValidIndex(x, y, z))
            {
                return kInvalidIndex;
            }

            return pointIndices[Flatten(x, y, z)];
        }

        COPPER_HOST_DEVICE void SetPointIndex(int x, int y, int z, int pointIndex)
        {
            if (!IsValidIndex(x, y, z))
            {
                return;
            }

            pointIndices[Flatten(x, y, z)] = pointIndex;
        }

        COPPER_HOST_DEVICE void Clear(int value = kInvalidIndex)
        {
            for (int i = 0; i < kVoxelSize; ++i)
            {
                pointIndices[i] = value;
            }
        }

        COPPER_HOST_DEVICE const int* GetPointIndices() const
        {
            return pointIndices;
        }

    private:
        COPPER_HOST_DEVICE static constexpr int Flatten(int x, int y, int z)
        {
            return (z * kResolution * kResolution) + (y * kResolution) + x;
        }

        COPPER_HOST_DEVICE static constexpr bool IsValidIndex(int x, int y, int z)
        {
            return x >= 0 && x < kResolution
                && y >= 0 && y < kResolution
                && z >= 0 && z < kResolution;
        }

        int pointIndices[kVoxelSize];
    };

    struct CuSparseVoxelKey
    {
        int x = 0;
        int y = 0;
        int z = 0;

        COPPER_HOST_DEVICE bool operator==(const CuSparseVoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    class CuSparseDatablock
    {
    public:
        COPPER_HOST_DEVICE CuSparseDatablock() = default;

        COPPER_HOST_DEVICE void SetStorage(CuSparseVoxelKey* keyStorage, CuSparseVoxel* voxelStorage, int voxelCapacity)
        {
            keys = keyStorage;
            voxels = voxelStorage;
            capacity = voxelCapacity;
            count = 0;
        }

        COPPER_HOST_DEVICE int Capacity() const
        {
            return capacity;
        }

        COPPER_HOST_DEVICE int Count() const
        {
            return count;
        }

        COPPER_HOST_DEVICE CuSparseVoxel* FindVoxel(int x, int y, int z)
        {
            const int index = FindIndex(x, y, z);
            if (index < 0)
            {
                return nullptr;
            }

            return &voxels[index];
        }

        COPPER_HOST_DEVICE const CuSparseVoxel* FindVoxel(int x, int y, int z) const
        {
            const int index = FindIndex(x, y, z);
            if (index < 0)
            {
                return nullptr;
            }

            return &voxels[index];
        }

        COPPER_HOST_DEVICE CuSparseVoxel* GetOrCreateVoxel(int x, int y, int z)
        {
            int index = FindIndex(x, y, z);
            if (index >= 0)
            {
                return &voxels[index];
            }

            if (count >= capacity || !keys || !voxels)
            {
                return nullptr;
            }

            index = count++;
            keys[index] = CuSparseVoxelKey{ x, y, z };
            voxels[index].Clear();
            return &voxels[index];
        }

        COPPER_HOST_DEVICE void Clear()
        {
            count = 0;
        }

        COPPER_HOST_DEVICE const CuSparseVoxelKey* GetKeys() const
        {
            return keys;
        }

        COPPER_HOST_DEVICE const CuSparseVoxel* GetVoxels() const
        {
            return voxels;
        }

    private:
        COPPER_HOST_DEVICE int FindIndex(int x, int y, int z) const
        {
            if (!keys)
            {
                return -1;
            }

            for (int i = 0; i < count; ++i)
            {
                if (keys[i].x == x && keys[i].y == y && keys[i].z == z)
                {
                    return i;
                }
            }

            return -1;
        }

        CuSparseVoxelKey* keys = nullptr;
        CuSparseVoxel* voxels = nullptr;
        int capacity = 0;
        int count = 0;
    };
}

#undef COPPER_HOST_DEVICE
