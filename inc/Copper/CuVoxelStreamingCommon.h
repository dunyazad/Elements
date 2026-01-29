#pragma once

#include <Copper/CopperCommon.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>

#define VOXEL_SIZE          0.1f
#define BLOCK_SIDE          8
#define BLOCK_SIZE_CUBED    512
#define TRUNCATION_DIST     1.0f

#define MAX_BLOCKS          500000
#define HASH_TABLE_SIZE     1000000

#define FREE_ENTRY          -1
#define SHORT_MAX_FLOAT     32767.0f

#define HASH_P1 73856093
#define HASH_P2 19349663
#define HASH_P3 83492791

namespace Copper
{

    struct VoxelBlock
    {
        short tsdf[BLOCK_SIZE_CUBED];
        unsigned char weight[BLOCK_SIZE_CUBED];
        short3 normal[BLOCK_SIZE_CUBED];
    };

    struct HashEntry
    {
        int3 position;
        int voxelBlockIndex;
        int offset;
    };

    struct HeapCounter
    {
        int count;
    };

    // ------------------------------------------------------------------
    // Device Helper Functions (Implementation in header for inlining)
    // ------------------------------------------------------------------

    __device__ inline int compute_hash(int3 pos)
    {
        int res = ((pos.x * HASH_P1) ^ (pos.y * HASH_P2) ^ (pos.z * HASH_P3)) % HASH_TABLE_SIZE;
        if (res < 0) res += HASH_TABLE_SIZE;
        return res;
    }

    __device__ inline short float_to_short_tsdf(float val)
    {
        float clamped = fmaxf(-1.0f, fminf(1.0f, val));
        return static_cast<short>(clamped * SHORT_MAX_FLOAT);
    }

    __device__ inline float short_to_float_tsdf(short val)
    {
        return static_cast<float>(val) / SHORT_MAX_FLOAT;
    }

    __device__ inline int3 world_to_block_coord(float3 p)
    {
        return make_int3(
            static_cast<int>(floorf(p.x / (VOXEL_SIZE * BLOCK_SIDE))),
            static_cast<int>(floorf(p.y / (VOXEL_SIZE * BLOCK_SIDE))),
            static_cast<int>(floorf(p.z / (VOXEL_SIZE * BLOCK_SIDE)))
        );
    }

}