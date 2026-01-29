#pragma once

#include <Copper/CopperCommon.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <tuple>
#include <vector>

#define VOXEL_SIZE          0.1f
#define BLOCK_SIDE          8
#define BLOCK_SIZE_CUBED    512
#define TRUNCATION_DIST     1.0f
#define SAMPLING_RADIUS     0.6f

#define MAX_BLOCKS          200000
#define HASH_TABLE_SIZE     1000000

#define FREE_ENTRY          -1
#define SHORT_MAX_FLOAT     32767.0f

#define HASH_P1 73856093
#define HASH_P2 19349663
#define HASH_P3 83492791

namespace Copper
{
    struct COPPER_API VoxelBlock
    {
        short tsdf[BLOCK_SIZE_CUBED];
        unsigned char weight[BLOCK_SIZE_CUBED];
        float3 normal[BLOCK_SIZE_CUBED];
        uchar3 color[BLOCK_SIZE_CUBED];
    };

    struct COPPER_API HashEntry
    {
        int3 position;
        int voxelBlockIndex;
        int offset;
    };

    struct COPPER_API HeapCounter
    {
        int count;
    };

    class COPPER_API CuVoxelStreaming
    {
    public:
        CuVoxelStreaming(int width, int height, float fx, float fy, float cx, float cy, float depth_scale, float depth_min, float depth_max, float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f);
        ~CuVoxelStreaming();

        void Reset();
        void ProcessFrame(const float* d_depth_map, const float* d_pose_matrix);
        void ProcessPointCloud(const float3* d_points, const float3* d_normals, const uchar3* d_colors, int point_count);

        int GetActiveBlockCount();
        std::vector<float3> GetActiveBlockCenters();
        std::vector<float3> GetActiveVoxelPositions();
        std::vector<std::pair<float3, uchar3>> GetActiveVoxelData();
        std::vector<std::tuple<float3, uchar3, float3>> GetActiveZeroCrossingPoints();

    private:
        struct CameraParams
        {
            float fx, fy, cx, cy;
            int width, height;
            float depth_scale, depth_min, depth_max;
            float k1, k2, p1, p2; // 누락된 멤버 추가
        };

        CameraParams cam_params;
        HashEntry* d_hash_table;
        VoxelBlock* d_voxel_blocks;
        HeapCounter* d_heap_counter;
        int3* d_visible_block_queue;
        int* d_visible_block_count;
    };
}
