#pragma once

#include <Core/Common/HostCommon.h>
#include <Core/Common/DeviceCommon.h>
#include <Core/Common/CUDAMath.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <thrust/device_vector.h>
#include <vector>
#include <string>

struct cached_allocator;

namespace Huvitz
{
    class PCD
    {
    public:
        PCD();
        PCD(size_t n);

        size_t size() const;
        size_t capacity() const;
        void resize(size_t n);
        void clear();

        void FromHostVectors(
            const std::vector<float3>& h_positions,
            const std::vector<float3>& h_normals,
            const std::vector<uchar3>& h_colors);

        void FromHostVectors(
            const std::vector<float3>& h_positions,
            const std::vector<float3>& h_normals,
            const std::vector<uchar3>& h_colors,
            const float3& h_aabbMin,
            const float3& h_aabbMax);

        void FromHostVectors(
            const std::vector<float3>& h_positions,
            const std::vector<float3>& h_normals,
            const std::vector<float4>& h_colors);

        void FromHostVectors(
            const std::vector<float3>& h_positions,
            const std::vector<float3>& h_normals,
            const std::vector<float4>& h_colors,
            const float3& h_aabbMin,
            const float3& h_aabbMax);

        void FromHostPointers(
            const float3* h_positions,
            const float3* h_normals,
            const uchar3* h_colors,
            size_t numberOfPositions);

        void FromHostPointers(
            const float3* h_positions,
            const float3* h_normals,
            const uchar3* h_colors,
            size_t numberOfPositions,
            const float3& h_aabbMin,
            const float3& h_aabbMax);

        void FromHostPointers(
            const float3* h_positions,
            const float3* h_normals,
            const float4* h_colors,
            size_t numberOfPositions);

        void FromHostPointers(
            const float3* h_positions,
            const float3* h_normals,
            const float4* h_colors,
            size_t numberOfPositions,
            const float3& h_aabbMin,
            const float3& h_aabbMax);

        void FromDevicePointers(
            const float3* d_positions,
            const float3* d_normals,
            const uchar3* d_colors,
            size_t numberOfPositions,
            CUstream_st* stream,
            cached_allocator* alloc);

        void FromDevicePointers(
            const float3* d_positions,
            const float3* d_normals,
            const float4* d_colors,
            size_t numberOfPositions,
            CUstream_st* stream,
            cached_allocator* alloc);

        void ToHostVectors(
            std::vector<float3>& h_positions,
            std::vector<float3>& h_normals,
            std::vector<uchar3>& h_colors);

        void ToHostVectors(
            std::vector<float3>& h_positions,
            std::vector<float3>& h_normals,
            std::vector<float4>& h_colors);

        inline size_t GetNumberOfPositions() const { return numberOfPositions; }
        inline void SetNumberOfPositions(size_t n) { numberOfPositions = n; }

        inline cuAABB& GetAABB() { return aabb; }
        inline const cuAABB& GetAABB() const { return aabb; }

        inline float3* GetPositions() { return positions.data().get(); }
        inline const float3* GetPositions() const { return positions.data().get(); }
        inline float3* GetNormals() { return normals.data().get(); }
        inline const float3* GetNormals() const { return normals.data().get(); }
        inline uchar3* GetColors() { return colors.data().get(); }
        inline const uchar3* GetColors() const { return colors.data().get(); }
        inline bool* GetIsAlive() { return isAlive.data().get(); }
        inline const bool* GetIsAlive() const { return isAlive.data().get(); }

        PCD ExtractByAABB(const cuAABB& region) const;
        PCD ExtractByAABB(const float3& regionMin, const float3& regionMax) const;

        void ExtractByAABB(const cuAABB& region, PCD& out) const;
        void ExtractByAABB(const float3& regionMin, const float3& regionMax, PCD& out) const;

        void FilterInvalidPositions();

        void Merge(const PCD& other);

        PCD  Downsample(float voxelSize) const;
        void Downsample(float voxelSize, PCD& out) const;

    protected:
        cuAABB aabb = {
            make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)
        };

        size_t allocated = 0;
        size_t numberOfPositions = 0;
        thrust::device_vector<float3> positions;
        thrust::device_vector<float3> normals;
        thrust::device_vector<uchar3> colors;
        thrust::device_vector<bool> isAlive;

        // robin_hood::unordered_flat_map<std::string, thrust::device_vector<float>> customFloatAttributes;
    };
}
