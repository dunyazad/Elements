#pragma once

#include <Core/Common/HostCommon.h>
#include <Core/Common/DeviceCommon.h>
#include <Core/Common/CUDAMath.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/DataStructures/SparseCells.h>
#include <thrust/device_vector.h>
#include <vector>
#include <string>

struct cached_allocator;

using namespace Huvitz::Core;

namespace Huvitz
{
    class PCD
    {
    public:
        PCD();
        PCD(size_t n);
        ~PCD();

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

        void FromDevicePointers(
            const float3* d_positions,
            const float3* d_normals,
            const uchar3* d_colors,
            size_t numberOfPositions,
            const  Huvitz::Core::cuAABB& localAABB,
            const float* inverseRT,
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

        inline Huvitz::Core::cuAABB& GetAABB() { return aabb; }
        inline const Huvitz::Core::cuAABB& GetAABB() const { return aabb; }

        inline float3* GetPositions() { return positions.data().get(); }
        inline const float3* GetPositions() const { return positions.data().get(); }
        inline float3* GetNormals() { return normals.data().get(); }
        inline const float3* GetNormals() const { return normals.data().get(); }
        inline uchar3* GetColors() { return colors.data().get(); }
        inline const uchar3* GetColors() const { return colors.data().get(); }
        inline bool* GetIsAlive() { return isAlive.data().get(); }
        inline const bool* GetIsAlive() const { return isAlive.data().get(); }

        inline unsigned int* GetLabels() { return labels.data().get(); }
        inline const unsigned int* GetLabels() const { return labels.data().get(); }
        inline unsigned int* GetSubLabels() { return subLabels.data().get(); }
        inline const unsigned int* GetSubLabels() const { return subLabels.data().get(); }

        PCD ExtractByAABB(const Huvitz::Core::cuAABB& region) const;
        PCD ExtractByAABB(const float3& regionMin, const float3& regionMax) const;

        void ExtractByAABB(const Huvitz::Core::cuAABB& region, PCD& out) const;
        void ExtractByAABB(const float3& regionMin, const float3& regionMax, PCD& out) const;

        void RebuildAABB();
        void RebuildAABB(CUstream_st* stream, cached_allocator* alloc);

        void FillIsAlive(bool value);
        void FillIsAlive(bool value, CUstream_st* stream, cached_allocator* alloc);

        void ClearLabels(CUstream_st* stream, cached_allocator* alloc);
        void ClearSubLabels(CUstream_st* stream, cached_allocator* alloc);

        void FilterInvalidPositions();

        void Merge(const PCD& other);

        void CropUsingLocalRTandAABB(
            const float* inverseRT,
            const Huvitz::Core::cuAABB& localAABB,
            PCD& out,
            cached_allocator* alloc,
            CUstream_st* stream);

        void Clustering(float cellSize, float clusterDistance, CUstream_st* stream, cached_allocator* alloc);
        void Clustering(float cellSize, float clusterDistance, float angleThreshold, CUstream_st* stream, cached_allocator* alloc);

        void ClusteringSubLabels(float cellSize, float clusterDistance, CUstream_st* stream, cached_allocator* alloc);
        void ClusteringSubLabels(float cellSize, float clusterDistance, float angleThreshold, CUstream_st* stream, cached_allocator* alloc);

        void BuildLabelCountTable();
        void BuildLabelCountTable(CUstream_st* stream);

        inline unsigned int* GetLabelCountTable() { return labelCountTable.data().get(); }
        inline const unsigned int* GetLabelCountTable() const { return labelCountTable.data().get(); }
        inline size_t GetLabelCountTableSize() const { return labelCountTable.size(); }

        void BuildPointCountTable();
        void BuildPointCountTable(CUstream_st* stream);

        inline unsigned int* GetPointCountTable() { return pointCountTable.data().get(); }
        inline const unsigned int* GetPointCountTable() const { return pointCountTable.data().get(); }

        void BuildSubLabelCountTable();
        void BuildSubLabelCountTable(CUstream_st* stream);

        inline unsigned int* GetSubLabelCountTable() { return subLabelCountTable.data().get(); }
        inline const unsigned int* GetSubLabelCountTable() const { return subLabelCountTable.data().get(); }
        inline size_t GetSubLabelCountTableSize() const { return subLabelCountTable.size(); }

        void BuildSubLabelPointCountTable();
        void BuildSubLabelPointCountTable(CUstream_st* stream);

        inline unsigned int* GetSubLabelPointCountTable() { return subLabelPointCountTable.data().get(); }
        inline const unsigned int* GetSubLabelPointCountTable() const { return subLabelPointCountTable.data().get(); }

        PCD  Downsample(float voxelSize) const;
        void Downsample(float voxelSize, PCD& out) const;

    protected:
        Huvitz::Core::cuAABB aabb = {
            make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)
        };

        size_t allocated = 0;
        size_t numberOfPositions = 0;
        thrust::device_vector<float3>        positions;
        thrust::device_vector<float3>        normals;
        thrust::device_vector<uchar3>        colors;
        thrust::device_vector<bool>          isAlive;
        thrust::device_vector<unsigned int>  labels;
        thrust::device_vector<unsigned int>  subLabels;
        thrust::device_vector<unsigned int>  labelCountTable;
        thrust::device_vector<unsigned int>  pointCountTable;
        thrust::device_vector<unsigned int>  subLabelCountTable;
        thrust::device_vector<unsigned int>  subLabelPointCountTable;
        SparseCells                          sparseCells;
    };
}