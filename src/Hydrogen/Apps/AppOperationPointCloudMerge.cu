#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdio.h>
#include <vector>

#include <robin_hood/robin_hood.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <Helium/IVisualDebugging.h>
using VD = IVisualDebugging;

#include <Helium/Serialization.hpp>
#include <Helium/Color.hpp>

#include <Core/Core.h>
#include <Core/DataStructures/SparseCells.h>
#include <Core/DataStructures/VoxelDataBase.h>
using namespace Huvitz;

#include <Core/DataStructures/CudaHashMap.cuh>

#include <Core/Operations/OperationPointCloudMerge.h>

namespace fs = std::filesystem;

using Morton64 = Huvitz::Morton64;

//struct Point
//{
//    float3 position{ 0.0f, 0.0f, 0.0f };
//    float3 normal{ 0.0f, 0.0f, 0.0f };
//    float4 color{ 0.0f, 0.0f, 0.0f, 0.0f };
//    unsigned int count = 0;
//};
//
//__global__ void insert_point_cloud_kernel(
//    DeviceHashMap<uint64_t, Point> map,
//    const float3* positions,
//    const float3* normals,
//    const float4* colors,
//    float voxel_size,
//    uint64_t count)
//{
//    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (index >= count) return;
//
//    float3 pos = positions[index];
//    if (abs(pos.x) > 1000.0f || abs(pos.y) > 1000.0f || abs(pos.z) > 1000.0f) return;
//
//    Morton64 morton_key = Morton64::FromPosition({ XYZ(pos) }, voxel_size);
//    uint64_t key = morton_key.code;
//
//    Point* target = map.InsertAndGet(key);
//
//    if (target != nullptr)
//    {
//        atomicAdd(&(target->position.x), pos.x);
//        atomicAdd(&(target->position.y), pos.y);
//        atomicAdd(&(target->position.z), pos.z);
//        atomicAdd(&(target->normal.x), normals[index].x);
//        atomicAdd(&(target->normal.y), normals[index].y);
//        atomicAdd(&(target->normal.z), normals[index].z);
//        atomicAdd(&(target->color.x), colors[index].x);
//        atomicAdd(&(target->color.y), colors[index].y);
//        atomicAdd(&(target->color.z), colors[index].z);
//        atomicAdd(&(target->color.w), colors[index].w);
//        atomicAdd(&(target->count), 1);
//    }
//}
//
//std::vector<Point> GetAllPointsFromHashMap(CudaHashMap<uint64_t, Point>& hash_map)
//{
//    cudaDeviceSynchronize();
//    uint64_t capacity = hash_map.GetCapacity();
//    std::vector<Slot<uint64_t, Point>> host_slots(capacity);
//    cudaMemcpy(host_slots.data(), hash_map.GetDeviceSlots(), capacity * sizeof(Slot<uint64_t, Point>), cudaMemcpyDeviceToHost);
//
//    std::vector<Point> result_points;
//    const uint64_t empty = CudaSentinel<uint64_t>::EmptyKey();
//
//    for (const auto& slot : host_slots)
//    {
//        if (slot.key != empty && slot.value.count > 0)
//        {
//            Point p = slot.value;
//            float inv = 1.0f / (float)p.count;
//            p.position.x *= inv; p.position.y *= inv; p.position.z *= inv;
//            p.normal.x *= inv; p.normal.y *= inv; p.normal.z *= inv;
//            p.color.x *= inv; p.color.y *= inv; p.color.z *= inv; p.color.w *= inv;
//            result_points.push_back(p);
//        }
//    }
//    return result_points;
//}

#include "Apps.h"

struct PointCloudData
{
    float3* positions = nullptr;
    float3* normals = nullptr;
    float4* colors = nullptr;
	unsigned int numberOfPoints = 0;
};

class AppOperationPointCloudMerge : public App
{
public:
    virtual void Execute() override
    {
        CUDAWarmUp();

		unsigned int numberOfSubmaps = 15;


        std::string root = "D:\\Debug\\MergingTest\\pointclouds\\";

#if 1
        std::vector<PointCloudData> pointClouds(numberOfSubmaps);

        unsigned int numberOfTotalPoints = 0;
        for (size_t i = 0; i < numberOfSubmaps; i++)
        {
            std::string filename = root + "SubMap" + std::to_string(i + 1) + ".ply";
            PLYFormat ply;
            if (ply.Deserialize(filename))
            {
                unsigned int count = ply.GetPoints().size();
                numberOfTotalPoints += count;
                pointClouds[i].numberOfPoints = count;
                cudaMalloc(&pointClouds[i].positions, count * sizeof(float3));
                cudaMalloc(&pointClouds[i].normals, count * sizeof(float3));
                cudaMalloc(&pointClouds[i].colors, count * sizeof(float4));
                cudaMemcpy(pointClouds[i].positions, ply.GetPoints().data(), count * sizeof(float3), cudaMemcpyHostToDevice);
                cudaMemcpy(pointClouds[i].normals, ply.GetNormals().data(), count * sizeof(float3), cudaMemcpyHostToDevice);
                cudaMemcpy(pointClouds[i].colors, ply.GetColors().data(), count * sizeof(float4), cudaMemcpyHostToDevice);
            }
        }

        CUDA_TS(Init);
        OperationPointCloudMerge om;
        CUDA_TE(Init);

        CUDA_TS(Resize);
        om.Resize(numberOfTotalPoints);
        CUDA_TE(Resize);

        CUDA_TS(MergeTotal);
        for (size_t i = 0; i < numberOfSubmaps; i++)
        {
            om.Merge(
                pointClouds[i].positions,
                pointClouds[i].normals,
                pointClouds[i].colors,
                pointClouds[i].numberOfPoints);
        }
        CUDA_TE(MergeTotal);

        for (size_t i = 0; i < numberOfSubmaps; i++)
        {
            cudaFree(pointClouds[i].positions);
            cudaFree(pointClouds[i].normals);
            cudaFree(pointClouds[i].colors);
        }

        PointCloudData mergedData;
        mergedData.numberOfPoints = numberOfTotalPoints;
        cudaMalloc(&mergedData.positions, numberOfTotalPoints * sizeof(float3));
        cudaMalloc(&mergedData.normals, numberOfTotalPoints * sizeof(float3));
        cudaMalloc(&mergedData.colors, numberOfTotalPoints * sizeof(float4));

        CUDA_TS(Retrieve);
        om.Extract(
            mergedData.positions,
            mergedData.normals,
            mergedData.colors,
            numberOfTotalPoints);
        CUDA_TE(Retrieve);

        float3* host_positions = new float3[numberOfTotalPoints];
        float3* host_normals = new float3[numberOfTotalPoints];
        float4* host_colors = new float4[numberOfTotalPoints];

        cudaMemcpy(host_positions, mergedData.positions, numberOfTotalPoints * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_normals, mergedData.normals, numberOfTotalPoints * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_colors, mergedData.colors, numberOfTotalPoints * sizeof(float4), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < numberOfTotalPoints; i++)
        {
            auto p = host_positions[i];
            auto n = host_normals[i];
            auto c = host_colors[i];

            //VD::AddSphere("MergedPointCloud", p, n, 0.05f, { c.x, c.y, c.z, c.w });
            VD::AddSphere("MergedPointCloud", p, n, 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });
        }

        delete[] host_positions;
        delete[] host_normals;
        delete[] host_colors;

        cudaFree(mergedData.positions);
        cudaFree(mergedData.normals);
        cudaFree(mergedData.colors);
#else
        PointCloudData pointCloudA;
		std::string filenameA = root + "MergedPointCloud1_5.ply";
        PLYFormat plyA;
		plyA.Deserialize(filenameA);
		pointCloudA.numberOfPoints = plyA.GetPoints().size();
        cudaMalloc(&pointCloudA.positions, pointCloudA.numberOfPoints * sizeof(float3));
        cudaMalloc(&pointCloudA.normals, pointCloudA.numberOfPoints * sizeof(float3));
        cudaMalloc(&pointCloudA.colors, pointCloudA.numberOfPoints * sizeof(float4));
        cudaMemcpy(pointCloudA.positions, plyA.GetPoints().data(), pointCloudA.numberOfPoints * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(pointCloudA.normals, plyA.GetNormals().data(), pointCloudA.numberOfPoints * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(pointCloudA.colors, plyA.GetColors().data(), pointCloudA.numberOfPoints * sizeof(float4), cudaMemcpyHostToDevice);

        PointCloudData pointCloudB;
        std::string filenameB = root + "SubMap6.ply";
        PLYFormat plyB;
        plyB.Deserialize(filenameB);
        pointCloudB.numberOfPoints = plyB.GetPoints().size();
        cudaMalloc(&pointCloudB.positions, pointCloudB.numberOfPoints * sizeof(float3));
        cudaMalloc(&pointCloudB.normals, pointCloudB.numberOfPoints * sizeof(float3));
        cudaMalloc(&pointCloudB.colors, pointCloudB.numberOfPoints * sizeof(float4));
        cudaMemcpy(pointCloudB.positions, plyB.GetPoints().data(), pointCloudB.numberOfPoints * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(pointCloudB.normals, plyB.GetNormals().data(), pointCloudB.numberOfPoints * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(pointCloudB.colors, plyB.GetColors().data(), pointCloudB.numberOfPoints * sizeof(float4), cudaMemcpyHostToDevice);

		Eigen::Vector3f center = plyB.GetAABBCenter();
        Huvitz::cuAABB aabb{ {center.x() - 20.f, center.y() - 20.f, center.z() - 20.f}, {center.x() + 20.f, center.y() + 20.f, center.z() + 20.f}};

        CUDA_TS(Init);
        OperationPointCloudMerge om;
        CUDA_TE(Init);

        CUDA_TS(Resize);
		om.Resize(pointCloudA.numberOfPoints + pointCloudB.numberOfPoints);
        CUDA_TE(Resize);

        CUDA_TS(MergeA);
		om.Merge(
			pointCloudA.positions,
			pointCloudA.normals,
			pointCloudA.colors,
			pointCloudA.numberOfPoints,
            aabb);
        CUDA_TE(MergeA);

        CUDA_TS(MergeB);
		om.Merge(
			pointCloudB.positions,
			pointCloudB.normals,
			pointCloudB.colors,
            pointCloudB.numberOfPoints,
            aabb);
        CUDA_TE(MergeB);

        PointCloudData mergedData;
        mergedData.numberOfPoints = pointCloudA.numberOfPoints + pointCloudB.numberOfPoints;
        cudaMalloc(&mergedData.positions, mergedData.numberOfPoints * sizeof(float3));
        cudaMalloc(&mergedData.normals, mergedData.numberOfPoints * sizeof(float3));
        cudaMalloc(&mergedData.colors, mergedData.numberOfPoints * sizeof(float4));

		CUDA_TS(Retrieve);
        om.Extract(
            mergedData.positions,
            mergedData.normals,
            mergedData.colors,
            mergedData.numberOfPoints);
        CUDA_TE(Retrieve);

        float3* host_positions = new float3[mergedData.numberOfPoints];
        float3* host_normals = new float3[mergedData.numberOfPoints];
        float4* host_colors = new float4[mergedData.numberOfPoints];
        cudaMemcpy(host_positions, mergedData.positions, mergedData.numberOfPoints * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_normals, mergedData.normals, mergedData.numberOfPoints * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_colors, mergedData.colors, mergedData.numberOfPoints * sizeof(float4), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < mergedData.numberOfPoints; i++)
        {
            auto p = host_positions[i];
            auto n = host_normals[i];
            auto c = host_colors[i];
            //VD::AddSphere("MergedPointCloud", p, n, 0.05f, { c.x, c.y, c.z, c.w });
            VD::AddSphere("MergedPointCloud", p, n, 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });
        }
        delete[] host_positions;
        delete[] host_normals;
		delete[] host_colors;
#endif // 0
    }

    void CUDAWarmUp()
    {
        nvDriverSetting.forceGPUPerformance();

        {
            thrust::device_vector<int> device_array(1 << 20);
            thrust::sequence(device_array.begin(), device_array.end());
            thrust::sort(device_array.begin(), device_array.end(), thrust::greater<int>());
            cudaDeviceSynchronize();
        }

        cudaStream_t stream = nullptr;
        cached_allocator* alloc = nullptr;

        cudaFree(0);
    }
};

REGISTER_APP(AppOperationPointCloudMerge, "AppOperationPointCloudMerge");