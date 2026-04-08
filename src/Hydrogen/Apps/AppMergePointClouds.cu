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

namespace fs = std::filesystem;

using Morton64 = Huvitz::Morton64;

struct Point
{
    float3 position{ 0.0f, 0.0f, 0.0f };
    float3 normal{ 0.0f, 0.0f, 0.0f };
    float4 color{ 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned int count = 0;
};

__global__ void insert_point_cloud_kernel(
    DeviceHashMap<uint64_t, Point> map,
    const float3* positions,
    const float3* normals,
    const float4* colors,
    float voxel_size,
    uint64_t count)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    float3 pos = positions[index];
    if (abs(pos.x) > 1000.0f || abs(pos.y) > 1000.0f || abs(pos.z) > 1000.0f) return;

    Morton64 morton_key = Morton64::FromPosition({ XYZ(pos) }, voxel_size);
    uint64_t key = morton_key.code;

    Point* target = map.InsertAndGet(key);

    if (target != nullptr)
    {
        atomicAdd(&(target->position.x), pos.x);
        atomicAdd(&(target->position.y), pos.y);
        atomicAdd(&(target->position.z), pos.z);
        atomicAdd(&(target->normal.x), normals[index].x);
        atomicAdd(&(target->normal.y), normals[index].y);
        atomicAdd(&(target->normal.z), normals[index].z);
        atomicAdd(&(target->color.x), colors[index].x);
        atomicAdd(&(target->color.y), colors[index].y);
        atomicAdd(&(target->color.z), colors[index].z);
        atomicAdd(&(target->color.w), colors[index].w);
        atomicAdd(&(target->count), 1);
    }
}

std::vector<Point> GetAllPointsFromHashMap(CudaHashMap<uint64_t, Point>& hash_map)
{
    cudaDeviceSynchronize();
    uint64_t capacity = hash_map.GetCapacity();
    std::vector<Slot<uint64_t, Point>> host_slots(capacity);
    cudaMemcpy(host_slots.data(), hash_map.GetDeviceSlotArray(), capacity * sizeof(Slot<uint64_t, Point>), cudaMemcpyDeviceToHost);

    std::vector<Point> result_points;
    const uint64_t empty = CudaSentinel<uint64_t>::EmptyKey();

    for (const auto& slot : host_slots)
    {
        if (slot.key != empty && slot.value.count > 0)
        {
            Point p = slot.value;
            float inv = 1.0f / (float)p.count;
            p.position.x *= inv; p.position.y *= inv; p.position.z *= inv;
            p.normal.x *= inv; p.normal.y *= inv; p.normal.z *= inv;
            p.color.x *= inv; p.color.y *= inv; p.color.z *= inv; p.color.w *= inv;
            result_points.push_back(p);
        }
    }
    return result_points;
}

#include "Apps.h"

class AppMergePointClouds : public App
{
public:
    virtual void Execute() override
    {
        CUDAWarmUp();

        unsigned int number_of_submaps = 16;
        float voxel_size = 0.1f;

        std::vector<float3*> device_positions(number_of_submaps);
        std::vector<float3*> device_normals(number_of_submaps);
        std::vector<float4*> device_colors(number_of_submaps);
        std::vector<uint64_t> point_counts(number_of_submaps);
        std::string root_path = "D:\\Debug\\MergingTest\\pointclouds";

        CheckDeviceMemory("Before CudaHashMap Initialization");

        const uint64_t reserved_count = 3000000;
        CudaHashMap<uint64_t, Point> voxel_map(reserved_count * 2);

        CheckDeviceMemory("After CudaHashMap Initialization");

        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
        int block_count = sm_count * 32;

        PLYFormat before_merge_ply;

        for (size_t index = 0; index < number_of_submaps; index++)
        {
            //if (5 <= index) continue;

            std::string filename = root_path + "\\SubMap" + std::to_string(index + 1) + ".ply";

            PLYFormat ply;
            ply.Deserialize(filename);

            uint64_t point_count = ply.GetPoints().size();
            point_counts[index] = point_count;
            printf("Loaded %s with %llu points\n", filename.c_str(), point_count);

            cudaMalloc(&device_positions[index], point_count * sizeof(float3));
            cudaMalloc(&device_normals[index], point_count * sizeof(float3));
            cudaMalloc(&device_colors[index], point_count * sizeof(float4));

            cudaMemcpy(device_positions[index], ply.GetPoints().data(), point_count * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(device_normals[index], ply.GetNormals().data(), point_count * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(device_colors[index], ply.GetColors().data(), point_count * sizeof(float4), cudaMemcpyHostToDevice);

            for (size_t i = 0; i < point_count; i++)
            {
                before_merge_ply.AddPoint(ply.GetPoints()[i]);
                before_merge_ply.AddNormal(ply.GetNormals()[i].x(), ply.GetNormals()[i].y(), ply.GetNormals()[i].z());
                before_merge_ply.AddColor(ply.GetColors()[i].x(), ply.GetColors()[i].y(), ply.GetColors()[i].z(), ply.GetColors()[i].w());
            }
        }

        before_merge_ply.Serialize(root_path + "\\BeforeMergePointCloud.ply");

        CUDA_TS(Total);
        for (size_t index = 0; index < number_of_submaps; index++)
        {
            uint64_t point_count = point_counts[index];

            CUDA_TS(Insert);

            insert_point_cloud_kernel << <block_count, 256 >> > (
                voxel_map.GetDeviceView(),
                device_positions[index],
                device_normals[index],
                device_colors[index],
                voxel_size,
                point_count
                );
            cudaDeviceSynchronize();

            CUDA_TE(Insert);
        }
        CUDA_TE(Total);

        uint64_t merged_point_count = voxel_map.CountHost();
        printf("Total merged unique voxels: %llu\n", merged_point_count);




        CUDA_TS(Clear);
        voxel_map.Clear();
        CUDA_TE(Clear);


        printf("Merging completed. Starting data download from GPU...\n");

        std::vector<Point> merged_points = GetAllPointsFromHashMap(voxel_map);

        printf("Successfully retrieved %zu merged points from GPU.\n", merged_points.size());

        PLYFormat merged_ply;
        if (!merged_points.empty())
        {
            for (const auto& p : merged_points)
            {
                VD::AddSphere("PointCloud", p.position, p.normal, 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });

                merged_ply.AddPoint(p.position.x, p.position.y, p.position.z);
                merged_ply.AddNormal(p.normal.x, p.normal.y, p.normal.z);
                merged_ply.AddColor(p.color.x, p.color.y, p.color.z, p.color.w);
            }

            printf("First point position: (%f, %f, %f)\n",
                merged_points[0].position.x,
                merged_points[0].position.y,
                merged_points[0].position.z);

            std::string merged_filename = root_path + "\\MergedPointCloud.ply";
            merged_ply.Serialize(merged_filename);
        }

        for (size_t index = 0; index < number_of_submaps; index++)
        {
            cudaFree(device_positions[index]);
            cudaFree(device_normals[index]);
            cudaFree(device_colors[index]);
        }






        for (size_t index = 0; index < number_of_submaps; index++)
        {
            cudaFree(device_positions[index]);
            cudaFree(device_normals[index]);
            cudaFree(device_colors[index]);
        }
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

REGISTER_APP(AppMergePointClouds, "AppMergePointClouds"); 