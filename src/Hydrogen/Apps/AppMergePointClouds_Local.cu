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

using Morton64 = Huvitz::Core::Morton64;

#include "Apps.h"

struct Point
{
    float3 position{ 0.0f, 0.0f, 0.0f };
    float3 normal{ 0.0f, 0.0f, 0.0f };
    float4 color{ 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned int count = 0;
};

__global__ void insert_point_cloud_kernel_Local(
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

std::vector<Point> GetAllPointsFromHashMapLocal(CudaHashMap<uint64_t, Point>& hash_map)
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

__global__ void count_valid_points_kernel(
    const Slot<uint64_t, Point>* slots,
    uint64_t capacity,
    uint64_t empty_key,
    unsigned int* valid_count)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= capacity)
    {
        return;
    }

    Slot<uint64_t, Point> slot = slots[index];
    if (slot.key != empty_key && slot.value.count > 0)
    {
        atomicAdd(valid_count, 1);
    }
}

__global__ void extract_points_kernel(
    const Slot<uint64_t, Point>* slots,
    uint64_t capacity,
    uint64_t empty_key,
    float3* positions,
    float3* normals,
    float4* colors,
    unsigned int* out_index)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= capacity)
    {
        return;
    }

    Slot<uint64_t, Point> slot = slots[index];
    if (slot.key != empty_key && slot.value.count > 0)
    {
        unsigned int current_index = atomicAdd(out_index, 1);
        float inverse_count = 1.0f / (float)slot.value.count;

        float3 position = slot.value.position;
        position.x *= inverse_count;
        position.y *= inverse_count;
        position.z *= inverse_count;
        positions[current_index] = position;

        float3 normal = slot.value.normal;
        normal.x *= inverse_count;
        normal.y *= inverse_count;
        normal.z *= inverse_count;
        normals[current_index] = normal;

        float4 color = slot.value.color;
        color.x *= inverse_count;
        color.y *= inverse_count;
        color.z *= inverse_count;
        color.w *= inverse_count;
        colors[current_index] = color;
    }
}

unsigned int GetValidPointCountDevice(CudaHashMap<uint64_t, Point>& hash_map)
{
    uint64_t capacity = hash_map.GetCapacity();
    int threads = 256;
    int blocks = (capacity + threads - 1) / threads;
    uint64_t empty_key = CudaSentinel<uint64_t>::EmptyKey();

    unsigned int* valid_count;
    cudaMalloc(&valid_count, sizeof(unsigned int));
    cudaMemset(valid_count, 0, sizeof(unsigned int));

    count_valid_points_kernel << <blocks, threads >> > (
        hash_map.GetDeviceSlotArray(),
        capacity,
        empty_key,
        valid_count
        );
    cudaDeviceSynchronize();

    unsigned int out_count = 0;
    cudaMemcpy(&out_count, valid_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(valid_count);

    return out_count;
}

void ExtractPointsFromHashMapDevice(
    CudaHashMap<uint64_t, Point>& hash_map,
    float3* positions,
    float3* normals,
    float4* colors,
    unsigned int* count)
{
    uint64_t capacity = hash_map.GetCapacity();
    int threads = 256;
    int blocks = (capacity + threads - 1) / threads;
    uint64_t empty_key = CudaSentinel<uint64_t>::EmptyKey();

    extract_points_kernel << <blocks, threads >> > (
        hash_map.GetDeviceSlotArray(),
        capacity,
        empty_key,
        positions,
        normals,
        colors,
        count
        );
    cudaDeviceSynchronize();
}
//
//void ProcessHashMap(CudaHashMap<uint64_t, Point>& voxel_map)
//{
//    unsigned int out_count = GetValidPointCountDevice(voxel_map);
//
//    float3* out_positions = nullptr;
//    float3* out_normals = nullptr;
//    float4* out_colors = nullptr;
//
//    if (out_count > 0)
//    {
//        cudaMalloc(&out_positions, out_count * sizeof(float3));
//        cudaMalloc(&out_normals, out_count * sizeof(float3));
//        cudaMalloc(&out_colors, out_count * sizeof(float4));
//
//        ExtractPointsFromHashMapDevice(voxel_map, out_positions, out_normals, out_colors);
//    }
//}

class AppMergePointClouds_Local : public App
{
public:
    void Tool()
    {
        CUDAWarmUp();

		Eigen::Vector3f center(0, 0, 0);

        std::string root_path = "D:\\Debug\\MergingTest\\pointclouds";
        
        {
            PLYFormat ply;
            ply.Deserialize(root_path + "\\SubMap6.ply");

            center = ply.GetAABBCenter();
        }

        printf("Center of SubMap6: (%f, %f, %f)\n", center.x(), center.y(), center.z());

        Eigen::AlignedBox3f aabb(center - Eigen::Vector3f(20.0f, 20.0f, 20.0f), center + Eigen::Vector3f(20.0f, 20.0f, 20.0f));

        {
            PLYFormat ply;
            ply.Deserialize(root_path + "\\MergedPointCloud1_5.ply");

            PLYFormat cropped;
            for (size_t i = 0; i < ply.GetPoints().size(); i++)
            {
				auto& p = ply.GetPoints()[i];
				auto& n = ply.GetNormals()[i];
				auto& c = ply.GetColors()[i];

                if(aabb.contains(p))
                {
                    cropped.AddPoint(p.x(), p.y(), p.z());
					cropped.AddNormal(n.x(), n.y(), n.z());
					cropped.AddColor(c.x(), c.y(), c.z(), c.w());
				}
            }

			cropped.Serialize(root_path + "\\MergedPointCloud_Cropped.ply");
        }
    }

    virtual void Execute() override
    {
        CUDAWarmUp();

        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
        int block_count = sm_count * 32;
		float voxel_size = 0.1f;

        Eigen::Vector3f center(0, 0, 0);

        std::string root_path = "D:\\Debug\\MergingTest\\pointclouds";

		float3* d_pointsA = nullptr;
		float3* d_normalsA = nullptr;
		float4* d_colorsA = nullptr;
		unsigned int numPointsA = 0;

        {
            PLYFormat ply;
            ply.Deserialize(root_path + "\\Cropped.ply");

			numPointsA = ply.GetPoints().size();
			cudaMalloc(&d_pointsA, numPointsA * sizeof(float3));
			cudaMalloc(&d_normalsA, numPointsA * sizeof(float3));
			cudaMalloc(&d_colorsA, numPointsA * sizeof(float4));
			cudaMemcpy(d_pointsA, ply.GetPoints().data(), numPointsA * sizeof(float3), cudaMemcpyHostToDevice);
			cudaMemcpy(d_normalsA, ply.GetNormals().data(), numPointsA * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_colorsA, ply.GetColors().data(), numPointsA * sizeof(float4), cudaMemcpyHostToDevice);
        }

        float3* d_pointsB = nullptr;
        float3* d_normalsB = nullptr;
        float4* d_colorsB = nullptr;
		unsigned int numPointsB = 0;

        {
            PLYFormat ply;
            ply.Deserialize(root_path + "\\SubMap6.ply");

            numPointsB = ply.GetPoints().size();
            cudaMalloc(&d_pointsB, numPointsB * sizeof(float3));
            cudaMalloc(&d_normalsB, numPointsB * sizeof(float3));
            cudaMalloc(&d_colorsB, numPointsB * sizeof(float4));
            cudaMemcpy(d_pointsB, ply.GetPoints().data(), numPointsB * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_normalsB, ply.GetNormals().data(), numPointsB * sizeof(float3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_colorsB, ply.GetColors().data(), numPointsB * sizeof(float4), cudaMemcpyHostToDevice);
        }

        CheckDeviceMemory("Before CudaHashMap Initialization");

        const uint64_t reserved_count = 300000;
        CudaHashMap<uint64_t, Point> voxel_map(reserved_count * 2);

        CheckDeviceMemory("After CudaHashMap Initialization");

        {
            CUDA_TS(InsertA);

            insert_point_cloud_kernel_Local << <block_count, 256 >> > (
                voxel_map.GetDeviceView(),
                d_pointsA,
                d_normalsA,
                d_colorsA,
                voxel_size,
                numPointsA
                );
            cudaDeviceSynchronize();

            CUDA_TE(InsertA);
        }

        {
            CUDA_TS(InsertB);

            insert_point_cloud_kernel_Local << <block_count, 256 >> > (
                voxel_map.GetDeviceView(),
                d_pointsB,
                d_normalsB,
                d_colorsB,
                voxel_size,
                numPointsB
                );
            cudaDeviceSynchronize();

            CUDA_TE(InsertB);
        }

        uint64_t merged_point_count = voxel_map.CountHost();
        printf("Total merged unique voxels: %llu\n", merged_point_count);

        printf("Merging completed. Starting data download from GPU...\n");

        
		float3* d_out_positions = nullptr;
		float3* d_out_normals = nullptr;
		float4* d_out_colors = nullptr;
        unsigned int* d_out_count = 0;
        
        ExtractPointsFromHashMapDevice(voxel_map, d_out_positions, d_out_normals, d_out_colors, d_out_count);
        
        CUDA_TS(Clear);
        voxel_map.Clear();
        CUDA_TE(Clear);

		unsigned int out_count = 0;
		cudaMemcpy(&out_count, d_out_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        std::vector<float3> h_positions(out_count);
        std::vector<float3> h_normals(out_count);
        std::vector<float4> h_colors(out_count);
        cudaMemcpy(h_positions.data(), d_out_positions, out_count * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals.data(), d_out_normals, out_count * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_colors.data(), d_out_colors, out_count * sizeof(float4), cudaMemcpyDeviceToHost);


        PLYFormat merged_ply;
        for (unsigned int i = 0; i < out_count; i++)
        {
            VD::AddSphere("PointCloud", h_positions[i], h_normals[i], 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });
         
            merged_ply.AddPoint(h_positions[i].x, h_positions[i].y, h_positions[i].z);
            merged_ply.AddNormal(h_normals[i].x, h_normals[i].y, h_normals[i].z);
            merged_ply.AddColor(h_colors[i].x, h_colors[i].y, h_colors[i].z, h_colors[i].w);
		}
        std::string merged_filename = root_path + "\\Merged.ply";
        merged_ply.Serialize(merged_filename);
        cudaFree(d_out_positions);
        cudaFree(d_out_normals);
		cudaFree(d_out_colors);

#if 0
        //std::vector<Point> merged_points = GetAllPointsFromHashMapLocal(voxel_map);

//printf("Successfully retrieved %zu merged points from GPU.\n", merged_points.size());

//PLYFormat merged_ply;
//if (!merged_points.empty())
//{
//    for (const auto& p : merged_points)
//    {
//        VD::AddSphere("PointCloud", p.position, p.normal, 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });

//        merged_ply.AddPoint(p.position.x, p.position.y, p.position.z);
//        merged_ply.AddNormal(p.normal.x, p.normal.y, p.normal.z);
//        merged_ply.AddColor(p.color.x, p.color.y, p.color.z, p.color.w);
//    }

//    std::string merged_filename = root_path + "\\LocalMergedPointCloud.ply";
//    merged_ply.Serialize(merged_filename);
//}  
#endif // 0


		cudaFree(d_pointsA);
		cudaFree(d_normalsA);
		cudaFree(d_colorsA);
		cudaFree(d_pointsB);
		cudaFree(d_normalsB);
        cudaFree(d_colorsB);
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

REGISTER_APP(AppMergePointClouds_Local, "AppMergePointClouds_Local");