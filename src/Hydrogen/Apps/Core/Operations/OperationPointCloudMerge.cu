#include <Core/Operations/OperationPointCloudMerge.h>

namespace Huvitz
{
    namespace Core
    {
        OperationPointCloudMerge::OperationPointCloudMerge(uint64_t capacityhint, cached_allocator* allocator, CUstream_st* stream)
            : pointHashMap(capacityhint, allocator, stream), allocatedSizeHint(capacityhint)
        {
        }

        OperationPointCloudMerge::~OperationPointCloudMerge()
        {
        }

        __global__ void Kernel_InsertPointCloud(
            DeviceHashMap<uint64_t, OperationPointCloudMerge::Point> map,
            const float3* positions,
            const float3* normals,
            const float4* colors,
            float voxel_size,
            uint64_t count,
            bool* outFullError)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count)
            {
                return;
            }

            float3 p = positions[index];
            if (false == VECTOR3F_VALID(p))
            {
                return;
            }

            Morton64 morton_key = Morton64::FromPosition({ XYZ(p) }, voxel_size);
            uint64_t key = morton_key.code;

            OperationPointCloudMerge::Point* voxel = map.InsertAndGet(key);

            if (voxel != nullptr)
            {
                atomicAdd(&(voxel->position.x), p.x);
                atomicAdd(&(voxel->position.y), p.y);
                atomicAdd(&(voxel->position.z), p.z);
                atomicAdd(&(voxel->normal.x), normals[index].x);
                atomicAdd(&(voxel->normal.y), normals[index].y);
                atomicAdd(&(voxel->normal.z), normals[index].z);
                atomicAdd(&(voxel->color.x), colors[index].x);
                atomicAdd(&(voxel->color.y), colors[index].y);
                atomicAdd(&(voxel->color.z), colors[index].z);
                atomicAdd(&(voxel->color.w), colors[index].w);
                atomicAdd(&(voxel->count), 1);
            }
            else if (outFullError)
            {
                *outFullError = true;
            }
        }

        __global__ void Kernel_InsertPointCloud(
            DeviceHashMap<uint64_t, OperationPointCloudMerge::Point> map,
            const float3* positions,
            const float3* normals,
            const float4* colors,
            float voxel_size,
            uint64_t count,
            cuAABB aabb,
            bool* outFullError)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count)
            {
                return;
            }

            float3 p = positions[index];
            if (false == VECTOR3F_VALID(p))
            {
                return;
            }
            if (false == aabb.contains(p))
            {
                return;
            }

            Morton64 morton_key = Morton64::FromPosition({ XYZ(p) }, voxel_size);
            uint64_t key = morton_key.code;

            OperationPointCloudMerge::Point* voxel = map.InsertAndGet(key);

            if (voxel != nullptr)
            {
                atomicAdd(&(voxel->position.x), p.x);
                atomicAdd(&(voxel->position.y), p.y);
                atomicAdd(&(voxel->position.z), p.z);
                atomicAdd(&(voxel->normal.x), normals[index].x);
                atomicAdd(&(voxel->normal.y), normals[index].y);
                atomicAdd(&(voxel->normal.z), normals[index].z);
                atomicAdd(&(voxel->color.x), colors[index].x);
                atomicAdd(&(voxel->color.y), colors[index].y);
                atomicAdd(&(voxel->color.z), colors[index].z);
                atomicAdd(&(voxel->color.w), colors[index].w);
                atomicAdd(&(voxel->count), 1);
            }
            else if (outFullError)
            {
                *outFullError = true;
            }
        }

        void OperationPointCloudMerge::Resize(uint64_t capacityhint, cached_allocator* allocator, CUstream_st* stream)
        {
            if (allocatedSizeHint >= capacityhint)
            {
                return;
            }

            allocatedSizeHint = capacityhint;
            pointHashMap.Resize(capacityhint, allocator, stream);

			printf("Resized hash map to capacity: %llu\n", pointHashMap.GetCapacity());
        }

        void OperationPointCloudMerge::Clear(cached_allocator* allocator, CUstream_st* stream)
        {
            pointHashMap.Clear(allocator, stream);
        }

        void OperationPointCloudMerge::Merge(
            const float3* points,
            const float3* normals,
            const float4* colors,
            unsigned int numberOfPoints,
            cached_allocator* allocator, CUstream_st* stream)
        {
            uint64_t currentCapacity = pointHashMap.GetCapacity();
            if ((uint64_t)numberOfPoints > (uint64_t)(currentCapacity * 0.7))
            {
                Resize(currentCapacity * 2, allocator, stream);
            }

            bool* deviceFullError = nullptr;
            cudaMallocManaged(&deviceFullError, sizeof(bool));

            float voxel_size = 0.1f;
            bool needsRetry = true;

            while (needsRetry)
            {
                *deviceFullError = false;
                int block_count = (numberOfPoints + 255) / 256;

                CUDA_TS(Insert);
                Kernel_InsertPointCloud << <block_count, 256, 0, stream >> > (
                    pointHashMap.GetDeviceView(),
                    points,
                    normals,
                    colors,
                    voxel_size,
                    numberOfPoints,
                    deviceFullError);

                if (stream)
                {
                    cudaStreamSynchronize(stream);
                }
                else
                {
                    cudaDeviceSynchronize();
                }
                CUDA_TE(Insert);

                if (*deviceFullError)
                {
                    uint64_t newCapacity = pointHashMap.GetCapacity() * 2;
                    Resize(newCapacity, allocator, stream);

                    Clear(allocator, stream);
                }
                else
                {
                    needsRetry = false;
                }
            }

            cudaFree(deviceFullError);
        }

        void OperationPointCloudMerge::Merge(
            const float3* points,
            const float3* normals,
            const float4* colors,
            unsigned int numberOfPoints,
            cuAABB aabb,
            cached_allocator* allocator, CUstream_st* stream)
        {
            uint64_t currentCapacity = pointHashMap.GetCapacity();
            if ((uint64_t)numberOfPoints > (uint64_t)(currentCapacity * 0.7))
            {
                Resize(currentCapacity * 2, allocator, stream);
            }

            bool* deviceFullError = nullptr;
            cudaMallocManaged(&deviceFullError, sizeof(bool));

            float voxel_size = 0.1f;
            bool needsRetry = true;

            while (needsRetry)
            {
                *deviceFullError = false;
                int block_count = (numberOfPoints + 255) / 256;

                CUDA_TS(Insert);
                Kernel_InsertPointCloud << <block_count, 256, 0, stream >> > (
                    pointHashMap.GetDeviceView(),
                    points,
                    normals,
                    colors,
                    voxel_size,
                    numberOfPoints,
                    aabb,
                    deviceFullError);

                if (stream)
                {
                    cudaStreamSynchronize(stream);
                }
                else
                {
                    cudaDeviceSynchronize();
                }
                CUDA_TE(Insert);

                if (*deviceFullError)
                {
                    Clear(allocator, stream);
                    uint64_t newCapacity = pointHashMap.GetCapacity() * 2;
                    Resize(newCapacity, allocator, stream);
                }
                else
                {
                    needsRetry = false;
                }
            }

            cudaFree(deviceFullError);
        }

        size_t OperationPointCloudMerge::GetNumberOfPoints(cached_allocator* allocator, CUstream_st* stream)
        {
            return pointHashMap.CountHost(allocator, stream);
        }

        __device__ unsigned int deviceNumberOfExtractedPoints = 0;

        __global__ void Kernel_ExtractPointCloud(
            const Slot<uint64_t, OperationPointCloudMerge::Point>* slotArray,
            uint64_t capacity,
            uint64_t empty_key,
            float3* positions,
            float3* normals,
            float4* colors)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= capacity)
            {
                return;
            }

            Slot<uint64_t, OperationPointCloudMerge::Point> slot = slotArray[index];
            if (slot.key != empty_key && slot.value.count > 0)
            {
                unsigned int current_index = atomicAdd(&deviceNumberOfExtractedPoints, 1);

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

        void OperationPointCloudMerge::Extract(
            float3* outPoints,
            float3* outNormals,
            float4* outColors,
            unsigned int& outNumberOfPoints,
            cached_allocator* allocator, CUstream_st* stream)
        {
            CUDA_TS(Extract);

            unsigned int initialCount = 0;

            if (stream)
            {
                cudaMemcpyToSymbolAsync(deviceNumberOfExtractedPoints, &initialCount, sizeof(unsigned int), 0, cudaMemcpyHostToDevice, stream);
            }
            else
            {
                cudaMemcpyToSymbol(deviceNumberOfExtractedPoints, &initialCount, sizeof(unsigned int));
            }

            uint64_t capacity = pointHashMap.GetCapacity();
            int block_count = (capacity + 255) / 256;

            Kernel_ExtractPointCloud << <block_count, 256, 0, stream >> > (
                pointHashMap.GetDeviceSlotArray(),
                capacity,
                CudaSentinel<uint64_t>::EmptyKey(),
                outPoints,
                outNormals,
                outColors);

            if (stream)
            {
                cudaStreamSynchronize(stream);
            }
            else
            {
                cudaDeviceSynchronize();
            }

            if (stream)
            {
                cudaMemcpyFromSymbolAsync(&outNumberOfPoints, deviceNumberOfExtractedPoints, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
            else
            {
                cudaMemcpyFromSymbol(&outNumberOfPoints, deviceNumberOfExtractedPoints, sizeof(unsigned int));
            }

            CUDA_TE(Extract);
        }
    }
}
