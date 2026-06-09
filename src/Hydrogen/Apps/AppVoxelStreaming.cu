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
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <cstring>

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

#ifndef CUDA_TS
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#endif

#ifndef CUDA_TE
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[<[%s]>] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(error));            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static constexpr int BLOCK_DIM = 8;
static constexpr int BLOCK_VOL = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;
static constexpr float VOXEL_SIZE = 0.1f;
static constexpr float BLOCK_SIZE = BLOCK_DIM * VOXEL_SIZE;

struct StreamingVoxel
{
    float sdf;
    float weight;
};

struct alignas(16) StreamingVoxelBlock
{
    StreamingVoxel voxels[BLOCK_VOL];
};
static_assert(sizeof(StreamingVoxelBlock) == 4096, "VoxelBlock must be 4KB");

struct BlockCoord
{
    int x, y, z;
    bool operator==(const BlockCoord& other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct BlockCoordHash
{
    size_t operator()(const BlockCoord& coord) const noexcept
    {
        size_t hash = 2166136261u;
        hash ^= (uint32_t)coord.x; hash *= 16777619u;
        hash ^= (uint32_t)coord.y; hash *= 16777619u;
        hash ^= (uint32_t)coord.z; hash *= 16777619u;
        return hash;
    }
};

struct CameraParams
{
    float3 position;
    float3 forward;
    float nearPlane;
    float farPlane;
};

inline __host__ __device__ float3 voxelWorldPos(int3 blockCoord, int localX, int localY, int localZ)
{
    return {
        (blockCoord.x * BLOCK_DIM + localX) * VOXEL_SIZE,
        (blockCoord.y * BLOCK_DIM + localY) * VOXEL_SIZE,
        (blockCoord.z * BLOCK_DIM + localZ) * VOXEL_SIZE
    };
}

__global__ void scatterBlocksKernel(StreamingVoxelBlock* devicePool, const StreamingVoxelBlock* deviceStaging, const int* deviceSlots, int count)
{
    int batchIndex = blockIdx.x;
    if (batchIndex >= count)
    {
        return;
    }
    int slot = deviceSlots[batchIndex];
    devicePool[slot].voxels[threadIdx.x] = deviceStaging[batchIndex].voxels[threadIdx.x];
}

__global__ void gatherBlocksKernel(StreamingVoxelBlock* deviceStaging, const StreamingVoxelBlock* devicePool, const int* deviceSlots, int count)
{
    int batchIndex = blockIdx.x;
    if (batchIndex >= count)
    {
        return;
    }
    int slot = deviceSlots[batchIndex];
    deviceStaging[batchIndex].voxels[threadIdx.x] = devicePool[slot].voxels[threadIdx.x];
}

__global__ void voxelUpdateKernel(StreamingVoxelBlock* devicePool, const int* deviceSlotIndex, const int3* deviceBlockCoords, int numBlocks, CameraParams cam)
{
    if (blockIdx.x >= numBlocks)
    {
        return;
    }

    int slot = deviceSlotIndex[blockIdx.x];
    int3 blockCoord = deviceBlockCoords[blockIdx.x];
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    int localZ = threadIdx.z;
    int voxelIndex = localZ * BLOCK_DIM * BLOCK_DIM + localY * BLOCK_DIM + localX;

    float3 worldPos = voxelWorldPos(blockCoord, localX, localY, localZ);
    float dx = worldPos.x - cam.position.x;
    float dy = worldPos.y - cam.position.y;
    float dz = worldPos.z - cam.position.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float depth = dx * cam.forward.x + dy * cam.forward.y + dz * cam.forward.z;

    const float trunc = 5.0f * VOXEL_SIZE;
    float measuredDepth = dist + 0.2f * VOXEL_SIZE;
    float newSDF = fmaxf(-1.0f, fminf(1.0f, (measuredDepth - depth) / trunc));

    StreamingVoxel& voxel = devicePool[slot].voxels[voxelIndex];
    float weight = voxel.weight;
    voxel.sdf = (voxel.sdf * weight + newSDF) / (weight + 1.0f);
    voxel.weight = fminf(weight + 1.0f, 255.0f);
}

static void launchVoxelKernel(dim3 grid, dim3 block, cudaStream_t stream, StreamingVoxelBlock* devicePool, const int* deviceSlotIndex, const int3* deviceBlockCoords, int numBlocks, CameraParams cam)
{
    voxelUpdateKernel << <grid, block, 0, stream >> > (devicePool, deviceSlotIndex, deviceBlockCoords, numBlocks, cam);
}

enum class BlockState : uint8_t
{
    HOST_ONLY,
    UPLOADING,
    ON_DEVICE,
    DOWNLOADING
};

struct BlockEntry
{
    StreamingVoxelBlock* hostPtr;
    int deviceSlot;
    int pinnedSlot;
    BlockState state;
    int lastSeenFrame;
    int evictDelayFrames;
};

class BlockManager
{
public:
    static constexpr int MAX_DEVICE_BLOCKS = 3072;
    static constexpr int MAX_PINNED_BLOCKS = 6144;
    static constexpr int EVICT_AFTER_FRAMES = 300;
    static constexpr int HYSTERESIS_FRAMES = 8;
    static constexpr int MAX_BATCH_SIZE = 1024;

    explicit BlockManager();
    ~BlockManager();
    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;

    void update(const std::vector<BlockCoord>& roiBlocks, const CameraParams& cam);
    const StreamingVoxel* readBlock(const BlockCoord& blockCoord) const;

private:
    robin_hood::unordered_flat_map<BlockCoord, BlockEntry, BlockCoordHash> registry;

    StreamingVoxelBlock* devicePool = nullptr;
    int* deviceSlotIndex = nullptr;
    int3* deviceBlockCoords = nullptr;
    std::stack<int> freeSlots;

    StreamingVoxelBlock* pinnedPool = nullptr;
    std::stack<int> freePinnedSlots;

    int* hostActiveSlots = nullptr;
    int3* hostActiveCoords = nullptr;

    StreamingVoxelBlock* hostStagingH2D = nullptr;
    StreamingVoxelBlock* deviceStagingH2D = nullptr;
    int* hostStagingSlotsH2D = nullptr;
    int* deviceStagingSlotsH2D = nullptr;

    StreamingVoxelBlock* hostStagingD2H = nullptr;
    StreamingVoxelBlock* deviceStagingD2H = nullptr;
    int* hostStagingSlotsD2H = nullptr;
    int* deviceStagingSlotsD2H = nullptr;

    cudaStream_t uploadStream;
    cudaStream_t computeStream;
    cudaStream_t downloadStream;

    cudaEvent_t frameComputeDone;
    cudaEvent_t uploadBatchDone;
    cudaEvent_t downloadBatchDone;
    bool frameComputeRecorded = false;

    robin_hood::unordered_flat_set<BlockCoord, BlockCoordHash> onDeviceSet;
    std::vector<BlockCoord> pendingDownloadCoords;

    int currentFrame = 0;

    void createBlock(const BlockCoord& coord);
    int allocDeviceSlot();
    void pruneRegistry();
};

BlockManager::BlockManager()
{
    CUDA_CHECK(cudaMalloc(&devicePool, (size_t)MAX_DEVICE_BLOCKS * sizeof(StreamingVoxelBlock)));
    CUDA_CHECK(cudaMalloc(&deviceSlotIndex, MAX_DEVICE_BLOCKS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceBlockCoords, MAX_DEVICE_BLOCKS * sizeof(int3)));

    for (int i = MAX_DEVICE_BLOCKS - 1; i >= 0; --i)
    {
        freeSlots.push(i);
    }

    CUDA_CHECK(cudaMallocHost(&pinnedPool, (size_t)MAX_PINNED_BLOCKS * sizeof(StreamingVoxelBlock)));
    memset(pinnedPool, 0, (size_t)MAX_PINNED_BLOCKS * sizeof(StreamingVoxelBlock));

    for (int i = MAX_PINNED_BLOCKS - 1; i >= 0; --i)
    {
        freePinnedSlots.push(i);
    }

    CUDA_CHECK(cudaMallocHost(&hostActiveSlots, MAX_DEVICE_BLOCKS * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&hostActiveCoords, MAX_DEVICE_BLOCKS * sizeof(int3)));

    CUDA_CHECK(cudaMallocHost(&hostStagingH2D, MAX_BATCH_SIZE * sizeof(StreamingVoxelBlock)));
    CUDA_CHECK(cudaMalloc(&deviceStagingH2D, MAX_BATCH_SIZE * sizeof(StreamingVoxelBlock)));
    CUDA_CHECK(cudaMallocHost(&hostStagingSlotsH2D, MAX_BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceStagingSlotsH2D, MAX_BATCH_SIZE * sizeof(int)));

    CUDA_CHECK(cudaMallocHost(&hostStagingD2H, MAX_BATCH_SIZE * sizeof(StreamingVoxelBlock)));
    CUDA_CHECK(cudaMalloc(&deviceStagingD2H, MAX_BATCH_SIZE * sizeof(StreamingVoxelBlock)));
    CUDA_CHECK(cudaMallocHost(&hostStagingSlotsD2H, MAX_BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceStagingSlotsD2H, MAX_BATCH_SIZE * sizeof(int)));

    CUDA_CHECK(cudaStreamCreate(&uploadStream));
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&downloadStream));

    CUDA_CHECK(cudaEventCreateWithFlags(&frameComputeDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&uploadBatchDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&downloadBatchDone, cudaEventDisableTiming));
}

BlockManager::~BlockManager()
{
    CUDA_CHECK(cudaStreamSynchronize(uploadStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(downloadStream));

    CUDA_CHECK(cudaFreeHost(pinnedPool));
    CUDA_CHECK(cudaFreeHost(hostActiveSlots));
    CUDA_CHECK(cudaFreeHost(hostActiveCoords));

    CUDA_CHECK(cudaFreeHost(hostStagingH2D));
    CUDA_CHECK(cudaFree(deviceStagingH2D));
    CUDA_CHECK(cudaFreeHost(hostStagingSlotsH2D));
    CUDA_CHECK(cudaFree(deviceStagingSlotsH2D));

    CUDA_CHECK(cudaFreeHost(hostStagingD2H));
    CUDA_CHECK(cudaFree(deviceStagingD2H));
    CUDA_CHECK(cudaFreeHost(hostStagingSlotsD2H));
    CUDA_CHECK(cudaFree(deviceStagingSlotsD2H));

    CUDA_CHECK(cudaFree(devicePool));
    CUDA_CHECK(cudaFree(deviceSlotIndex));
    CUDA_CHECK(cudaFree(deviceBlockCoords));

    CUDA_CHECK(cudaStreamDestroy(uploadStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(downloadStream));

    CUDA_CHECK(cudaEventDestroy(frameComputeDone));
    CUDA_CHECK(cudaEventDestroy(uploadBatchDone));
    CUDA_CHECK(cudaEventDestroy(downloadBatchDone));
}

void BlockManager::createBlock(const BlockCoord& coord)
{
    if (freePinnedSlots.empty())
    {
        pruneRegistry();
        if (freePinnedSlots.empty())
        {
            return;
        }
    }

    int pinnedSlot = freePinnedSlots.top();
    freePinnedSlots.pop();

    BlockEntry entry{};
    entry.hostPtr = pinnedPool + pinnedSlot;
    entry.pinnedSlot = pinnedSlot;
    entry.deviceSlot = -1;
    entry.state = BlockState::HOST_ONLY;
    entry.lastSeenFrame = currentFrame;
    entry.evictDelayFrames = HYSTERESIS_FRAMES;

    registry[coord] = entry;
}

int BlockManager::allocDeviceSlot()
{
    if (freeSlots.empty())
    {
        fprintf(stderr, "GPU block pool exhausted - increase MAX_DEVICE_BLOCKS\n");
        return -1;
    }
    int slot = freeSlots.top();
    freeSlots.pop();
    return slot;
}

void BlockManager::pruneRegistry()
{
    std::vector<BlockCoord> toDelete;
    for (auto& pair : registry)
    {
        if (pair.second.state == BlockState::HOST_ONLY && (currentFrame - pair.second.lastSeenFrame) > EVICT_AFTER_FRAMES)
        {
            toDelete.push_back(pair.first);
        }
    }
    for (const auto& coord : toDelete)
    {
        auto& entry = registry[coord];
        freePinnedSlots.push(entry.pinnedSlot);
        registry.erase(coord);
    }
}

void BlockManager::update(const std::vector<BlockCoord>& roiBlocks, const CameraParams& cam)
{
    ++currentFrame;

    if (!pendingDownloadCoords.empty())
    {
        CUDA_CHECK(cudaEventSynchronize(downloadBatchDone));
        for (size_t i = 0; i < pendingDownloadCoords.size(); ++i)
        {
            BlockCoord coord = pendingDownloadCoords[i];
            auto it = registry.find(coord);
            if (it != registry.end())
            {
                memcpy(it->second.hostPtr, &hostStagingD2H[i], sizeof(StreamingVoxelBlock));
                freeSlots.push(it->second.deviceSlot);
                it->second.deviceSlot = -1;
                it->second.state = BlockState::HOST_ONLY;
            }
        }
        pendingDownloadCoords.clear();
    }

    robin_hood::unordered_flat_set<BlockCoord, BlockCoordHash> roiSet;
    roiSet.reserve(roiBlocks.size());
    for (const auto& coord : roiBlocks)
    {
        roiSet.insert(coord);
    }

    std::vector<BlockCoord> toEvict;
    for (const auto& coord : onDeviceSet)
    {
        if (roiSet.count(coord))
        {
            continue;
        }
        auto it = registry.find(coord);
        if (it == registry.end())
        {
            continue;
        }
        if (--it->second.evictDelayFrames <= 0)
        {
            toEvict.push_back(coord);
        }
    }

    int downloadCount = 0;
    for (const auto& coord : toEvict)
    {
        if (downloadCount >= MAX_BATCH_SIZE)
        {
            break;
        }
        auto& entry = registry[coord];
        entry.state = BlockState::DOWNLOADING;
        onDeviceSet.erase(coord);
        hostStagingSlotsD2H[downloadCount] = entry.deviceSlot;
        pendingDownloadCoords.push_back(coord);
        downloadCount++;
    }

    if (downloadCount > 0)
    {
        if (frameComputeRecorded)
        {
            CUDA_CHECK(cudaStreamWaitEvent(downloadStream, frameComputeDone, 0));
        }
        CUDA_CHECK(cudaMemcpyAsync(deviceStagingSlotsD2H, hostStagingSlotsD2H, downloadCount * sizeof(int), cudaMemcpyHostToDevice, downloadStream));
        gatherBlocksKernel << <downloadCount, 512, 0, downloadStream >> > (deviceStagingD2H, devicePool, deviceStagingSlotsD2H, downloadCount);
        CUDA_CHECK(cudaMemcpyAsync(hostStagingD2H, deviceStagingD2H, downloadCount * sizeof(StreamingVoxelBlock), cudaMemcpyDeviceToHost, downloadStream));
        CUDA_CHECK(cudaEventRecord(downloadBatchDone, downloadStream));
    }

    int uploadCount = 0;
    for (const auto& coord : roiBlocks)
    {
        if (!registry.count(coord))
        {
            createBlock(coord);
        }

        auto it = registry.find(coord);
        if (it == registry.end())
        {
            continue;
        }

        auto& entry = it->second;
        entry.lastSeenFrame = currentFrame;
        entry.evictDelayFrames = HYSTERESIS_FRAMES;

        if (entry.state == BlockState::HOST_ONLY)
        {
            if (freeSlots.empty() || uploadCount >= MAX_BATCH_SIZE)
            {
                continue;
            }

            int slot = allocDeviceSlot();
            entry.deviceSlot = slot;
            entry.state = BlockState::ON_DEVICE;
            onDeviceSet.insert(coord);

            memcpy(&hostStagingH2D[uploadCount], entry.hostPtr, sizeof(StreamingVoxelBlock));
            hostStagingSlotsH2D[uploadCount] = slot;
            uploadCount++;
        }
    }

    if (uploadCount > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(deviceStagingH2D, hostStagingH2D, uploadCount * sizeof(StreamingVoxelBlock), cudaMemcpyHostToDevice, uploadStream));
        CUDA_CHECK(cudaMemcpyAsync(deviceStagingSlotsH2D, hostStagingSlotsH2D, uploadCount * sizeof(int), cudaMemcpyHostToDevice, uploadStream));
        scatterBlocksKernel << <uploadCount, 512, 0, uploadStream >> > (devicePool, deviceStagingH2D, deviceStagingSlotsH2D, uploadCount);
        CUDA_CHECK(cudaEventRecord(uploadBatchDone, uploadStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, uploadBatchDone, 0));
    }

    int activeCount = 0;
    for (const auto& coord : roiBlocks)
    {
        auto it = registry.find(coord);
        if (it == registry.end())
        {
            continue;
        }
        if (it->second.state == BlockState::ON_DEVICE)
        {
            hostActiveSlots[activeCount] = it->second.deviceSlot;
            hostActiveCoords[activeCount] = { coord.x, coord.y, coord.z };
            activeCount++;
        }
    }

    if (activeCount > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(deviceSlotIndex, hostActiveSlots, activeCount * sizeof(int), cudaMemcpyHostToDevice, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(deviceBlockCoords, hostActiveCoords, activeCount * sizeof(int3), cudaMemcpyHostToDevice, computeStream));

        launchVoxelKernel(dim3(activeCount, 1, 1), dim3(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM), computeStream, devicePool, deviceSlotIndex, deviceBlockCoords, activeCount, cam);

        CUDA_CHECK(cudaEventRecord(frameComputeDone, computeStream));
        frameComputeRecorded = true;
    }

    if (currentFrame % 30 == 0)
    {
        pruneRegistry();
    }
}

const StreamingVoxel* BlockManager::readBlock(const BlockCoord& blockCoord) const
{
    auto it = registry.find(blockCoord);
    if (it == registry.end())
    {
        return nullptr;
    }
    if (!it->second.hostPtr)
    {
        return nullptr;
    }
    return it->second.hostPtr->voxels;
}

struct FrustumPlane
{
    float nx, ny, nz, d;
};

inline float3 normalize3(float x, float y, float z)
{
    float len = sqrtf(x * x + y * y + z * z) + 1e-9f;
    return { x / len, y / len, z / len };
}

inline std::vector<BlockCoord> frustumCull(const CameraParams& cam, float halfFovH = 0.523f, float halfFovV = 0.393f)
{
    float3 fwd = normalize3(cam.forward.x, cam.forward.y, cam.forward.z);
    float3 wu = { 0.f, 1.f, 0.f };
    if (fabsf(fwd.x * wu.x + fwd.y * wu.y + fwd.z * wu.z) > 0.99f)
    {
        wu = { 0.f, 0.f, 1.f };
    }

    float3 right = normalize3(
        fwd.y * wu.z - fwd.z * wu.y,
        fwd.z * wu.x - fwd.x * wu.z,
        fwd.x * wu.y - fwd.y * wu.x);
    float3 up = normalize3(
        right.y * fwd.z - right.z * fwd.y,
        right.z * fwd.x - right.x * fwd.z,
        right.x * fwd.y - right.y * fwd.x);

    const float3& p = cam.position;
    float fp = fwd.x * p.x + fwd.y * p.y + fwd.z * p.z;
    float rp = right.x * p.x + right.y * p.y + right.z * p.z;
    float upp = up.x * p.x + up.y * p.y + up.z * p.z;

    float ch = cosf(halfFovH), sh = sinf(halfFovH);
    float cv = cosf(halfFovV), sv = sinf(halfFovV);

    FrustumPlane planes[6];
    planes[0] = { fwd.x, fwd.y, fwd.z, -(fp + cam.nearPlane) };
    planes[1] = { -fwd.x, -fwd.y, -fwd.z, fp + cam.farPlane };
    planes[2] = { -ch * right.x + sh * fwd.x, -ch * right.y + sh * fwd.y,
                  -ch * right.z + sh * fwd.z, ch * rp - sh * fp };
    planes[3] = { ch * right.x + sh * fwd.x, ch * right.y + sh * fwd.y,
                   ch * right.z + sh * fwd.z, -ch * rp - sh * fp };
    planes[4] = { -cv * up.x + sv * fwd.x, -cv * up.y + sv * fwd.y,
                  -cv * up.z + sv * fwd.z, cv * upp - sv * fp };
    planes[5] = { cv * up.x + sv * fwd.x, cv * up.y + sv * fwd.y,
                   cv * up.z + sv * fwd.z, -cv * upp - sv * fp };

    int cbx = (int)floorf(p.x / BLOCK_SIZE);
    int cby = (int)floorf(p.y / BLOCK_SIZE);
    int cbz = (int)floorf(p.z / BLOCK_SIZE);
    int rx = (int)ceilf((cam.farPlane * tanf(halfFovH) + BLOCK_SIZE) / BLOCK_SIZE) + 1;
    int ry = (int)ceilf((cam.farPlane * tanf(halfFovV) + BLOCK_SIZE) / BLOCK_SIZE) + 1;
    int rz = (int)ceilf((cam.farPlane + BLOCK_SIZE) / BLOCK_SIZE) + 1;

    std::vector<BlockCoord> result;
    result.reserve(2048);

    for (int bz = cbz - rz; bz <= cbz + rz; ++bz)
    {
        for (int by = cby - ry; by <= cby + ry; ++by)
        {
            for (int bx = cbx - rx; bx <= cbx + rx; ++bx)
            {
                float cx = (bx + 0.5f) * BLOCK_SIZE;
                float cy = (by + 0.5f) * BLOCK_SIZE;
                float cz = (bz + 0.5f) * BLOCK_SIZE;
                float hs = BLOCK_SIZE * 0.5f;

                bool inside = true;
                for (int pi = 0; pi < 6 && inside; ++pi)
                {
                    const auto& pl = planes[pi];
                    float pvx = cx + hs * (pl.nx >= 0.f ? 1.f : -1.f);
                    float pvy = cy + hs * (pl.ny >= 0.f ? 1.f : -1.f);
                    float pvz = cz + hs * (pl.nz >= 0.f ? 1.f : -1.f);
                    if (pl.nx * pvx + pl.ny * pvy + pl.nz * pvz + pl.d < 0.f)
                    {
                        inside = false;
                    }
                }
                if (inside)
                {
                    result.push_back({ bx, by, bz });
                }
            }
        }
    }
    return result;
}

#include "Apps.h"
class AppVoxelStreaming : public App
{
public:
    virtual void Initialize() override
    {
    }

    virtual void Execute() override
    {
        nvDriverSetting.forceGPUPerformance();

        {
            thrust::device_vector<int> dummyVector(1 << 20);
            thrust::sequence(dummyVector.begin(), dummyVector.end());
            thrust::sort(dummyVector.begin(), dummyVector.end(), thrust::greater<int>());
            cudaDeviceSynchronize();
        }

        cudaFree(0);
        CheckDeviceMemory("Initialization");

        BlockManager manager;

        for (int frame = 0; frame < 200; ++frame)
        {
            CUDA_TS(Frame);

            CameraParams cam;
            cam.position = { 0.0f, 0.0f, frame * BLOCK_SIZE };
            cam.forward = { 0.0f, 0.0f, 1.0f };
            cam.nearPlane = VOXEL_SIZE;
            cam.farPlane = 15.0f * BLOCK_SIZE;

            auto roiBlocks = frustumCull(cam);
            manager.update(roiBlocks, cam);

            printf("frame %3d: roi=%zu blocks, pos=%.2fmm\n", frame, roiBlocks.size(), cam.position.z);

            BlockCoord targetCoord = { 0, 0, frame - 5 };
            const StreamingVoxel* blockData = manager.readBlock(targetCoord);
            if (blockData)
            {
                printf("  block(%d,%d,%d) center voxel sdf=%.4f w=%.1f\n",
                    targetCoord.x, targetCoord.y, targetCoord.z,
                    blockData[4 * BLOCK_DIM * BLOCK_DIM + 4 * BLOCK_DIM + 4].sdf,
                    blockData[4 * BLOCK_DIM * BLOCK_DIM + 4 * BLOCK_DIM + 4].weight);
            }

            CUDA_TE(Frame);
        }
    }
};

REGISTER_APP(AppVoxelStreaming, "AppVoxelStreaming");