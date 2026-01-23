#include <Copper/OperatorCollection/CuOperatorPointCloudClustering.h>
#include <Copper/OperatorCollection/CuOperatorCommonDevice.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuSparseDataBlock.h>
#include <Copper/CuTransferFunction.h>

#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#ifdef __CUDACC__
#define LDG(ptr, idx) __ldg(&(ptr)[idx])
#else
#define LDG(ptr, idx) (ptr)[idx]
#endif

#define FETCH(ptr, idx) fetch_val(ptr, idx)

// Helper Device Functions
//namespace {
//    template <typename T>
//    __device__ __forceinline__ T fetch_val(const T* ptr, int idx) {
//        return LDG(ptr, idx);
//    }
//    template <>
//    __device__ __forceinline__ float3 fetch_val(const float3* ptr, int idx) {
//        float3 ret;
//        ret.x = LDG(ptr, idx).x; ret.y = LDG(ptr, idx).y; ret.z = LDG(ptr, idx).z;
//        return ret;
//    }
//
//    __device__ __forceinline__ float getDistSq(float3 a, float3 b) {
//        float dx = a.x - b.x; float dy = a.y - b.y; float dz = a.z - b.z;
//        return dx * dx + dy * dy + dz * dz;
//    }
//}

// --------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------

__global__ void initClusterIndicesKernel(uint64_t* clusterIndices, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        clusterIndices[idx] = (uint64_t)idx;
    }
}

__global__ void clusterHookKernel(
    const float3* __restrict__ positions,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    uint64_t* __restrict__ clusterIndices,
    bool* __restrict__ changed,
    int numPoints, float radiusSq, float cellSize, int3 gridSize, float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    float3 myPos = FETCH(positions, index);

    // Find Root
    uint64_t myRoot = clusterIndices[index];
    while (myRoot != clusterIndices[myRoot]) {
        myRoot = clusterIndices[myRoot];
    }

    // Grid Search
    float invCellSize = 1.0f / cellSize;
    int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    int searchRange = (int)ceilf(sqrtf(radiusSq) * invCellSize);

    for (int z = -searchRange; z <= searchRange; ++z) {
        for (int y = -searchRange; y <= searchRange; ++y) {
            for (int x = -searchRange; x <= searchRange; ++x) {
                int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z) {
                    int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                    int start = FETCH(cellStart, hash);
                    if (start != -1) {
                        int end = FETCH(cellEnd, hash);
                        for (int j = start; j < end; ++j) {
                            if (index == j) continue;

                            float3 otherPos = FETCH(positions, j);
                            if (getDistSq(myPos, otherPos) <= radiusSq) {
                                // Union
                                uint64_t neighborRoot = clusterIndices[j];
                                while (neighborRoot != clusterIndices[neighborRoot]) {
                                    neighborRoot = clusterIndices[neighborRoot];
                                }

                                if (myRoot != neighborRoot) {
                                    uint64_t minID = (myRoot < neighborRoot) ? myRoot : neighborRoot;
                                    uint64_t maxID = (myRoot > neighborRoot) ? myRoot : neighborRoot;

                                    // atomicMin supports unsigned long long (uint64_t)
                                    uint64_t old = atomicMin((unsigned long long*) & clusterIndices[maxID], (unsigned long long)minID);

                                    if (old != minID) *changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void clusterCompressKernel(uint64_t* clusterIndices, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    uint64_t parent = clusterIndices[idx];
    uint64_t grandParent = clusterIndices[parent];

    if (parent != grandParent) {
        clusterIndices[idx] = grandParent;
    }
}

__global__ void clusterFinalFlattenKernel(uint64_t* clusterIndices, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    uint64_t root = clusterIndices[idx];
    while (root != clusterIndices[root]) {
        root = clusterIndices[root];
    }
    clusterIndices[idx] = root;
}

__global__ void calculateClusterSizesKernel(const uint64_t* clusterIndices, int* clusterSizes, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    uint64_t root = clusterIndices[idx];
    atomicAdd(&clusterSizes[root], 1);
}

__global__ void filterClustersKernel(uint64_t* clusterIndices, const int* clusterSizes, int numPoints, int minSize, int maxSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    uint64_t root = clusterIndices[idx];
    int size = clusterSizes[root];

    if (size < minSize || size > maxSize) {
        clusterIndices[idx] = UINT64_MAX; // Noise
    }
}

// --------------------------------------------------------------------------
// Host Implementation
// --------------------------------------------------------------------------

void CuOperatorPointCloudClustering::Execute(const CuOperatorParameters& params, std::vector<uint64_t>& result)
{
    thrust::device_vector<uint64_t> d_indices = ExecuteDevice(params);

    if (d_indices.empty()) {
        result.clear();
        return;
    }

    result.resize(d_indices.size());
    thrust::copy(d_indices.begin(), d_indices.end(), result.begin());
}

thrust::device_vector<uint64_t> CuOperatorPointCloudClustering::ExecuteDevice(const CuOperatorParameters& params)
{
    thrust::device_vector<uint64_t> deviceResult;

    CuPointCloud* pointCloud = params.GetParameter("pointCloud", static_cast<CuPointCloud*>(nullptr));
    CuSparseDataBlock* sparseBlock = params.GetParameter("sparseDataBlock", static_cast<CuSparseDataBlock*>(nullptr));

    float radius = params.GetParameter("radius", 0.5f);
    int minClusterSize = params.GetParameter("minClusterSize", 10);
    int maxClusterSize = params.GetParameter("maxClusterSize", 100000);

    if (!pointCloud || !sparseBlock || pointCloud->size() == 0) {
        return deviceResult;
    }

    int numPoints = (int)pointCloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    deviceResult.resize(numPoints);
    initClusterIndicesKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(deviceResult.data()), numPoints);
    cudaDeviceSynchronize();

    bool h_changed = true;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    float radiusSq = radius * radius;
    int iter = 0;
    int maxIter = 100;

    while (h_changed && iter < maxIter) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        clusterHookKernel << <numBlocks, blockSize >> > (
            thrust::raw_pointer_cast(pointCloud->points.data()),
            thrust::raw_pointer_cast(sparseBlock->cellStartIndices.data()),
            thrust::raw_pointer_cast(sparseBlock->cellEndIndices.data()),
            thrust::raw_pointer_cast(deviceResult.data()),
            d_changed,
            numPoints, radiusSq, sparseBlock->cellSize, sparseBlock->gridSize, sparseBlock->worldOrigin
            );
        cudaDeviceSynchronize();

        clusterCompressKernel << <numBlocks, blockSize >> > (
            thrust::raw_pointer_cast(deviceResult.data()),
            numPoints
            );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;
    }
    cudaFree(d_changed);

    clusterFinalFlattenKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(deviceResult.data()), numPoints);
    cudaDeviceSynchronize();

    thrust::device_vector<int> clusterSizes(numPoints, 0);

    calculateClusterSizesKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(deviceResult.data()),
        thrust::raw_pointer_cast(clusterSizes.data()),
        numPoints
        );
    cudaDeviceSynchronize();

    filterClustersKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(deviceResult.data()),
        thrust::raw_pointer_cast(clusterSizes.data()),
        numPoints, minClusterSize, maxClusterSize
        );
    cudaDeviceSynchronize();

    return deviceResult;
}