#include <Copper/CuSparseDataBlock.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuTransferFunction.h> 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

#ifdef __CUDACC__
#define LDG(ptr, idx) __ldg(&(ptr)[idx])
#else
#define LDG(ptr, idx) (ptr)[idx]
#endif

// float3 Å¸ÀÔ Áö¿ø¿ë ÇïÆÛ (__ldg »ç¿ë)
namespace {
    template <typename T>
    __device__ __forceinline__ T fetch_val(const T* ptr, int idx) {
        return __ldg(&ptr[idx]);
    }
    template <>
    __device__ __forceinline__ float3 fetch_val(const float3* ptr, int idx) {
        float3 ret;
        ret.x = __ldg(&ptr[idx].x);
        ret.y = __ldg(&ptr[idx].y);
        ret.z = __ldg(&ptr[idx].z);
        return ret;
    }
}
#define FETCH(ptr, idx) fetch_val(ptr, idx)

namespace
{
    constexpr int MAX_K = 32;

    __device__ __forceinline__ float getDistSq(float3 a, float3 b)
    {
        float dx = a.x - b.x; float dy = a.y - b.y; float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __forceinline__ void solveEigen3x3_Fast(
        float xx, float xy, float xz, float yy, float yz, float zz, float3& outNormal)
    {
        float m = (xx + yy + zz) * 0.333333f;
        xx -= m; yy -= m; zz -= m;

        float3 r0 = make_float3(xx, xy, xz);
        float3 r1 = make_float3(xy, yy, yz);

        float3 c0 = make_float3(r0.y * r1.z - r0.z * r1.y, r0.z * r1.x - r0.x * r1.z, r0.x * r1.y - r0.y * r1.x);
        float lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;

        if (lenSq < 1e-10f) {
            float3 r2 = make_float3(xz, yz, zz);
            c0 = make_float3(r0.y * r2.z - r0.z * r2.y, r0.z * r2.x - r0.x * r2.z, r0.x * r2.y - r0.y * r2.x);
            lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;
        }

        if (lenSq > 1e-12f) {
            float invLen = rsqrtf(lenSq);
            outNormal.x = c0.x * invLen; outNormal.y = c0.y * invLen; outNormal.z = c0.z * invLen;
        }
        else {
            outNormal = make_float3(0.0f, 0.0f, 1.0f);
        }
    }

    // [SOR Kernel]
    __global__ void computeMeanDistanceKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outMeanDists, int numParticles, int k_neighbors, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);
        float invCellSize = 1.0f / cellSize;

        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        float dists[MAX_K];
        for (int i = 0; i < k_neighbors; ++i) dists[i] = 1.0e20f;
        float currentMaxDist = 1.0e20f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z) {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1) {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j) {
                                if (j == index) continue;
                                float3 otherPos = FETCH(positions, j);
                                float d2 = getDistSq(myPos, otherPos);

                                if (d2 < currentMaxDist) {
                                    dists[currentMaxIdx] = d2;
                                    float newMax = -1.0f;
                                    int newIdx = -1;
                                    for (int k = 0; k < k_neighbors; ++k) {
                                        if (dists[k] > newMax) {
                                            newMax = dists[k];
                                            newIdx = k;
                                        }
                                    }
                                    currentMaxDist = newMax;
                                    currentMaxIdx = newIdx;
                                }
                            }
                        }
                    }
                }
            }
        }

        float sumDist = 0.0f;
        int validCount = 0;
        for (int i = 0; i < k_neighbors; ++i) {
            if (dists[i] < 1.0e19f) {
                sumDist += sqrtf(dists[i]);
                validCount++;
            }
        }

        outMeanDists[index] = (validCount > 0) ? (sumDist / (float)validCount) : 0.0f;
    }

    // [PFOR Kernel]
    __global__ void computePlaneDistanceKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outPlaneDists, int numParticles, int k_neighbors, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);
        float invCellSize = 1.0f / cellSize;

        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        int neighbors[MAX_K];
        float dists[MAX_K];
        for (int i = 0; i < k_neighbors; ++i) dists[i] = 1.0e20f;
        float currentMaxDist = 1.0e20f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z) {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1) {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j) {
                                if (j == index) continue;
                                float3 p = FETCH(positions, j);
                                float d2 = getDistSq(myPos, p);

                                if (d2 < currentMaxDist) {
                                    dists[currentMaxIdx] = d2;
                                    neighbors[currentMaxIdx] = j;

                                    float newMax = -1.0f;
                                    int newIdx = -1;
                                    for (int k = 0; k < k_neighbors; ++k) {
                                        if (dists[k] > newMax) {
                                            newMax = dists[k];
                                            newIdx = k;
                                        }
                                    }
                                    currentMaxDist = newMax;
                                    currentMaxIdx = newIdx;
                                }
                            }
                        }
                    }
                }
            }
        }

        float3 sumDiff = make_float3(0.0f, 0.0f, 0.0f);
        float s_xx = 0.0f, s_xy = 0.0f, s_xz = 0.0f;
        float s_yy = 0.0f, s_yz = 0.0f, s_zz = 0.0f;
        int count = 0;

        for (int i = 0; i < k_neighbors; ++i) {
            if (dists[i] >= 1.0e19f) continue;
            float3 p = FETCH(positions, neighbors[i]);

            float dx = p.x - myPos.x;
            float dy = p.y - myPos.y;
            float dz = p.z - myPos.z;

            sumDiff.x += dx; sumDiff.y += dy; sumDiff.z += dz;
            s_xx += dx * dx; s_xy += dx * dy; s_xz += dx * dz;
            s_yy += dy * dy; s_yz += dy * dz; s_zz += dz * dz;
            count++;
        }

        if (count < 3) {
            outPlaneDists[index] = 0.0f;
            return;
        }

        float invN = 1.0f / (float)count;
        float3 relCentroid = make_float3(sumDiff.x * invN, sumDiff.y * invN, sumDiff.z * invN);

        float xx = s_xx * invN - relCentroid.x * relCentroid.x;
        float xy = s_xy * invN - relCentroid.x * relCentroid.y;
        float xz = s_xz * invN - relCentroid.x * relCentroid.z;
        float yy = s_yy * invN - relCentroid.y * relCentroid.y;
        float yz = s_yz * invN - relCentroid.y * relCentroid.z;
        float zz = s_zz * invN - relCentroid.z * relCentroid.z;

        float3 normal;
        solveEigen3x3_Fast(xx, xy, xz, yy, yz, zz, normal);

        float dist = fabsf(normal.x * relCentroid.x + normal.y * relCentroid.y + normal.z * relCentroid.z);
        outPlaneDists[index] = dist;
    }

    __global__ void markOutliersKernel(bool* isAlive, const float* __restrict__ vals, float threshold, int numParticles) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;
        if (vals[index] > threshold) isAlive[index] = false;
    }

    struct SquareOp { __host__ __device__ float operator()(float x) const { return x * x; } };
    struct Float3ToPair { __host__ __device__ thrust::pair<float3, float3> operator()(const float3& x) const { return thrust::make_pair(x, x); } };
    struct Float3MinMax {
        __host__ __device__ thrust::pair<float3, float3> operator()(const thrust::pair<float3, float3>& a, const thrust::pair<float3, float3>& b) const {
            float3 minVal = { fminf(a.first.x, b.first.x), fminf(a.first.y, b.first.y), fminf(a.first.z, b.first.z) };
            float3 maxVal = { fmaxf(a.second.x, b.second.x), fmaxf(a.second.y, b.second.y), fmaxf(a.second.z, b.second.z) };
            return thrust::make_pair(minVal, maxVal);
        }
    };

    // 1. NND Kernel (Nearest Neighbor Distance) - Best K
    __global__ void computeNNDKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outValues, int numParticles, int k, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);
        float invCellSize = 1.0f / cellSize;
        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        float dists[MAX_K];
        for (int i = 0; i < k; ++i) dists[i] = 1.0e20f;
        float currentMaxDist = 1.0e20f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z)
        {
            for (int y = -1; y <= 1; ++y)
            {
                for (int x = -1; x <= 1; ++x)
                {
                    int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1)
                        {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j)
                            {
                                if (j == index) continue;
                                float d2 = getDistSq(myPos, FETCH(positions, j));
                                if (d2 < currentMaxDist)
                                {
                                    dists[currentMaxIdx] = d2;
                                    float newMax = -1.0f;
                                    int newIdx = -1;
                                    for (int kk = 0; kk < k; ++kk)
                                    {
                                        if (dists[kk] > newMax)
                                        {
                                            newMax = dists[kk];
                                            newIdx = kk;
                                        }
                                    }
                                    currentMaxDist = newMax; currentMaxIdx = newIdx;
                                }
                            }
                        }
                    }
                }
            }
        }
        float sumDist = 0.0f;
        int count = 0;
        for (int i = 0; i < k; ++i) {
            if (dists[i] < 1.0e19f) { sumDist += sqrtf(dists[i]); count++; }
        }
        outValues[index] = (count > 0) ? (sumDist / count) : 0.0f;
    }

    // 2. Density Kernel (Radius Search)
    __global__ void computeDensityKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outValues, int numParticles, float radius, int mode, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);
        float rSq = radius * radius;
        float invHSq = 1.0f / ((radius * 0.5f) * (radius * 0.5f));

        float invCellSize = 1.0f / cellSize;
        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        int searchRange = (int)ceilf(radius * invCellSize);
        float density = 0.0f;

        for (int z = -searchRange; z <= searchRange; ++z)
        {
            for (int y = -searchRange; y <= searchRange; ++y)
            {
                for (int x = -searchRange; x <= searchRange; ++x)
                {
                    int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1)
                        {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j)
                            {
                                float d2 = getDistSq(myPos, FETCH(positions, j));
                                if (d2 <= rSq)
                                {
                                    if (mode == 0) density += 1.0f; // LDE
                                    else density += expf(-d2 * invHSq); // KDE
                                }
                            }
                        }
                    }
                }
            }
        }
        outValues[index] = density;
    }

    __global__ void applyTransferFunctionKernel(uchar3* colors, const float* values, int numParticles, CuTransferFunction tf)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        uchar4 c4 = tf.Map(values[index]);
        colors[index] = make_uchar3(c4.x, c4.y, c4.z);
    }
}

__global__ void computeHashKernel(
    const float3* positions, int* particleHash, int numParticles, float cellSize, int3 gridSize, float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    float3 p = positions[index];
    int gridX = (int)((p.x - worldOrigin.x) / cellSize);
    int gridY = (int)((p.y - worldOrigin.y) / cellSize);
    int gridZ = (int)((p.z - worldOrigin.z) / cellSize);
    if (gridX < 0) gridX = 0; else if (gridX >= gridSize.x) gridX = gridSize.x - 1;
    if (gridY < 0) gridY = 0; else if (gridY >= gridSize.y) gridY = gridSize.y - 1;
    if (gridZ < 0) gridZ = 0; else if (gridZ >= gridSize.z) gridZ = gridSize.z - 1;
    particleHash[index] = (gridZ * gridSize.y + gridY) * gridSize.x + gridX;
}

__global__ void findCellStartEndKernel(const int* particleHash, int* cellStart, int* cellEnd, int numParticles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    int currentHash = particleHash[index];
    if (index == 0) cellStart[currentHash] = index;
    else {
        int prevHash = particleHash[index - 1];
        if (currentHash != prevHash) { cellEnd[prevHash] = index; cellStart[currentHash] = index; }
    }
    if (index == numParticles - 1) cellEnd[currentHash] = numParticles;
}

CuSparseDataBlock::CuSparseDataBlock() {}

float CuSparseDataBlock::computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier) {
    if (points.empty()) return 1.0f;
    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(points.begin(), points.end(), Float3ToPair(), init, Float3MinMax());
    float dx = bbox.second.x - bbox.first.x; float dy = bbox.second.y - bbox.first.y; float dz = bbox.second.z - bbox.first.z;
    float maxDim = fmaxf(dx, fmaxf(dy, dz));
    if (maxDim <= 0.0f) maxDim = 1.0f;
    return (maxDim / powf((float)points.size(), 1.0f / 3.0f)) * multiplier;
}

void CuSparseDataBlock::Build(CuPointCloud* cloud) {
    if (cloud == nullptr || cloud->size() == 0) return;
    cellSize = computeAutoCellSize(cloud->points, 1.5f);
    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(cloud->points.begin(), cloud->points.end(), Float3ToPair(), init, Float3MinMax());
    worldOrigin = bbox.first; float3 maxP = bbox.second;
    float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };
    gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };
    numberOfCells = gridSize.x * gridSize.y * gridSize.z;
    size_t numPoints = cloud->size();
    if (hashCodes.size() != numPoints) hashCodes.resize(numPoints);
    if (cellStartIndices.size() != numberOfCells) cellStartIndices.resize(numberOfCells);
    if (cellEndIndices.size() != numberOfCells) cellEndIndices.resize(numberOfCells);
    thrust::fill(cellStartIndices.begin(), cellStartIndices.end(), -1);
    thrust::fill(cellEndIndices.begin(), cellEndIndices.end(), -1);
    int blockSize = 256; int numBlocks = (int)((numPoints + blockSize - 1) / blockSize);
    computeHashKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(cloud->points.data()), thrust::raw_pointer_cast(hashCodes.data()), (int)numPoints, cellSize, gridSize, worldOrigin);
    cudaDeviceSynchronize();
    thrust::sort_by_key(hashCodes.begin(), hashCodes.end(), thrust::make_zip_iterator(thrust::make_tuple(cloud->points.begin(), cloud->normals.begin(), cloud->colors.begin(), cloud->isAlive.begin())));
    findCellStartEndKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(hashCodes.data()), thrust::raw_pointer_cast(cellStartIndices.data()), thrust::raw_pointer_cast(cellEndIndices.data()), (int)numPoints);
    cudaDeviceSynchronize();
}

thrust::device_vector<float> CuSparseDataBlock::ApplySOR(CuPointCloud* cloud, int k, float stdDevMult)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    if (k > MAX_K) k = MAX_K;
    int numPoints = (int)cloud->size();
    int blockSize = 128;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_meanDists(numPoints);

    computeMeanDistanceKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_meanDists.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    float sum = thrust::reduce(d_meanDists.begin(), d_meanDists.end(), 0.0f, thrust::plus<float>());
    float mean = sum / numPoints;
    float sumSq = thrust::transform_reduce(d_meanDists.begin(), d_meanDists.end(), SquareOp(), 0.0f, thrust::plus<float>());
    float variance = (sumSq / numPoints) - (mean * mean);
    float stdDev = sqrtf(variance > 0 ? variance : 0.0f);
    float threshold = mean + stdDevMult * stdDev;

    markOutliersKernel << <numBlocks, 256 >> > (
        thrust::raw_pointer_cast(cloud->isAlive.data()),
        thrust::raw_pointer_cast(d_meanDists.data()),
        threshold, numPoints
        );
    cudaDeviceSynchronize();

    return d_meanDists;
}

thrust::device_vector<float> CuSparseDataBlock::ApplyPFOR(CuPointCloud* cloud, int k, float distanceThreshold)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    if (k > MAX_K) k = MAX_K;
    int numPoints = (int)cloud->size();
    int blockSize = 128;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_planeDists(numPoints);

    computePlaneDistanceKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_planeDists.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    markOutliersKernel << <numBlocks, 256 >> > (
        thrust::raw_pointer_cast(cloud->isAlive.data()),
        thrust::raw_pointer_cast(d_planeDists.data()),
        distanceThreshold, numPoints
        );
    cudaDeviceSynchronize();

    return d_planeDists;
}

// [NND] Invert=false
thrust::device_vector<float> CuSparseDataBlock::ApplyNND(CuPointCloud* cloud, int k)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_values(numPoints);

    computeNNDKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(d_values.begin(), d_values.end());

    // Transfer Function »ý¼º
    CuTransferFunction tf(*minmax.first, *minmax.second);
    tf.invert = false; // °¡±î¿ò=Low=ÆÄ¶û

    applyTransferFunctionKernel << <numBlocks, blockSize >> > (
        (uchar3*)thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, tf
        );
    cudaDeviceSynchronize();

    return d_values;
}

// [LDE] Invert=true
thrust::device_vector<float> CuSparseDataBlock::ApplyLDE(CuPointCloud* cloud, float radius)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_values(numPoints);

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, radius, 0, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(d_values.begin(), d_values.end());

    // Transfer Function »ý¼º
    CuTransferFunction tf(*minmax.first, *minmax.second);
    tf.SetGray();
    //tf.invert = true; // ¹Ðµµ ³·À½=Low=»¡°­

    applyTransferFunctionKernel << <numBlocks, blockSize >> > (
        (uchar3*)thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, tf
        );
    cudaDeviceSynchronize();

    return d_values;
}

// [KDE] Invert=true
//thrust::device_vector<float> CuSparseDataBlock::ApplyKDE(CuPointCloud* cloud, float bandwidth)
//{
//    if (cloud == nullptr || cloud->size() == 0)
//    {
//        return thrust::device_vector<float>();
//    }
//
//    int numPoints = (int)cloud->size();
//    int blockSize = 256;
//    int numBlocks = (numPoints + blockSize - 1) / blockSize;
//
//    thrust::device_vector<float> d_values(numPoints);
//    float searchRadius = bandwidth * 3.0f;
//
//    computeDensityKernel << <numBlocks, blockSize >> > (
//        thrust::raw_pointer_cast(cloud->points.data()),
//        thrust::raw_pointer_cast(cellStartIndices.data()),
//        thrust::raw_pointer_cast(cellEndIndices.data()),
//        thrust::raw_pointer_cast(d_values.data()),
//        numPoints, searchRadius, 1, cellSize, gridSize, worldOrigin
//        );
//    cudaDeviceSynchronize();
//
//    auto minmax = thrust::minmax_element(d_values.begin(), d_values.end());
//
//    // Transfer Function »ý¼º
//    CuTransferFunction tf(*minmax.first, *minmax.second);
//    tf.SetJet();
//    tf.invert = true; // ¹Ðµµ ³·À½=Low=»¡°­
//
//    applyTransferFunctionKernel << <numBlocks, blockSize >> > (
//        (uchar3*)thrust::raw_pointer_cast(cloud->colors.data()),
//        thrust::raw_pointer_cast(d_values.data()),
//        numPoints, tf
//        );
//    cudaDeviceSynchronize();
//
//    return d_values;
//}
