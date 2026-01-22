#include <Copper/CuSparseDataBlock.h>
#include <Copper/CuPointCloud.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

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
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    // ----------------------------------------------------------------------
    // 3x3 Eigen Solver (Analytical)
    // ----------------------------------------------------------------------
    __device__ __forceinline__ void solveEigen3x3_Fast(
        float xx, float xy, float xz,
        float yy, float yz,
        float zz,
        float3& outNormal)
    {
        float m = (xx + yy + zz) * 0.333333f;
        xx -= m; yy -= m; zz -= m;

        float3 r0 = make_float3(xx, xy, xz);
        float3 r1 = make_float3(xy, yy, yz);
        float3 r2 = make_float3(xz, yz, zz);

        float3 c0 = make_float3(r0.y * r1.z - r0.z * r1.y, r0.z * r1.x - r0.x * r1.z, r0.x * r1.y - r0.y * r1.x);
        float3 c1 = make_float3(r0.y * r2.z - r0.z * r2.y, r0.z * r2.x - r0.x * r2.z, r0.x * r2.y - r0.y * r2.x);
        float3 c2 = make_float3(r1.y * r2.z - r1.z * r2.y, r1.z * r2.x - r1.x * r2.z, r1.x * r2.y - r1.y * r2.x);

        float len0 = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;
        float len1 = c1.x * c1.x + c1.y * c1.y + c1.z * c1.z;
        float len2 = c2.x * c2.x + c2.y * c2.y + c2.z * c2.z;

        if (len0 >= len1 && len0 >= len2) outNormal = c0;
        else if (len1 >= len0 && len1 >= len2) outNormal = c1;
        else outNormal = c2;

        float len = sqrtf(outNormal.x * outNormal.x + outNormal.y * outNormal.y + outNormal.z * outNormal.z);
        if (len > 1e-12f) {
            float invLen = rsqrtf(len);
            outNormal.x *= invLen; outNormal.y *= invLen; outNormal.z *= invLen;
        }
        else {
            outNormal = make_float3(0.0f, 0.0f, 1.0f);
        }
    }

    // ----------------------------------------------------------------------
    // [SOR Kernel]
    // ----------------------------------------------------------------------
    __global__ void computeMeanDistanceKernel(
        const float3* __restrict__ positions,
        const int* __restrict__ cellStart,
        const int* __restrict__ cellEnd,
        float* __restrict__ outMeanDists,
        int numParticles,
        int k_neighbors,
        float cellSize,
        int3 gridSize,
        float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);

        float invCellSize = 1.0f / cellSize;
        int gridX = (int)((myPos.x - worldOrigin.x) * invCellSize);
        int gridY = (int)((myPos.y - worldOrigin.y) * invCellSize);
        int gridZ = (int)((myPos.z - worldOrigin.z) * invCellSize);

        if (gridX < 0) gridX = 0; else if (gridX >= gridSize.x) gridX = gridSize.x - 1;
        if (gridY < 0) gridY = 0; else if (gridY >= gridSize.y) gridY = gridSize.y - 1;
        if (gridZ < 0) gridZ = 0; else if (gridZ >= gridSize.z) gridZ = gridSize.z - 1;

        float dists[MAX_K];
        for (int i = 0; i < k_neighbors; ++i) dists[i] = 1.0e10f;

        float currentMaxDist = 1.0e10f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z) {
#pragma unroll
            for (int y = -1; y <= 1; ++y) {
#pragma unroll
                for (int x = -1; x <= 1; ++x) {
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;

                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
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
            if (dists[i] < 1.0e9f) {
                sumDist += sqrtf(dists[i]);
                validCount++;
            }
        }

        if (validCount > 0)
            outMeanDists[index] = sumDist / (float)validCount;
        else
            outMeanDists[index] = 0.0f;
    }

    // ----------------------------------------------------------------------
    // [PFOR Kernel]
    // ----------------------------------------------------------------------
    __global__ void computePlaneDistanceKernel(
        const float3* __restrict__ positions,
        const int* __restrict__ cellStart,
        const int* __restrict__ cellEnd,
        float* __restrict__ outPlaneDists,
        int numParticles,
        int k_neighbors,
        float cellSize,
        int3 gridSize,
        float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);

        float invCellSize = 1.0f / cellSize;
        int gridX = (int)((myPos.x - worldOrigin.x) * invCellSize);
        int gridY = (int)((myPos.y - worldOrigin.y) * invCellSize);
        int gridZ = (int)((myPos.z - worldOrigin.z) * invCellSize);

        if (gridX < 0) gridX = 0; else if (gridX >= gridSize.x) gridX = gridSize.x - 1;
        if (gridY < 0) gridY = 0; else if (gridY >= gridSize.y) gridY = gridSize.y - 1;
        if (gridZ < 0) gridZ = 0; else if (gridZ >= gridSize.z) gridZ = gridSize.z - 1;

        int neighbors[MAX_K];
        float dists[MAX_K];

        for (int i = 0; i < k_neighbors; ++i) dists[i] = 1.0e10f;
        float currentMaxDist = 1.0e10f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z) {
#pragma unroll
            for (int y = -1; y <= 1; ++y) {
#pragma unroll
                for (int x = -1; x <= 1; ++x) {
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;

                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);

                        if (start != -1) {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j) {
                                float3 otherPos = FETCH(positions, j);
                                float d2 = getDistSq(myPos, otherPos);

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

        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        float s_xx = 0.0f, s_xy = 0.0f, s_xz = 0.0f;
        float s_yy = 0.0f, s_yz = 0.0f, s_zz = 0.0f;
        int validCount = 0;

        for (int i = 0; i < k_neighbors; ++i) {
            if (dists[i] >= 1.0e9f) continue;

            validCount++;
            // 여기서도 float3 FETCH 사용
            float3 p = FETCH(positions, neighbors[i]);

            sum.x += p.x; sum.y += p.y; sum.z += p.z;
            s_xx += p.x * p.x; s_xy += p.x * p.y; s_xz += p.x * p.z;
            s_yy += p.y * p.y; s_yz += p.y * p.z; s_zz += p.z * p.z;
        }

        if (validCount < 3) {
            outPlaneDists[index] = 1e30f;
            return;
        }

        float invN = 1.0f / (float)validCount;
        float3 centroid = make_float3(sum.x * invN, sum.y * invN, sum.z * invN);

        float xx = s_xx * invN - centroid.x * centroid.x;
        float xy = s_xy * invN - centroid.x * centroid.y;
        float xz = s_xz * invN - centroid.x * centroid.z;
        float yy = s_yy * invN - centroid.y * centroid.y;
        float yz = s_yz * invN - centroid.y * centroid.z;
        float zz = s_zz * invN - centroid.z * centroid.z;

        float3 normal;
        solveEigen3x3_Fast(xx, xy, xz, yy, yz, zz, normal);

        float dist = fabsf(normal.x * (myPos.x - centroid.x) +
            normal.y * (myPos.y - centroid.y) +
            normal.z * (myPos.z - centroid.z));

        outPlaneDists[index] = dist;
    }

    // 마킹 커널
    __global__ void markOutliersKernel(
        bool* isAlive,
        const float* __restrict__ meanDists,
        float threshold,
        int numParticles)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;
        if (meanDists[index] > threshold) isAlive[index] = false;
    }

    struct SquareOp {
        __host__ __device__ float operator()(float x) const { return x * x; }
    };

    struct Float3ToPair {
        __host__ __device__ thrust::pair<float3, float3> operator()(const float3& x) const { return thrust::make_pair(x, x); }
    };

    struct Float3MinMax {
        __host__ __device__ thrust::pair<float3, float3> operator()(const thrust::pair<float3, float3>& a, const thrust::pair<float3, float3>& b) const {
            float3 minVal = { fminf(a.first.x, b.first.x), fminf(a.first.y, b.first.y), fminf(a.first.z, b.first.z) };
            float3 maxVal = { fmaxf(a.second.x, b.second.x), fmaxf(a.second.y, b.second.y), fmaxf(a.second.z, b.second.z) };
            return thrust::make_pair(minVal, maxVal);
        }
    };
} // namespace

// ------------------------------------------------------------------
// [Hash & Build Kernels]
// ------------------------------------------------------------------
__global__ void computeHashKernel(
    const float3* positions, int* particleHash, int numParticles,
    float cellSize, int3 gridSize, float3 worldOrigin)
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

__global__ void findCellStartEndKernel(
    const int* particleHash, int* cellStart, int* cellEnd, int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;
    int currentHash = particleHash[index];
    if (index == 0) cellStart[currentHash] = index;
    else {
        int prevHash = particleHash[index - 1];
        if (currentHash != prevHash) {
            cellEnd[prevHash] = index;
            cellStart[currentHash] = index;
        }
    }
    if (index == numParticles - 1) cellEnd[currentHash] = numParticles;
}

CuSparseDataBlock::CuSparseDataBlock() {}

float CuSparseDataBlock::computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier)
{
    if (points.empty()) return 1.0f;
    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(points.begin(), points.end(), Float3ToPair(), init, Float3MinMax());

    float dx = bbox.second.x - bbox.first.x;
    float dy = bbox.second.y - bbox.first.y;
    float dz = bbox.second.z - bbox.first.z;
    if (dx <= 0) dx = 1.0f; if (dy <= 0) dy = 1.0f; if (dz <= 0) dz = 1.0f;
    float volume = dx * dy * dz;

    return powf(volume / (float)points.size(), 1.0f / 3.0f) * multiplier;
}

void CuSparseDataBlock::Build(CuPointCloud* cloud)
{
    if (cloud == nullptr || cloud->size() == 0) return;

    cellSize = computeAutoCellSize(cloud->points, 1.5f);

    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(cloud->points.begin(), cloud->points.end(), Float3ToPair(), init, Float3MinMax());
    worldOrigin = bbox.first;
    float3 maxP = bbox.second;

    float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };
    gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };
    numberOfCells = gridSize.x * gridSize.y * gridSize.z;

    size_t numPoints = cloud->size();
    if (hashCodes.size() != numPoints) hashCodes.resize(numPoints);
    if (cellStartIndices.size() != numberOfCells) cellStartIndices.resize(numberOfCells);
    if (cellEndIndices.size() != numberOfCells) cellEndIndices.resize(numberOfCells);

    thrust::fill(cellStartIndices.begin(), cellStartIndices.end(), -1);
    thrust::fill(cellEndIndices.begin(), cellEndIndices.end(), -1);

    int blockSize = 256;
    int numBlocks = (int)((numPoints + blockSize - 1) / blockSize);

    computeHashKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(hashCodes.data()),
        (int)numPoints, cellSize, gridSize, worldOrigin);
    cudaDeviceSynchronize();

    thrust::sort_by_key(
        hashCodes.begin(), hashCodes.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            cloud->points.begin(), cloud->normals.begin(), cloud->colors.begin(), cloud->isAlive.begin()))
    );

    findCellStartEndKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(hashCodes.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        (int)numPoints);
    cudaDeviceSynchronize();
}

void CuSparseDataBlock::ApplySOR(CuPointCloud* cloud, int k, float stdDevMult)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    if (k > MAX_K) k = MAX_K;

    int numPoints = (int)cloud->size();
    int blockSize = 256;
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
}

void CuSparseDataBlock::ApplyPFOR(CuPointCloud* cloud, int k, float distanceThreshold)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    if (k > MAX_K) k = MAX_K;

    int numPoints = (int)cloud->size();
    int blockSize = 256;
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
}

namespace
{
    // [Heatmap] Scalar 값을 Jet Colormap (Blue->Red)으로 변환
    __device__ uchar3 scalarToColor(float value, float minVal, float maxVal, bool invert = false)
    {
        if (maxVal - minVal < 1e-6f) return make_uchar3(0, 255, 0); // Green

        float t = (value - minVal) / (maxVal - minVal);
        t = fmaxf(0.0f, fminf(1.0f, t));
        if (invert) t = 1.0f - t;

        float r = 0.0f, g = 0.0f, b = 0.0f;

        if (t < 0.25f) { r = 0.0f; g = 4.0f * t; b = 1.0f; }
        else if (t < 0.5f) { r = 0.0f; g = 1.0f; b = 1.0f - 4.0f * (t - 0.25f); }
        else if (t < 0.75f) { r = 4.0f * (t - 0.5f); g = 1.0f; b = 0.0f; }
        else { r = 1.0f; g = 1.0f - 4.0f * (t - 0.75f); b = 0.0f; }

        return make_uchar3((unsigned char)(r * 255), (unsigned char)(g * 255), (unsigned char)(b * 255));
    }

    // 1. NND Kernel (Nearest Neighbor Distance)
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

        float sumDist = 0.0f;
        int count = 0;

        // Fast K-NN Search (No Sort, just Accumulate)
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
                                float d2 = getDistSq(myPos, FETCH(positions, j));
                                sumDist += sqrtf(d2);
                                count++;
                                if (count >= k) goto done_nnd;
                            }
                        }
                    }
                }
            }
        }
    done_nnd:
        outValues[index] = (count > 0) ? (sumDist / count) : 0.0f;
    }

    // 2. LDE/KDE Kernel (Radius Search)
    // mode 0: LDE (Simple Count), mode 1: KDE (Gaussian Weight)
    __global__ void computeDensityKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outValues, int numParticles, float radius, int mode, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = FETCH(positions, index);
        float rSq = radius * radius;
        // KDE Bandwidth (h) = radius / 2.0 roughly
        float invHSq = 1.0f / ((radius * 0.5f) * (radius * 0.5f));

        float invCellSize = 1.0f / cellSize;
        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        // Search range based on radius (optimized)
        int searchRange = (int)ceilf(radius * invCellSize);

        float density = 0.0f;

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
                                float d2 = getDistSq(myPos, FETCH(positions, j));
                                if (d2 <= rSq) {
                                    if (mode == 0) { // LDE
                                        density += 1.0f;
                                    }
                                    else { // KDE (Gaussian)
                                        density += expf(-d2 * invHSq);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        outValues[index] = density;
    }

    // [Colorize Kernel] 계산된 값을 색상으로 변환
    __global__ void applyColorMapKernel(
        uchar3* colors, const float* values, int numParticles, float minVal, float maxVal, bool invert)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;
        colors[index] = scalarToColor(values[index], minVal, maxVal, invert);
    }
}

void CuSparseDataBlock::ApplyNND(CuPointCloud* cloud, int k)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    int numPoints = (int)cloud->size();
    int blockSize = 256; int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_values(numPoints);

    computeNNDKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    // Min/Max 찾기 (오토 스케일링)
    auto minmax = thrust::minmax_element(d_values.begin(), d_values.end());
    float minVal = *minmax.first;
    float maxVal = *minmax.second;

    // 색상 적용 (Invert=false: 작은값(가까움)이 파랑, 큰값(멈)이 빨강)
    // 보통 거리가 먼 것(구멍)을 빨강으로 표시함.
    applyColorMapKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->colors.data()), // uchar3*
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, minVal, maxVal, false
        );
    cudaDeviceSynchronize();
}

void CuSparseDataBlock::ApplyLDE(CuPointCloud* cloud, float radius)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    int numPoints = (int)cloud->size();
    int blockSize = 256; int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_values(numPoints);

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, radius, 0, cellSize, gridSize, worldOrigin // Mode 0 = LDE
        );
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(d_values.begin(), d_values.end());

    // 색상 적용 (Invert=false: 적은 것 파랑, 많은 것 빨강)
    applyColorMapKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, *minmax.first, *minmax.second, true
        );
    cudaDeviceSynchronize();
}

// [KDE: Kernel Density Estimation] -> 부드러운 밀도
// 파랑: 밀도 낮음, 빨강: 밀도 높음
void CuSparseDataBlock::ApplyKDE(CuPointCloud* cloud, float bandwidth)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    int numPoints = (int)cloud->size();
    int blockSize = 256; int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> d_values(numPoints);

    // Bandwidth를 기준으로 검색 반경 설정 (보통 3*bandwidth까지 유효)
    float searchRadius = bandwidth * 3.0f;

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, searchRadius, 1, cellSize, gridSize, worldOrigin // Mode 1 = KDE
        );
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(d_values.begin(), d_values.end());

    applyColorMapKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(d_values.data()),
        numPoints, *minmax.first, *minmax.second, true
        );
    cudaDeviceSynchronize();
}
