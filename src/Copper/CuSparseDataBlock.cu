#include <Copper/CuSparseDataBlock.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuTransferFunction.h> 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

#ifdef __CUDACC__
#define LDG(ptr, idx) __ldg(&(ptr)[idx])
#else
#define LDG(ptr, idx) (ptr)[idx]
#endif

// float3 타입 지원용 헬퍼 (__ldg 사용)
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

    __device__ uchar3 hashToColor(int hash) {
        // 간단한 비트 연산으로 랜덤 색상 생성 (RGB)
        unsigned char r = (hash * 1664525 + 1013904223) & 0xFF;
        unsigned char g = (hash * 25214903917 + 11) & 0xFF;
        unsigned char b = (hash * 8253729 + 2396403) & 0xFF;
        return make_uchar3(r, g, b);
    }

    __global__ void colorizeByHashKernel(
        uchar3* colors, const int* hashCodes, int numParticles)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        int hash = hashCodes[index];
        colors[index] = hashToColor(hash);
    }

    // 3x3 대칭 행렬의 6개 성분 저장
    struct CovarianceData {
        float xx, yy, zz, xy, yz, zx;

        __host__ __device__ CovarianceData()
            : xx(0), yy(0), zz(0), xy(0), yz(0), zx(0) {
        }

        __host__ __device__ CovarianceData(float3 p) {
            xx = p.x * p.x; yy = p.y * p.y; zz = p.z * p.z;
            xy = p.x * p.y; yz = p.y * p.z; zx = p.z * p.x;
        }
    };

    // CovarianceData 덧셈 연산자
    struct CovarianceSumOp {
        __host__ __device__ CovarianceData operator()(const CovarianceData& a, const CovarianceData& b) const {
            CovarianceData r;
            r.xx = a.xx + b.xx; r.yy = a.yy + b.yy; r.zz = a.zz + b.zz;
            r.xy = a.xy + b.xy; r.yz = a.yz + b.yz; r.zx = a.zx + b.zx;
            return r;
        }
    };

    // float3 -> CovarianceData 변환
    struct PositionToCovariance {
        __host__ __device__ CovarianceData operator()(const float3& p) const {
            return CovarianceData(p);
        }
    };

    // float3 덧셈
    struct Float3SumOp {
        __host__ __device__ float3 operator()(const float3& a, const float3& b) const {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
    };

    // [CPU Helper] 3x3 행렬의 최소 고유벡터(Normal) 계산
    // 기존 PFOR 커널의 로직을 CPU로 포팅
    void solveEigen3x3_CPU(float xx, float xy, float xz, float yy, float yz, float zz, float3& outNormal)
    {
        // 1. Shift by Mean to avoid precision loss
        float m = (xx + yy + zz) * 0.333333f;
        xx -= m; yy -= m; zz -= m;

        // 2. Analytical approximation for smallest eigenvector
        // (행렬의 행 벡터들의 외적을 이용해 수직 벡터 찾기)
        float3 r0 = make_float3(xx, xy, xz);
        float3 r1 = make_float3(xy, yy, yz);
        float3 r2 = make_float3(xz, yz, zz);

        // 첫 번째 시도: Row0 x Row1
        float3 c0 = make_float3(
            r0.y * r1.z - r0.z * r1.y,
            r0.z * r1.x - r0.x * r1.z,
            r0.x * r1.y - r0.y * r1.x
        );
        float lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;

        // 만약 Row0, Row1이 평행하면 결과가 0이 되므로 다른 조합 시도
        if (lenSq < 1e-10f) {
            // 두 번째 시도: Row0 x Row2
            c0 = make_float3(
                r0.y * r2.z - r0.z * r2.y,
                r0.z * r2.x - r0.x * r2.z,
                r0.x * r2.y - r0.y * r2.x
            );
            lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;
        }

        if (lenSq < 1e-10f) {
            // 세 번째 시도: Row1 x Row2
            c0 = make_float3(
                r1.y * r2.z - r1.z * r2.y,
                r1.z * r2.x - r1.x * r2.z,
                r1.x * r2.y - r1.y * r2.x
            );
            lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;
        }

        if (lenSq > 1e-12f) {
            float invLen = 1.0f / sqrtf(lenSq);
            outNormal.x = c0.x * invLen;
            outNormal.y = c0.y * invLen;
            outNormal.z = c0.z * invLen;
        }
        else {
            // Degenerate case (모든 포인트가 한 직선 위에 있거나 점 하나일 때)
            outNormal = make_float3(0.0f, 1.0f, 0.0f);
        }
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

void CuSparseDataBlock::Build(CuPointCloud* cloud)
{
    if (cloud == nullptr || cloud->size() == 0) return;

    cellSize = computeAutoCellSize(cloud->points, 1.5f);

	printf("Computed cell size: %f\n", cellSize);

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
    
    int blockSize = 256; int numBlocks = (int)((numPoints + blockSize - 1) / blockSize);
    
    computeHashKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(cloud->points.data()), thrust::raw_pointer_cast(hashCodes.data()), (int)numPoints, cellSize, gridSize, worldOrigin);
    
    cudaDeviceSynchronize();
    
    thrust::sort_by_key(hashCodes.begin(), hashCodes.end(), thrust::make_zip_iterator(thrust::make_tuple(cloud->points.begin(), cloud->normals.begin(), cloud->colors.begin(), cloud->isAlive.begin())));
    
    findCellStartEndKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(hashCodes.data()), thrust::raw_pointer_cast(cellStartIndices.data()), thrust::raw_pointer_cast(cellEndIndices.data()), (int)numPoints);
    
    cudaDeviceSynchronize();
}

void CuSparseDataBlock::Build(CuPointCloud* cloud, float cellSize)
{
    if (cloud == nullptr || cloud->size() == 0) return;

    this->cellSize = cellSize;

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

    // Transfer Function 생성
    CuTransferFunction tf(*minmax.first, *minmax.second);
    tf.invert = false; // 가까움=Low=파랑

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

    // Transfer Function 생성
    CuTransferFunction tf(*minmax.first, *minmax.second);
    tf.SetGray();
    //tf.invert = true; // 밀도 낮음=Low=빨강

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
//    // Transfer Function 생성
//    CuTransferFunction tf(*minmax.first, *minmax.second);
//    tf.SetJet();
//    tf.invert = true; // 밀도 낮음=Low=빨강
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

std::vector<std::pair<float3, float3>> CuSparseDataBlock::GetActiveCellBounds()
{
    std::vector<std::pair<float3, float3>> activeBoxes;

    // 1. GPU에 있는 cellStartIndices를 CPU로 가져옴
    // 활성화된 셀인지 확인하기 위해 (값이 -1이 아니면 활성)
    thrust::host_vector<int> h_cellStart = cellStartIndices;

    // 2. 전체 그리드 셀을 순회하며 활성화된 셀 찾기
    // (GPU에서 컴팩션 할 수도 있지만, 디버깅 용도이므로 CPU 루프가 구현이 간단함)
    for (int hash = 0; hash < numberOfCells; ++hash)
    {
        if (h_cellStart[hash] != -1) // 포인트가 하나라도 들어있는 셀
        {
            // 역해싱 (Inverse Hashing): Hash -> Grid Index (x, y, z)
            int z = hash / (gridSize.x * gridSize.y);
            int rem = hash % (gridSize.x * gridSize.y);
            int y = rem / gridSize.x;
            int x = rem % gridSize.x;

            // 월드 좌표 계산
            float3 minPos;
            minPos.x = worldOrigin.x + x * cellSize;
            minPos.y = worldOrigin.y + y * cellSize;
            minPos.z = worldOrigin.z + z * cellSize;

            float3 maxPos;
            maxPos.x = minPos.x + cellSize;
            maxPos.y = minPos.y + cellSize;
            maxPos.z = minPos.z + cellSize;

            activeBoxes.push_back(std::make_pair(minPos, maxPos));
        }
    }

    return activeBoxes;
}

void CuSparseDataBlock::ColorizePointsByCell(CuPointCloud* cloud)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    if (hashCodes.empty()) return; // Build()가 먼저 호출되어야 함

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    colorizeByHashKernel << <numBlocks, blockSize >> > (
        (uchar3*)thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(hashCodes.data()),
        numPoints
        );
    cudaDeviceSynchronize();
}

std::vector<CuCellStats> CuSparseDataBlock::GetActiveCellStats(CuPointCloud* cloud)
{
    std::vector<CuCellStats> results;

    if (cloud == nullptr || cloud->size() == 0) return results;
    if (hashCodes.empty()) return results;

    int numPoints = (int)cloud->size();

    // -------------------------------------------------------
    // 1. GPU: Prepare Data
    // -------------------------------------------------------

    // (A) Covariance Data 변환 (Point -> Cov Data)
    thrust::device_vector<CovarianceData> d_covData(numPoints);
    thrust::transform(cloud->points.begin(), cloud->points.end(), d_covData.begin(), PositionToCovariance());

    // -------------------------------------------------------
    // 2. GPU: Reduce By Key (집계)
    // -------------------------------------------------------

    // 결과 저장용 버퍼
    thrust::device_vector<int> d_uniqueHashes(numPoints);
    thrust::device_vector<float3> d_sumPositions(numPoints);
    thrust::device_vector<float3> d_sumNormals(numPoints);
    thrust::device_vector<CovarianceData> d_sumCov(numPoints); // [New] 공분산 합
    thrust::device_vector<int> d_counts(numPoints);

    // [Step A] Position Sum (Centroid용)
    auto endPairPos = thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        cloud->points.begin(),
        d_uniqueHashes.begin(),
        d_sumPositions.begin(),
        thrust::equal_to<int>(),
        Float3SumOp()
    );

    // 활성 셀 개수 확인
    int activeCellCount = (int)(endPairPos.first - d_uniqueHashes.begin());

    // [Step B] Normal Sum (Orientation 참조용)
    thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        cloud->normals.begin(),
        thrust::make_discard_iterator(),
        d_sumNormals.begin(),
        thrust::equal_to<int>(),
        Float3SumOp()
    );

    // [Step C] Covariance Sum (PCA용)
    thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        d_covData.begin(),
        thrust::make_discard_iterator(),
        d_sumCov.begin(),
        thrust::equal_to<int>(),
        CovarianceSumOp()
    );

    // [Step D] Count
    thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        thrust::make_constant_iterator(1),
        thrust::make_discard_iterator(),
        d_counts.begin()
    );

    // -------------------------------------------------------
    // 3. Data Download (GPU -> CPU)
    // -------------------------------------------------------
    thrust::host_vector<int> h_hashes(d_uniqueHashes.begin(), d_uniqueHashes.begin() + activeCellCount);
    thrust::host_vector<float3> h_sumPos(d_sumPositions.begin(), d_sumPositions.begin() + activeCellCount);
    thrust::host_vector<float3> h_sumNorm(d_sumNormals.begin(), d_sumNormals.begin() + activeCellCount);
    thrust::host_vector<CovarianceData> h_sumCov(d_sumCov.begin(), d_sumCov.begin() + activeCellCount);
    thrust::host_vector<int> h_counts(d_counts.begin(), d_counts.begin() + activeCellCount);

    // -------------------------------------------------------
    // 4. Result Generation (CPU)
    // -------------------------------------------------------
    results.reserve(activeCellCount);

    for (int i = 0; i < activeCellCount; ++i)
    {
        int count = h_counts[i];
        if (count < 3) continue; // PCA는 최소 3점 이상 필요

        CuCellStats stats;
        stats.pointCount = count;

        // 1) Centroid & Avg Normal Calculation
        float3 sp = h_sumPos[i];
        float3 sn = h_sumNorm[i];
        float invN = 1.0f / (float)count;

        float3 c = make_float3(sp.x * invN, sp.y * invN, sp.z * invN);
        stats.pointCentroid = c;

        // Avg Normal (정규화)
        float lenN = sqrtf(sn.x * sn.x + sn.y * sn.y + sn.z * sn.z);
        float3 avgN = (lenN > 1e-6f) ? make_float3(sn.x / lenN, sn.y / lenN, sn.z / lenN) : make_float3(0, 1, 0);
        stats.avgNormal = avgN;

        // 2) Covariance Matrix Calculation (E[XX] - E[X]*E[X])
        CovarianceData sumC = h_sumCov[i];
        float xx = sumC.xx * invN - c.x * c.x;
        float yy = sumC.yy * invN - c.y * c.y;
        float zz = sumC.zz * invN - c.z * c.z;
        float xy = sumC.xy * invN - c.x * c.y;
        float yz = sumC.yz * invN - c.y * c.z;
        float zx = sumC.zx * invN - c.z * c.x;

        // 3) Solve Eigen System for Normal
        solveEigen3x3_CPU(xx, xy, zx, yy, yz, zz, stats.pcaNormal);

        // 4) Orient PCA Normal (방향 맞추기)
        // PCA Normal은 부호가 없으므로 Avg Normal과 내적하여 같은 방향으로 뒤집음
        float dot = stats.pcaNormal.x * avgN.x + stats.pcaNormal.y * avgN.y + stats.pcaNormal.z * avgN.z;
        if (dot < 0.0f) {
            stats.pcaNormal.x = -stats.pcaNormal.x;
            stats.pcaNormal.y = -stats.pcaNormal.y;
            stats.pcaNormal.z = -stats.pcaNormal.z;
        }

        // 5) Grid AABB Calculation
        int hash = h_hashes[i];
        int area = gridSize.x * gridSize.y;
        int z = hash / area;
        int rem = hash % area;
        int y = rem / gridSize.x;
        int x = rem % gridSize.x;

        stats.cellMin.x = worldOrigin.x + (x * cellSize);
        stats.cellMin.y = worldOrigin.y + (y * cellSize);
        stats.cellMin.z = worldOrigin.z + (z * cellSize);

        stats.cellMax.x = stats.cellMin.x + cellSize;
        stats.cellMax.y = stats.cellMin.y + cellSize;
        stats.cellMax.z = stats.cellMin.z + cellSize;

        results.push_back(stats);
    }

    return results;
}
