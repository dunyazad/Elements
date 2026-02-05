#include <Copper/CuSparseCells.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuTransferFunction.h>
#include <Copper/CUDAMath.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

#include <algorithm>
#ifdef __CUDACC__
using std::max;
using std::min;
#endif

namespace
{
    constexpr int MAX_K = 32;

    __device__ __forceinline__ void solveEigen3x3_Fast(
        float xx, float xy, float xz, float yy, float yz, float zz, float3& outNormal)
    {
        float m = (xx + yy + zz) * 0.333333f;
        xx -= m;
        yy -= m;
        zz -= m;

        float3 r0 = make_float3(xx, xy, xz);
        float3 r1 = make_float3(xy, yy, yz);

        float3 c0 = Cross(r0, r1);
        float lenSq = LengthSq(c0);

        if (lenSq < 1e-10f)
        {
            float3 r2 = make_float3(xz, yz, zz);
            c0 = Cross(r0, r2);
            lenSq = LengthSq(c0);
        }

        if (lenSq > 1e-12f)
        {
            outNormal = Normalize(c0);
        }
        else
        {
            outNormal = make_float3(0.0f, 0.0f, 1.0f);
        }
    }

    // [SOR Kernel]
    __global__ void computeMeanDistanceKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outMeanDists, int numParticles, int k_neighbors, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }

        float3 myPos = FETCH(positions, index);
        float invCellSize = 1.0f / cellSize;

        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        float dists[MAX_K];
        for (int i = 0; i < k_neighbors; ++i)
        {
            dists[i] = 1.0e20f;
        }
        float currentMaxDist = 1.0e20f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z)
        {
            for (int y = -1; y <= 1; ++y)
            {
                for (int x = -1; x <= 1; ++x)
                {
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1)
                        {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j)
                            {
                                if (j == index)
                                {
                                    continue;
                                }
                                float3 otherPos = FETCH(positions, j);
                                float d2 = DistanceSq(myPos, otherPos);

                                if (d2 < currentMaxDist)
                                {
                                    dists[currentMaxIdx] = d2;
                                    float newMax = -1.0f;
                                    int newIdx = -1;
                                    for (int k = 0; k < k_neighbors; ++k)
                                    {
                                        if (dists[k] > newMax)
                                        {
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
        for (int i = 0; i < k_neighbors; ++i)
        {
            if (dists[i] < 1.0e19f)
            {
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
        if (index >= numParticles)
        {
            return;
        }

        float3 myPos = FETCH(positions, index);
        float invCellSize = 1.0f / cellSize;

        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        int neighbors[MAX_K];
        float dists[MAX_K];
        for (int i = 0; i < k_neighbors; ++i)
        {
            dists[i] = 1.0e20f;
        }
        float currentMaxDist = 1.0e20f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z)
        {
            for (int y = -1; y <= 1; ++y)
            {
                for (int x = -1; x <= 1; ++x)
                {
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1)
                        {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j)
                            {
                                if (j == index)
                                {
                                    continue;
                                }
                                float3 p = FETCH(positions, j);
                                float d2 = DistanceSq(myPos, p);

                                if (d2 < currentMaxDist)
                                {
                                    dists[currentMaxIdx] = d2;
                                    neighbors[currentMaxIdx] = j;

                                    float newMax = -1.0f;
                                    int newIdx = -1;
                                    for (int k = 0; k < k_neighbors; ++k)
                                    {
                                        if (dists[k] > newMax)
                                        {
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

        for (int i = 0; i < k_neighbors; ++i)
        {
            if (dists[i] >= 1.0e19f)
            {
                continue;
            }
            float3 p = FETCH(positions, neighbors[i]);

            float dx = p.x - myPos.x;
            float dy = p.y - myPos.y;
            float dz = p.z - myPos.z;

            sumDiff.x += dx;
            sumDiff.y += dy;
            sumDiff.z += dz;
            s_xx += dx * dx;
            s_xy += dx * dy;
            s_xz += dx * dz;
            s_yy += dy * dy;
            s_yz += dy * dz;
            s_zz += dz * dz;
            count++;
        }

        if (count < 3)
        {
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

        float dist = fabsf(Dot(normal, relCentroid));
        outPlaneDists[index] = dist;
    }

    __global__ void markOutliersKernel(bool* isAlive, const float* __restrict__ vals, float threshold, int numParticles)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }
        if (vals[index] > threshold)
        {
            isAlive[index] = false;
        }
    }

    struct SquareOp
    {
        __host__ __device__ float operator()(float x) const
        {
            return x * x;
        }
    };
    struct Float3ToPair
    {
        __host__ __device__ thrust::pair<float3, float3> operator()(const float3& x) const
        {
            return thrust::make_pair(x, x);
        }
    };
    struct Float3MinMax
    {
        __host__ __device__ thrust::pair<float3, float3> operator()(const thrust::pair<float3, float3>& a, const thrust::pair<float3, float3>& b) const
        {
            float3 minVal = { fminf(a.first.x, b.first.x), fminf(a.first.y, b.first.y), fminf(a.first.z, b.first.z) };
            float3 maxVal = { fmaxf(a.second.x, b.second.x), fmaxf(a.second.y, b.second.y), fmaxf(a.second.z, b.second.z) };
            return thrust::make_pair(minVal, maxVal);
        }
    };

    // 1. NND Kernel (Nearest Neighbor Distance) - Best K
    __global__ void computeNNDKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outValues, int numParticles, int k, float cellSize, int3 gridSize, float3 worldOrigin
    )
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }

        float3 myPos = FETCH(positions, index);
        float invCellSize = 1.0f / cellSize;
        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

        float dists[MAX_K];
        for (int i = 0; i < k; ++i)
        {
            dists[i] = 1.0e20f;
        }
        float currentMaxDist = 1.0e20f;
        int currentMaxIdx = 0;

#pragma unroll
        for (int z = -1; z <= 1; ++z)
        {
            for (int y = -1; y <= 1; ++y)
            {
                for (int x = -1; x <= 1; ++x)
                {
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1)
                        {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j)
                            {
                                if (j == index)
                                {
                                    continue;
                                }
                                float d2 = DistanceSq(myPos, FETCH(positions, j));
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
        int count = 0;
        for (int i = 0; i < k; ++i)
        {
            if (dists[i] < 1.0e19f)
            {
                sumDist += sqrtf(dists[i]);
                count++;
            }
        }
        outValues[index] = (count > 0) ? (sumDist / count) : 0.0f;
    }

    // 2. Density Kernel (Radius Search)
    __global__ void computeDensityKernel(
        const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
        float* __restrict__ outValues, int numParticles, float radius, int mode, float cellSize, int3 gridSize, float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }

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
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;
                    if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = FETCH(cellStart, hash);
                        if (start != -1)
                        {
                            int end = FETCH(cellEnd, hash);
                            for (int j = start; j < end; ++j)
                            {
                                float d2 = DistanceSq(myPos, FETCH(positions, j));
                                if (d2 <= rSq)
                                {
                                    if (mode == 0)
                                    {
                                        density += 1.0f; // LDE
                                    }
                                    else
                                    {
                                        density += expf(-d2 * invHSq); // KDE
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

    __global__ void applyTransferFunctionKernel(uchar3* colors, const float* values, int numParticles, CuTransferFunction tf)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }

        uchar4 c4 = tf.Map(values[index]);
        colors[index] = make_uchar3(c4.x, c4.y, c4.z);
    }

    __device__ uchar3 hashToColor(int hash)
    {
        unsigned char r = (hash * 1664525 + 1013904223) & 0xFF;
        unsigned char g = (hash * 25214903917 + 11) & 0xFF;
        unsigned char b = (hash * 8253729 + 2396403) & 0xFF;
        return make_uchar3(r, g, b);
    }

    __global__ void colorizeByHashKernel(
        uchar3* colors, const int* hashCodes, int numParticles)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }

        int hash = hashCodes[index];
        colors[index] = hashToColor(hash);
    }

    struct CovarianceData
    {
        float xx, yy, zz, xy, yz, zx;

        __host__ __device__ CovarianceData()
            : xx(0), yy(0), zz(0), xy(0), yz(0), zx(0)
        {
        }

        __host__ __device__ CovarianceData(float3 p)
        {
            xx = p.x * p.x;
            yy = p.y * p.y;
            zz = p.z * p.z;
            xy = p.x * p.y;
            yz = p.y * p.z;
            zx = p.z * p.x;
        }
    };

    struct CovarianceSumOp
    {
        __host__ __device__ CovarianceData operator()(const CovarianceData& a, const CovarianceData& b) const
        {
            CovarianceData r;
            r.xx = a.xx + b.xx;
            r.yy = a.yy + b.yy;
            r.zz = a.zz + b.zz;
            r.xy = a.xy + b.xy;
            r.yz = a.yz + b.yz;
            r.zx = a.zx + b.zx;
            return r;
        }
    };

    struct PositionToCovariance
    {
        __host__ __device__ CovarianceData operator()(const float3& p) const
        {
            return CovarianceData(p);
        }
    };

    struct Float3SumOp
    {
        __host__ __device__ float3 operator()(const float3& a, const float3& b) const
        {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
    };

    void solveEigen3x3_CPU(float xx, float xy, float xz, float yy, float yz, float zz, float3& outNormal)
    {
        float m = (xx + yy + zz) * 0.333333f;
        xx -= m;
        yy -= m;
        zz -= m;

        float3 r0 = make_float3(xx, xy, xz);
        float3 r1 = make_float3(xy, yy, yz);
        float3 r2 = make_float3(xz, yz, zz);

        float3 c0 = make_float3(
            r0.y * r1.z - r0.z * r1.y,
            r0.z * r1.x - r0.x * r1.z,
            r0.x * r1.y - r0.y * r1.x
        );
        float lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;

        if (lenSq < 1e-10f)
        {
            c0 = make_float3(
                r0.y * r2.z - r0.z * r2.y,
                r0.z * r2.x - r0.x * r2.z,
                r0.x * r2.y - r0.y * r2.x
            );
            lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;
        }

        if (lenSq < 1e-10f)
        {
            c0 = make_float3(
                r1.y * r2.z - r1.z * r2.y,
                r1.z * r2.x - r1.x * r2.z,
                r1.x * r2.y - r1.y * r2.x
            );
            lenSq = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z;
        }

        if (lenSq > 1e-12f)
        {
            float invLen = 1.0f / sqrtf(lenSq);
            outNormal.x = c0.x * invLen;
            outNormal.y = c0.y * invLen;
            outNormal.z = c0.z * invLen;
        }
        else
        {
            outNormal = make_float3(0.0f, 1.0f, 0.0f);
        }
    }
}

__global__ void computeHashKernel(
    const float3* positions, int* particleHash, int numParticles, float cellSize, int3 gridSize, float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
    {
        return;
    }
    float3 p = positions[index];
    int gridX = (int)((p.x - worldOrigin.x) / cellSize);
    int gridY = (int)((p.y - worldOrigin.y) / cellSize);
    int gridZ = (int)((p.z - worldOrigin.z) / cellSize);
    if (gridX < 0)
    {
        gridX = 0;
    }
    else if (gridX >= gridSize.x)
    {
        gridX = gridSize.x - 1;
    }
    if (gridY < 0)
    {
        gridY = 0;
    }
    else if (gridY >= gridSize.y)
    {
        gridY = gridSize.y - 1;
    }
    if (gridZ < 0)
    {
        gridZ = 0;
    }
    else if (gridZ >= gridSize.z)
    {
        gridZ = gridSize.z - 1;
    }
    particleHash[index] = (gridZ * gridSize.y + gridY) * gridSize.x + gridX;
}

__global__ void findCellStartEndKernel(const int* particleHash, int* cellStart, int* cellEnd, int numParticles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles)
    {
        return;
    }
    int currentHash = particleHash[index];
    if (index == 0)
    {
        cellStart[currentHash] = index;
    }
    else
    {
        int prevHash = particleHash[index - 1];
        if (currentHash != prevHash)
        {
            cellEnd[prevHash] = index;
            cellStart[currentHash] = index;
        }
    }
    if (index == numParticles - 1)
    {
        cellEnd[currentHash] = numParticles;
    }
}

CuSparseCells::CuSparseCells()
{
}

float CuSparseCells::computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier)
{
    if (points.empty())
    {
        return 1.0f;
    }
    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(points.begin(), points.end(), Float3ToPair(), init, Float3MinMax());
    float dx = bbox.second.x - bbox.first.x;
    float dy = bbox.second.y - bbox.first.y;
    float dz = bbox.second.z - bbox.first.z;
    float maxDim = fmaxf(dx, fmaxf(dy, dz));
    if (maxDim <= 0.0f)
    {
        maxDim = 1.0f;
    }

    float cellSize = (maxDim / powf((float)points.size(), 1.0f / 3.0f)) * multiplier;

    printf("Auto cell size computed: %f\n", cellSize);

    return cellSize;
}

void CuSparseCells::Build(CuPointCloud* cloud)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return;
    }

    cellSize = computeAutoCellSize(cloud->points, 1.5f);

    printf("Computed cell size: %f\n", cellSize);

    worldOrigin = cloud->aabbMin;
    float3 maxP = cloud->aabbMax;
    float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };

    gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };

    numberOfCells = gridSize.x * gridSize.y * gridSize.z;

    size_t numPoints = cloud->size();

    if (hashCodes.size() != numPoints)
    {
        hashCodes.resize(numPoints);
    }

    if (cellStartIndices.size() != numberOfCells)
    {
        cellStartIndices.resize(numberOfCells);
    }

    if (cellEndIndices.size() != numberOfCells)
    {
        cellEndIndices.resize(numberOfCells);
    }

    thrust::fill(cellStartIndices.begin(), cellStartIndices.end(), -1);
    thrust::fill(cellEndIndices.begin(), cellEndIndices.end(), -1);

    int blockSize = 256;
    int numBlocks = (int)((numPoints + blockSize - 1) / blockSize);

    computeHashKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(cloud->points.data()), thrust::raw_pointer_cast(hashCodes.data()), (int)numPoints, cellSize, gridSize, worldOrigin);

    cudaDeviceSynchronize();

    thrust::sort_by_key(hashCodes.begin(), hashCodes.end(), thrust::make_zip_iterator(thrust::make_tuple(cloud->points.begin(), cloud->normals.begin(), cloud->colors.begin(), cloud->isAlive.begin())));

    findCellStartEndKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(hashCodes.data()), thrust::raw_pointer_cast(cellStartIndices.data()), thrust::raw_pointer_cast(cellEndIndices.data()), (int)numPoints);

    cudaDeviceSynchronize();
}

void CuSparseCells::Build(CuPointCloud* cloud, float cellSize)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return;
    }

    CUDA_TS(CuSParseCellsBuild);

    this->cellSize = cellSize;

    worldOrigin = cloud->aabbMin;
    float3 maxP = cloud->aabbMax;
    float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };

    gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };

    numberOfCells = gridSize.x * gridSize.y * gridSize.z;

    size_t numPoints = cloud->size();

    if (hashCodes.size() != numPoints)
    {
        hashCodes.resize(numPoints);
    }

    if (cellStartIndices.size() != numberOfCells)
    {
        cellStartIndices.resize(numberOfCells);
    }

    if (cellEndIndices.size() != numberOfCells)
    {
        cellEndIndices.resize(numberOfCells);
    }

    CUDA_TS(CuSParseCellsBuild_Fill);
    thrust::fill(cellStartIndices.begin(), cellStartIndices.end(), -1);
    thrust::fill(cellEndIndices.begin(), cellEndIndices.end(), -1);
    CUDA_TE(CuSParseCellsBuild_Fill);

    CUDA_TS(CuSParseCellsBuild_Hash);
    int blockSize = 256;
    int numBlocks = (int)((numPoints + blockSize - 1) / blockSize);

    computeHashKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(cloud->points.data()), thrust::raw_pointer_cast(hashCodes.data()), (int)numPoints, cellSize, gridSize, worldOrigin);

    cudaDeviceSynchronize();
    CUDA_TE(CuSParseCellsBuild_Hash);

    CUDA_TS(CuSParseCellsBuild_Sort);
    thrust::sort_by_key(hashCodes.begin(), hashCodes.end(), thrust::make_zip_iterator(thrust::make_tuple(cloud->points.begin(), cloud->normals.begin(), cloud->colors.begin(), cloud->isAlive.begin())));
    CUDA_TE(CuSParseCellsBuild_Sort);

    CUDA_TS(CuSParseCellsBuild_Find);
    findCellStartEndKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(hashCodes.data()), thrust::raw_pointer_cast(cellStartIndices.data()), thrust::raw_pointer_cast(cellEndIndices.data()), (int)numPoints);

    cudaDeviceSynchronize();
    CUDA_TE(CuSParseCellsBuild_Find);

    CUDA_TE(CuSParseCellsBuild);
}

__global__ void initUnionFindKernel(unsigned int* labels, int numPoints)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numPoints)
    {
        labels[index] = (unsigned int)index;
    }
}

__global__ void unionClustersKernel(
    const float3* __restrict__ positions,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    unsigned int* __restrict__ labels,
    int numPoints,
    float clusterDistSq,
    float cellSize,
    int3 gridSize,
    float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    float3 myPos = FETCH(positions, index);
    float invCellSize = 1.0f / cellSize;

    int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int nx = gridX + x;
                int ny = gridY + y;
                int nz = gridZ + z;

                if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                {
                    int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                    int start = FETCH(cellStart, hash);
                    if (start != -1)
                    {
                        int end = FETCH(cellEnd, hash);
                        for (int j = start; j < end; ++j)
                        {
                            if (j <= index) continue;

                            float3 otherPos = FETCH(positions, j);
                            float d2 = DistanceSq(myPos, otherPos);

                            if (d2 <= clusterDistSq)
                            {
                                unsigned int rootA = index;
                                unsigned int rootB = j;

                                while (rootA != labels[rootA]) rootA = labels[rootA];
                                while (rootB != labels[rootB]) rootB = labels[rootB];

                                if (rootA < rootB) atomicMin(&labels[rootB], rootA);
                                else if (rootB < rootA) atomicMin(&labels[rootA], rootB);
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void flattenLabelsKernel(unsigned int* labels, bool* changed, int numPoints)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    unsigned int parent = labels[index];
    unsigned int root = labels[parent];

    if (root != parent)
    {
        labels[index] = root;
        *changed = true;
    }
}

__global__ void flattenLabelsFinalKernel(unsigned int* labels, int numPoints)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints)
    {
        return;
    }

    unsigned int curr = labels[index];
    unsigned int next = labels[curr];

    while (curr != next)
    {
        curr = next;
        next = labels[curr];
    }
    labels[index] = curr;
}

void CuSparseCells::ApplyClustering(CuPointCloud* cloud, unsigned int* d_outLabels, float clusterDistance)
{
    if (cloud == nullptr || cloud->size() == 0 || d_outLabels == nullptr)
    {
        return;
    }

    CUDA_TS(Clustering_UnionFind);

    int numPoints = (int)cloud->size();
    float clusterDistSq = clusterDistance * clusterDistance;
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    initUnionFindKernel << <numBlocks, blockSize >> > (d_outLabels, numPoints);

    unionClustersKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        d_outLabels,
        numPoints,
        clusterDistSq,
        cellSize,
        gridSize,
        worldOrigin
        );

    flattenLabelsFinalKernel << <numBlocks, blockSize >> > (d_outLabels, numPoints);

    cudaDeviceSynchronize();
    CUDA_TE(Clustering_UnionFind);
}

thrust::device_vector<float> CuSparseCells::ApplySOR(CuPointCloud* cloud, int k, float stdDevMult)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    if (k > MAX_K)
    {
        k = MAX_K;
    }
    int numPoints = (int)cloud->size();
    int blockSize = 128;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> meanDists(numPoints);

    computeMeanDistanceKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(meanDists.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    float sum = thrust::reduce(meanDists.begin(), meanDists.end(), 0.0f, thrust::plus<float>());
    float mean = sum / numPoints;
    float sumSq = thrust::transform_reduce(meanDists.begin(), meanDists.end(), SquareOp(), 0.0f, thrust::plus<float>());
    float variance = (sumSq / numPoints) - (mean * mean);
    float stdDev = sqrtf(variance > 0 ? variance : 0.0f);
    float threshold = mean + stdDevMult * stdDev;

    markOutliersKernel << <numBlocks, 256 >> > (
        thrust::raw_pointer_cast(cloud->isAlive.data()),
        thrust::raw_pointer_cast(meanDists.data()),
        threshold, numPoints
        );
    cudaDeviceSynchronize();

    return meanDists;
}

thrust::device_vector<float> CuSparseCells::ApplyPFOR(CuPointCloud* cloud, int k, float distanceThreshold)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    if (k > MAX_K)
    {
        k = MAX_K;
    }
    int numPoints = (int)cloud->size();
    int blockSize = 128;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> planeDists(numPoints);

    computePlaneDistanceKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(planeDists.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    markOutliersKernel << <numBlocks, 256 >> > (
        thrust::raw_pointer_cast(cloud->isAlive.data()),
        thrust::raw_pointer_cast(planeDists.data()),
        distanceThreshold, numPoints
        );
    cudaDeviceSynchronize();

    return planeDists;
}

thrust::device_vector<float> CuSparseCells::ApplyNND(CuPointCloud* cloud, int k)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> values(numPoints);

    computeNNDKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(values.data()),
        numPoints, k, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(values.begin(), values.end());

    CuTransferFunction tf(*minmax.first, *minmax.second);
    tf.invert = false;

    applyTransferFunctionKernel << <numBlocks, blockSize >> > (
        (uchar3*)thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(values.data()),
        numPoints, tf
        );
    cudaDeviceSynchronize();

    return values;
}

thrust::device_vector<float> CuSparseCells::ApplyLDE(CuPointCloud* cloud, float radius)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return thrust::device_vector<float>();
    }

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    thrust::device_vector<float> values(numPoints);

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(values.data()),
        numPoints, radius, 0, cellSize, gridSize, worldOrigin
        );
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(values.begin(), values.end());

    CuTransferFunction tf(*minmax.first, *minmax.second);
    tf.SetGray();

    applyTransferFunctionKernel << <numBlocks, blockSize >> > (
        (uchar3*)thrust::raw_pointer_cast(cloud->colors.data()),
        thrust::raw_pointer_cast(values.data()),
        numPoints, tf
        );
    cudaDeviceSynchronize();

    return values;
}

std::vector<std::pair<float3, float3>> CuSparseCells::GetActiveCellBounds()
{
    std::vector<std::pair<float3, float3>> activeBoxes;

    thrust::host_vector<int> cellStartHost = cellStartIndices;

    for (int hash = 0; hash < numberOfCells; ++hash)
    {
        if (cellStartHost[hash] != -1)
        {
            int z = hash / (gridSize.x * gridSize.y);
            int rem = hash % (gridSize.x * gridSize.y);
            int y = rem / gridSize.x;
            int x = rem % gridSize.x;

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

void CuSparseCells::ColorizePointsByCell(CuPointCloud* cloud)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return;
    }
    if (hashCodes.empty())
    {
        return;
    }

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

std::vector<CuCellStats> CuSparseCells::GetActiveCellStats(CuPointCloud* cloud)
{
    std::vector<CuCellStats> results;

    if (cloud == nullptr || cloud->size() == 0)
    {
        return results;
    }
    if (hashCodes.empty())
    {
        return results;
    }

    int numPoints = (int)cloud->size();

    thrust::device_vector<CovarianceData> covData(numPoints);
    thrust::transform(cloud->points.begin(), cloud->points.end(), covData.begin(), PositionToCovariance());

    thrust::device_vector<int> uniqueHashes(numPoints);
    thrust::device_vector<float3> sumPositions(numPoints);
    thrust::device_vector<float3> sumNormals(numPoints);
    thrust::device_vector<CovarianceData> sumCov(numPoints);
    thrust::device_vector<int> counts(numPoints);

    auto endPairPos = thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        cloud->points.begin(),
        uniqueHashes.begin(),
        sumPositions.begin(),
        thrust::equal_to<int>(),
        Float3SumOp()
    );

    int activeCellCount = (int)(endPairPos.first - uniqueHashes.begin());

    thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        cloud->normals.begin(),
        thrust::make_discard_iterator(),
        sumNormals.begin(),
        thrust::equal_to<int>(),
        Float3SumOp()
    );

    thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        covData.begin(),
        thrust::make_discard_iterator(),
        sumCov.begin(),
        thrust::equal_to<int>(),
        CovarianceSumOp()
    );

    thrust::reduce_by_key(
        hashCodes.begin(), hashCodes.end(),
        thrust::make_constant_iterator(1),
        thrust::make_discard_iterator(),
        counts.begin()
    );

    thrust::host_vector<int> hashes(uniqueHashes.begin(), uniqueHashes.begin() + activeCellCount);
    thrust::host_vector<float3> sumPos(sumPositions.begin(), sumPositions.begin() + activeCellCount);
    thrust::host_vector<float3> sumNorm(sumNormals.begin(), sumNormals.begin() + activeCellCount);
    thrust::host_vector<CovarianceData> sumCovHost(sumCov.begin(), sumCov.begin() + activeCellCount);
    thrust::host_vector<int> countsHost(counts.begin(), counts.begin() + activeCellCount);

    results.reserve(activeCellCount);

    for (int i = 0; i < activeCellCount; ++i)
    {
        int count = countsHost[i];
        if (count < 3)
        {
            continue;
        }

        CuCellStats stats;
        stats.pointCount = count;

        float3 sp = sumPos[i];
        float3 sn = sumNorm[i];
        float invN = 1.0f / (float)count;

        float3 c = make_float3(sp.x * invN, sp.y * invN, sp.z * invN);
        stats.pointCentroid = c;

        float lenN = sqrtf(sn.x * sn.x + sn.y * sn.y + sn.z * sn.z);
        float3 avgN = (lenN > 1e-6f) ? make_float3(sn.x / lenN, sn.y / lenN, sn.z / lenN) : make_float3(0, 1, 0);
        stats.avgNormal = avgN;

        CovarianceData sumC = sumCovHost[i];
        float xx = sumC.xx * invN - c.x * c.x;
        float yy = sumC.yy * invN - c.y * c.y;
        float zz = sumC.zz * invN - c.z * c.z;
        float xy = sumC.xy * invN - c.x * c.y;
        float yz = sumC.yz * invN - c.y * c.z;
        float zx = sumC.zx * invN - c.z * c.x;

        solveEigen3x3_CPU(xx, xy, zx, yy, yz, zz, stats.pcaNormal);

        float dot = stats.pcaNormal.x * avgN.x + stats.pcaNormal.y * avgN.y + stats.pcaNormal.z * avgN.z;
        if (dot < 0.0f)
        {
            stats.pcaNormal.x = -stats.pcaNormal.x;
            stats.pcaNormal.y = -stats.pcaNormal.y;
            stats.pcaNormal.z = -stats.pcaNormal.z;
        }

        int hash = hashes[i];
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

__global__ void smoothProtrusionEdgePreservingKernel(
    float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    int numPoints,
    float searchRadius,
    float smoothFactor,
    float edgeThreshold, // 0.0 ~ 1.0 (낮을수록 Edge 보호 강도가 높음, 예: 0.8)
    float cellSize,
    int3 gridSize,
    float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    float3 myPos = FETCH(positions, index);
    float3 myNormal = FETCH(normals, index);
    float rSq = searchRadius * searchRadius;
    float invCellSize = 1.0f / cellSize;

    int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    float3 weightedSum = make_float3(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                if (nx < 0 || nx >= gridSize.x || ny < 0 || ny >= gridSize.y || nz < 0 || nz >= gridSize.z) continue;

                int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                int start = FETCH(cellStart, hash);
                if (start == -1) continue;

                int end = FETCH(cellEnd, hash);
                for (int j = start; j < end; ++j)
                {
                    float3 otherPos = FETCH(positions, j);
                    float d2 = DistanceSq(myPos, otherPos);

                    if (d2 < rSq)
                    {
                        float3 otherNormal = FETCH(normals, j);
                        // 두 점 사이의 법선 벡터 유사도 (내적)
                        float dot = fabsf(Dot(myNormal, otherNormal));

                        // 1. Bilateral 가중치: 거리가 가깝고 + 법선 방향이 비슷할수록 높은 가중치
                        // 2. 만약 법선 방향이 너무 다르면(Edge 부위), 가중치를 급격히 낮춤
                        float weight = expf(-d2 / (0.5f * rSq)) * powf(dot, 8.0f);

                        weightedSum.x += otherPos.x * weight;
                        weightedSum.y += otherPos.y * weight;
                        weightedSum.z += otherPos.z * weight;
                        totalWeight += weight;
                    }
                }
            }
        }
    }

    if (totalWeight > 1e-6f)
    {
        float3 targetPos = make_float3(weightedSum.x / totalWeight, weightedSum.y / totalWeight, weightedSum.z / totalWeight);

        // 투영(Projection) 방식: 
        // 단순히 이동하는 게 아니라, 현재 노멀 방향 성분의 오차만 줄임으로써 평면상의 디테일 유지
        float3 diff = make_float3(targetPos.x - myPos.x, targetPos.y - myPos.y, targetPos.z - myPos.z);
        float distToSurface = Dot(diff, myNormal);

        // Edge 부위(dot 유사도가 낮은 곳)라면 이동을 억제하는 임계치 로직
        // 여기서는 간단하게 smoothFactor에 법선 정렬도를 곱해 적용
        positions[index].x += myNormal.x * distToSurface * smoothFactor;
        positions[index].y += myNormal.y * distToSurface * smoothFactor;
        positions[index].z += myNormal.z * distToSurface * smoothFactor;
    }
}

/**
 * @brief Edge를 보존하면서 돌출된 노이즈를 평탄화하는 함수
 * @param cloud 대상 포인트 클라우드 (normals 데이터가 포함되어 있어야 함)
 * @param radius 탐색 반경 (노이즈 덩어리 크기보다 커야 함)
 * @param factor 보정 강도 (0.0 ~ 1.0, 보통 0.5 권장)
 * @param edgeThreshold 법선 유사도 임계치 (높을수록 모서리 보호 강함, 예: 0.9)
 * @param iterations 반복 횟수
 */
void CuSparseCells::ApplyEdgePreservingSmoothing(
    CuPointCloud* cloud,
    float radius,
    float factor,
    float edgeThreshold,
    int iterations)
{
    if (cloud == nullptr || cloud->size() == 0)
    {
        return;
    }

    // 법선 데이터가 비어있는지 체크 (Edge 보존의 핵심 데이터)
    if (cloud->normals.size() != cloud->size())
    {
        printf("Error: Normals are required for edge-preserving smoothing.\n");
        return;
    }

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    // CUDA 타임스탬프 기록 (기존 코드 스타일 유지)
    CUDA_TS(ApplyEdgeSmoothing);

    for (int i = 0; i < iterations; ++i)
    {
        smoothProtrusionEdgePreservingKernel << <numBlocks, blockSize >> > (
            thrust::raw_pointer_cast(cloud->points.data()),
            thrust::raw_pointer_cast(cloud->normals.data()),
            thrust::raw_pointer_cast(cellStartIndices.data()),
            thrust::raw_pointer_cast(cellEndIndices.data()),
            numPoints,
            radius,
            factor,
            edgeThreshold,
            cellSize,
            gridSize,
            worldOrigin
            );

        // 반복 사이 동기화 (다음 루프에서 수정된 좌표 참조를 위해)
        cudaDeviceSynchronize();
    }

    CUDA_TE(ApplyEdgeSmoothing);
}

/**
 * @brief 에너지 최소화를 통해 Edge를 보존하며 돌출 노이즈를 제거하는 커널
 */
__global__ void applyEnergyBasedSmoothingKernel(
    float3* __restrict__ positions,
    const float3* __restrict__ originalPositions,
    const float3* __restrict__ normals,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    int numPoints,
    float searchRadius,
    float dataWeight,    // 원래 위치를 유지하려는 힘 (보통 0.1~0.3)
    float smoothWeight,  // 표면을 평탄화하려는 힘 (보통 0.7~0.9)
    float cellSize,
    int3 gridSize,
    float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    float3 p_curr = positions[index];
    float3 p_orig = originalPositions[index];
    float3 n_curr = normals[index];

    float rSq = searchRadius * searchRadius;
    float invCellSize = 1.0f / cellSize;

    int gridX = max(0, min((int)((p_curr.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((p_curr.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((p_curr.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    float3 smoothTarget = make_float3(0, 0, 0);
    float totalWeight = 0.0f;

    // 1. Smooth Term 계산: 주변 이웃들이 정의하는 가상의 표면 찾기
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int hash = ((gridZ + z) * gridSize.y + (gridY + y)) * gridSize.x + (gridX + x);
                if (gridX + x < 0 || gridX + x >= gridSize.x || gridY + y < 0 || gridY + y >= gridSize.y || gridZ + z < 0 || gridZ + z >= gridSize.z) continue;

                int start = FETCH(cellStart, hash);
                if (start == -1) continue;
                int end = FETCH(cellEnd, hash);

                for (int j = start; j < end; ++j)
                {
                    float3 p_neigh = FETCH(originalPositions, j);
                    float d2 = DistanceSq(p_curr, p_neigh);

                    if (d2 < rSq && j != index)
                    {
                        float3 n_neigh = FETCH(normals, j);
                        float dot = fabsf(Dot(n_curr, n_neigh));

                        // Edge 보존 가중치: 법선이 비슷할수록 평탄화 에너지를 강하게 적용
                        float w = expf(-d2 / (0.5f * rSq)) * powf(dot, 12.0f);

                        // 이웃 점을 현재 평면에 투영시킨 위치의 기여도 계산
                        smoothTarget.x += p_neigh.x * w;
                        smoothTarget.y += p_neigh.y * w;
                        smoothTarget.z += p_neigh.z * w;
                        totalWeight += w;
                    }
                }
            }
        }
    }

    // 2. 에너지 최소화 해(Solution) 업데이트
    if (totalWeight > 1e-6f)
    {
        smoothTarget.x /= totalWeight;
        smoothTarget.y /= totalWeight;
        smoothTarget.z /= totalWeight;

        // 최종 위치 = (DataWeight * 원래위치 + SmoothWeight * 평탄화타겟) / (DataWeight + SmoothWeight)
        float invSum = 1.0f / (dataWeight + smoothWeight);
        positions[index].x = (dataWeight * p_orig.x + smoothWeight * smoothTarget.x) * invSum;
        positions[index].y = (dataWeight * p_orig.y + smoothWeight * smoothTarget.y) * invSum;
        positions[index].z = (dataWeight * p_orig.z + smoothWeight * smoothTarget.z) * invSum;
    }
}

__global__ void applyProjectedEnergySmoothingKernel(
    float3* __restrict__ positions,
    const float3* __restrict__ originalPositions,
    const float3* __restrict__ normals,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    int numPoints,
    float searchRadius,
    float dataWeight,
    float smoothWeight,
    float cellSize,
    int3 gridSize,
    float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    float3 p_curr = positions[index];
    float3 p_orig = originalPositions[index];
    float3 n_orig = normals[index]; // 원래의 정교한 법선 참조

    float rSq = searchRadius * searchRadius;
    float invCellSize = 1.0f / cellSize;

    int gridX = max(0, min((int)((p_curr.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((p_curr.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((p_curr.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    float3 smoothTarget = make_float3(0, 0, 0);
    float totalWeight = 0.0f;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int hash = ((gridZ + z) * gridSize.y + (gridY + y)) * gridSize.x + (gridX + x);
                if (gridX + x < 0 || gridX + x >= gridSize.x || gridY + y < 0 || gridY + y >= gridSize.y || gridZ + z < 0 || gridZ + z >= gridSize.z) continue;

                int start = FETCH(cellStart, hash);
                if (start == -1) continue;
                int end = FETCH(cellEnd, hash);

                for (int j = start; j < end; ++j)
                {
                    float3 p_neigh = FETCH(originalPositions, j);
                    float d2 = DistanceSq(p_curr, p_neigh);

                    if (d2 < rSq && j != index)
                    {
                        float3 n_neigh = FETCH(normals, j);
                        float dot = fabsf(Dot(n_orig, n_neigh));

                        // 지수를 높여(24.0f) 급격한 굴곡에서의 간섭을 원천 차단
                        float w = expf(-d2 / (0.5f * rSq)) * powf(dot, 24.0f);

                        // 주변 점들을 현재 점의 '평면'에 투영한 위치를 타겟으로 삼음
                        float3 diff = make_float3(p_neigh.x - p_curr.x, p_neigh.y - p_curr.y, p_neigh.z - p_curr.z);
                        float dist = Dot(diff, n_orig);
                        float3 projectedPoint = make_float3(p_neigh.x - n_orig.x * dist, p_neigh.y - n_orig.y * dist, p_neigh.z - n_orig.z * dist);

                        smoothTarget.x += projectedPoint.x * w;
                        smoothTarget.y += projectedPoint.y * w;
                        smoothTarget.z += projectedPoint.z * w;
                        totalWeight += w;
                    }
                }
            }
        }
    }

    if (totalWeight > 1e-6f)
    {
        smoothTarget.x /= totalWeight;
        smoothTarget.y /= totalWeight;
        smoothTarget.z /= totalWeight;

        float invSum = 1.0f / (dataWeight + smoothWeight);
        float3 p_new;
        p_new.x = (dataWeight * p_orig.x + smoothWeight * smoothTarget.x) * invSum;
        p_new.y = (dataWeight * p_orig.y + smoothWeight * smoothTarget.y) * invSum;
        p_new.z = (dataWeight * p_orig.z + smoothWeight * smoothTarget.z) * invSum;

        // 핵심: 점이 옆으로 흐르지 않게, 오직 Normal 방향으로만 이동 제한
        float3 finalDiff = make_float3(p_new.x - p_orig.x, p_new.y - p_orig.y, p_new.z - p_orig.z);
        float projectionDist = Dot(finalDiff, n_orig);

        positions[index].x = p_orig.x + n_orig.x * projectionDist;
        positions[index].y = p_orig.y + n_orig.y * projectionDist;
        positions[index].z = p_orig.z + n_orig.z * projectionDist;
    }
}

__global__ void applyStableEnergySmoothingKernel(
    float3* __restrict__ positions,
    const float3* __restrict__ originalPositions,
    const float3* __restrict__ normals,
    const int* __restrict__ cellStart,
    const int* __restrict__ cellEnd,
    int numPoints,
    float searchRadius,
    float dataWeight,
    float smoothWeight,
    float cellSize,
    int3 gridSize,
    float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPoints) return;

    float3 p_orig = originalPositions[index];
    float3 n_orig = normals[index]; // 기준이 되는 원래 법선

    float rSq = searchRadius * searchRadius;
    float invCellSize = 1.0f / cellSize;

    int gridX = max(0, min((int)((p_orig.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((p_orig.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((p_orig.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    float weightedDistSum = 0.0f;
    float totalWeight = 0.0f;

    // 1. 점을 좌표로 이동시키는 대신, "법선 방향의 이동 거리(Scalar)"만 계산
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int hash = ((gridZ + z) * gridSize.y + (gridY + y)) * gridSize.x + (gridX + x);
                if (gridX + x < 0 || gridX + x >= gridSize.x || gridY + y < 0 || gridY + y >= gridSize.y || gridZ + z < 0 || gridZ + z >= gridSize.z) continue;

                int start = FETCH(cellStart, hash);
                if (start == -1) continue;
                int end = FETCH(cellEnd, hash);

                for (int j = start; j < end; ++j) {
                    float3 p_neigh = FETCH(originalPositions, j);
                    float d2 = DistanceSq(p_orig, p_neigh);

                    if (d2 < rSq && j != index) {
                        float3 n_neigh = FETCH(normals, j);
                        float dot = fabsf(Dot(n_orig, n_neigh));

                        // 법선 유사도가 매우 높은 점들만 참조 (치아 엣지 보존)
                        float w = expf(-d2 / (0.5f * rSq)) * powf(dot, 32.0f);

                        // 현재 점의 평면으로부터 이웃 점까지의 수직 거리 계산
                        float3 diff = make_float3(p_neigh.x - p_orig.x, p_neigh.y - p_orig.y, p_neigh.z - p_orig.z);
                        float dist = Dot(diff, n_orig);

                        weightedDistSum += dist * w;
                        totalWeight += w;
                    }
                }
            }
        }
    }

    // 2. 최종 위치 결정
    if (totalWeight > 1e-6f) {
        // 목표로 하는 수직 이동 거리 (평균 평면으로의 거리)
        float targetDist = weightedDistSum / totalWeight;

        // 에너지를 고려한 최종 이동량 제어
        // dataWeight는 0(원래 위치), smoothWeight는 1(평균 평면)로 끌어당김
        float finalMoveDist = targetDist * (smoothWeight / (dataWeight + smoothWeight));

        // 결과 반영: 오직 원래 법선 방향으로만 점을 미세하게 이동
        // 이렇게 하면 절대 점들이 옆으로 뭉칠 수 없음
        positions[index].x = p_orig.x + n_orig.x * finalMoveDist;
        positions[index].y = p_orig.y + n_orig.y * finalMoveDist;
        positions[index].z = p_orig.z + n_orig.z * finalMoveDist;
    }
}

/**
 * @brief 에너지 기반의 포인트 클라우드 평탄화 함수
 * @param cloud 대상 포인트 클라우드
 * @param radius 탐색 반경
 * @param dataWeight 원래 위치 유지 가중치
 * @param smoothWeight 평탄화 가중치
 * @param iterations 반복 횟수
 */
void CuSparseCells::ApplyEnergySmoothing(CuPointCloud* cloud, float radius, float dataWeight, float smoothWeight, int iterations)
{
    if (cloud == nullptr || cloud->size() == 0) return;

    int numPoints = (int)cloud->size();

    // 원래 위치를 보존하기 위한 임시 버퍼
    thrust::device_vector<float3> originalPositions = cloud->points;

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    CUDA_TS(EnergySmoothing);
    for (int i = 0; i < iterations; ++i)
    {
        //applyEnergyBasedSmoothingKernel << <numBlocks, blockSize >> > (
        //applyEnergyBasedSmoothingKernel << <numBlocks, blockSize >> > (
        applyStableEnergySmoothingKernel << <numBlocks, blockSize >> > (
            thrust::raw_pointer_cast(cloud->points.data()),
            thrust::raw_pointer_cast(originalPositions.data()),
            thrust::raw_pointer_cast(cloud->normals.data()),
            thrust::raw_pointer_cast(cellStartIndices.data()),
            thrust::raw_pointer_cast(cellEndIndices.data()),
            numPoints, radius, dataWeight, smoothWeight,
            cellSize, gridSize, worldOrigin
            );
        cudaDeviceSynchronize();
    }
    CUDA_TE(EnergySmoothing);
}