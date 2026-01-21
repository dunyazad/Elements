#include <Copper/CuSparseDataBlock.h>
#include <Copper/CuPointCloud.h>

#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

struct Float3MinMax
{
    __host__ __device__
        thrust::pair<float3, float3> operator()(const thrust::pair<float3, float3>& a, const thrust::pair<float3, float3>& b) const
    {
        float3 minVal = { fminf(a.first.x, b.first.x), fminf(a.first.y, b.first.y), fminf(a.first.z, b.first.z) };
        float3 maxVal = { fmaxf(a.second.x, b.second.x), fmaxf(a.second.y, b.second.y), fmaxf(a.second.z, b.second.z) };
        return thrust::make_pair(minVal, maxVal);
    }
};

struct Float3ToPair
{
    __host__ __device__
        thrust::pair<float3, float3> operator()(const float3& x) const { return thrust::make_pair(x, x); }
};

__global__ void computeHashKernel(
    const float3* positions, int* particleHash, int numParticles,
    float cellSize, int3 gridSize, float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    float3 p = positions[index];
    int gridX = max(0, min((int)((p.x - worldOrigin.x) / cellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((p.y - worldOrigin.y) / cellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((p.z - worldOrigin.z) / cellSize), gridSize.z - 1));

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

CuSparseDataBlock::CuSparseDataBlock()
{
}

void CuSparseDataBlock::Build(CuPointCloud* cloud)
{
    if (cloud == nullptr || cloud->size() == 0) return;

    cellSize = computeAutoCellSize(cloud->points, 4.0f);

    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(
        cloud->points.begin(), cloud->points.end(), Float3ToPair(), init, Float3MinMax()
    );

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
        hashCodes.begin(),
        hashCodes.end(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                cloud->points.begin(),
                cloud->normals.begin(),
                cloud->colors.begin(),
                cloud->isAlive.begin()
            )
        )
    );

    findCellStartEndKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(hashCodes.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        (int)numPoints
        );
    cudaDeviceSynchronize();
}

namespace
{
    // 최대 K값 제한 (레지스터 배열 크기)
    constexpr int MAX_K = 64;

    __device__ float getDistSq(float3 a, float3 b)
    {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    // 각 포인트의 평균 이웃 거리 계산 커널
    __global__ void computeMeanDistanceKernel(
        const float3* sortedPos,
        const int* cellStart,
        const int* cellEnd,
        float* outMeanDists,
        int numParticles,
        int k_neighbors,
        float cellSize,
        int3 gridSize,
        float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        float3 myPos = sortedPos[index];

        // 1. Grid Index 계산
        int gridX = max(0, min((int)((myPos.x - worldOrigin.x) / cellSize), gridSize.x - 1));
        int gridY = max(0, min((int)((myPos.y - worldOrigin.y) / cellSize), gridSize.y - 1));
        int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) / cellSize), gridSize.z - 1));

        // 2. kNN 검색용 로컬 버퍼 (Insertion Sort용)
        // 가장 먼 거리가 마지막 인덱스에 오도록 유지
        float nearestDistSq[MAX_K];
        for (int i = 0; i < k_neighbors; ++i) nearestDistSq[i] = 1e30f;

        int foundCount = 0;

        // 3. 3x3x3 인접 셀 탐색
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    int nx = gridX + x;
                    int ny = gridY + y;
                    int nz = gridZ + z;

                    if (nx >= 0 && nx < gridSize.x &&
                        ny >= 0 && ny < gridSize.y &&
                        nz >= 0 && nz < gridSize.z)
                    {
                        int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                        int start = cellStart[hash];
                        int end = cellEnd[hash];

                        if (start == -1) continue;

                        for (int j = start; j < end; ++j)
                        {
                            if (j == index) continue; // 자기 자신 제외

                            float distSq = getDistSq(myPos, sortedPos[j]);

                            // k번째(가장 먼) 값보다 작으면 삽입
                            if (distSq < nearestDistSq[k_neighbors - 1])
                            {
                                int insertPos = k_neighbors - 1;
                                // 자리 찾기 (뒤에서부터 밀기)
                                while (insertPos > 0 && nearestDistSq[insertPos - 1] > distSq)
                                {
                                    nearestDistSq[insertPos] = nearestDistSq[insertPos - 1];
                                    insertPos--;
                                }
                                nearestDistSq[insertPos] = distSq;
                            }
                        }
                    }
                }
            }
        }

        // 4. 평균 거리 계산
        // 실제 찾은 개수는 정확히 알 수 없으나, 초기값(1e30f)이 아닌 것만 합산
        double sumDist = 0.0;
        int validCount = 0;
        for (int i = 0; i < k_neighbors; ++i)
        {
            if (nearestDistSq[i] < 1e29f)
            {
                sumDist += sqrtf(nearestDistSq[i]);
                validCount++;
            }
        }

        if (validCount > 0)
            outMeanDists[index] = (float)(sumDist / validCount);
        else
            outMeanDists[index] = 0.0f; // 고립된 점
    }

    // Threshold 기반 마킹 커널
    __global__ void markOutliersKernel(
        bool* isAlive,
        const float* meanDists,
        float threshold,
        int numParticles)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles) return;

        // 평균 거리가 임계값보다 크면 삭제
        if (meanDists[index] > threshold)
        {
            isAlive[index] = false;
        }
    }

    // Thrust용 제곱 Functor
    struct SquareOp {
        __host__ __device__ float operator()(float x) const { return x * x; }
    };
}

void CuSparseDataBlock::ApplySOR(CuPointCloud* cloud, int k, float stdDevMult)
{
    if (cloud == nullptr || cloud->size() == 0) return;
    if (k > MAX_K) k = MAX_K; // 안전장치

    int numPoints = (int)cloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    // 1. 평균 거리 저장용 임시 버퍼
    thrust::device_vector<float> d_meanDists(numPoints);

    // 2. 평균 거리 계산 커널 실행
    computeMeanDistanceKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->points.data()),
        thrust::raw_pointer_cast(cellStartIndices.data()),
        thrust::raw_pointer_cast(cellEndIndices.data()),
        thrust::raw_pointer_cast(d_meanDists.data()),
        numPoints,
        k,
        cellSize,
        gridSize,
        worldOrigin
        );
    cudaDeviceSynchronize();

    // 3. 전체 통계 계산 (Mean & StdDev)
    // 3-1. 합계
    float sum = thrust::reduce(d_meanDists.begin(), d_meanDists.end(), 0.0f, thrust::plus<float>());
    float mean = sum / numPoints;

    // 3-2. 분산 (Variance) -> E[X^2] - (E[X])^2
    float sumSq = thrust::transform_reduce(
        d_meanDists.begin(),
        d_meanDists.end(),
        SquareOp(),
        0.0f,
        thrust::plus<float>()
    );
    float variance = (sumSq / numPoints) - (mean * mean);
    float stdDev = sqrtf(variance > 0 ? variance : 0.0f);

    // 4. 임계값 설정
    float threshold = mean + stdDevMult * stdDev;

    // Debugging (옵션)
    // printf("[SOR] Mean: %.4f, StdDev: %.4f, Threshold: %.4f\n", mean, stdDev, threshold);

    // 5. Outlier 마킹 (isAlive = false)
    markOutliersKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(cloud->isAlive.data()),
        thrust::raw_pointer_cast(d_meanDists.data()),
        threshold,
        numPoints
        );
    cudaDeviceSynchronize();
}

float CuSparseDataBlock::computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier)
{
    if (points.empty()) return 1.0f;

    thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
    thrust::pair<float3, float3> bbox = thrust::transform_reduce(
        points.begin(), points.end(), Float3ToPair(), init, Float3MinMax()
    );

    float dx = bbox.second.x - bbox.first.x;
    float dy = bbox.second.y - bbox.first.y;
    float dz = bbox.second.z - bbox.first.z;
    if (dx <= 0) dx = 1.0f; if (dy <= 0) dy = 1.0f; if (dz <= 0) dz = 1.0f;

    float volume = dx * dy * dz;
    return powf(volume / (float)points.size(), 1.0f / 3.0f) * multiplier;
}
