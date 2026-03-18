#include <Core/DataStructures/SparseCells.h>
#include <Core/Common/CUDAMath.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/DataStructures/PCD.h>

#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace Huvitz
{
    __global__
        void computeHashKernel(const float3* positions, int* positionHash, int numParticles, float cellSize, int3 gridSize, float3 worldOrigin)
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
        positionHash[index] = (gridZ * gridSize.y + gridY) * gridSize.x + gridX;
    }

    __global__
        void findCellStartEndKernel(const int* positionHash, int* cellStart, int* cellEnd, int numParticles)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numParticles)
        {
            return;
        }
        int currentHash = positionHash[index];
        if (index == 0)
        {
            cellStart[currentHash] = index;
        }
        else
        {
            int prevHash = positionHash[index - 1];
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

    __global__
        void initUnionFindKernel(unsigned int* labels, int numberOfPositions)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < numberOfPositions)
        {
            labels[index] = (unsigned int)index;
        }
    }

    __global__
        void unionClustersKernel(
            const float3* __restrict__ positions,
            const int* __restrict__ cellStart,
            const int* __restrict__ cellEnd,
            unsigned int* __restrict__ labels,
            int numberOfPositions,
            float clusterDistSq,
            float cellSize,
            int3 gridSize,
            float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numberOfPositions) return;

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

    __global__
        void unionClustersWithNormalKernel(
            const float3* __restrict__ positions,
            const float3* __restrict__ normals,
            const int* __restrict__ cellStart,
            const int* __restrict__ cellEnd,
            unsigned int* __restrict__ labels,
            int numberOfPositions,
            float clusterDistSq,
            float minNormalDot,
            float cellSize,
            int3 gridSize,
            float3 worldOrigin)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numberOfPositions) return;

        float3 myPos = FETCH(positions, index);
        float3 myNormal = FETCH(normals, index);
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
                                    float3 otherNormal = FETCH(normals, j);
                                    float dot = myNormal.x * otherNormal.x + myNormal.y * otherNormal.y + myNormal.z * otherNormal.z;

                                    if (dot >= minNormalDot)
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
    }

    __global__
        void flattenLabelsKernel(unsigned int* labels, bool* changed, int numberOfPositions)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numberOfPositions) return;

        unsigned int parent = labels[index];
        unsigned int root = labels[parent];

        if (root != parent)
        {
            labels[index] = root;
            *changed = true;
        }
    }

    __global__
        void flattenLabelsFinalKernel(unsigned int* labels, int numberOfPositions)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numberOfPositions)
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

    SparseCells::SparseCells()
    {
    }

    float SparseCells::computeAutoCellSize(const thrust::device_vector<float3>& positions, float multiplier, CUstream_st* stream)
    {
        if (positions.empty())
        {
            return 1.0f;
        }
        thrust::pair<float3, float3> init = thrust::make_pair(make_float3(1e30f, 1e30f, 1e30f), make_float3(-1e30f, -1e30f, -1e30f));
        thrust::pair<float3, float3> bbox = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            positions.begin(), positions.end(),
            Float3ToPair(), init, Float3MinMax()
        );

        float dx = bbox.second.x - bbox.first.x;
        float dy = bbox.second.y - bbox.first.y;
        float dz = bbox.second.z - bbox.first.z;
        float maxDim = fmaxf(dx, fmaxf(dy, dz));
        if (maxDim <= 0.0f) maxDim = 1.0f;

        float cellSize = (maxDim / powf((float)positions.size(), 1.0f / 3.0f)) * multiplier;
        return cellSize;
    }

    float SparseCells::computeAutoCellSize(float3* positions, size_t size, float multiplier, CUstream_st* stream)
    {
        if (positions == nullptr || size == 0)
        {
            return 1.0f;
        }

        thrust::device_ptr<float3> beginPtr = thrust::device_pointer_cast(positions);
        thrust::device_ptr<float3> endPtr = beginPtr + size;

        thrust::pair<float3, float3> init = thrust::make_pair(
            make_float3(1e30f, 1e30f, 1e30f),
            make_float3(-1e30f, -1e30f, -1e30f)
        );

        thrust::pair<float3, float3> bbox = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            beginPtr, endPtr,
            Float3ToPair(), init, Float3MinMax()
        );

        float dx = bbox.second.x - bbox.first.x;
        float dy = bbox.second.y - bbox.first.y;
        float dz = bbox.second.z - bbox.first.z;

        float maxDim = fmaxf(dx, fmaxf(dy, dz));
        if (maxDim <= 0.0f)
        {
            maxDim = 1.0f;
        }

        float cellSize = (maxDim / powf((float)size, 1.0f / 3.0f)) * multiplier;

        return cellSize;
    }

    void SparseCells::Build(PCD* cloud, CUstream_st* stream)
    {
        if (cloud == nullptr || cloud->size() == 0)
        {
            return;
        }

        cellSize = computeAutoCellSize(cloud->GetPositions(), cloud->size(), 1.5f, stream);

        printf("Computed cell size: %f\n", cellSize);

        worldOrigin = cloud->GetAABB().min;
        float3 maxP = cloud->GetAABB().max;
        float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };

        gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };

        numberOfCells = gridSize.x * gridSize.y * gridSize.z;

        size_t numberOfPositions = cloud->size();

        if (hashCodes.size() != numberOfPositions)
        {
            hashCodes.resize(numberOfPositions);
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
        int numBlocks = (int)((numberOfPositions + blockSize - 1) / blockSize);

        computeHashKernel << <numBlocks, blockSize >> > (cloud->GetPositions(), thrust::raw_pointer_cast(hashCodes.data()), (int)numberOfPositions, cellSize, gridSize, worldOrigin);

        cudaDeviceSynchronize();

        thrust::device_ptr<float3> d_positions = thrust::device_pointer_cast(cloud->GetPositions());
        thrust::device_ptr<float3> d_normals = thrust::device_pointer_cast(cloud->GetNormals());
        thrust::device_ptr<uchar3> d_colors = thrust::device_pointer_cast(cloud->GetColors());
        thrust::device_ptr<bool> d_isAlive = thrust::device_pointer_cast(cloud->GetIsAlive());

        thrust::sort_by_key(hashCodes.begin(), hashCodes.end(), thrust::make_zip_iterator(thrust::make_tuple(d_positions, d_normals, d_colors, d_isAlive)));

        findCellStartEndKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(hashCodes.data()), thrust::raw_pointer_cast(cellStartIndices.data()), thrust::raw_pointer_cast(cellEndIndices.data()), (int)numberOfPositions);

        cudaDeviceSynchronize();
    }

    void SparseCells::Build(PCD* cloud, float cellSize, CUstream_st* stream)
    {
        if (cloud == nullptr || cloud->size() == 0)
        {
            return;
        }

        CUDA_TS(CuSParseCellsBuild);

        this->cellSize = cellSize;

        worldOrigin = cloud->GetAABB().min;
        float3 maxP = cloud->GetAABB().max;
        float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };

        gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };

        numberOfCells = gridSize.x * gridSize.y * gridSize.z;

        size_t numberOfPositions = cloud->size();

        if (hashCodes.size() != numberOfPositions)
        {
            hashCodes.resize(numberOfPositions);
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
        int numBlocks = (int)((numberOfPositions + blockSize - 1) / blockSize);

        computeHashKernel << <numBlocks, blockSize >> > (cloud->GetPositions(), thrust::raw_pointer_cast(hashCodes.data()), (int)numberOfPositions, cellSize, gridSize, worldOrigin);

        cudaDeviceSynchronize();
        CUDA_TE(CuSParseCellsBuild_Hash);

        CUDA_TS(CuSParseCellsBuild_Sort);
        thrust::device_ptr<float3> d_positions = thrust::device_pointer_cast(cloud->GetPositions());
        thrust::device_ptr<float3> d_normals = thrust::device_pointer_cast(cloud->GetNormals());
        thrust::device_ptr<uchar3> d_colors = thrust::device_pointer_cast(cloud->GetColors());
        thrust::device_ptr<bool> d_isAlive = thrust::device_pointer_cast(cloud->GetIsAlive());

        thrust::sort_by_key(hashCodes.begin(), hashCodes.end(), thrust::make_zip_iterator(thrust::make_tuple(d_positions, d_normals, d_colors, d_isAlive)));
        CUDA_TE(CuSParseCellsBuild_Sort);

        CUDA_TS(CuSParseCellsBuild_Find);
        findCellStartEndKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(hashCodes.data()), thrust::raw_pointer_cast(cellStartIndices.data()), thrust::raw_pointer_cast(cellEndIndices.data()), (int)numberOfPositions);

        cudaDeviceSynchronize();
        CUDA_TE(CuSParseCellsBuild_Find);

        CUDA_TE(CuSParseCellsBuild);
    }

    void SparseCells::Build(float3* d_positions, size_t numberOfPositions, float cellSize, CUstream_st* stream)
    {
        if (d_positions == nullptr || numberOfPositions == 0)
        {
            return;
        }

        CUDA_TS(CuSParseCellsBuild);

        this->cellSize = cellSize;

        thrust::device_ptr<float3> d_ptr = thrust::device_pointer_cast(d_positions);
        float3 h_aabbMin = make_float3(1e30f, 1e30f, 1e30f);
        float3 h_aabbMax = make_float3(-1e30f, -1e30f, -1e30f);
        thrust::pair<float3, float3> init = thrust::make_pair(h_aabbMin, h_aabbMax);

        thrust::pair<float3, float3> bbox = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            d_ptr, d_ptr + numberOfPositions,
            Float3ToPair(), init, Float3MinMax()
        );

        worldOrigin = bbox.first;
        float3 maxP = bbox.second;


        float3 gridDimf = { (maxP.x - worldOrigin.x) / cellSize, (maxP.y - worldOrigin.y) / cellSize, (maxP.z - worldOrigin.z) / cellSize };

        gridSize = { (int)ceilf(gridDimf.x) + 1, (int)ceilf(gridDimf.y) + 1, (int)ceilf(gridDimf.z) + 1 };

        numberOfCells = gridSize.x * gridSize.y * gridSize.z;

        if (hashCodes.size() != numberOfPositions)
        {
            hashCodes.resize(numberOfPositions);
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
        int numBlocks = (int)((numberOfPositions + blockSize - 1) / blockSize);

        computeHashKernel << <numBlocks, blockSize, 0, stream >> > (
            d_positions,
            thrust::raw_pointer_cast(hashCodes.data()),
            (int)numberOfPositions, cellSize, gridSize, worldOrigin
            );

        cudaStreamSynchronize(stream);
        CUDA_TE(CuSParseCellsBuild_Hash);

        CUDA_TS(CuSParseCellsBuild_Find);
        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            hashCodes.begin(), hashCodes.begin() + numberOfPositions,
            d_ptr
        );

        findCellStartEndKernel << <numBlocks, blockSize, 0, stream >> > (
            thrust::raw_pointer_cast(hashCodes.data()),
            thrust::raw_pointer_cast(cellStartIndices.data()),
            thrust::raw_pointer_cast(cellEndIndices.data()),
            (int)numberOfPositions
            );

        CUDA_TE(CuSParseCellsBuild_Find);

        CUDA_TE(CuSParseCellsBuild);
    }

    void SparseCells::ApplyClustering(PCD* cloud, unsigned int* d_outLabels, float clusterDistance, CUstream_st* stream)
    {
        if (cloud == nullptr) return;

        this->ApplyClustering(
            cloud->GetPositions(),
            cloud->size(),
            d_outLabels,
            clusterDistance,
            stream
        );
    }

    void SparseCells::ApplyClustering(float3* d_positions, size_t numberOfPositions, unsigned int* d_outLabels, float clusterDistance, CUstream_st* stream)
    {
        if (d_positions == nullptr || numberOfPositions == 0 || d_outLabels == nullptr)
        {
            return;
        }

        float clusterDistSq = clusterDistance * clusterDistance;
        int blockSize = 256;
        int numBlocks = (int)((numberOfPositions + blockSize - 1) / blockSize);

        initUnionFindKernel << <numBlocks, blockSize, 0, stream >> > (d_outLabels, (int)numberOfPositions);

        for (size_t i = 0; i < 3; i++)
        {
            unionClustersKernel << <numBlocks, blockSize, 0, stream >> > (
                d_positions,
                thrust::raw_pointer_cast(cellStartIndices.data()),
                thrust::raw_pointer_cast(cellEndIndices.data()),
                d_outLabels,
                (int)numberOfPositions,
                clusterDistSq,
                cellSize,
                gridSize,
                worldOrigin
                );

            flattenLabelsFinalKernel << <numBlocks, blockSize, 0, stream >> > (d_outLabels, (int)numberOfPositions);
        }

        flattenLabelsFinalKernel << <numBlocks, blockSize, 0, stream >> > (d_outLabels, (int)numberOfPositions);

        cudaStreamSynchronize(stream);
    }
}
