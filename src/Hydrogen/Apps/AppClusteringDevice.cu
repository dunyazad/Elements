#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <device_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <nvtx3/nvToolsExt.h>

#include <Helium/Color.hpp>
#include <Helium/Serialization.hpp>
#include <Helium/IVisualDebugging.h>
using VD = IVisualDebugging;

#include <Core/CodingSugar.h>
#include <Core/Common/DeviceCommon.h>

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

__device__ inline uint64_t DeviceGetKeyFromIndices(int64_t xi, int64_t yi, int64_t zi)
{
	return ((uint64_t)(xi & 0x1FFFFF) << 42) |
		((uint64_t)(yi & 0x1FFFFF) << 21) |
		((uint64_t)(zi & 0x1FFFFF));
}

struct PointToKeyFunctor
{
	const Eigen::Vector3f* points;
	float invVoxelSize;

	PointToKeyFunctor(const Eigen::Vector3f* p, float invVS) : points(p), invVoxelSize(invVS) {}

	__device__ uint64_t operator()(unsigned int idx) const
	{
		Eigen::Vector3f pt = points[idx];
		int64_t xi = static_cast<int64_t>(floorf(pt.x() * invVoxelSize));
		int64_t yi = static_cast<int64_t>(floorf(pt.y() * invVoxelSize));
		int64_t zi = static_cast<int64_t>(floorf(pt.z() * invVoxelSize));

		return ((uint64_t)(xi & 0x1FFFFF) << 42) |
			((uint64_t)(yi & 0x1FFFFF) << 21) |
			((uint64_t)(zi & 0x1FFFFF));
	}
};

#pragma region FlatClustering
__global__ void UnionBlocksKernel(
	size_t numTotalBlocks,
	size_t offset,
	const uint64_t* blockKeys,
	const unsigned int* blockOffsets,
	const unsigned int* blockCounts,
	const unsigned int* sortedIndices,
	const Eigen::Vector3f* points,
	float sqDistThreshold,
	int* parent)
{
	// 현재 호출된 청크 내에서의 인덱스 + 시작 offset
	int i = blockIdx.x * blockDim.x + threadIdx.x + (int)offset;
	if (i >= numTotalBlocks) return;

	uint64_t key = blockKeys[i];
	unsigned int curOffset = blockOffsets[i];
	unsigned int curCount = blockCounts[i];

	int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
	int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
	int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

	float distThreshold = sqrtf(sqDistThreshold);

	for (int64_t dz = -1; dz <= 1; ++dz)
	{
		for (int64_t dy = -1; dy <= 1; ++dy)
		{
			for (int64_t dx = -1; dx <= 1; ++dx)
			{
				uint64_t nKey = DeviceGetKeyFromIndices(xi + dx, yi + dy, zi + dz);
				if (nKey == key) continue;

				int left = 0;
				int right = (int)numTotalBlocks - 1;
				int found_ni = -1;

				while (left <= right)
				{
					int mid = left + (right - left) / 2;
					if (blockKeys[mid] == nKey) { found_ni = mid; break; }
					else if (blockKeys[mid] < nKey) left = mid + 1;
					else right = mid - 1;
				}

				if (found_ni != -1 && i < found_ni)
				{
					unsigned int nOffset = blockOffsets[found_ni];
					unsigned int nCount = blockCounts[found_ni];
					bool isConnected = false;

					for (unsigned int p1 = 0; p1 < curCount; ++p1)
					{
						Eigen::Vector3f pt1 = points[sortedIndices[curOffset + p1]];
						for (unsigned int p2 = 0; p2 < nCount; ++p2)
						{
							Eigen::Vector3f pt2 = points[sortedIndices[nOffset + p2]];

							if (fabsf(pt1.x() - pt2.x()) > distThreshold) continue;
							if (fabsf(pt1.y() - pt2.y()) > distThreshold) continue;
							if (fabsf(pt1.z() - pt2.z()) > distThreshold) continue;

							float dx_f = pt1.x() - pt2.x();
							float dy_f = pt1.y() - pt2.y();
							float dz_f = pt1.z() - pt2.z();
							if (dx_f * dx_f + dy_f * dy_f + dz_f * dz_f <= sqDistThreshold)
							{
								isConnected = true;
								break;
							}
						}
						if (isConnected) break;
					}

					if (isConnected)
					{
						int root1 = i;
						int root2 = found_ni;
						while (parent[root1] != root1) root1 = parent[root1];
						while (parent[root2] != root2) root2 = parent[root2];

						if (root1 != root2)
						{
							if (root1 < root2) atomicMin(&parent[root2], root1);
							else atomicMin(&parent[root1], root2);
						}
					}
				}
			}
		}
	}
}

__global__ void FlattenParentKernel(int numBlocks, int* parent)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numBlocks) return;

	int root = i;
	while (parent[root] != root) root = parent[root];
	parent[i] = root;
}

__global__ void LabelPointsKernel(
	size_t numBlocks,
	const unsigned int* blockOffsets,
	const unsigned int* blockCounts,
	const unsigned int* sortedIndices,
	const int* parent,
	unsigned int* labels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numBlocks)
	{
		return;
	}

	unsigned int clusterId = static_cast<unsigned int>(parent[i]);
	unsigned int offset = blockOffsets[i];
	unsigned int count = blockCounts[i];

	for (unsigned int p = 0; p < count; ++p)
	{
		unsigned int pointIdx = sortedIndices[offset + p];
		labels[pointIdx] = clusterId;
	}
}

class DeviceMemoryPool
{
public:
	DeviceMemoryPool(size_t capacity)
	{
		AllocateInternal(capacity);
	}

	~DeviceMemoryPool()
	{
		FreeInternal();
	}

	void EnsureCapacity(size_t requestedCapacity)
	{
		if (requestedCapacity > totalSize)
		{
			FreeInternal();
			AllocateInternal(requestedCapacity);
		}
		else
		{
			Clear();
		}
	}

	void Clear()
	{
		offset = 0;
	}

	void* Allocate(size_t bytes)
	{
		size_t alignedBytes = (bytes + 255) & ~255;
		if (offset + alignedBytes > totalSize)
		{
			return nullptr;
		}
		void* ptr = static_cast<char*>(basePointer) + offset;
		offset += alignedBytes;
		return ptr;
	}

private:
	void AllocateInternal(size_t capacity)
	{
		totalSize = capacity;
		offset = 0;
		if (cudaMalloc(&basePointer, totalSize) != cudaSuccess)
		{
			basePointer = nullptr;
			totalSize = 0;
			throw std::runtime_error("Failed to allocate GPU memory pool");
		}
	}

	void FreeInternal()
	{
		if (basePointer)
		{
			cudaFree(basePointer);
			basePointer = nullptr;
		}
	}

	void* basePointer = nullptr;
	size_t offset = 0;
	size_t totalSize = 0;
};

template <typename T>
struct FlatClusteringAllocator
{
	typedef T value_type;
	typedef thrust::device_ptr<T> pointer;
	typedef thrust::device_ptr<const T> const_pointer;
	typedef thrust::device_reference<T> reference;
	typedef thrust::device_reference<const T> const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	DeviceMemoryPool* pool;

	FlatClusteringAllocator(DeviceMemoryPool* p) : pool(p) {}

	template <typename U>
	FlatClusteringAllocator(const FlatClusteringAllocator<U>& other) : pool(other.pool) {}

	pointer allocate(std::size_t n)
	{
		if (n == 0)
		{
			return pointer(nullptr);
		}

		void* ptr = pool->Allocate(n * sizeof(T));
		if (!ptr)
		{
			throw std::bad_alloc();
		}

		return thrust::device_pointer_cast(static_cast<T*>(ptr));
	}

	void deallocate(pointer ptr, std::size_t n)
	{
	}

	template <typename U>
	struct rebind
	{
		typedef FlatClusteringAllocator<U> other;
	};

	bool operator==(const FlatClusteringAllocator& other) const { return pool == other.pool; }
	bool operator!=(const FlatClusteringAllocator& other) const { return pool != other.pool; }
};

class FlatClusteringDevice
{
public:
	FlatClusteringDevice(size_t poolSize = 200ULL * 1024 * 1024)
		: memoryPool(poolSize),
		deviceBlockKeys(FlatClusteringAllocator<uint64_t>(&memoryPool)),
		deviceBlockOffsets(FlatClusteringAllocator<unsigned int>(&memoryPool)),
		deviceBlockCounts(FlatClusteringAllocator<unsigned int>(&memoryPool)),
		deviceSortedIndices(FlatClusteringAllocator<unsigned int>(&memoryPool)),
		deviceParent(FlatClusteringAllocator<int>(&memoryPool))
	{
	}

	template <typename T>
	auto GetPolicy(FlatClusteringAllocator<T>& alloc, cudaStream_t stream)
	{
		return thrust::cuda::par(alloc).on(stream);
	}

	void BuildDevice(const Eigen::Vector3f* points, size_t pointCount, float blockSize, cudaStream_t stream = 0)
	{
		if (pointCount == 0) return;

		//if (stream == 0) cudaDeviceSynchronize();
		//else cudaStreamSynchronize(stream);

		cudaStreamSynchronize(stream);

		deviceBlockKeys.clear();     deviceBlockKeys.shrink_to_fit();
		deviceBlockOffsets.clear();  deviceBlockOffsets.shrink_to_fit();
		deviceBlockCounts.clear();   deviceBlockCounts.shrink_to_fit();
		deviceSortedIndices.clear(); deviceSortedIndices.shrink_to_fit();
		deviceParent.clear();        deviceParent.shrink_to_fit();

		size_t bytesPerPoint = 80;
		size_t estimatedRequired = pointCount * bytesPerPoint;
		memoryPool.EnsureCapacity(std::max(200ULL * 1024 * 1024, estimatedRequired));

		this->voxelSize = blockSize;

		FlatClusteringAllocator<uint8_t> poolAlloc(&memoryPool);
		auto policy = GetPolicy(poolAlloc, stream);

		deviceSortedIndices.resize(pointCount);
		deviceParent.resize(pointCount);

		float invVoxelSize = 1.0f / blockSize;

		thrust::device_vector<uint64_t, FlatClusteringAllocator<uint64_t>> deviceKeys(pointCount, FlatClusteringAllocator<uint64_t>(&memoryPool));
		auto countIter = thrust::make_counting_iterator<unsigned int>(0);

		thrust::copy(policy, countIter, countIter + pointCount, deviceSortedIndices.begin());
		thrust::transform(policy, countIter, countIter + pointCount, deviceKeys.begin(),
			[points, invVoxelSize] __device__(unsigned int i) {
			const Eigen::Vector3f& pt = points[i];
			int64_t xi = static_cast<int64_t>(floorf(pt.x() * invVoxelSize));
			int64_t yi = static_cast<int64_t>(floorf(pt.y() * invVoxelSize));
			int64_t zi = static_cast<int64_t>(floorf(pt.z() * invVoxelSize));
			return ((uint64_t)(xi & 0x1FFFFF) << 42) | ((uint64_t)(yi & 0x1FFFFF) << 21) | ((uint64_t)(zi & 0x1FFFFF));
		});

		thrust::sort_by_key(policy, deviceKeys.begin(), deviceKeys.end(), deviceSortedIndices.begin());

		thrust::device_vector<unsigned int, FlatClusteringAllocator<unsigned int>> deviceOnes(pointCount, 1, FlatClusteringAllocator<unsigned int>(&memoryPool));
		deviceBlockKeys.resize(pointCount);
		deviceBlockCounts.resize(pointCount);

		auto newEnd = thrust::reduce_by_key(policy, deviceKeys.begin(), deviceKeys.end(), deviceOnes.begin(), deviceBlockKeys.begin(), deviceBlockCounts.begin());
		size_t numBlocks = newEnd.first - deviceBlockKeys.begin();

		deviceBlockKeys.resize(numBlocks);
		deviceBlockCounts.resize(numBlocks);
		deviceBlockOffsets.resize(numBlocks);
		thrust::exclusive_scan(policy, deviceBlockCounts.begin(), deviceBlockCounts.end(), deviceBlockOffsets.begin());

		deviceParent.resize(numBlocks);
	}

	void ExtractClusterLabelsDevice(const Eigen::Vector3f* points, unsigned int* labels, float distThreshold, cudaStream_t stream = 0)
	{
		if (deviceBlockKeys.empty()) return;

		FlatClusteringAllocator<uint8_t> poolAlloc(&memoryPool);
		auto policy = GetPolicy(poolAlloc, stream);
		size_t numBlocks = deviceBlockKeys.size();

		thrust::sequence(policy, deviceParent.begin(), deviceParent.end());

		const size_t chunkSize = 10000;
		int threads = 256;

		for (size_t offset = 0; offset < numBlocks; offset += chunkSize)
		{
			size_t currentBatchSize = (numBlocks - offset < chunkSize) ? (numBlocks - offset) : chunkSize;
			int blocks = (int)((currentBatchSize + threads - 1) / threads);

			UnionBlocksKernel << <blocks, threads, 0, stream >> > (
				numBlocks,
				offset,
				thrust::raw_pointer_cast(deviceBlockKeys.data()),
				thrust::raw_pointer_cast(deviceBlockOffsets.data()),
				thrust::raw_pointer_cast(deviceBlockCounts.data()),
				thrust::raw_pointer_cast(deviceSortedIndices.data()),
				points,
				distThreshold * distThreshold,
				thrust::raw_pointer_cast(deviceParent.data())
				);

			if ((offset / chunkSize) % 4 == 0)
			{
				std::this_thread::yield();
			}
		}

		int totalBlocks = (int)((numBlocks + threads - 1) / threads);
		FlattenParentKernel << <totalBlocks, threads, 0, stream >> > ((int)numBlocks, thrust::raw_pointer_cast(deviceParent.data()));

		LabelPointsKernel << <totalBlocks, threads, 0, stream >> > (
			numBlocks,
			thrust::raw_pointer_cast(deviceBlockOffsets.data()),
			thrust::raw_pointer_cast(deviceBlockCounts.data()),
			thrust::raw_pointer_cast(deviceSortedIndices.data()),
			thrust::raw_pointer_cast(deviceParent.data()),
			labels
			);
	}

private:
	float voxelSize = 0.1f;
	DeviceMemoryPool memoryPool;
	thrust::device_vector<uint64_t, FlatClusteringAllocator<uint64_t>> deviceBlockKeys;
	thrust::device_vector<unsigned int, FlatClusteringAllocator<unsigned int>> deviceBlockOffsets;
	thrust::device_vector<unsigned int, FlatClusteringAllocator<unsigned int>> deviceBlockCounts;
	thrust::device_vector<unsigned int, FlatClusteringAllocator<unsigned int>> deviceSortedIndices;
	thrust::device_vector<int, FlatClusteringAllocator<int>> deviceParent;
};
#pragma endregion


class AppClusteringDevice : public App
{
public:
	virtual void Initialize() override
	{
	}

	virtual void Execute() override
	{
		CheckDeviceMemory("Initial");

		PLYFormat ply;
		ply.Deserialize("D:\\Debug\\Compound_Full.ply");
		if (ply.GetPoints().empty())
		{
			printf("Failed to load point cloud.\n");
			return;
		}

		nvtxRangePushA("Copying points to device");

		thrust::device_vector<Eigen::Vector3f> d_points(ply.GetPoints().begin(), ply.GetPoints().end());

		thrust::device_vector<unsigned int> d_labels(ply.GetPoints().size());

		CheckDeviceMemory("After copying points to device");

		nvtxRangePop();

		//CUDA_TS(Total);
		nvtxRangePushA("Total");

		nvtxRangePushA("Creating FlatClustering object");

		static FlatClusteringDevice clustering;

		CheckDeviceMemory("After creating FlatClustering object");

		nvtxRangePop();

		for (size_t i = 0; i < 10; i++)
		{
			nvtxRangePushA("building clusters");

			//CUDA_TS(Build);
			clustering.BuildDevice(thrust::raw_pointer_cast(d_points.data()), d_points.size(), 0.1f);
			//CUDA_TE(Build);

			CheckDeviceMemory("After building clusters");

			nvtxRangePop();

			nvtxRangePushA("Extracting clusters");

			//CUDA_TS(Extract);
			clustering.ExtractClusterLabelsDevice(
				thrust::raw_pointer_cast(d_points.data()),
				thrust::raw_pointer_cast(d_labels.data()),
				0.175f);
			//CUDA_TE(Extract);

			CheckDeviceMemory("After extracting clusters");

			nvtxRangePop();
		}

		//CUDA_TE(Total);

		nvtxRangePop();

		CUDA_TS(CopyDToH);
		thrust::host_vector<unsigned int> h_labels = d_labels;
		CUDA_TE(CopyDToH);

		CUDA_TS(ClusterSizeCounting);
		robin_hood::unordered_map<unsigned int, unsigned int> clusterSizes;
		clusterSizes.reserve(h_labels.size());
		for (size_t i = 0; i < h_labels.size(); ++i)
		{
			clusterSizes[h_labels[i]]++;
		}
		CUDA_TE(ClusterSizeCounting);

		CUDA_TS(Sorting);
		std::vector<std::pair<unsigned int, unsigned int>> sortedClusters;
		sortedClusters.reserve(clusterSizes.size());
		for (const auto& [label, size] : clusterSizes)
		{
			sortedClusters.push_back({ label, size });
		}

		std::sort(sortedClusters.begin(), sortedClusters.end(),
			[](const std::pair<unsigned int, unsigned int>& a, const std::pair<unsigned int, unsigned int>& b)
			{
				return a.second > b.second;
			});
		CUDA_TE(Sorting);

		CUDA_TS(Visualizing);
		auto colors = Color::GetContrastingColorsWithoutBWRGB(128);

		// 최대 라벨 값을 찾아 배열(Vector) 기반의 빠른 룩업 테이블 생성
		unsigned int maxLabel = 0;
		for (const auto& [label, size] : clusterSizes)
		{
			if (label > maxLabel)
			{
				maxLabel = label;
			}
		}

		std::vector<Eigen::Vector4f> colorLookup(maxLabel + 1, { 1.0f, 1.0f, 1.0f, 1.0f });
		std::vector<unsigned int> sizeLookup(maxLabel + 1, 0);

		for (size_t i = 0; i < sortedClusters.size(); ++i)
		{
			unsigned int label = sortedClusters[i].first;
			colorLookup[label] = colors[i % colors.size()];
			sizeLookup[label] = sortedClusters[i].second;
		}

		size_t pointCount = ply.GetPoints().size();
		const auto& points = ply.GetPoints();
		const auto& normals = ply.GetNormals();

		// Batch 처리를 위한 버퍼 준비
		std::vector<float3> smallCenters;
		std::vector<float3> smallNormals;
		std::vector<float4> smallColors;

		std::vector<float3> largeCenters;
		std::vector<float3> largeNormals;
		std::vector<float4> largeColors;

		// 재할당 오버헤드 방지를 위한 메모리 예약
		smallCenters.reserve(pointCount);
		smallNormals.reserve(pointCount);
		smallColors.reserve(pointCount);
		largeCenters.reserve(pointCount);
		largeNormals.reserve(pointCount);
		largeColors.reserve(pointCount);

		// 포인트들을 크기 조건에 따라 두 그룹으로 분류
		for (size_t i = 0; i < pointCount; i++)
		{
			const auto& p = points[i];
			const auto& n = normals[i];
			unsigned int label = h_labels[i];

			const auto& c = colorLookup[label];
			unsigned int cSize = sizeLookup[label];

			float3 pos = { XYZ_(p) };
			float3 norm = { XYZ_(n) };
			float4 col = { XYZW_(c) };

			if (10 > cSize)
			{
				smallCenters.push_back(pos);
				smallNormals.push_back(norm);
				smallColors.push_back(col);
			}
			else
			{
				largeCenters.push_back(pos);
				largeNormals.push_back(norm);
				largeColors.push_back(col);
			}
		}

		// 단 2번의 호출로 모든 포인트 클라우드 일괄 렌더링
		if (!smallCenters.empty())
		{
			VD::AddSphereBatch("Points_Small", smallCenters, smallNormals, 0.1f, smallColors);
		}
		if (!largeCenters.empty())
		{
			VD::AddSphereBatch("Points_Large", largeCenters, largeNormals, 0.05f, largeColors);
		}

		CUDA_TE(Visualizing);
	}
};

REGISTER_APP(AppClusteringDevice, "AppClusteringDevice");
