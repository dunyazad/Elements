#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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

__global__ void CheckAdjacencyKernel(
	size_t numBlocks,
	const uint64_t* blockKeys,
	const unsigned int* blockOffsets,
	const unsigned int* blockCounts,
	const unsigned int* sortedIndices,
	const Eigen::Vector3f* points,
	float sqDistThreshold,
	int* adjacencyMatrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numBlocks) return;

	uint64_t key = blockKeys[i];
	unsigned int curOffset = blockOffsets[i];
	unsigned int curCount = blockCounts[i];

	int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
	int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
	int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

	int adj_count = 0;
	int base_idx = i * 27;

	for (int64_t dz = -1; dz <= 1; ++dz)
	{
		for (int64_t dy = -1; dy <= 1; ++dy)
		{
			for (int64_t dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;

				uint64_t nKey = DeviceGetKeyFromIndices(xi + dx, yi + dy, zi + dz);

				int left = 0;
				int right = numBlocks - 1;
				int found_ni = -1;

				while (left <= right)
				{
					int mid = left + (right - left) / 2;
					if (blockKeys[mid] == nKey)
					{
						found_ni = mid;
						break;
					}
					else if (blockKeys[mid] < nKey)
					{
						left = mid + 1;
					}
					else
					{
						right = mid - 1;
					}
				}

				if (found_ni != -1)
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
							float dx_f = pt1.x() - pt2.x();
							float dy_f = pt1.y() - pt2.y();
							float dz_f = pt1.z() - pt2.z();
							float distSq = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f;

							if (distSq <= sqDistThreshold)
							{
								isConnected = true;
								break;
							}
						}
						if (isConnected) break;
					}

					if (isConnected)
					{
						adjacencyMatrix[base_idx + adj_count] = found_ni;
						adj_count++;
					}
				}
			}
		}
	}
	adjacencyMatrix[base_idx + adj_count] = -1;
}

__global__ void CheckAdjacencyNormalKernel(
	size_t numBlocks,
	const uint64_t* blockKeys,
	const unsigned int* blockOffsets,
	const unsigned int* blockCounts,
	const unsigned int* sortedIndices,
	const Eigen::Vector3f* points,
	const Eigen::Vector3f* normals,
	float sqDistThreshold,
	float normalThreshold,
	int* adjacencyMatrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numBlocks) return;

	uint64_t key = blockKeys[i];
	unsigned int curOffset = blockOffsets[i];
	unsigned int curCount = blockCounts[i];

	int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
	int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
	int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

	int adj_count = 0;
	int base_idx = i * 27;

	for (int64_t dz = -1; dz <= 1; ++dz)
	{
		for (int64_t dy = -1; dy <= 1; ++dy)
		{
			for (int64_t dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;

				uint64_t nKey = DeviceGetKeyFromIndices(xi + dx, yi + dy, zi + dz);

				int left = 0;
				int right = numBlocks - 1;
				int found_ni = -1;

				while (left <= right)
				{
					int mid = left + (right - left) / 2;
					if (blockKeys[mid] == nKey)
					{
						found_ni = mid;
						break;
					}
					else if (blockKeys[mid] < nKey)
					{
						left = mid + 1;
					}
					else
					{
						right = mid - 1;
					}
				}

				if (found_ni != -1)
				{
					unsigned int nOffset = blockOffsets[found_ni];
					unsigned int nCount = blockCounts[found_ni];
					bool isConnected = false;

					for (unsigned int p1 = 0; p1 < curCount; ++p1)
					{
						unsigned int idx1 = sortedIndices[curOffset + p1];
						Eigen::Vector3f pt1 = points[idx1];
						Eigen::Vector3f nm1 = normals[idx1];

						for (unsigned int p2 = 0; p2 < nCount; ++p2)
						{
							unsigned int idx2 = sortedIndices[nOffset + p2];
							Eigen::Vector3f pt2 = points[idx2];

							float dx_f = pt1.x() - pt2.x();
							float dy_f = pt1.y() - pt2.y();
							float dz_f = pt1.z() - pt2.z();
							float distSq = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f;

							if (distSq <= sqDistThreshold)
							{
								Eigen::Vector3f nm2 = normals[idx2];
								float dotProd = nm1.x() * nm2.x() + nm1.y() * nm2.y() + nm1.z() * nm2.z();
								if (fabs(dotProd) >= normalThreshold)
								{
									isConnected = true;
									break;
								}
							}
						}
						if (isConnected) break;
					}

					if (isConnected)
					{
						adjacencyMatrix[base_idx + adj_count] = found_ni;
						adj_count++;
					}
				}
			}
		}
	}
	adjacencyMatrix[base_idx + adj_count] = -1;
}

#pragma region FlatClustering
__global__ void UnionBlocksKernel(
	size_t numBlocks,
	const uint64_t* blockKeys,
	const unsigned int* blockOffsets,
	const unsigned int* blockCounts,
	const unsigned int* sortedIndices,
	const Eigen::Vector3f* points,
	float sqDistThreshold,
	int* parent)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numBlocks) return;

	uint64_t key = blockKeys[i];
	unsigned int curOffset = blockOffsets[i];
	unsigned int curCount = blockCounts[i];

	int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
	int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
	int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

	for (int64_t dz = -1; dz <= 1; ++dz)
	{
		for (int64_t dy = -1; dy <= 1; ++dy)
		{
			for (int64_t dx = -1; dx <= 1; ++dx)
			{
				if (dx == 0 && dy == 0 && dz == 0) continue;

				uint64_t nKey = ((uint64_t)((xi + dx) & 0x1FFFFF) << 42) |
					((uint64_t)((yi + dy) & 0x1FFFFF) << 21) |
					((uint64_t)((zi + dz) & 0x1FFFFF));

				// Binary Search로 이웃 블록 인덱스 찾기
				int found_ni = -1;
				int left = 0, right = (int)numBlocks - 1;
				while (left <= right)
				{
					int mid = left + (right - left) / 2;
					if (blockKeys[mid] == nKey) { found_ni = mid; break; }
					else if (blockKeys[mid] < nKey) left = mid + 1;
					else right = mid - 1;
				}

				if (found_ni != -1 && i < found_ni) // 중복 검사 방지
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
							float dxf = pt1.x() - pt2.x();
							float dyf = pt1.y() - pt2.y();
							float dzf = pt1.z() - pt2.z();
							if (dxf * dxf + dyf * dyf + dzf * dzf <= sqDistThreshold)
							{
								isConnected = true;
								break;
							}
						}
						if (isConnected) break;
					}

					if (isConnected)
					{
						// Union 연산 (Atomic 기반)
						int root1 = i;
						int root2 = found_ni;
						while (parent[root1] != root1) root1 = parent[root1];
						while (parent[root2] != root2) root2 = parent[root2];

						if (root1 != root2)
						{
							int high = root1 > root2 ? root1 : root2;
							int low = root1 > root2 ? root2 : root1;
							atomicMin(&parent[high], low);
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
		totalSize = capacity;
		offset = 0;
		if (cudaMalloc(&basePointer, totalSize) != cudaSuccess)
		{
			throw std::runtime_error("Failed to allocate GPU memory pool");
		}
	}

	~DeviceMemoryPool()
	{
		if (basePointer)
		{
			cudaFree(basePointer);
		}
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

	void Reset()
	{
		offset = 0;
	}

private:
	void* basePointer = nullptr;
	size_t offset = 0;
	size_t totalSize = 0;
};

template <typename T>
struct FastAllocator
{
	typedef T value_type;
	typedef thrust::device_ptr<T> pointer;
	DeviceMemoryPool* pool;

	FastAllocator(DeviceMemoryPool* p) : pool(p) {}
	template <typename U>
	FastAllocator(const FastAllocator<U>& other) : pool(other.pool) {}

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
		return pointer(static_cast<T*>(ptr));
	}

	void deallocate(pointer ptr, std::size_t n)
	{
	}

	bool operator==(const FastAllocator& other) const
	{
		return pool == other.pool;
	}

	bool operator!=(const FastAllocator& other) const
	{
		return pool != other.pool;
	}
};

class FlatClustering
{
public:
	FlatClustering(size_t poolSize = 200ULL * 1024 * 1024)
		: memoryPool(poolSize),
		deviceBlockKeys(FastAllocator<uint64_t>(&memoryPool)),
		deviceBlockOffsets(FastAllocator<unsigned int>(&memoryPool)),
		deviceBlockCounts(FastAllocator<unsigned int>(&memoryPool)),
		deviceSortedIndices(FastAllocator<unsigned int>(&memoryPool))
	{
	}

	void BuildDevice(const Eigen::Vector3f* points, size_t pointCount, float blockSize = 0.3f)
	{
		memoryPool.Reset();
		this->voxelSize = blockSize;
		if (pointCount == 0)
		{
			return;
		}

		float invVoxelSize = 1.0f / blockSize;
		thrust::device_vector<uint64_t, FastAllocator<uint64_t>> deviceKeys(pointCount, FastAllocator<uint64_t>(&memoryPool));
		deviceSortedIndices.resize(pointCount);

		auto countIter = thrust::make_counting_iterator<unsigned int>(0);
		thrust::copy(thrust::device, countIter, countIter + pointCount, deviceSortedIndices.begin());

		thrust::transform(thrust::device, countIter, countIter + pointCount, deviceKeys.begin(),
			[points, invVoxelSize] __device__(unsigned int i) {
			const Eigen::Vector3f& pt = points[i];
			int64_t xi = static_cast<int64_t>(floorf(pt.x() * invVoxelSize));
			int64_t yi = static_cast<int64_t>(floorf(pt.y() * invVoxelSize));
			int64_t zi = static_cast<int64_t>(floorf(pt.z() * invVoxelSize));
			return ((uint64_t)(xi & 0x1FFFFF) << 42) | ((uint64_t)(yi & 0x1FFFFF) << 21) | ((uint64_t)(zi & 0x1FFFFF));
		});

		thrust::sort_by_key(thrust::device, deviceKeys.begin(), deviceKeys.end(), deviceSortedIndices.begin());

		thrust::device_vector<unsigned int, FastAllocator<unsigned int>> deviceOnes(pointCount, 1, FastAllocator<unsigned int>(&memoryPool));
		deviceBlockKeys.resize(pointCount);
		deviceBlockCounts.resize(pointCount);

		auto newEnd = thrust::reduce_by_key(thrust::device, deviceKeys.begin(), deviceKeys.end(), deviceOnes.begin(), deviceBlockKeys.begin(), deviceBlockCounts.begin());
		size_t numBlocks = newEnd.first - deviceBlockKeys.begin();
		deviceBlockKeys.resize(numBlocks);
		deviceBlockCounts.resize(numBlocks);

		deviceBlockOffsets.resize(numBlocks);
		thrust::exclusive_scan(thrust::device, deviceBlockCounts.begin(), deviceBlockCounts.end(), deviceBlockOffsets.begin());
	}

	void ExtractClusterLabelsDevice(const Eigen::Vector3f* points, unsigned int* labels, float distThreshold = 0.1f)
	{
		if (deviceBlockKeys.empty())
		{
			return;
		}

		size_t numBlocks = deviceBlockKeys.size();
		thrust::device_vector<int, FastAllocator<int>> deviceParent(numBlocks, FastAllocator<int>(&memoryPool));
		thrust::sequence(thrust::device, deviceParent.begin(), deviceParent.end());

		int threads = 256;
		int blocks = (int)((numBlocks + threads - 1) / threads);

		UnionBlocksKernel << <blocks, threads >> > (
			numBlocks,
			thrust::raw_pointer_cast(deviceBlockKeys.data()),
			thrust::raw_pointer_cast(deviceBlockOffsets.data()),
			thrust::raw_pointer_cast(deviceBlockCounts.data()),
			thrust::raw_pointer_cast(deviceSortedIndices.data()),
			points,
			distThreshold * distThreshold,
			thrust::raw_pointer_cast(deviceParent.data())
			);
		cudaDeviceSynchronize();

		FlattenParentKernel << <blocks, threads >> > (numBlocks, thrust::raw_pointer_cast(deviceParent.data()));
		cudaDeviceSynchronize();

		LabelPointsKernel << <blocks, threads >> > (
			numBlocks,
			thrust::raw_pointer_cast(deviceBlockOffsets.data()),
			thrust::raw_pointer_cast(deviceBlockCounts.data()),
			thrust::raw_pointer_cast(deviceSortedIndices.data()),
			thrust::raw_pointer_cast(deviceParent.data()),
			labels
			);
		cudaDeviceSynchronize();
	}

private:
	float voxelSize = 0.1f;
	DeviceMemoryPool memoryPool;
	thrust::device_vector<uint64_t, FastAllocator<uint64_t>> deviceBlockKeys;
	thrust::device_vector<unsigned int, FastAllocator<unsigned int>> deviceBlockOffsets;
	thrust::device_vector<unsigned int, FastAllocator<unsigned int>> deviceBlockCounts;
	thrust::device_vector<unsigned int, FastAllocator<unsigned int>> deviceSortedIndices;
};
#pragma endregion


class AppClusteringDevice : public App
{
public:
	virtual void Execute() override
	{
		CheckDeviceMemory("Initial");

		PLYFormat ply;
		ply.Deserialize("D:\\Debug\\Compound_Full.ply");
		//ply.SetDataType(PLYFormat::PLYDataType::BINARY);
		//ply.Serialize("D:\\Debug\\Compound_Full.ply");
		if (ply.GetPoints().empty())
		{
			printf("Failed to load point cloud.\n");
			return;
		}

		nvtxRangePushA("Copying points to device");

		// 포인트 데이터 복사
		thrust::device_vector<Eigen::Vector3f> d_points(ply.GetPoints().begin(), ply.GetPoints().end());

		// [CORRECTION]: d_labels는 포인트 값으로 초기화하는 것이 아니라, 개수만큼 할당만 해야 합니다.
		thrust::device_vector<unsigned int> d_labels(ply.GetPoints().size());

		CheckDeviceMemory("After copying points to device");

		nvtxRangePop();

		CUDA_TS(Total);
		nvtxRangePushA("Total");

		nvtxRangePushA("Creating FlatClustering object");

		static FlatClustering clustering;

		CheckDeviceMemory("After creating FlatClustering object");

		nvtxRangePop();

		nvtxRangePushA("building clusters");

		CUDA_TS(Build);
		clustering.BuildDevice(thrust::raw_pointer_cast(d_points.data()), d_points.size(), 0.1f);
		CUDA_TE(Build);

		CheckDeviceMemory("After building clusters");

		nvtxRangePop();

		nvtxRangePushA("Extracting clusters");

		CUDA_TS(Extract);
		clustering.ExtractClusterLabelsDevice(
			thrust::raw_pointer_cast(d_points.data()),
			thrust::raw_pointer_cast(d_labels.data()),
			0.175f);
		CUDA_TE(Extract);

		CheckDeviceMemory("After extracting clusters");

		nvtxRangePop();

		CUDA_TE(Total);

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

		CUDA_TS(Visualizing);
		auto colors = Color::GetContrastingColorsWithoutBWRGB(128);

		for (size_t i = 0; i < ply.GetPoints().size(); i++)
		{
			auto& p = ply.GetPoints()[i];
			auto& n = ply.GetNormals()[i];
			auto label = h_labels[i];

			if (10 > clusterSizes[label])
			{
				VD::AddSphere("Points", { XYZ_(p) }, { XYZ_(n) }, 0.5f, { 1.0f, 0.0f, 0.0f, 1.0f });
			}
			else
			{
				VD::AddSphere("Points", { XYZ_(p) }, { XYZ_(n) }, 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });
			}
		}
		CUDA_TE(Visualizing);

		//CheckDeviceMemory("After extracting clusters");

		//nvtxRangePop();

		//CUDA_TE(Total);

		//nvtxRangePop();

		//std::map<unsigned int, unsigned int> clusterSizes;
		//for (size_t i = 0; i < d_labels.size(); ++i)
		//{
		//	clusterSizes[d_labels[i]]++;
		//}

		//thrust::host_vector<unsigned int> h_labels = d_labels;

		//auto colors = Color::GetContrastingColorsWithoutBWRGB(128);

		//for (size_t i = 0; i < ply.GetPoints().size(); i++)
		//{
		//	auto& p = ply.GetPoints()[i];
		//	auto& n = ply.GetNormals()[i];
		//	auto label = h_labels[i];

		//	if(10 > clusterSizes[label])
		//	{
		//		VD::AddSphere("Cluster_" + std::to_string(label), { XYZ_(p) }, { XYZ_(n) }, 0.5f, { 1.0f, 0.0f, 0.0f, 1.0f });
		//	}
		//	else
		//	{
		//		VD::AddSphere("Cluster_" + std::to_string(label), { XYZ_(p) }, { XYZ_(n) }, 0.05f, { 1.0f, 1.0f, 1.0f, 1.0f });
		//	}
		//}
	}
};

REGISTER_APP(AppClusteringDevice, "AppClusteringDevice");
