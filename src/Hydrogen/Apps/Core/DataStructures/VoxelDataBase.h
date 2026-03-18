#pragma once

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/Common/CUDAMath.h>

#include <Core/DataStructures/VolumeBase.h>

#include <Helium/Serialization.hpp>

namespace Huvitz
{
	using BlockID = uint32_t;
	static constexpr BlockID INVALID_BLOCK = 0xFFFFFFFF;

	static constexpr float TSDF_TRUNC_DIST = 1.0f;
	static constexpr float MAX_WEIGHT = 100.0f;

	template <typename T>
	class VoxelDataBase;

	struct ExtractedVoxel
	{
		Eigen::Vector3f position;
		Eigen::Vector3f normal;
		uint8_t color[3];
		float weight;
	};

	template <typename T>
	struct VoxelBlock
	{
		static constexpr int BLOCK_SIZE = 8;
		static constexpr int VOXELS_PER_BLOCK = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
		T voxels[VOXELS_PER_BLOCK];
		//uint32_t lastTouchedFrameId = 0xFFFFFFFF;
	};

	class VoxelDataBaseIntegrationParameters : public IntegrationParameters
	{
	public:
		unsigned int mapWidth = 0;
		unsigned int mapHeight = 0;

		Eigen::Vector3f* d_depthMap = nullptr;
		Eigen::Vector3f* d_normalMap = nullptr;
		//Eigen::Vector3b* d_colorMap = nullptr;
		unsigned int* d_colorMap = nullptr;
		Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

		float voxelSize = 0.1f;
	};

	struct NoiseFilterParameters
	{
		// 2D XY 깊이 캐시 (GPU 버퍼, 초기값 FLT_MAX)
		float* d_depthCache;
		uint32_t dimX;
		uint32_t dimY;
		float    originX;
		float    originY;
		float    cellSize;

		// 깊이맵 소스 (Kernel_Integrate 의 d_depthMap 과 동일)
		const Eigen::Vector3f* d_depthMap;
		uint32_t               mapWidth;
		uint32_t               mapHeight;
		Eigen::Matrix4f        transform;
	};

	class VoxelDataBaseExtractionParameters : public ExtractionParameters
	{
	public:
		enum class Mode
		{
			AllOccupied,     // valueCount > 0 인 모든 복셀 추출
			ZeroCrossing     // TSDF 부호 변화 지점 (표면) 추출
		};

		Mode mode = Mode::ZeroCrossing;
		ExtractedVoxel* d_out = nullptr;
		uint32_t* d_count = nullptr;
		uint32_t maxOut = 0;
		Eigen::Vector3f cacheMin;
		Eigen::Vector3f cacheMax;
	};

	template <typename T>
	__device__ inline float GetVoxelValue(VoxelDataBase<T>& db, const Eigen::Vector3f& pos);

	template <typename T>
	__global__ void Kernel_Clear(VoxelDataBase<T> db);

	template <typename T>
	__global__ void Kernel_Integrate(VoxelDataBase<T> db, VoxelDataBaseIntegrationParameters parameters);

	template <typename T>
	__global__ void Kernel_BuildNoiseFilter2DCache(NoiseFilterParameters nf);

	template <typename T>
	__global__ void Kernel_IntraRegionNoiseFilter(VoxelDataBase<T> db, NoiseFilterParameters nf);

	template <typename T>
	__global__ void InsertKernel(VoxelDataBase<T> db, Eigen::Matrix4f rt, const Eigen::Vector3f* points, const Eigen::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId);

	template <typename T>
	__global__ void TSDFIntegrateKernel(VoxelDataBase<T> db, Eigen::Matrix4f rt, const Eigen::Vector3f* points, const Eigen::Vector3f* normals, const Eigen::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId);

	template <typename T>
	__global__ void Kernel_ExtractAllOccupied(
		VoxelDataBase<T> db, float blockSize,
		ExtractedVoxel* out, uint32_t* count, uint32_t maxOut);

	template <typename T>
	__global__ void Kernel_ExtractZeroCrossing(
		VoxelDataBase<T> db, float blockSize,
		ExtractedVoxel* out, uint32_t* count, uint32_t maxOut);

	template <typename T>
	__global__ void Kernel_ExtractIntraRegion(
		VoxelDataBase<T> db, float blockSize,
		VoxelDataBaseExtractionParameters::Mode mode,
		Eigen::Vector3f aabbMin, Eigen::Vector3f aabbMax,
		ExtractedVoxel* out, uint32_t* count, uint32_t maxOut);

	template <typename T>
	__global__ void Kernel_Serialize(
		VoxelDataBase<T> db,
		float3* positions,
		float3* normals,
		uchar3* colors,
		uint32_t* numberOfVoxels);

	template <typename T>
	class VoxelDataBase : public VolumeBase
	{
	public:
		void Initialize(uint32_t maxBlocks, float blockSize = 0.8f);

		void Terminate();

		void IntegrateTSDF(const Eigen::Matrix4f& rt, const Eigen::Vector3f* d_points, const Eigen::Vector3f* d_normals, const Eigen::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId);

		__host__ __device__ inline VoxelBlock<T>* GetBlocks() { return d_blocks; }
		__host__ __device__ inline uint64_t* GetHashTable() { return d_hashTable; }
		__host__ __device__ inline uint32_t* GetBlockCount() { return d_blockCount; }
		__host__ __device__ inline uint32_t GetMaxBlockCount() { return maxBlockCount; }
		__host__ __device__ inline float GetBlockSize() { return blockSize; }

		__device__ inline VoxelBlock<T>* GetVoxelBlock(const Eigen::Vector3f& position);

		__device__ inline VoxelBlock<T>* GetOrCreateVoxelBlock(const Eigen::Vector3f& position);

		__device__ inline T* GetVoxel(const Eigen::Vector3f& position);

		__device__ inline T* GetOrCreateVoxel(const Eigen::Vector3f& position);

		void Serialize(const std::wstring& filename);
		void Deserialize(const std::wstring& filename);

		virtual void Clear(cached_allocator* allocator, CUstream_st* stream) override
		{
#ifdef __CUDACC__
			nvtxRangePushA("Clear Voxel Data Base");
			printf("Clear Voxel Data Base\n");
			Kernel_Clear << <(maxBlockCount + 255) / 256, 256, 0, stream >> > (*this);

			nvtxRangePop();
#endif
			//cudaMemsetAsync(d_blocks, 0, sizeof(VoxelBlock<T>) * maxBlockCount, stream);
			//cudaMemsetAsync(d_hashTable, 0, sizeof(uint64_t) * maxBlockCount, stream);
			//cudaMemsetAsync(d_blockCount, 0, sizeof(uint32_t), stream);
		}

		virtual void Integrate(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream) override
		{
#ifdef __CUDACC__
			//printf("!!!Integrate\n");

			auto params = dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
			if(nullptr == params)
			{
				return;
			}

			int threads = 256;
			int blocks = (params->mapWidth * params->mapHeight + threads - 1) / threads;

			nvtxRangePushA("!!!Integrate");
			Kernel_Integrate<<<blocks, threads, 0, stream>>>(*this, *params);

			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
		}

		void ApplyIntraRegionNoiseFilter(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
			auto params = dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
			if (nullptr == params)
			{
				return;
			}

			NoiseFilterParameters nfParams;
			nfParams.d_depthMap = params->d_depthMap;
			nfParams.mapWidth = params->mapWidth;
			nfParams.mapHeight = params->mapHeight;
			nfParams.transform = params->transform;
			nfParams.cellSize = params->voxelSize;
			nfParams.dimX = 256;
			nfParams.dimY = 256;
			float camX = nfParams.transform(0, 3);
			float camY = nfParams.transform(1, 3);
			nfParams.originX = camX - nfParams.dimX * nfParams.cellSize * 0.5f;
			nfParams.originY = camY - nfParams.dimY * nfParams.cellSize * 0.5f;

			cudaMalloc(&nfParams.d_depthCache, sizeof(float) * nfParams.dimX * nfParams.dimY);

#ifdef __CUDACC__
			int threads = 256;

			// Step 1: 2D 캐시 초기화 (0x7F7F7F7F = FLT_MAX bit pattern)
			cudaMemset(nfParams.d_depthCache, 0x7F,
				sizeof(float) * nfParams.dimX * nfParams.dimY);

			int pixelCount = (int)(nfParams.mapWidth * nfParams.mapHeight);

			nvtxRangePushA("ApplyIntraRegionNoiseFilter");
			Kernel_BuildNoiseFilter2DCache<T> << <(pixelCount + threads - 1) / threads, threads >> > (nfParams);

			// Step 2: 복셀 필터링
			Kernel_IntraRegionNoiseFilter << <(maxBlockCount + threads - 1) / threads, threads >> > (*this, nfParams);

			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
			cudaFree(nfParams.d_depthCache);
		}

		virtual void Extract(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream) override
		{
#ifdef __CUDACC__
			//CUDA_TS(ExtractVoxelDataBase);

			auto params = dynamic_cast<VoxelDataBaseExtractionParameters*>(parameters);
			if (nullptr == params)
				return;

			if (nullptr == params->d_out || nullptr == params->d_count || params->maxOut == 0)
				return;

			nvtxRangePushA("ExtetractVoxelDataBase");

			cudaMemsetAsync(params->d_count, 0, sizeof(uint32_t), stream);

			int threads = 256;
			int numBlocks = (maxBlockCount + threads - 1) / threads;

			switch (params->mode)
			{
			case VoxelDataBaseExtractionParameters::Mode::AllOccupied:
				Kernel_ExtractAllOccupied << <numBlocks, threads, 0, stream >> > (
					*this, blockSize, params->d_out, params->d_count, params->maxOut);
				break;

			case VoxelDataBaseExtractionParameters::Mode::ZeroCrossing:
				Kernel_ExtractZeroCrossing << <numBlocks, threads, 0, stream >> > (
					*this, blockSize, params->d_out, params->d_count, params->maxOut);
				break;

			default:
				break;
			}

			cudaStreamSynchronize(stream);

			nvtxRangePop();

			//CUDA_TE(ExtractVoxelDataBase);
#endif
		}

		virtual void ExtractIntraRegion(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
#ifdef __CUDACC__
			//CUDA_TS(ExtractIntraRegion);

			auto params = dynamic_cast<VoxelDataBaseExtractionParameters*>(parameters);
			if (nullptr == params)
				return;
			if (nullptr == params->d_out || nullptr == params->d_count || params->maxOut == 0)
				return;
			
			nvtxRangePushA("ExtractIntraRegion");
			
			cudaMemsetAsync(params->d_count, 0, sizeof(uint32_t), stream);
			
			int threads = 256;
			int numBlocks = (maxBlockCount + threads - 1) / threads;
			
			Kernel_ExtractIntraRegion << <numBlocks, threads, 0, stream >> > (
				*this, blockSize, params->mode,
				params->cacheMin, params->cacheMax,
				params->d_out, params->d_count, params->maxOut);
			
			cudaStreamSynchronize(stream);
			
			nvtxRangePop();
			
			//CUDA_TE(ExtractIntraRegion);
#endif
		}

	private:
		VoxelBlock<T>* d_blocks = nullptr;
		uint64_t* d_hashTable = nullptr;
		uint32_t* d_blockCount = nullptr;
		uint32_t maxBlockCount = 0;
		float blockSize = 0.8f;

		__device__ inline uint32_t FindBlockSlot(const Eigen::Vector3f& position);

		__device__ inline uint32_t GetOrCreateBlockSlot(const Eigen::Vector3f& position);
	};
}
