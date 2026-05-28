#pragma once

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/Common/CUDAMath.h>

#include <Core/DataStructures/VolumeBase.h>

//#include <HSerialization.h>
#include <Helium/Serialization.hpp>

#include <DataFrameIO/DataFrameIO.hpp>

namespace Huvitz
{
	namespace Core
	{
		using BlockID = uint32_t;
		static constexpr BlockID INVALID_BLOCK = 0xFFFFFFFF;

		static constexpr float TSDF_TRUNC_DIST = 1.0f;
		static constexpr float MAX_WEIGHT = 100.0f;

		template <typename T>
		class VoxelDataBase;

		typedef unsigned char uchar;
		typedef float voxel_value_t;

			struct VoxelExtraAttrib {
				static const VoxelExtraAttrib Zero;
		
				uchar deepLearningClass; // enum DL_Class_Names
				uchar materialID; // 0: other 255: tooth
				unsigned short startPatchID;	// 복셀의 조합에 영향을 미친 첫 패치 번호
				unsigned int flags : 2; // 복셀의 잠금상태 등의 상태를 저장.VOXEL_FLAG_**** _BIT 로 비교
				unsigned int label : 30;	// 클러스터링 된 레이블 번호
		
				uint8_t colorMap[4]; // colors[3], alpha (신뢰도)
			};
		
			struct Voxel {
				voxel_value_t		value;
				unsigned short		valueCount;
				//Eigen::Vector3f		position; // 사용하지 않음
				Eigen::Vector3f		normal;
				Eigen::Vector3b		color;
		
				//	mscho	@20250806
				Eigen::Vector3b		color_list[3];
				uint8_t				color_score[3];
		
				//float				colorScore; // 사용하지 않음
				char				segmentation;
				VoxelExtraAttrib	extraAttrib;
		#ifdef USE_EXPERIMENTAL_COLOR_OPT2
				ColorReconData		voxelColorReconData;
		#endif //USE_EXPERIMENTAL_COLOR_OPT2
			};

		struct DummyVoxel
		{
			float           value = 0.f;
			uint16_t        valueCount = 0;
			uint8_t         _pad[2] = {};
			Vector3f normal = Vector3f::Zero();
			Vector3b color = Vector3b::Zero();
		};

		// Direction indices: X+, X-, Y+, Y-, Z+, Z-
		static constexpr int DIR_XP = 0;
		static constexpr int DIR_XN = 1;
		static constexpr int DIR_YP = 2;
		static constexpr int DIR_YN = 3;
		static constexpr int DIR_ZP = 4;
		static constexpr int DIR_ZN = 5;
		static constexpr int DIR_COUNT = 6;

		__device__ __host__ inline Vector3f GetDirectionVector(int d)
		{
			switch (d)
			{
			case DIR_XP: return { 1.f,  0.f,  0.f };
			case DIR_XN: return { -1.f,  0.f,  0.f };
			case DIR_YP: return { 0.f,  1.f,  0.f };
			case DIR_YN: return { 0.f, -1.f,  0.f };
			case DIR_ZP: return { 0.f,  0.f,  1.f };
			case DIR_ZN: return { 0.f,  0.f, -1.f };
			default:     return { 0.f,  0.f,  0.f };
			}
		}

		static constexpr float kDirThreshold = 0.3827f;

		template<typename T>
		struct DirectionalVoxel : public T
		{
			float    dirValue[DIR_COUNT];
			float    weight[DIR_COUNT];
			float    accSd[DIR_COUNT];
			float    accSw[DIR_COUNT];
			Vector3f normal;
			Vector3b color;
			uint8_t  validMask;

			__device__ __host__ inline bool HasDirection(int d) const
			{
				return (validMask >> d) & 1u;
			}

			__device__ __host__ inline float GetValue(int d) const
			{
				return HasDirection(d) ? dirValue[d] : FLT_MAX;
			}
		};

		struct ExtractedVoxel
		{
			Vector3f position;
			Vector3f normal;
			uint8_t color[3];
			float weight;
		};

		template <typename T>
		struct VoxelBlock
		{
			static constexpr int BLOCK_SIZE = 8;
			static constexpr int VOXELS_PER_BLOCK = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
			T voxels[VOXELS_PER_BLOCK];

			__device__ __host__ inline T* GetVoxel(int x, int y, int z)
			{
				return &voxels[z * BLOCK_SIZE * BLOCK_SIZE + y * BLOCK_SIZE + x];
			}

			__device__ __host__ inline const T* GetVoxel(int x, int y, int z) const
			{
				return &voxels[z * BLOCK_SIZE * BLOCK_SIZE + y * BLOCK_SIZE + x];
			}

			__device__ __host__ inline T* GetVoxel(uint32_t index)
			{
				if (index >= VOXELS_PER_BLOCK) return nullptr;
				return &voxels[index];
			}
		};

		class VoxelDataBaseIntegrationParameters : public IntegrationParameters
		{
		public:
			unsigned int mapWidth = 0;
			unsigned int mapHeight = 0;

			Vector3f* d_depthMap = nullptr;
			Vector3f* d_normalMap = nullptr;
			unsigned int* d_colorMap = nullptr;
			Matrix4f transform = Matrix4f::Identity();

			float voxelSize = 0.1f;

			cuAABB aabb;
		};

		struct NoiseFilterParameters
		{
			float* d_depthCache;
			uint32_t dimX;
			uint32_t dimY;
			float    originX;
			float    originY;
			float    cellSize;

			const Vector3f* d_depthMap;
			uint32_t mapWidth;
			uint32_t mapHeight;
			Matrix4f transform;
		};

		class VoxelDataBaseExtractionParameters : public ExtractionParameters
		{
		public:
			enum class Mode
			{
				AllOccupied,
				ZeroCrossing
			};

			Mode mode = Mode::ZeroCrossing;
			ExtractedVoxel* d_out = nullptr;
			uint32_t* d_count = nullptr;
			uint32_t maxOut = 0;
			Vector3f cacheMin;
			Vector3f cacheMax;
		};

		template <typename T>
		__device__ inline float GetVoxelValue(VoxelDataBase<T>& db, VoxelBlock<T>* block, const Vector3f& pos)
		{
			T* v = db.GetVoxel(block, pos);
			if (v != nullptr && v->valueCount > 0)
			{
				return v->value;
			}
			return FLT_MAX;
		}

		template <typename T>
		__global__ void Kernel_Clear(VoxelDataBase<T> db);

		template <typename T>
		__global__ void Kernel_Integrate(VoxelDataBase<T> db, VoxelDataBaseIntegrationParameters parameters);

		template <typename T>
		__global__ void Kernel_IntegrateSurfaceNormal(VoxelDataBase<T> db, VoxelDataBaseIntegrationParameters parameters);

		template <typename T>
		__global__ void Kernel_BuildNoiseFilter2DCache(NoiseFilterParameters nf);

		template <typename T>
		__global__ void InsertKernel(VoxelDataBase<T> db, Matrix4f rt, const Vector3f* points, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId);

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
			Vector3f aabbMin, Vector3f aabbMax,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut);

		template <typename T>
		__global__ void Kernel_Serialize(
			VoxelDataBase<T> db,
			float3* positions,
			float3* normals,
			uchar3* colors,
			uint32_t* numberOfVoxels,
			uint32_t maxOut);

		__global__ void Kernel_IntegrateDirectional_Phase2_Voxel(
			VoxelDataBase<DirectionalVoxel<Voxel>> db,
			uint32_t* occupiedSlots,
			uint32_t occupiedCount);

		__global__ void Kernel_IntegrateDirectional_Phase2_DummyVoxel(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db,
			uint32_t* occupiedSlots,
			uint32_t occupiedCount);

		__global__ void Kernel_ExtractZeroCrossing_Directional(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db, float blockSize,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut);

		__global__ void Kernel_ExtractSurface_Directional(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db, float blockSize,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut);

		template <typename T>
		__global__ void Kernel_PerFrameFilter(
			VoxelDataBase<T> db,
			VoxelDataBaseIntegrationParameters parameters);

		template <typename T>
		class VoxelDataBase : public VolumeBase
		{
		public:
			bool Initialize(uint32_t maxBlocks, float blockSize = 0.8f);

			void Terminate();

			__host__ __device__ inline VoxelBlock<T>* GetBlocks() { return d_blocks; }
			__host__ __device__ inline uint64_t* GetHashTable() { return d_hashTable; }
			__host__ __device__ inline uint32_t* GetBlockCount() { return d_blockCount; }
			__host__ __device__ inline uint32_t GetMaxBlockCount() { return maxBlockCount; }
			__host__ __device__ inline float GetBlockSize() { return blockSize; }

			__host__ __device__ inline uint32_t* GetOccupiedSlots() { return d_occupiedSlots; }

			__device__ inline VoxelBlock<T>* GetVoxelBlock(const Vector3f& position)
			{
				uint32_t slot = FindBlockSlot(position);
				return (slot != INVALID_BLOCK) ? &d_blocks[slot] : nullptr;
			}

			__host__ __device__ inline Vector3f GetBlockCenter(const VoxelBlock<T>* block) const
			{
				uint32_t slot = static_cast<uint32_t>(block - d_blocks);
				Morton64 bKey(d_hashTable[slot]);
				return bKey.ToPosition(blockSize);
			}

			__device__ inline VoxelBlock<T>* GetOrCreateVoxelBlock(const Vector3f& position)
			{
				uint32_t slot = GetOrCreateBlockSlot(position);
				return (slot != INVALID_BLOCK) ? &d_blocks[slot] : nullptr;
			}

			__device__ inline T* GetVoxel(const VoxelBlock<T>* block, const Vector3f& position)
			{
				const float vSize = blockSize / 8.0f;
				const float halfBS = blockSize * 0.5f;
				Vector3f    bc = GetBlockCenter(block);

				int lx = static_cast<int>(floorf((position.x() - (bc.x() - halfBS)) / vSize + 1e-5f));
				int ly = static_cast<int>(floorf((position.y() - (bc.y() - halfBS)) / vSize + 1e-5f));
				int lz = static_cast<int>(floorf((position.z() - (bc.z() - halfBS)) / vSize + 1e-5f));

				if (lx < 0 || lx > 7 || ly < 0 || ly > 7 || lz < 0 || lz > 7)
					return nullptr;

				return const_cast<T*>(&block->voxels[(lz << 6) | (ly << 3) | lx]);
			}

			__device__ T* GetOrCreateVoxel(const Vector3f& position)
			{
				VoxelBlock<T>* block = GetOrCreateVoxelBlock(position);
				if (block == nullptr) return nullptr;

				const float vSize = blockSize / 8.0f;

				Morton64 blockKey = Morton64::FromPosition(position, blockSize);
				Vector3f bc = blockKey.ToPosition(blockSize);

				int lx = static_cast<int>(floorf((position.x() - (bc.x() - blockSize * 0.5f)) / vSize + 1e-4f));
				int ly = static_cast<int>(floorf((position.y() - (bc.y() - blockSize * 0.5f)) / vSize + 1e-4f));
				int lz = static_cast<int>(floorf((position.z() - (bc.z() - blockSize * 0.5f)) / vSize + 1e-4f));
				lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
				ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
				lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

				return &(block->voxels[(lz << 6) | (ly << 3) | lx]);
			}

			void Serialize(const std::wstring& filename);
			void Deserialize(const std::wstring& filename);
			void SaveToPLY(const std::wstring& filename);

			virtual void Clear(cached_allocator* allocator, CUstream_st* stream) override
			{
#ifdef __CUDACC__
				nvtxRangePushA("Clear Voxel Data Base");
				printf("Clear Voxel Data Base\n");
				Kernel_Clear << <(maxBlockCount + 255) / 256, 256, 0, stream >> > (*this);
				cudaMemsetAsync(d_blockCount, 0, sizeof(uint32_t), stream);
				nvtxRangePop();
#endif
			}

			virtual void Integrate(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream) override;
			void Integrate_Recording(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream);
			void IntegrateSurfaceNormal(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream);
			void IntegrateDirectional(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream);
			void ApplyIntraRegionNoiseFilter(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream);
			virtual void Extract(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream) override;
			virtual void ExtractIntraRegion(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream);
			void ExtractDirectional(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream);
			void PerFrameFilter(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream);

		private:
			VoxelBlock<T>* d_blocks = nullptr;
			uint64_t* d_hashTable = nullptr;
			uint32_t* d_blockCount = nullptr;
			uint32_t  maxBlockCount = 0;
			float     blockSize = 0.8f;

			uint32_t* d_occupiedSlots = nullptr;

			uint32_t* d_dirtySlots = nullptr;
			uint32_t* d_dirtyCount = nullptr;

			__device__ inline uint32_t VoxelDataBase<T>::FindBlockSlot(const Vector3f& position)
			{
				Morton64 blockKey = Morton64::FromPosition(position, blockSize);
				uint64_t key = blockKey.code;
				if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

				uint32_t slot = StrongHash(key, maxBlockCount);
				uint32_t start = slot;

				while (true)
				{
					uint64_t hKey = d_hashTable[slot];
					if (hKey == key) return slot;
					if (hKey == 0)   return INVALID_BLOCK;
					slot = (slot + 1) % maxBlockCount;
					if (slot == start) break;
				}
				return INVALID_BLOCK;
			}

			__device__ inline uint32_t VoxelDataBase<T>::GetOrCreateBlockSlot(const Vector3f& position)
			{
				Morton64 blockKey = Morton64::FromPosition(position, blockSize);
				uint64_t key = blockKey.code;
				if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

				uint32_t slot = StrongHash(key, maxBlockCount);
				uint32_t start = slot;

				while (true)
				{
					unsigned long long* slotPtr = (unsigned long long*) & d_hashTable[slot];
					unsigned long long  prev = atomicCAS(slotPtr, 0ULL, (unsigned long long)key);

					if (prev == 0)
					{
						uint32_t idx = atomicAdd(d_blockCount, 1);
						d_occupiedSlots[idx] = slot;

						uint32_t dirtyIdx = atomicAdd(d_dirtyCount, 1);
						d_dirtySlots[dirtyIdx] = slot;
						return slot;
					}
					if (prev == (unsigned long long)key)
					{
						uint32_t dirtyIdx = atomicAdd(d_dirtyCount, 1);
						d_dirtySlots[dirtyIdx] = slot;
						return slot;
					}

					slot = (slot + 1) % maxBlockCount;
					if (slot == start) break;
				}
				return INVALID_BLOCK;
			}

			static std::unique_ptr<DataFrameRecorder<VoxelDataBaseIntegrationParameters>> recorder;
		};
	}
}
