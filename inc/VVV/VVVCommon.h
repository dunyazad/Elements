#pragma once

#ifdef VVV_EXPORTS
#define VVV_API __declspec(dllexport)
#else
#define VVV_API __declspec(dllimport)
#endif

#include <cstdint>
#include <vector>
#include <cmath>

// CUDA 매크로 정의를 네임스페이스 밖에서 확실히 처리
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_HOST
#endif

namespace VVV
{
	struct Vector3f
	{
		float x, y, z;
	};

	struct Vector3b
	{
		uint8_t x, y, z;
	};

	struct VoxelExtraAttrib
	{
		uint8_t deepLearningClass;
		uint8_t materialID;
		unsigned short startPatchID;
		unsigned int flags : 2;
		unsigned int label : 30;
	};

	struct Voxel
	{
		float value;
		unsigned short valueCount;
		Vector3f normal;
		Vector3b color;
		Vector3b color_list[3];
		uint8_t color_score[3];
		char segmentation;
		VoxelExtraAttrib extraAttrib;

		uint16_t octa1;
		uint16_t octa2;
	};

	struct ExtractedVoxel
	{
		Vector3f position;
		Vector3f normal;
		uint8_t color[3];
		float weight;
	};

	struct VVV_API Morton64
	{
		uint64_t code = 0;
		static constexpr int AXIS_BITS = 21;
		static constexpr int32_t AXIS_BIAS = 1 << (AXIS_BITS - 1);
		static constexpr uint64_t AXIS_MASK = (1ull << AXIS_BITS) - 1ull;

		CUDA_HOST_DEVICE Morton64() = default;
		CUDA_HOST_DEVICE explicit Morton64(uint64_t c) : code(c) {}

		CUDA_HOST_DEVICE inline Morton64(int32_t x, int32_t y, int32_t z)
		{
			uint32_t ux = static_cast<uint32_t>(x + AXIS_BIAS);
			uint32_t uy = static_cast<uint32_t>(y + AXIS_BIAS);
			uint32_t uz = static_cast<uint32_t>(z + AXIS_BIAS);
			code = Encode(ux, uy, uz);
		}

		CUDA_HOST_DEVICE inline bool operator==(const Morton64& other) const { return code == other.code; }
		CUDA_HOST_DEVICE inline operator uint64_t() const { return code; }

		CUDA_HOST_DEVICE static inline int32_t ToBlockCoord(float x, float blockSize)
		{
			return static_cast<int32_t>(floorf(x / blockSize));
		}

		CUDA_HOST_DEVICE static inline Morton64 FromPosition(const Vector3f& p, float blockSize)
		{
			return Morton64(ToBlockCoord(p.x, blockSize), ToBlockCoord(p.y, blockSize), ToBlockCoord(p.z, blockSize));
		}

		CUDA_HOST_DEVICE Vector3f ToPosition(float blockSize) const;

	private:
		CUDA_HOST_DEVICE static inline uint64_t ExpandBits(uint32_t v)
		{
			uint64_t x = v & AXIS_MASK;
			x = (x | (x << 32)) & 0x1f00000000ffffull;
			x = (x | (x << 16)) & 0x1f0000ff0000ffull;
			x = (x | (x << 8)) & 0x100f00f00f00f00full;
			x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
			x = (x | (x << 2)) & 0x1249249249249249ull;
			return x;
		}

		CUDA_HOST_DEVICE static inline uint64_t Encode(uint32_t x, uint32_t y, uint32_t z)
		{
			return (ExpandBits(x) << 0) | (ExpandBits(y) << 1) | (ExpandBits(z) << 2);
		}
	};

	struct VoxelBlock
	{
		static constexpr int BLOCK_SIZE = 8;
		static constexpr int VOXELS_PER_BLOCK = 512;
		Voxel voxels[VOXELS_PER_BLOCK];
		uint32_t lastTouchedFrameId = 0xFFFFFFFF;
		uint16_t activeVoxelCount = 0;
	};

	using BlockID = uint32_t;
	static constexpr BlockID INVALID_BLOCK = 0xFFFFFFFF;

	class VoxelDataBase
	{
	public:
		VoxelBlock* d_blocks = nullptr;
		uint64_t* d_hashTable = nullptr;
		uint32_t* d_blockCount = nullptr;
		uint32_t maxBlockCount = 0;

		VoxelDataBase() = default;
		void InternalAllocate(uint32_t maxBlocks);
		void InternalFree();
	};
}

extern "C"
{
	VVV_API void VVV_Allocate(VVV::VoxelDataBase& db, uint32_t maxBlocks);
	VVV_API void VVV_Free(VVV::VoxelDataBase& db);
	VVV_API void VVV_UpdateVoxelFromPoints(VVV::VoxelDataBase& db, const VVV::Vector3f* points, const VVV::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId);
	VVV_API uint32_t VVV_ExtractActiveVoxelsToHost(VVV::VoxelDataBase& db, float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut);
}