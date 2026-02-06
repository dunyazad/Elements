#pragma once

#ifdef VVV_EXPORTS
#define VVV_API __declspec(dllexport)
#else
#define VVV_API __declspec(dllimport)
#endif

#include <cstdint>
#include <vector>
#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#if !defined(__CUDACC__) && !defined(__host__)
#define __host__
#endif
#if !defined(__CUDACC__) && !defined(__device__)
#define __device__
#endif

#ifndef CUDA_TS
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#endif

#ifndef CUDA_TE
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[<[%s]>] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#endif

namespace Eigen
{
	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class Matrix;

	typedef Matrix<float, 3, 3, 0, 3, 3> Matrix3f;
	typedef Matrix<float, 4, 4, 0, 4, 4> Matrix4f;
	typedef Matrix<float, 3, 1, 0, 3, 1> Vector3f;
	typedef Matrix<uint8_t, 3, 1, 0, 3, 1> Vector3b;
	typedef Matrix<int32_t, 3, 1, 0, 3, 1> Vector3i;
	typedef Matrix<uint32_t, 3, 1, 0, 3, 1> Vector3ui;
}

namespace VVV
{
	__device__ inline uint32_t StrongHash(uint64_t key, uint32_t maxBlocks)
	{
		key ^= key >> 33;
		key *= 0xff51afd7ed558ccdULL;
		key ^= key >> 33;
		key *= 0xc4ceb9fe1a85ec53ULL;
		key ^= key >> 33;
		return static_cast<uint32_t>(key % maxBlocks);
	}

	struct VVV_API Vector3b
	{
		uint8_t x, y, z;

		__host__ __device__
		inline Vector3b() : x(0), y(0), z(0) {}

		__host__ __device__
		inline Vector3b(uint8_t _x, uint8_t _y, uint8_t _z) : x(_x), y(_y), z(_z) {}

		__host__ __device__
		inline Vector3b(const Eigen::Vector3b& other)
		{
			const uint8_t* data = reinterpret_cast<const uint8_t*>(&other);
			x = data[0]; y = data[1]; z = data[2];
		}

		__host__ __device__
		inline Vector3b& operator=(const Eigen::Vector3b& other)
		{
			const uint8_t* data = reinterpret_cast<const uint8_t*>(&other);
			x = data[0]; y = data[1]; z = data[2];
			return *this;
		}

		__host__ __device__
		inline operator Eigen::Vector3b& () { return *reinterpret_cast<Eigen::Vector3b*>(this); }
	};

	struct VVV_API Vector3i
	{
		int32_t x, y, z;

		__host__ __device__
		inline Vector3i() : x(0), y(0), z(0) {}

		__host__ __device__
		inline Vector3i(int32_t _x, int32_t _y, int32_t _z) : x(_x), y(_y), z(_z) {}

		__host__ __device__
		inline Vector3i(const Eigen::Vector3i& other)
		{
			const int32_t* data = reinterpret_cast<const int32_t*>(&other);
			x = data[0]; y = data[1]; z = data[2];
		}

		__host__ __device__
		inline Vector3i& operator=(const Eigen::Vector3i& other)
		{
			const int32_t* data = reinterpret_cast<const int32_t*>(&other);
			x = data[0]; y = data[1]; z = data[2];
			return *this;
		}

		__host__ __device__
		inline operator Eigen::Vector3i& () { return *reinterpret_cast<Eigen::Vector3i*>(this); }
	};

	struct VVV_API Vector3f
	{
		float x, y, z;

		__host__ __device__
		inline Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}

		__host__ __device__
		inline Vector3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

		__host__ __device__
		inline Vector3f(const Eigen::Vector3f& other)
		{
			const float* data = reinterpret_cast<const float*>(&other);
			x = data[0]; y = data[1]; z = data[2];
		}

		__host__ __device__
		inline Vector3f& operator=(const Eigen::Vector3f& other)
		{
			const float* data = reinterpret_cast<const float*>(&other);
			x = data[0]; y = data[1]; z = data[2];
			return *this;
		}

		__host__ __device__
		inline operator Eigen::Vector3f& () { return *reinterpret_cast<Eigen::Vector3f*>(this); }
	};

	struct VVV_API Matrix3f
	{
		float data[9];

		__host__ __device__
			inline Matrix3f()
		{
			for (int i = 0; i < 9; ++i) data[i] = 0.0f;
		}

		__host__ __device__
			inline Matrix3f(const Eigen::Matrix3f& other)
		{
			const float* src = reinterpret_cast<const float*>(&other);
			for (int i = 0; i < 9; ++i) data[i] = src[i];
		}

		// 행렬 원소 접근을 위한 연산자 추가
		__host__ __device__
			inline float& operator()(int row, int col)
		{
			return data[col * 3 + row];
		}

		__host__ __device__
			inline const float& operator()(int row, int col) const
		{
			return data[col * 3 + row];
		}

		__host__ __device__
			inline Matrix3f& operator=(const Eigen::Matrix3f& other)
		{
			const float* src = reinterpret_cast<const float*>(&other);
			for (int i = 0; i < 9; ++i) data[i] = src[i];
			return *this;
		}

		__host__ __device__
			inline operator Eigen::Matrix3f& ()
		{
			return *reinterpret_cast<Eigen::Matrix3f*>(data);
		}

		__host__ __device__
			inline operator const Eigen::Matrix3f& () const
		{
			return *reinterpret_cast<const Eigen::Matrix3f*>(data);
		}

		__host__ __device__
			static inline Matrix3f Identity()
		{
			Matrix3f mat;
			mat.data[0] = 1.0f; mat.data[4] = 1.0f; mat.data[8] = 1.0f;
			return mat;
		}

		__host__ __device__
			static inline Matrix3f Zero()
		{
			return Matrix3f();
		}
	};

	struct VVV_API Matrix4f
	{
		float data[16];

		__host__ __device__
			inline Matrix4f()
		{
			for (int i = 0; i < 16; ++i) data[i] = 0.0f;
		}

		__host__ __device__
			inline Matrix4f(const Eigen::Matrix4f& other)
		{
			const float* src = reinterpret_cast<const float*>(&other);
			for (int i = 0; i < 16; ++i) data[i] = src[i];
		}

		__host__ __device__
			inline Matrix4f& operator=(const Eigen::Matrix4f& other)
		{
			const float* src = reinterpret_cast<const float*>(&other);
			for (int i = 0; i < 16; ++i) data[i] = src[i];
			return *this;
		}

		// 행렬 원소 접근을 위한 연산자 추가 (Column-major 기반)
		__host__ __device__
			inline float& operator()(int row, int col)
		{
			return data[col * 4 + row];
		}

		__host__ __device__
			inline const float& operator()(int row, int col) const
		{
			return data[col * 4 + row];
		}

		__host__ __device__
			inline operator Eigen::Matrix4f& ()
		{
			return *reinterpret_cast<Eigen::Matrix4f*>(data);
		}

		__host__ __device__
			inline operator const Eigen::Matrix4f& () const
		{
			return *reinterpret_cast<const Eigen::Matrix4f*>(data);
		}

		__host__ __device__
			inline Vector3f Transform(const Vector3f& vec) const
		{
			float x = data[0] * vec.x + data[4] * vec.y + data[8] * vec.z + data[12];
			float y = data[1] * vec.x + data[5] * vec.y + data[9] * vec.z + data[13];
			float z = data[2] * vec.x + data[6] * vec.y + data[10] * vec.z + data[14];
			float w = data[3] * vec.x + data[7] * vec.y + data[11] * vec.z + data[15];

			return { x / w, y / w, z / w };
		}

		__host__ __device__
			inline Vector3f TransformNormal(const Vector3f& vec) const
		{
			float x = data[0] * vec.x + data[4] * vec.y + data[8] * vec.z;
			float y = data[1] * vec.x + data[5] * vec.y + data[9] * vec.z;
			float z = data[2] * vec.x + data[6] * vec.y + data[10] * vec.z;
			float w = data[3] * vec.x + data[7] * vec.y + data[11] * vec.z;

			return { x / w, y / w, z / w };
		}

		__host__ __device__
			static inline Matrix4f Identity()
		{
			Matrix4f mat;
			mat.data[0] = 1.0f; mat.data[5] = 1.0f; mat.data[10] = 1.0f; mat.data[15] = 1.0f;
			return mat;
		}

		__host__ __device__
			static inline Matrix4f Zero()
		{
			return Matrix4f();
		}
	};

	struct VVV_API VoxelExtraAttrib
	{
		uint8_t deepLearningClass;
		uint8_t materialID;
		unsigned short startPatchID;
		unsigned int flags : 2;
		unsigned int label : 30;
	};

	struct VVV_API Voxel
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

	struct VVV_API ExtractedVoxel
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

		__host__ __device__ Morton64() {};
		__host__ __device__ explicit Morton64(uint64_t c) : code(c) {}

		__host__ __device__ inline Morton64(int32_t x, int32_t y, int32_t z)
		{
			uint32_t ux = static_cast<uint32_t>(x + AXIS_BIAS);
			uint32_t uy = static_cast<uint32_t>(y + AXIS_BIAS);
			uint32_t uz = static_cast<uint32_t>(z + AXIS_BIAS);
			code = Encode(ux, uy, uz);
		}

		__host__ __device__ inline bool operator==(const Morton64& other) const { return code == other.code; }
		__host__ __device__ inline operator uint64_t() const { return code; }

		__host__ __device__ static inline int32_t ToBlockCoord(float x, float blockSize)
		{
			return static_cast<int32_t>(floorf(x / blockSize));
		}

		__host__ __device__ static inline Morton64 FromPosition(const Vector3f& p, float blockSize)
		{
			return Morton64(ToBlockCoord(p.x, blockSize), ToBlockCoord(p.y, blockSize), ToBlockCoord(p.z, blockSize));
		}

		__host__ __device__ inline Vector3f ToPosition(float blockSize)
		{
			uint32_t ux = CompactBits(code >> 0);
			uint32_t uy = CompactBits(code >> 1);
			uint32_t uz = CompactBits(code >> 2);
			int32_t vx = static_cast<int32_t>(ux) - AXIS_BIAS;
			int32_t vy = static_cast<int32_t>(uy) - AXIS_BIAS;
			int32_t vz = static_cast<int32_t>(uz) - AXIS_BIAS;
			return Vector3f{ (vx + 0.5f) * blockSize, (vy + 0.5f) * blockSize, (vz + 0.5f) * blockSize };
		}

		__host__ __device__ static inline uint32_t CompactBits(uint64_t v)
		{
			v &= 0x1249249249249249ull;
			v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ull;
			v = (v ^ (v >> 4)) & 0x100f00f00f00f00full;
			v = (v ^ (v >> 8)) & 0x1f0000ff0000ffull;
			v = (v ^ (v >> 16)) & 0x1f00000000ffffull;
			v = (v ^ (v >> 32)) & 0x001FFFFFull;
			return static_cast<uint32_t>(v);
		}

		__host__ __device__ static inline uint64_t ExpandBits(uint32_t v)
		{
			uint64_t x = v & AXIS_MASK;
			x = (x | (x << 32)) & 0x1f00000000ffffull;
			x = (x | (x << 16)) & 0x1f0000ff0000ffull;
			x = (x | (x << 8)) & 0x100f00f00f00f00full;
			x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
			x = (x | (x << 2)) & 0x1249249249249249ull;
			return x;
		}

		__host__ __device__ static inline uint64_t Encode(uint32_t x, uint32_t y, uint32_t z)
		{
			return (ExpandBits(x) << 0) | (ExpandBits(y) << 1) | (ExpandBits(z) << 2);
		}
	};

	struct VVV_API VoxelBlock
	{
		static constexpr int BLOCK_SIZE = 8;
		static constexpr int VOXELS_PER_BLOCK = 512;
		Voxel voxels[VOXELS_PER_BLOCK];
		uint32_t lastTouchedFrameId = 0xFFFFFFFF;
		uint16_t activeVoxelCount = 0;
	};

	using BlockID = uint32_t;
	static constexpr BlockID INVALID_BLOCK = 0xFFFFFFFF;

	class VVV_API VoxelDataBase
	{
	public:
		VoxelBlock* d_blocks = nullptr;
		uint64_t* d_hashTable = nullptr;
		uint32_t* d_blockCount = nullptr;
		uint32_t maxBlockCount = 0;

		VoxelDataBase() = default;
		void Allocate(uint32_t maxBlocks);
		void Free();
		void OccupyVoxelFromPoints(const VVV::Matrix4f& rt, const VVV::Vector3f* points, const VVV::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId);
		void IntegrateTSDF(const VVV::Matrix4f& rt, const VVV::Vector3f* d_points, const VVV::Vector3f* d_normals, const VVV::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId);
		void IntegrateESDF(const VVV::Matrix4f& rt, const VVV::Vector3f* d_points, const VVV::Vector3f* d_normals, const VVV::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId);
		uint32_t ExtractActiveVoxelsToHost(float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut);
		uint32_t ExtractZeroCrossingVoxelsToHost(float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut);

		__device__ inline uint32_t FindBlockSlot(const Vector3f& position)
		{
			const float bSize = 0.8f;
			Morton64 blockKey = Morton64::FromPosition(position, bSize);
			uint64_t key = blockKey.code;
			if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

			uint32_t slot = StrongHash(key, maxBlockCount);
			uint32_t start = slot;

			while (true)
			{
				uint64_t hKey = d_hashTable[slot];
				if (hKey == key) return slot;
				if (hKey == 0) return INVALID_BLOCK;
				slot = (slot + 1) % maxBlockCount;
				if (slot == start) break;
			}
			return INVALID_BLOCK;
		}

		__device__ inline VoxelBlock* GetVoxelBlock(const Vector3f& position)
		{
			uint32_t slot = FindBlockSlot(position);
			return (slot != INVALID_BLOCK) ? &d_blocks[slot] : nullptr;
		}

		__device__ inline Voxel* GetVoxel(const Vector3f& position)
		{
			uint32_t slot = FindBlockSlot(position);
			if (slot == INVALID_BLOCK) return nullptr;

			const float bSize = 0.8f;
			const float vSize = bSize / 8.0f;

			Morton64 blockKey = Morton64::FromPosition(position, bSize);
			Vector3f bc = blockKey.ToPosition(bSize);

			int lx = static_cast<int>(floorf((position.x - (bc.x - bSize * 0.5f)) / vSize + 1e-5f));
			int ly = static_cast<int>(floorf((position.y - (bc.y - bSize * 0.5f)) / vSize + 1e-5f));
			int lz = static_cast<int>(floorf((position.z - (bc.z - bSize * 0.5f)) / vSize + 1e-5f));

			lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
			ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
			lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

			return &d_blocks[slot].voxels[(lz << 6) | (ly << 3) | lx];
		}

		__device__ inline uint32_t GetOrCreateBlockSlot(const Vector3f& position)
		{
#if defined(__CUDA_ARCH__)
			const float bSize = 0.8f;
			Morton64 blockKey = Morton64::FromPosition(position, bSize);
			uint64_t key = blockKey.code;
			if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

			uint32_t slot = StrongHash(key, maxBlockCount);
			uint32_t start = slot;

			while (true)
			{
				unsigned long long* slotPtr = (unsigned long long*) & d_hashTable[slot];
				unsigned long long prev = atomicCAS(slotPtr, 0ULL, (unsigned long long)key);

				if (prev == 0)
				{
					atomicAdd(d_blockCount, 1);
					return slot;
				}
				if (prev == key) return slot;

				slot = (slot + 1) % maxBlockCount;
				if (slot == start) break;
			}
#endif
			return INVALID_BLOCK;
		}

		__device__ inline Voxel* GetOrCreateVoxel(const Vector3f& position)
		{
#if defined(__CUDA_ARCH__)
			uint32_t slot = GetOrCreateBlockSlot(position);
			if (slot == INVALID_BLOCK) return nullptr;

			const float bSize = 0.8f;
			const float vSize = bSize / 8.0f;
			Morton64 blockKey = Morton64::FromPosition(position, bSize);
			Vector3f bc = blockKey.ToPosition(bSize);

			int lx = static_cast<int>(floorf((position.x - (bc.x - bSize * 0.5f)) / vSize + 1e-5f));
			int ly = static_cast<int>(floorf((position.y - (bc.y - bSize * 0.5f)) / vSize + 1e-5f));
			int lz = static_cast<int>(floorf((position.z - (bc.z - bSize * 0.5f)) / vSize + 1e-5f));

			lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
			ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
			lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

			return &d_blocks[slot].voxels[(lz << 6) | (ly << 3) | lx];
#else
			return nullptr;
#endif
		}
	};
}
