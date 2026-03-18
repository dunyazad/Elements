#pragma once

#include <cuda_runtime.h>

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/CUDAMath.h>

#include <Eigen/Core>

namespace Eigen
{
	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class Matrix;

	typedef Matrix<float, 3, 3, 0, 3, 3> Matrix3f;
	typedef Matrix<float, 4, 4, 0, 4, 4> Matrix4f;
	typedef Matrix<float, 3, 1, 0, 3, 1> Vector3f;
	typedef Matrix<float, 4, 1, 0, 4, 1> Vector4f;
	typedef Matrix<uint8_t, 3, 1, 0, 3, 1> Vector3b;
	typedef Matrix<int32_t, 3, 1, 0, 3, 1> Vector3i;
	typedef Matrix<uint32_t, 3, 1, 0, 3, 1> Vector3ui;
}

namespace Huvitz
{
	struct Float4ToUChar3 {
		__host__ __device__ uchar3 operator()(const float4& c) const {
			return make_uchar3(
				(unsigned char)(c.x * 255.0f),
				(unsigned char)(c.y * 255.0f),
				(unsigned char)(c.z * 255.0f)
			);
		}
	};

	struct UChar3ToFloat4 {
		__host__ __device__ float4 operator()(const uchar3& c) const {
			return make_float4(
				c.x / 255.0f,
				c.y / 255.0f,
				c.z / 255.0f,
				1.0f
			);
		}
	};

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

	struct Morton64
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

		__host__ __device__ static inline Morton64 FromPosition(const Eigen::Vector3f& p, float blockSize)
		{
			return Morton64(ToBlockCoord(p.x(), blockSize), ToBlockCoord(p.y(), blockSize), ToBlockCoord(p.z(), blockSize));
		}

		__host__ __device__ inline Eigen::Vector3f ToPosition(float blockSize)
		{
			uint32_t ux = CompactBits(code >> 0);
			uint32_t uy = CompactBits(code >> 1);
			uint32_t uz = CompactBits(code >> 2);
			int32_t vx = static_cast<int32_t>(ux) - AXIS_BIAS;
			int32_t vy = static_cast<int32_t>(uy) - AXIS_BIAS;
			int32_t vz = static_cast<int32_t>(uz) - AXIS_BIAS;
			return Eigen::Vector3f{ (vx + 0.5f) * blockSize, (vy + 0.5f) * blockSize, (vz + 0.5f) * blockSize };
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

	struct cuAABB
	{
		float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		__host__ __device__ __forceinline__
			void expand(const float3& p)
		{
			min.x = fminf(min.x, p.x);
			min.y = fminf(min.y, p.y);
			min.z = fminf(min.z, p.z);

			max.x = fmaxf(max.x, p.x);
			max.y = fmaxf(max.y, p.y);
			max.z = fmaxf(max.z, p.z);
		}

		__host__ __device__ __forceinline__
			void expand(const cuAABB& b)
		{
			min.x = std::min(min.x, b.min.x);
			min.y = std::min(min.y, b.min.y);
			min.z = std::min(min.z, b.min.z);

			max.x = std::max(max.x, b.max.x);
			max.y = std::max(max.y, b.max.y);
			max.z = std::max(max.z, b.max.z);
		}

		__host__ __device__ __forceinline__
			float volume() const
		{
			if (min.x > max.x) return 0.0f;
			return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
		}

		__host__ __device__ __forceinline__
			bool contains(const float3& p) const
		{
			return (p.x >= min.x && p.x <= max.x &&
				p.y >= min.y && p.y <= max.y &&
				p.z >= min.z && p.z <= max.z);
		}

		__host__ __device__ __forceinline__
			bool contains(const cuAABB& other) const
		{
			return (other.min.x >= min.x && other.max.x <= max.x &&
				other.min.y >= min.y && other.max.y <= max.y &&
				other.min.z >= min.z && other.max.z <= max.z);
		}

		__host__ __device__ __forceinline__
			float3 center() const
		{
			return make_float3((min.x + max.x) * 0.5f,
				(min.y + max.y) * 0.5f,
				(min.z + max.z) * 0.5f);
		}

		__host__ __device__ __forceinline__ static
			cuAABB merge(const cuAABB& a, const cuAABB& b)
		{
			cuAABB out;
			out.min = make_float3(fminf(a.min.x, b.min.x),
				fminf(a.min.y, b.min.y),
				fminf(a.min.z, b.min.z));
			out.max = make_float3(fmaxf(a.max.x, b.max.x),
				fmaxf(a.max.y, b.max.y),
				fmaxf(a.max.z, b.max.z));
			return out;
		}

		__host__ __device__ __forceinline__
			float3 lengths() const
		{
			return make_float3(
				max.x - min.x,
				max.y - min.y,
				max.z - min.z);
		}

		__host__ __device__ __forceinline__ static float Distance2(const cuAABB& aabb, const float3& p)
		{
			float3 clamped;
			clamped.x = fmaxf(aabb.min.x, fminf(p.x, aabb.max.x));
			clamped.y = fmaxf(aabb.min.y, fminf(p.y, aabb.max.y));
			clamped.z = fmaxf(aabb.min.z, fminf(p.z, aabb.max.z));
			return LengthSq(p - clamped);
		}
	};
}
