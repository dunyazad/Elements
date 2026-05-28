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
	namespace Core
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
			__host__ __device__ float operator()(float s) const
			{
				return s * s;
			}
		};

		struct Float3ToPair
		{
			__host__ __device__ thrust::pair<float3, float3> operator()(const float3& v) const
			{
				return thrust::make_pair(v, v);
			}
		};

		struct Float3MinMax
		{
			__host__ __device__ thrust::pair<float3, float3> operator()(
				const thrust::pair<float3, float3>& a,
				const thrust::pair<float3, float3>& b) const
			{
				float3 minVal = { fminf(a.first.x, b.first.x), fminf(a.first.y, b.first.y), fminf(a.first.z, b.first.z) };
				float3 maxVal = { fmaxf(a.second.x, b.second.x), fmaxf(a.second.y, b.second.y), fmaxf(a.second.z, b.second.z) };
				return thrust::make_pair(minVal, maxVal);
			}
		};

		// ---------- Vector3b ----------

		struct Vector3b
		{
			uint8_t data[3];

			__host__ __device__ inline Vector3b() : data{ 0, 0, 0 } {}
			__host__ __device__ inline Vector3b(uint8_t _x, uint8_t _y, uint8_t _z) : data{ _x, _y, _z } {}

			__host__ __device__ inline Vector3b(const Eigen::Vector3b& other)
			{
				const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&other);
				data[0] = ptr[0]; data[1] = ptr[1]; data[2] = ptr[2];
			}

			__host__ __device__ inline uint8_t& x() { return data[0]; }
			__host__ __device__ inline uint8_t& y() { return data[1]; }
			__host__ __device__ inline uint8_t& z() { return data[2]; }
			__host__ __device__ inline const uint8_t& x() const { return data[0]; }
			__host__ __device__ inline const uint8_t& y() const { return data[1]; }
			__host__ __device__ inline const uint8_t& z() const { return data[2]; }

			__host__ __device__ inline operator Eigen::Vector3b& () { return *reinterpret_cast<Eigen::Vector3b*>(this); }

			__host__ __device__ static inline Vector3b Zero() { return { 0, 0, 0 }; }
		};

		// ---------- Vector3f ----------

		struct Vector3f
		{
			float data[3];

			__host__ __device__ inline Vector3f() : data{ 0.0f, 0.0f, 0.0f } {}
			__host__ __device__ inline Vector3f(float _x, float _y, float _z) : data{ _x, _y, _z } {}

			__host__ __device__ inline Vector3f(const Eigen::Vector3f& other)
			{
				const float* ptr = reinterpret_cast<const float*>(&other);
				data[0] = ptr[0]; data[1] = ptr[1]; data[2] = ptr[2];
			}

			__host__ __device__ inline float& x() { return data[0]; }
			__host__ __device__ inline float& y() { return data[1]; }
			__host__ __device__ inline float& z() { return data[2]; }
			__host__ __device__ inline const float& x() const { return data[0]; }
			__host__ __device__ inline const float& y() const { return data[1]; }
			__host__ __device__ inline const float& z() const { return data[2]; }

			__host__ __device__ inline operator Eigen::Vector3f& () { return *reinterpret_cast<Eigen::Vector3f*>(this); }

			__host__ __device__ inline float squaredNorm() const
			{
				return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
			}

			__host__ __device__ inline float norm() const { return sqrtf(squaredNorm()); }

			__host__ __device__ inline Vector3f normalized() const
			{
				float n = norm();
				const float epsilon = 1e-8f;
				if (n > epsilon)
					return Vector3f{ data[0] / n, data[1] / n, data[2] / n };
				return Vector3f{ 0.0f, 0.0f, 0.0f };
			}

			__host__ __device__ inline float Dot(const Vector3f& o) const
			{
				return data[0] * o.data[0] + data[1] * o.data[1] + data[2] * o.data[2];
			}

			__host__ __device__ inline Vector3f Cross(const Vector3f& o) const
			{
				return Vector3f{
					data[1] * o.data[2] - data[2] * o.data[1],
					data[2] * o.data[0] - data[0] * o.data[2],
					data[0] * o.data[1] - data[1] * o.data[0]
				};
			}

			__host__ __device__ inline Vector3f operator+(const Vector3f& o) const { return { data[0] + o.data[0], data[1] + o.data[1], data[2] + o.data[2] }; }
			__host__ __device__ inline Vector3f operator-(const Vector3f& o) const { return { data[0] - o.data[0], data[1] - o.data[1], data[2] - o.data[2] }; }
			__host__ __device__ inline Vector3f operator*(const Vector3f& o) const { return { data[0] * o.data[0], data[1] * o.data[1], data[2] * o.data[2] }; }
			__host__ __device__ inline Vector3f operator/(const Vector3f& o) const { return { data[0] / o.data[0], data[1] / o.data[1], data[2] / o.data[2] }; }

			__host__ __device__ inline Vector3f operator*(float s) const { return { data[0] * s, data[1] * s, data[2] * s }; }
			__host__ __device__ inline Vector3f operator/(float s) const { float inv = 1.0f / s; return { data[0] * inv, data[1] * inv, data[2] * inv }; }
			__host__ __device__ inline Vector3f operator-() const { return { -data[0], -data[1], -data[2] }; }

			__host__ __device__ inline Vector3f& operator+=(const Vector3f& o) { data[0] += o.data[0]; data[1] += o.data[1]; data[2] += o.data[2]; return *this; }
			__host__ __device__ inline Vector3f& operator-=(const Vector3f& o) { data[0] -= o.data[0]; data[1] -= o.data[1]; data[2] -= o.data[2]; return *this; }
			__host__ __device__ inline Vector3f& operator*=(float s) { data[0] *= s; data[1] *= s; data[2] *= s; return *this; }
			__host__ __device__ inline Vector3f& operator/=(float s) { float inv = 1.0f / s; data[0] *= inv; data[1] *= inv; data[2] *= inv; return *this; }

			__host__ __device__ inline bool operator==(const Vector3f& o) const { return data[0] == o.data[0] && data[1] == o.data[1] && data[2] == o.data[2]; }
			__host__ __device__ inline bool operator!=(const Vector3f& o) const { return !(*this == o); }

			__host__ __device__ static inline Vector3f Zero() { return { 0.0f, 0.0f, 0.0f }; }
			__host__ __device__ static inline Vector3f UnitX() { return { 1.0f, 0.0f, 0.0f }; }
			__host__ __device__ static inline Vector3f UnitY() { return { 0.0f, 1.0f, 0.0f }; }
			__host__ __device__ static inline Vector3f UnitZ() { return { 0.0f, 0.0f, 1.0f }; }
		};

		__host__ __device__ inline Vector3f operator*(float s, const Vector3f& v)
		{
			return v * s;
		}

		// ---------- Vector4f ----------

		struct Vector4f
		{
			float data[4];

			__host__ __device__ inline Vector4f() : data{ 0.0f, 0.0f, 0.0f, 0.0f } {}
			__host__ __device__ inline Vector4f(float _x, float _y, float _z, float _w) : data{ _x, _y, _z, _w } {}

			__host__ __device__ inline Vector4f(const Eigen::Vector4f& other)
			{
				const float* ptr = reinterpret_cast<const float*>(&other);
				data[0] = ptr[0]; data[1] = ptr[1]; data[2] = ptr[2]; data[3] = ptr[3];
			}

			__host__ __device__ inline float& x() { return data[0]; }
			__host__ __device__ inline float& y() { return data[1]; }
			__host__ __device__ inline float& z() { return data[2]; }
			__host__ __device__ inline float& w() { return data[3]; }
			__host__ __device__ inline const float& x() const { return data[0]; }
			__host__ __device__ inline const float& y() const { return data[1]; }
			__host__ __device__ inline const float& z() const { return data[2]; }
			__host__ __device__ inline const float& w() const { return data[3]; }

			__host__ __device__ inline operator Eigen::Vector4f& () { return *reinterpret_cast<Eigen::Vector4f*>(this); }

			__host__ __device__ inline Vector4f operator+(const Vector4f& o) const { return { data[0] + o.data[0], data[1] + o.data[1], data[2] + o.data[2], data[3] + o.data[3] }; }
			__host__ __device__ inline Vector4f operator-(const Vector4f& o) const { return { data[0] - o.data[0], data[1] - o.data[1], data[2] - o.data[2], data[3] - o.data[3] }; }
			__host__ __device__ inline Vector4f operator*(const Vector4f& o) const { return { data[0] * o.data[0], data[1] * o.data[1], data[2] * o.data[2], data[3] * o.data[3] }; }
			__host__ __device__ inline Vector4f operator/(const Vector4f& o) const { return { data[0] / o.data[0], data[1] / o.data[1], data[2] / o.data[2], data[3] / o.data[3] }; }

			__host__ __device__ inline Vector4f operator*(float s) const { return { data[0] * s, data[1] * s, data[2] * s, data[3] * s }; }
			__host__ __device__ inline Vector4f operator/(float s) const { float inv = 1.0f / s; return { data[0] * inv, data[1] * inv, data[2] * inv, data[3] * inv }; }
			__host__ __device__ inline Vector4f operator-() const { return { -data[0], -data[1], -data[2], -data[3] }; }
			__host__ __device__ inline Vector4f& operator+=(const Vector4f& o) { data[0] += o.data[0]; data[1] += o.data[1]; data[2] += o.data[2]; data[3] += o.data[3]; return *this; }
			__host__ __device__ inline Vector4f& operator-=(const Vector4f& o) { data[0] -= o.data[0]; data[1] -= o.data[1]; data[2] -= o.data[2]; data[3] -= o.data[3]; return *this; }
			__host__ __device__ inline Vector4f& operator*=(float s) { data[0] *= s; data[1] *= s; data[2] *= s; data[3] *= s; return *this; }
			__host__ __device__ inline Vector4f& operator/=(float s) { float inv = 1.0f / s; data[0] *= inv; data[1] *= inv; data[2] *= inv; data[3] *= inv; return *this; }
			__host__ __device__ inline bool operator==(const Vector4f& o) const { return data[0] == o.data[0] && data[1] == o.data[1] && data[2] == o.data[2] && data[3] == o.data[3]; }
			__host__ __device__ inline bool operator!=(const Vector4f& o) const { return !(*this == o); }

			__host__ __device__ static inline Vector4f Zero() { return { 0.0f, 0.0f, 0.0f, 0.0f }; }
			__host__ __device__ static inline Vector4f UnitX() { return { 1.0f, 0.0f, 0.0f, 0.0f }; }
			__host__ __device__ static inline Vector4f UnitY() { return { 0.0f, 1.0f, 0.0f, 0.0f }; }
			__host__ __device__ static inline Vector4f UnitZ() { return { 0.0f, 0.0f, 1.0f, 0.0f }; }
			__host__ __device__ static inline Vector4f UnitW() { return { 0.0f, 0.0f, 0.0f, 1.0f }; }
		};

		// ---------- Matrix3f ----------

		struct Matrix3f
		{
			float data[9];  // col-major

			__host__ __device__ inline Matrix3f()
			{
				for (int i = 0; i < 9; ++i) data[i] = 0.0f;
			}

			__host__ __device__ inline float& operator()(int row, int col) { return data[col * 3 + row]; }
			__host__ __device__ inline const float& operator()(int row, int col) const { return data[col * 3 + row]; }

			template<int R, int C>
			__host__ __device__ inline auto block(int startRow, int startCol) const
			{
				if constexpr (R == 3 && C == 1)
				{
					Vector3f res;
					res.data[0] = (*this)(startRow + 0, startCol);
					res.data[1] = (*this)(startRow + 1, startCol);
					res.data[2] = (*this)(startRow + 2, startCol);
					return res;
				}
			}
		};

		// ---------- Matrix4f ----------

		struct Matrix4f
		{
			float data[16];  // col-major

			__host__ __device__ inline Matrix4f()
			{
				for (int i = 0; i < 16; ++i) data[i] = 0.0f;
			}

			__host__ __device__ inline Matrix4f(float* ptr)
			{
				for (int i = 0; i < 16; ++i) data[i] = ptr[i];
			}

			inline Matrix4f(const Eigen::Matrix4f& other)
			{
				const float* src = other.data();
				for (int i = 0; i < 16; ++i) data[i] = src[i];
			}

			__host__ __device__ inline Matrix4f& operator=(const Eigen::Matrix4f& other)
			{
				const float* src = other.data();
				for (int i = 0; i < 16; ++i) data[i] = src[i];
				return *this;
			}

			__host__ inline operator Eigen::Matrix4f() const
			{
				return Eigen::Map<const Eigen::Matrix4f>(data);
			}

			__host__ __device__ inline float& operator[](int index) { return data[index]; }
			__host__ __device__ inline float  operator[](int index) const { return data[index]; }

			__host__ __device__ inline float& operator()(int row, int col) { return data[col * 4 + row]; }
			__host__ __device__ inline const float& operator()(int row, int col) const { return data[col * 4 + row]; }

			__host__ __device__ static inline Matrix4f Identity()
			{
				Matrix4f mat;
				mat.data[0] = 1.0f; mat.data[5] = 1.0f; mat.data[10] = 1.0f; mat.data[15] = 1.0f;
				return mat;
			}

			__host__ __device__ static inline Matrix4f Zero() { return Matrix4f(); }

			template<int R, int C>
			__host__ __device__ inline auto block(int startRow, int startCol) const
			{
				if constexpr (R == 3 && C == 3)
				{
					Matrix3f res;
					for (int j = 0; j < 3; ++j)
						for (int i = 0; i < 3; ++i)
							res(i, j) = (*this)(startRow + i, startCol + j);
					return res;
				}
				else if constexpr (R == 3 && C == 1)
				{
					Vector3f res;
					res.data[0] = (*this)(startRow + 0, startCol);
					res.data[1] = (*this)(startRow + 1, startCol);
					res.data[2] = (*this)(startRow + 2, startCol);
					return res;
				}
			}

			__host__ __device__ inline Vector3f Transform(const Vector3f& vec) const
			{
				float rx = data[0] * vec.data[0] + data[4] * vec.data[1] + data[8] * vec.data[2] + data[12];
				float ry = data[1] * vec.data[0] + data[5] * vec.data[1] + data[9] * vec.data[2] + data[13];
				float rz = data[2] * vec.data[0] + data[6] * vec.data[1] + data[10] * vec.data[2] + data[14];
				float rw = data[3] * vec.data[0] + data[7] * vec.data[1] + data[11] * vec.data[2] + data[15];

				const float epsilon = 1e-8f;
				float invW = (fabsf(rw) > epsilon) ? (1.0f / rw) : 1.0f;
				return { rx * invW, ry * invW, rz * invW };
			}

			__host__ __device__ inline Vector3f TransformNormal(const Vector3f& vec) const
			{
				float rx = data[0] * vec.data[0] + data[4] * vec.data[1] + data[8] * vec.data[2];
				float ry = data[1] * vec.data[0] + data[5] * vec.data[1] + data[9] * vec.data[2];
				float rz = data[2] * vec.data[0] + data[6] * vec.data[1] + data[10] * vec.data[2];
				return { rx, ry, rz };
			}
		};

		// ---------- Morton64 ----------

		struct Morton64
		{
			uint64_t code = 0;
			static constexpr int AXIS_BITS = 21;
			static constexpr int32_t AXIS_BIAS = 1 << (AXIS_BITS - 1);
			static constexpr uint64_t AXIS_MASK = (1ull << AXIS_BITS) - 1ull;

			__host__ __device__ Morton64() {}
			__host__ __device__ explicit Morton64(uint64_t c) : code(c) {}

			__host__ __device__ inline Morton64(int32_t x, int32_t y, int32_t z)
			{
				uint32_t ux = static_cast<uint32_t>(x + AXIS_BIAS);
				uint32_t uy = static_cast<uint32_t>(y + AXIS_BIAS);
				uint32_t uz = static_cast<uint32_t>(z + AXIS_BIAS);
				code = Encode(ux, uy, uz);
			}

			__host__ __device__ inline Morton64(const Morton64& other) : code(other.code) {}
			__host__ __device__ inline Morton64(const volatile Morton64& other) : code(other.code) {}

			__host__ __device__ inline Morton64& operator=(const Morton64& other)
			{
				code = other.code;
				return *this;
			}

			__host__ __device__ inline void operator=(const Morton64& other) volatile
			{
				code = other.code;
			}

			__host__ __device__ inline Morton64& operator=(const volatile Morton64& other)
			{
				code = other.code;
				return *this;
			}

			__host__ __device__ inline bool operator==(const Morton64& other) const { return code == other.code; }
			__host__ __device__ inline operator uint64_t() const { return code; }

			__host__ __device__ static inline int32_t ToBlockCoord(float v, float blockSize)
			{
				return static_cast<int32_t>(floorf(v / blockSize));
			}

			__host__ __device__ static inline Morton64 FromPosition(const Vector3f& p, float blockSize)
			{
				return Morton64(
					ToBlockCoord(p.data[0], blockSize),
					ToBlockCoord(p.data[1], blockSize),
					ToBlockCoord(p.data[2], blockSize));
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

		// ---------- cuAABB ----------

		struct cuAABB
		{
			float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			__host__ __device__ __forceinline__ void expand(const float3& p)
			{
				min.x = fminf(min.x, p.x); min.y = fminf(min.y, p.y); min.z = fminf(min.z, p.z);
				max.x = fmaxf(max.x, p.x); max.y = fmaxf(max.y, p.y); max.z = fmaxf(max.z, p.z);
			}

			__host__ __device__ __forceinline__ void expand(const cuAABB& b)
			{
				min.x = fminf(min.x, b.min.x); min.y = fminf(min.y, b.min.y); min.z = fminf(min.z, b.min.z);
				max.x = fmaxf(max.x, b.max.x); max.y = fmaxf(max.y, b.max.y); max.z = fmaxf(max.z, b.max.z);
			}

			__host__ __device__ __forceinline__ float volume() const
			{
				if (min.x > max.x) return 0.0f;
				return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
			}

			__host__ __device__ __forceinline__ bool contains(const float3& p) const
			{
				return (p.x >= min.x && p.x <= max.x &&
					p.y >= min.y && p.y <= max.y &&
					p.z >= min.z && p.z <= max.z);
			}

			__host__ __device__ __forceinline__ bool contains(const cuAABB& other) const
			{
				return (other.min.x >= min.x && other.max.x <= max.x &&
					other.min.y >= min.y && other.max.y <= max.y &&
					other.min.z >= min.z && other.max.z <= max.z);
			}

			__host__ __device__ __forceinline__ float3 center() const
			{
				return make_float3(
					(min.x + max.x) * 0.5f,
					(min.y + max.y) * 0.5f,
					(min.z + max.z) * 0.5f);
			}

			__host__ __device__ __forceinline__ static cuAABB merge(const cuAABB& a, const cuAABB& b)
			{
				cuAABB out;
				out.min = make_float3(fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z));
				out.max = make_float3(fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z));
				return out;
			}

			__host__ __device__ __forceinline__ float3 lengths() const
			{
				return make_float3(max.x - min.x, max.y - min.y, max.z - min.z);
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
}
