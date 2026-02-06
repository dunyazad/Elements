#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <robin_hood/robin_hood.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <Helium/IVisualDebugging.h>
using VD = IVisualDebugging;

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

namespace VDB
{
	template <typename T>
	class VoxelDataBase;

	static constexpr float TSDF_TRUNC_DIST = 1.0f;
	static constexpr float MAX_WEIGHT = 100.0f;

	using BlockID = uint32_t;
	static constexpr BlockID INVALID_BLOCK = 0xFFFFFFFF;

	__device__ inline unsigned short atomicAddUShort(unsigned short* address, unsigned short val)
	{
		unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
		unsigned int selectors[] = { 0, 16 };
		unsigned int sel = selectors[((size_t)address & 2) >> 1];
		unsigned int old, assumed, sum, new_val;

		old = *base_address;

		do
		{
			assumed = old;
			unsigned short old_us = (unsigned short)((old >> sel) & 0xFFFF);
			unsigned short sum_us = old_us + val;

			if (old_us + val > 0xFFFF) sum_us = 0xFFFF;

			new_val = (old & ~(0xFFFF << sel)) | (sum_us << sel);
			old = atomicCAS(base_address, assumed, new_val);
		} while (assumed != old);

		return (unsigned short)((old >> sel) & 0xFFFF);
	}

	__device__ inline float atomicMinFloat(float* addr, float val)
	{
		int* addr_as_int = (int*)addr;
		int old = *addr_as_int, assumed;

		while (val < __int_as_float(old))
		{
			assumed = old;
			old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
			if (assumed == old)
			{
				break;
			}
		}
		return __int_as_float(old);
	}

	__device__ inline uint32_t StrongHash(uint64_t key, uint32_t maxBlocks)
	{
		key ^= key >> 33;
		key *= 0xff51afd7ed558ccdULL;
		key ^= key >> 33;
		key *= 0xc4ceb9fe1a85ec53ULL;
		key ^= key >> 33;
		return static_cast<uint32_t>(key % maxBlocks);
	}

#pragma region Primitive Types
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

	struct Vector3b
	{
		uint8_t x, y, z;

		__host__ __device__
			inline Vector3b() : x(0), y(0), z(0) {}

		__host__ __device__
			inline Vector3b(uint8_t _x, uint8_t _y, uint8_t _z) : x(_x), y(_y), z(_z) {}

		__host__ __device__
			inline Vector3b(const Eigen::Vector3b& other)
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&other);
			x = ptr[0]; y = ptr[1]; z = ptr[2];
		}

		__host__ __device__
			inline operator Eigen::Vector3b& () { return *reinterpret_cast<Eigen::Vector3b*>(this); }
	};

	union Vector3f
	{
		struct { float data[3]; };
		struct { float x, y, z; };

		__host__ __device__
			inline Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}

		__host__ __device__
			inline Vector3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

		__host__ __device__
			inline Vector3f(const Eigen::Vector3f& other)
		{
			const float* ptr = reinterpret_cast<const float*>(&other);
			x = ptr[0]; y = ptr[1]; z = ptr[2];
		}

		__host__ __device__
			inline operator Eigen::Vector3f& () { return *reinterpret_cast<Eigen::Vector3f*>(this); }
	};

	union Vector4f
	{
		struct { float data[4]; };
		struct { float x, y, z, w; };

		__host__ __device__
			inline Vector4f() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}

		__host__ __device__
			inline Vector4f(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}

		__host__ __device__
			inline Vector4f(const Eigen::Vector4f& other)
		{
			const float* ptr = reinterpret_cast<const float*>(&other);
			x = ptr[0]; y = ptr[1]; z = ptr[2]; w = ptr[3];
		}

		__host__ __device__
			inline operator Eigen::Vector4f& () { return *reinterpret_cast<Eigen::Vector4f*>(this); }
	};

	struct Matrix3f
	{
		float data[9];

		__host__ __device__
			inline Matrix3f()
		{
			for (int i = 0; i < 9; ++i) data[i] = 0.0f;
		}

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

		template<int R, int C>
		__host__ __device__
			inline auto block(int startRow, int startCol) const
		{
			if constexpr (R == 3 && C == 1)
			{
				Vector3f res;
				res.x = (*this)(startRow + 0, startCol);
				res.y = (*this)(startRow + 1, startCol);
				res.z = (*this)(startRow + 2, startCol);
				return res;
			}
		}
	};

	struct Matrix4f
	{
		float data[16];

		__host__ __device__
			inline Matrix4f()
		{
			for (int i = 0; i < 16; ++i) data[i] = 0.0f;
		}

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

		template<int R, int C>
		__host__ __device__
			inline auto block(int startRow, int startCol) const
		{
			if constexpr (R == 3 && C == 3)
			{
				Matrix3f res;
				for (int j = 0; j < 3; ++j)
				{
					for (int i = 0; i < 3; ++i)
					{
						res(i, j) = (*this)(startRow + i, startCol + j);
					}
				}
				return res;
			}
			else if constexpr (R == 3 && C == 1)
			{
				Vector3f res;
				res.x = (*this)(startRow + 0, startCol);
				res.y = (*this)(startRow + 1, startCol);
				res.z = (*this)(startRow + 2, startCol);
				return res;
			}
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
#pragma endregion

#pragma region CuPointCloud
	struct PickResult
	{
		float distance;
		int index;
		float3 position;
	};

	struct CuPointCloud
	{
		float3 aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		float3 aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		thrust::device_vector<float3> points;
		thrust::device_vector<float3> normals;
		thrust::device_vector<uchar3> colors;
		thrust::device_vector<bool> isAlive;

		robin_hood::unordered_flat_map<std::string, thrust::device_vector<float>> customFloatAttributes;

		CuPointCloud() {}
		CuPointCloud(size_t n) { resize(n); }

		size_t size() const { return points.size(); }
		void resize(size_t n)
		{
			points.resize(n);
			normals.resize(n);
			colors.resize(n);
			isAlive.resize(n);
		}

		void clear()
		{
			points.clear();
			normals.clear();
			colors.clear();
			isAlive.clear();
			customFloatAttributes.clear();
		}

		void FromHostVectors(
			const std::vector<float3>& h_points,
			const std::vector<float3>& h_normals,
			const std::vector<uchar3>& h_colors,
			const float3& h_aabbMin,
			const float3& h_aabbMax)
		{
			size_t n = h_points.size();
			resize(n);

			if (!h_points.empty())
				thrust::copy(h_points.begin(), h_points.end(), points.begin());

			if (!h_normals.empty())
				thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
			else
				thrust::fill(normals.begin(), normals.end(), make_float3(0.0f, 0.0f, 1.0f));

			if (!h_colors.empty())
				thrust::copy(h_colors.begin(), h_colors.end(), colors.begin());
			else
				thrust::fill(colors.begin(), colors.end(), make_uchar3(255, 255, 255));

			thrust::fill(isAlive.begin(), isAlive.end(), true);

			aabbMin = h_aabbMin;
			aabbMax = h_aabbMax;
		}

		void FromHostPointers(
			const float3* h_points,
			const float3* h_normals,
			const uchar3* h_colors,
			size_t numPoints,
			const float3& h_aabbMin,
			const float3& h_aabbMax)
		{
			resize(numPoints);

			if (h_points)
				thrust::copy(h_points, h_points + numPoints, points.begin());

			if (h_normals)
				thrust::copy(h_normals, h_normals + numPoints, normals.begin());
			else
				thrust::fill(normals.begin(), normals.end(), make_float3(0.0f, 0.0f, 1.0f));

			if (h_colors)
				thrust::copy(h_colors, h_colors + numPoints, colors.begin());
			else
				thrust::fill(colors.begin(), colors.end(), make_uchar3(255, 255, 255));

			thrust::fill(isAlive.begin(), isAlive.end(), true);

			aabbMin = h_aabbMin;
			aabbMax = h_aabbMax;
		}

		void FromHostPointers(
			const float3* h_points,
			const float3* h_normals,
			const float4* h_colors,
			size_t numPoints,
			const float3& h_aabbMin,
			const float3& h_aabbMax)
		{
			resize(numPoints);

			if (h_points)
				thrust::copy(h_points, h_points + numPoints, points.begin());

			if (h_normals)
				thrust::copy(h_normals, h_normals + numPoints, normals.begin());
			else
				thrust::fill(normals.begin(), normals.end(), make_float3(0.0f, 0.0f, 1.0f));

			if (h_colors) {
				std::vector<uchar3> temp_colors(numPoints);
				for (size_t i = 0; i < numPoints; i++)
				{
					auto& c = h_colors[i];
					temp_colors[i] = {
						(unsigned char)(c.x * 255.0f),
						(unsigned char)(c.y * 255.0f),
						(unsigned char)(c.z * 255.0f)
					};
				}
				thrust::copy(temp_colors.begin(), temp_colors.end(), colors.begin());
			}
			else {
				thrust::fill(colors.begin(), colors.end(), make_uchar3(255, 255, 255));
			}

			thrust::fill(isAlive.begin(), isAlive.end(), true);

			aabbMin = h_aabbMin;
			aabbMax = h_aabbMax;
		}

		void ToHostVectors(
			std::vector<float3>& h_points,
			std::vector<float3>& h_normals,
			std::vector<uchar3>& h_colors)
		{
			size_t aliveCount = thrust::count(isAlive.begin(), isAlive.end(), true);

			h_points.resize(aliveCount);
			h_normals.resize(aliveCount);
			h_colors.resize(aliveCount);

			if (aliveCount == 0) return;

			thrust::device_vector<float3> d_temp_p(aliveCount);
			thrust::device_vector<float3> d_temp_n(aliveCount);
			thrust::device_vector<uchar3> d_temp_c(aliveCount);

			thrust::copy_if(points.begin(), points.end(), isAlive.begin(), d_temp_p.begin(), cuda::std::identity());
			thrust::copy_if(normals.begin(), normals.end(), isAlive.begin(), d_temp_n.begin(), cuda::std::identity());
			thrust::copy_if(colors.begin(), colors.end(), isAlive.begin(), d_temp_c.begin(), cuda::std::identity());

			thrust::copy(d_temp_p.begin(), d_temp_p.end(), h_points.begin());
			thrust::copy(d_temp_n.begin(), d_temp_n.end(), h_normals.begin());
			thrust::copy(d_temp_c.begin(), d_temp_c.end(), h_colors.begin());
		}

		void ToHostVectors(
			std::vector<float3>& h_points,
			std::vector<float3>& h_normals,
			std::vector<float4>& h_colors)
		{
			size_t aliveCount = thrust::count(isAlive.begin(), isAlive.end(), true);

			h_points.resize(aliveCount);
			h_normals.resize(aliveCount);
			h_colors.resize(aliveCount);

			if (aliveCount == 0) return;

			thrust::device_vector<float3> d_temp_p(aliveCount);
			thrust::device_vector<float3> d_temp_n(aliveCount);
			thrust::device_vector<uchar3> d_temp_c(aliveCount);

			thrust::copy_if(points.begin(), points.end(), isAlive.begin(), d_temp_p.begin(), cuda::std::identity());
			thrust::copy_if(normals.begin(), normals.end(), isAlive.begin(), d_temp_n.begin(), cuda::std::identity());
			thrust::copy_if(colors.begin(), colors.end(), isAlive.begin(), d_temp_c.begin(), cuda::std::identity());

			thrust::copy(d_temp_p.begin(), d_temp_p.end(), h_points.begin());
			thrust::copy(d_temp_n.begin(), d_temp_n.end(), h_normals.begin());

			std::vector<uchar3> temp_h_colors(aliveCount);
			thrust::copy(d_temp_c.begin(), d_temp_c.end(), temp_h_colors.begin());

			for (size_t i = 0; i < aliveCount; ++i)
			{
				uchar3 c = temp_h_colors[i];
				h_colors[i] = {
					(float)c.x / 255.0f,
					(float)c.y / 255.0f,
					(float)c.z / 255.0f,
					1.0f
				};
			}
		}

		struct PickTransformOp
		{
			float3 origin;
			float3 dir;
			float thresholdSq;

			__host__ __device__
				PickTransformOp(float3 o, float3 d, float t)
				: origin(o), dir(d), thresholdSq(t* t) {
			}

			__host__ __device__
				PickResult operator()(const thrust::tuple<int, float3, bool>& t) const
			{
				int idx = thrust::get<0>(t);
				float3 p = thrust::get<1>(t);
				bool alive = thrust::get<2>(t);

				if (!alive) return { FLT_MAX, -1, make_float3(0,0,0) };

				// 1. 벡터 P - Origin
				float3 v = make_float3(p.x - origin.x, p.y - origin.y, p.z - origin.z);

				// 2. Ray 방향으로의 투영 거리 (t)
				float t_proj = v.x * dir.x + v.y * dir.y + v.z * dir.z;

				// 카메라 뒤에 있는 점은 무시
				if (t_proj < 0.0f) return { FLT_MAX, -1, make_float3(0,0,0) };

				// 3. Ray 상의 가장 가까운 점 (Closest Point on Ray)
				float3 cp = make_float3(
					origin.x + t_proj * dir.x,
					origin.y + t_proj * dir.y,
					origin.z + t_proj * dir.z
				);

				// 4. Ray와 포인트 사이의 수직 거리 제곱 계산
				float dx = p.x - cp.x;
				float dy = p.y - cp.y;
				float dz = p.z - cp.z;
				float distSq = dx * dx + dy * dy + dz * dz;

				// 허용 오차(반지름) 밖이면 무시
				if (distSq > thresholdSq) return { FLT_MAX, -1, make_float3(0,0,0) };

				// 허용 오차 안이라면 결과 반환
				// [수정] 실제 포인트 위치 p를 함께 반환
				return { t_proj, idx, p };
			}
		};

		// 두 결과 중 더 가까운(depth가 작은) 것을 선택하는 Functor
		struct PickReduceOp
		{
			__host__ __device__
				PickResult operator()(const PickResult& a, const PickResult& b) const
			{
				return (a.distance < b.distance) ? a : b;
			}
		};

		PickResult Pick(const float3& rayOrigin, const float3& rayDir, float tolerance)
		{
			if (points.empty()) return { -1, -1, make_float3(0,0,0) };

			// 1. Iterator 준비 (Index, Point, IsAlive를 묶어서 전달)
			auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(
				thrust::counting_iterator<int>(0),
				points.begin(),
				isAlive.begin()
			));

			// 2. Transform Reduce 실행
			PickResult initVal = { FLT_MAX, -1, make_float3(0,0,0) };

			PickResult result = thrust::transform_reduce(
				zip_iter,
				zip_iter + points.size(),
				PickTransformOp(rayOrigin, rayDir, tolerance),
				initVal,
				PickReduceOp()
			);

			return result;
		}
	};
#pragma endregion

	struct CuCellStats
	{
		float3 cellMin;
		float3 cellMax;
		int pointCount;
		float3 pointCentroid;
		float3 avgNormal;
		float3 pcaNormal;
	};

	struct CuSparseCells
	{
		int3 gridSize;
		int numberOfCells = 0;
		float cellSize = 0.0f;
		float3 worldOrigin;

		thrust::device_vector<int> hashCodes;
		thrust::device_vector<int> cellStartIndices;
		thrust::device_vector<int> cellEndIndices;

		CuSparseCells();

		void Build(CuPointCloud* cloud);
		void Build(CuPointCloud* cloud, float cellSize);

		void ApplyClustering(CuPointCloud* cloud, unsigned int* d_outLabels, float clusterDistance = 0.1f);

		thrust::device_vector<float> ApplySOR(CuPointCloud* cloud, int k = 30, float stdDevMult = 1.0f);

		thrust::device_vector<float> ApplyPFOR(CuPointCloud* cloud, int k = 30, float distanceThreshold = 0.085f);

		thrust::device_vector<float> ApplyNND(CuPointCloud* cloud, int k = 30);

		thrust::device_vector<float> ApplyLDE(CuPointCloud* cloud, float radius);

		//thrust::device_vector<float> ApplyKDE(CuPointCloud* cloud, float bandwidth);

		std::vector<std::pair<float3, float3>> GetActiveCellBounds();

		void ColorizePointsByCell(CuPointCloud* cloud);

		std::vector<CuCellStats> GetActiveCellStats(CuPointCloud* cloud);

		void ApplyEdgePreservingSmoothing(
			CuPointCloud* cloud,
			float radius,
			float factor,
			float edgeThreshold,
			int iterations);

		void ApplyEnergySmoothing(CuPointCloud* cloud, float radius, float dataWeight, float smoothWeight, int iterations);

	private:
		float computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier);
	};

#pragma region Domain Data Types
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
#pragma endregion

#pragma region Voxel Data Base
	template <typename T>
	struct VoxelBlock
	{
		static constexpr int BLOCK_SIZE = 8;
		static constexpr int VOXELS_PER_BLOCK = 512;
		T voxels[VOXELS_PER_BLOCK];
		uint32_t lastTouchedFrameId = 0xFFFFFFFF;
		uint16_t activeVoxelCount = 0;
	};

	template <typename T>
	__device__ inline float GetVoxelValueSafe(VoxelDataBase<T>& db, const Vector3f& pos)
	{
		Voxel* v = db.GetVoxel(pos);
		if (v != nullptr && v->valueCount > 0)
		{
			return v->value;
		}
		return NAN;
	}

	template <typename T>
	__global__ void InsertKernel(VoxelDataBase<T> db, Matrix4f rt, const Vector3f* points, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= count)
		{
			return;
		}

		Vector3f p = rt.Transform(points[idx]);

		Voxel* v = db.GetOrCreateVoxel(p);

		if (v != nullptr)
		{
			atomicAdd(&(v->value), 1.0f);

			if (v->valueCount < 65535)
			{
				atomicAddUShort(&(v->valueCount), 1);
			}

			v->color = colors[idx];

			BlockID bid = db.GetOrCreateBlockSlot(p);
			if (bid != INVALID_BLOCK)
			{
				db.d_blocks[bid].lastTouchedFrameId = frameId;
			}
		}
	}

	template <typename T>
	__global__ void TSDFIntegrateKernel(VoxelDataBase<T> db, Matrix4f rt, const Vector3f* points, const Vector3f* normals, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
	{
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= count)
		{
			return;
		}

		Vector3f p_local = points[index];
		Vector3f n_local = normals[index];
		Vector3b color = colors[index];

		Vector3f p_world = rt.Transform(p_local);
		Vector3f n_world = {
			rt.data[0] * n_local.x + rt.data[4] * n_local.y + rt.data[8] * n_local.z,
			rt.data[1] * n_local.x + rt.data[5] * n_local.y + rt.data[9] * n_local.z,
			rt.data[2] * n_local.x + rt.data[6] * n_local.y + rt.data[10] * n_local.z
		};

		float n_len = rsqrtf(n_world.x * n_world.x + n_world.y * n_world.y + n_world.z * n_world.z + 1e-10f);
		n_world.x *= n_len; n_world.y *= n_len; n_world.z *= n_len;

		float voxel_size = blockSize / 8.0f;
		float inv_voxel_size = 1.0f / voxel_size;
		float trunc_dist = voxel_size * 5.0f;

		Vector3f start_pos = {
			p_world.x - n_world.x * trunc_dist,
			p_world.y - n_world.y * trunc_dist,
			p_world.z - n_world.z * trunc_dist
		};

		int curr_x = __float2int_rd(start_pos.x * inv_voxel_size);
		int curr_y = __float2int_rd(start_pos.y * inv_voxel_size);
		int curr_z = __float2int_rd(start_pos.z * inv_voxel_size);

		int step_x = (n_world.x > 0) ? 1 : -1;
		int step_y = (n_world.y > 0) ? 1 : -1;
		int step_z = (n_world.z > 0) ? 1 : -1;

		auto calc_t_max = [&](float pos, float dir, int step, int curr) {
			if (fabsf(dir) < 1e-7f) return 1e30f;
			float border = (float)(curr + (step > 0 ? 1 : 0)) * voxel_size;
			return (border - pos) / dir;
			};

		float t_max_x = calc_t_max(start_pos.x, n_world.x, step_x, curr_x);
		float t_max_y = calc_t_max(start_pos.y, n_world.y, step_y, curr_y);
		float t_max_z = calc_t_max(start_pos.z, n_world.z, step_z, curr_z);

		float t_delta_x = (fabsf(n_world.x) > 1e-7f) ? fabsf(voxel_size / n_world.x) : 1e30f;
		float t_delta_y = (fabsf(n_world.y) > 1e-7f) ? fabsf(voxel_size / n_world.y) : 1e30f;
		float t_delta_z = (fabsf(n_world.z) > 1e-7f) ? fabsf(voxel_size / n_world.z) : 1e30f;

		float max_t = 2.0f * trunc_dist;
		float t = 0.0f;

		while (t <= max_t)
		{
			Vector3f voxel_center = { (curr_x + 0.5f) * voxel_size, (curr_y + 0.5f) * voxel_size, (curr_z + 0.5f) * voxel_size };
			Voxel* voxel_ptr = db.GetOrCreateVoxel(voxel_center);

			if (voxel_ptr != nullptr)
			{
				float dist = (voxel_center.x - p_world.x) * n_world.x +
					(voxel_center.y - p_world.y) * n_world.y +
					(voxel_center.z - p_world.z) * n_world.z;

				// [수정됨] valueCount atomicAddUShort로 안전하게 증가
				unsigned short old_w_us = atomicAddUShort(&(voxel_ptr->valueCount), 1);

				float old_w = fminf((float)old_w_us, MAX_WEIGHT);
				float new_w = old_w + 1.0f;

				float weight_factor = old_w / new_w;
				float new_factor = 1.0f / new_w;

				// 1. TSDF 값 통합
				voxel_ptr->value = (voxel_ptr->value * weight_factor) + (dist * new_factor);

				// 2. Color 통합
				voxel_ptr->color.x = (uint8_t)((float)voxel_ptr->color.x * weight_factor + (float)color.x * new_factor + 0.5f);
				voxel_ptr->color.y = (uint8_t)((float)voxel_ptr->color.y * weight_factor + (float)color.y * new_factor + 0.5f);
				voxel_ptr->color.z = (uint8_t)((float)voxel_ptr->color.z * weight_factor + (float)color.z * new_factor + 0.5f);

				// 3. Normal 통합
				Vector3f blended_n = {
					voxel_ptr->normal.x * weight_factor + n_world.x * new_factor,
					voxel_ptr->normal.y * weight_factor + n_world.y * new_factor,
					voxel_ptr->normal.z * weight_factor + n_world.z * new_factor
				};
				float bn_len = rsqrtf(blended_n.x * blended_n.x + blended_n.y * blended_n.y + blended_n.z * blended_n.z + 1e-10f);
				voxel_ptr->normal = { blended_n.x * bn_len, blended_n.y * bn_len, blended_n.z * bn_len };

				VoxelBlock<T>* block = db.GetVoxelBlock(voxel_center);
				if (block)
				{
					block->lastTouchedFrameId = frameId;
				}
			}

			if (t_max_x < t_max_y)
			{
				if (t_max_x < t_max_z) { t = t_max_x; t_max_x += t_delta_x; curr_x += step_x; }
				else { t = t_max_z; t_max_z += t_delta_z; curr_z += step_z; }
			}
			else
			{
				if (t_max_y < t_max_z) { t = t_max_y; t_max_y += t_delta_y; curr_y += step_y; }
				else { t = t_max_z; t_max_z += t_delta_z; curr_z += step_z; }
			}
		}
	}

	template <typename T>
	__global__ void ESDFIntegrateKernel(
		VoxelDataBase<T> db,
		Matrix4f rt,
		const Vector3f* points,
		const Vector3f* normals,
		const Vector3b* colors,
		uint32_t count,
		float blockSize,
		uint32_t frameId)
	{
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= count)
		{
			return;
		}

		Vector3f p_world = rt.Transform(points[index]);
		Vector3f n_world = rt.TransformNormal(normals[index]);
		Vector3b color = colors[index];

		Vector3f cam_pos = { rt.data[12], rt.data[13], rt.data[14] };
		Vector3f ray_vec = { p_world.x - cam_pos.x, p_world.y - cam_pos.y, p_world.z - cam_pos.z };
		float ray_len = sqrtf(ray_vec.x * ray_vec.x + ray_vec.y * ray_vec.y + ray_vec.z * ray_vec.z);

		if (ray_len < 1e-6f)
		{
			return;
		}
		Vector3f ray_dir = { ray_vec.x / ray_len, ray_vec.y / ray_len, ray_vec.z / ray_len };

		float voxel_size = blockSize / 8.0f;
		float trunc_dist = voxel_size * 4.0f;

		float t_start = fmaxf(0.0f, ray_len - trunc_dist);
		float t_end = ray_len + trunc_dist;
		float step = voxel_size * 0.5f;

		for (float t = t_start; t <= t_end; t += step)
		{
			Vector3f sample_pos = {
				cam_pos.x + ray_dir.x * t,
				cam_pos.y + ray_dir.y * t,
				cam_pos.z + ray_dir.z * t
			};

			Voxel* voxel_ptr = db.GetOrCreateVoxel(sample_pos);
			if (voxel_ptr != nullptr)
			{
				float current_sdf = ray_len - t;
				float esdf_val = current_sdf;
				unsigned short current_w = voxel_ptr->valueCount;

				if (current_w == 0)
				{
					if (atomicAddUShort(&(voxel_ptr->valueCount), 1) == 0)
					{
						voxel_ptr->value = esdf_val;
						voxel_ptr->color = color;
						voxel_ptr->normal = n_world;
					}
					else
					{
						goto UPDATE_VOXEL;
					}
				}
				else if (current_sdf >= -trunc_dist)
				{
				UPDATE_VOXEL:

					if (current_w > 5)
					{
						if (fabsf(voxel_ptr->value - esdf_val) > trunc_dist * 0.8f)
						{
							continue;
						}
					}

					unsigned short old_w_us = atomicAddUShort(&(voxel_ptr->valueCount), 1);

					float old_w = fminf((float)old_w_us, 15.0f);
					float alpha = 1.0f / (old_w + 1.0f);

					voxel_ptr->value = voxel_ptr->value * (1.0f - alpha) + esdf_val * alpha;

					voxel_ptr->value = fmaxf(-trunc_dist, fminf(trunc_dist, voxel_ptr->value));

					atomicMinFloat(&(voxel_ptr->value), esdf_val);

					voxel_ptr->color.x = (uint8_t)((float)voxel_ptr->color.x * (1.0f - alpha) + (float)color.x * alpha + 0.5f);
					voxel_ptr->color.y = (uint8_t)((float)voxel_ptr->color.y * (1.0f - alpha) + (float)color.y * alpha + 0.5f);
					voxel_ptr->color.z = (uint8_t)((float)voxel_ptr->color.z * (1.0f - alpha) + (float)color.z * alpha + 0.5f);

					voxel_ptr->normal.x = voxel_ptr->normal.x * (1.0f - alpha) + n_world.x * alpha;
					voxel_ptr->normal.y = voxel_ptr->normal.y * (1.0f - alpha) + n_world.y * alpha;
					voxel_ptr->normal.z = voxel_ptr->normal.z * (1.0f - alpha) + n_world.z * alpha;
				}

				VoxelBlock<T>* block = db.GetVoxelBlock(sample_pos);
				if (block)
				{
					block->lastTouchedFrameId = frameId;
				}
			}
		}
	}

	template <typename T>
	__global__ void ExtractKernel(VoxelDataBase<T> db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.maxBlockCount) return;

		uint64_t key = db.d_hashTable[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

		Morton64 bKey(key);
		Vector3f bc = bKey.ToPosition(blockSize);
		float vSize = blockSize / 8.0f;

		for (int i = 0; i < 512; ++i)
		{
			Voxel& v = db.d_blocks[slot].voxels[i];
			if (v.valueCount > 0)
			{
				uint32_t idx = atomicAdd(count, 1);
				if (idx < maxOut)
				{
					int lz = i / 64;
					int ly = (i % 64) / 8;
					int lx = i % 8;

					out[idx].position = {
						(bc.x - blockSize * 0.5f) + (lx + 0.5f) * vSize,
						(bc.y - blockSize * 0.5f) + (ly + 0.5f) * vSize,
						(bc.z - blockSize * 0.5f) + (lz + 0.5f) * vSize
					};
					out[idx].weight = (float)v.valueCount;
					out[idx].color[0] = v.color.x;
					out[idx].color[1] = v.color.y;
					out[idx].color[2] = v.color.z;
				}
			}
		}
	}

	template <typename T>
	__global__ void ExtractZeroCrossingKernel(VoxelDataBase<T> db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;

		if (slot >= db.maxBlockCount)
		{
			return;
		}

		uint64_t key = db.d_hashTable[slot];

		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
		{
			return;
		}

		Morton64 bKey(key);
		Vector3f bc = bKey.ToPosition(blockSize);
		Vector3f origin = { bc.x - blockSize * 0.5f, bc.y - blockSize * 0.5f, bc.z - blockSize * 0.5f };
		float voxelSize = blockSize / 8.0f;
		float eps = voxelSize;

		for (int lz = 0; lz < 8; ++lz)
		{
			for (int ly = 0; ly < 8; ++ly)
			{
				for (int lx = 0; lx < 8; ++lx)
				{
					int idx0 = (lz << 6) | (ly << 3) | lx;
					Voxel& v0 = db.d_blocks[slot].voxels[idx0];

					if (v0.valueCount < 5)
					{
						continue;
					}

					Vector3f p0 = {
						origin.x + (lx + 0.5f) * voxelSize,
						origin.y + (ly + 0.5f) * voxelSize,
						origin.z + (lz + 0.5f) * voxelSize
					};

					int neighbors[3] = { 1, 8, 64 };
					for (int axis = 0; axis < 3; ++axis)
					{
						bool isInternal = false;
						if (axis == 0 && lx < 7) isInternal = true;
						else if (axis == 1 && ly < 7) isInternal = true;
						else if (axis == 2 && lz < 7) isInternal = true;

						Voxel* v1_ptr = nullptr;
						Vector3f p1 = p0;

						if (axis == 0) p1.x += voxelSize;
						else if (axis == 1) p1.y += voxelSize;
						else p1.z += voxelSize;

						if (isInternal)
						{
							v1_ptr = &db.d_blocks[slot].voxels[idx0 + neighbors[axis]];
						}
						else
						{
							v1_ptr = db.GetVoxel(p1);
						}

						if (v1_ptr != nullptr && v1_ptr->valueCount >= 5)
						{
							float v0v = v0.value;
							float v1v = v1_ptr->value;

							if (v0v * v1v < 0.0f)
							{
								uint32_t outIdx = atomicAdd(count, 1);
								if (outIdx < maxOut)
								{
									float mu = -v0v / (v1v - v0v);
									mu = fminf(fmaxf(mu, 0.0f), 1.0f);

									Vector3f interpPos = {
										p0.x + mu * (p1.x - p0.x),
										p0.y + mu * (p1.y - p0.y),
										p0.z + mu * (p1.z - p0.z)
									};

									out[outIdx].position = interpPos;

									float val_x_p = GetVoxelValueSafe<T>(db, { interpPos.x + eps, interpPos.y, interpPos.z });
									float val_x_m = GetVoxelValueSafe<T>(db, { interpPos.x - eps, interpPos.y, interpPos.z });
									float val_y_p = GetVoxelValueSafe<T>(db, { interpPos.x, interpPos.y + eps, interpPos.z });
									float val_y_m = GetVoxelValueSafe<T>(db, { interpPos.x, interpPos.y - eps, interpPos.z });
									float val_z_p = GetVoxelValueSafe<T>(db, { interpPos.x, interpPos.y, interpPos.z + eps });
									float val_z_m = GetVoxelValueSafe<T>(db, { interpPos.x, interpPos.y, interpPos.z - eps });

									Vector3f finalNormal;
									bool hasGradient = !isnan(val_x_p) && !isnan(val_x_m) &&
										!isnan(val_y_p) && !isnan(val_y_m) &&
										!isnan(val_z_p) && !isnan(val_z_m);

									if (hasGradient)
									{
										Vector3f grad = { val_x_p - val_x_m, val_y_p - val_y_m, val_z_p - val_z_m };
										float len = sqrtf(grad.x * grad.x + grad.y * grad.y + grad.z * grad.z);
										if (len > 1e-6f)
										{
											finalNormal = { grad.x / len, grad.y / len, grad.z / len };
										}
										else
										{
											finalNormal = { 0, 1, 0 };
										}
									}
									else
									{
										Vector3f n0 = v0.normal;
										Vector3f n1 = v1_ptr->normal;
										Vector3f blended = {
											n0.x + mu * (n1.x - n0.x),
											n0.y + mu * (n1.y - n0.y),
											n0.z + mu * (n1.z - n0.z)
										};
										float len = sqrtf(blended.x * blended.x + blended.y * blended.y + blended.z * blended.z);
										if (len > 1e-6f)
											finalNormal = { blended.x / len, blended.y / len, blended.z / len };
										else
											finalNormal = n0;
									}

									out[outIdx].normal = finalNormal;

									out[outIdx].color[0] = (uint8_t)(v0.color.x + mu * (static_cast<float>(v1_ptr->color.x) - v0.color.x));
									out[outIdx].color[1] = (uint8_t)(v0.color.y + mu * (static_cast<float>(v1_ptr->color.y) - v0.color.y));
									out[outIdx].color[2] = (uint8_t)(v0.color.z + mu * (static_cast<float>(v1_ptr->color.z) - v0.color.z));

									out[outIdx].weight = (float)v0.valueCount;
								}
							}
						}
					}
				}
			}
		}
	}

	template <typename T>
	class VoxelDataBase
	{
	public:
		VoxelBlock<T>* d_blocks = nullptr;
		uint64_t* d_hashTable = nullptr;
		uint32_t* d_blockCount = nullptr;
		uint32_t maxBlockCount = 0;

		void Allocate(uint32_t maxBlocks)
		{
			maxBlockCount = maxBlocks;
			cudaMalloc(&d_blocks, sizeof(VoxelBlock<T>) * maxBlocks);
			cudaMalloc(&d_hashTable, sizeof(uint64_t) * maxBlocks);
			cudaMalloc(&d_blockCount, sizeof(uint32_t));
			cudaMemset(d_blocks, 0, sizeof(VoxelBlock<T>) * maxBlocks);
			cudaMemset(d_hashTable, 0, sizeof(uint64_t) * maxBlocks);
			cudaMemset(d_blockCount, 0, sizeof(uint32_t));
		}

		void Free()
		{
			if (d_blocks) cudaFree(d_blocks);
			if (d_hashTable) cudaFree(d_hashTable);
			if (d_blockCount) cudaFree(d_blockCount);
			d_blocks = nullptr;
			d_hashTable = nullptr;
			d_blockCount = nullptr;
		}

		void IntegrateTSDF(const Matrix4f& rt, const Vector3f* d_points, const Vector3f* d_normals, const Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId)
		{
			int threads = 256;
			int blocks = (count + threads - 1) / threads;
			TSDFIntegrateKernel << <blocks, threads >> > (*this, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
			cudaDeviceSynchronize();
		}

		void IntegrateESDF(
			const Matrix4f& rt,
			const Vector3f* d_points,
			const Vector3f* d_normals,
			const Vector3b* d_colors,
			uint32_t count,
			float blockSize,
			uint32_t frameId)
		{
			int threads = 256;
			int blocks = (count + threads - 1) / threads;

			ESDFIntegrateKernel << <blocks, threads >> > (*this, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
			cudaDeviceSynchronize();
		}

		uint32_t ExtractActiveVoxelsToHost(float blockSize, ExtractedVoxel* hostBuffer, uint32_t maxOut)
		{
			ExtractedVoxel* d_out;
			uint32_t* d_cnt;
			cudaMalloc(&d_out, sizeof(ExtractedVoxel) * maxOut);
			cudaMalloc(&d_cnt, sizeof(uint32_t));
			cudaMemset(d_cnt, 0, sizeof(uint32_t));

			int threadsPerBlock = 256;
			int blocksPerGrid = (maxBlockCount + threadsPerBlock - 1) / threadsPerBlock;
			ExtractKernel << <blocksPerGrid, threadsPerBlock >> > (*this, blockSize, d_out, d_cnt, maxOut);
			cudaDeviceSynchronize();

			uint32_t res;
			cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			uint32_t copyAmt = (res > maxOut) ? maxOut : res;
			cudaMemcpy(hostBuffer, d_out, sizeof(ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);

			cudaFree(d_out);
			cudaFree(d_cnt);
			return res;
		}

		uint32_t ExtractZeroCrossingVoxelsToHost(float blockSize, ExtractedVoxel* hostBuffer, uint32_t maxOut)
		{
			ExtractedVoxel* d_out;
			uint32_t* d_cnt;
			cudaMalloc(&d_out, sizeof(ExtractedVoxel) * maxOut);
			cudaMalloc(&d_cnt, sizeof(uint32_t));
			cudaMemset(d_cnt, 0, sizeof(uint32_t));

			int threadsPerBlock = 256;
			int blocksPerGrid = (maxBlockCount + threadsPerBlock - 1) / threadsPerBlock;
			ExtractZeroCrossingKernel << <blocksPerGrid, threadsPerBlock >> > (*this, blockSize, d_out, d_cnt, maxOut);
			cudaDeviceSynchronize();

			uint32_t res;
			cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			uint32_t copyAmt = (res > maxOut) ? maxOut : res;
			cudaMemcpy(hostBuffer, d_out, sizeof(ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);

			cudaFree(d_out);
			cudaFree(d_cnt);
			return res;
		}

		void VisualizeVoxelsBatch_Surface(VoxelDataBase<T>& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
		{
			uint32_t maxOut = 10000000;
			std::vector<ExtractedVoxel> hostOut(maxOut);

			uint32_t finalCnt = voxelDb.ExtractZeroCrossingVoxelsToHost(blockSize, hostOut.data(), maxOut);

			if (finalCnt > 0)
			{
				uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;

				std::vector<float3> surfacePoints;
				std::vector<float3> surfaceNormals;
				std::vector<float4> surfaceColors;
				surfacePoints.reserve(limit);
				surfaceNormals.reserve(limit);
				surfaceColors.reserve(limit);

				float3 aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
				float3 aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

				for (uint32_t i = 0; i < limit; i++)
				{
					surfacePoints.emplace_back(make_float3(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z));

					surfaceNormals.emplace_back(make_float3(hostOut[i].normal.x, hostOut[i].normal.y, hostOut[i].normal.z));

					surfaceColors.emplace_back(make_float4(
						hostOut[i].color[0] / 255.f,
						hostOut[i].color[1] / 255.f,
						hostOut[i].color[2] / 255.f,
						1.f));

					aabbMin.x = std::min(aabbMin.x, hostOut[i].position.x);
					aabbMin.y = std::min(aabbMin.y, hostOut[i].position.y);
					aabbMin.z = std::min(aabbMin.z, hostOut[i].position.z);

					aabbMax.x = std::max(aabbMax.x, hostOut[i].position.x);
					aabbMax.y = std::max(aabbMax.y, hostOut[i].position.y);
					aabbMax.z = std::max(aabbMax.z, hostOut[i].position.z);
				}

				VD::AddDiskBatch("ZeroCrossingSurface", surfacePoints, surfaceNormals, 0.05f, 16, surfaceColors, true);

				//std::vector<uint64_t> hostHashTable(maxBlocks);
				//cudaMemcpy(hostHashTable.data(), voxelDb.d_hashTable, sizeof(uint64_t) * maxBlocks, cudaMemcpyDeviceToHost);

				//std::vector<float3> blockCenters;
				//blockCenters.reserve(activeBlocksCount);

				//for (uint32_t i = 0; i < maxBlocks; ++i)
				//{
				//    uint64_t key = hostHashTable[i];
				//    if (key != 0 && key != 0xFFFFFFFFFFFFFFFFULL)
				//    {
				//        Morton64 morton(key);
				//        Vector3f blockPos = morton.ToPosition(blockSize);
				//        blockCenters.emplace_back(make_float3(blockPos.x, blockPos.y, blockPos.z));
				//    }
				//}
				//VD::AddWiredBoxBatch("SparseDataBlocks", blockCenters, make_float3(blockSize, blockSize, blockSize), make_float4(0, 1, 0, 0.1f));
			}

			printf("\n>>> [Iso-surface 리포트]\n");
			printf("    - 추출된 Zero-crossing 정점 : %u 개\n", finalCnt);
			printf("    - 해시 적재율             : %.2f%% (%u / %u)\n",
				(double)activeBlocksCount / maxBlocks * 100.0, activeBlocksCount, maxBlocks);
		}

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

		__device__ inline VoxelBlock<T>* GetVoxelBlock(const Vector3f& position)
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
#pragma endregion

	typedef struct CamInfo_
	{
		float cfx;
		float cfy;
		float ccx;
		float ccy;
		int cx;
		int cy;
		int img_width;
		int img_height;
		double R[9];
		double T[3];
		Vector3f dlpPos;
		Vector3f camPos;
		Matrix3f invMatTilt;
		Matrix3f matTilt;

		Matrix4f GetViewMatrix(const CamInfo_& info)
		{
			Matrix4f view = Matrix4f::Identity();
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					view(i, j) = (float)info.R[i * 3 + j];
				}
				view(i, 3) = (float)info.T[i];
			}
			return view;
		}

		Matrix4f GetProjectionMatrix(const CamInfo_& info, float n, float f)
		{
			Matrix4f proj = Matrix4f::Zero();
			proj(0, 0) = 2.0f * info.cfx / info.img_width;
			proj(0, 2) = 1.0f - (2.0f * info.ccx / info.img_width);
			proj(1, 1) = 2.0f * info.cfy / info.img_height;
			proj(1, 2) = (2.0f * info.ccy / info.img_height) - 1.0f;
			proj(2, 2) = -(f + n) / (f - n);
			proj(2, 3) = -(2.0f * f * n) / (f - n);
			proj(3, 2) = -1.0f;
			return proj;
		}
	} CamInfo_;

	void ExecuteAppVVV()
	{
		VoxelDataBase<Voxel> voxelDb;
		voxelDb.Allocate(160000);

		std::ifstream ifs("D:\\Resources\\Default\\Patches.bin", std::ios::binary);
		if (!ifs.is_open())
		{
			voxelDb.Free();
			return;
		}

		CamInfo_ cam;
		Matrix4f camRT;
		ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
		ifs.read(reinterpret_cast<char*>(camRT.data), sizeof(float) * 16);

		size_t numberOfPatches = 0;
		ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));

		Vector3f* d_points0 = nullptr;
		Vector3f* d_normals0 = nullptr;
		Vector3b* d_colors0 = nullptr;
		unsigned int numberOfPoints0 = 0;

		Vector3f* d_points45 = nullptr;
		Vector3f* d_normals45 = nullptr;
		Vector3b* d_colors45 = nullptr;
		unsigned int numberOfPoints45 = 0;

		CUDA_TS(PatchTotal);
		for (size_t i = 0; i < numberOfPatches; i++)
		{
			CUDA_TS(patch);

			size_t patchIndex = 0;
			ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t));
			Matrix4f rt0;
			ifs.read(reinterpret_cast<char*>(rt0.data), sizeof(float) * 16);
			Vector3f aabbMin0, aabbMax0;
			ifs.read(reinterpret_cast<char*>(aabbMin0.data), sizeof(float) * 3);
			ifs.read(reinterpret_cast<char*>(aabbMax0.data), sizeof(float) * 3);
			Matrix4f rt45;
			ifs.read(reinterpret_cast<char*>(rt45.data), sizeof(float) * 16);
			Vector3f aabbMin45, aabbMax45;
			ifs.read(reinterpret_cast<char*>(aabbMin45.data), sizeof(float) * 3);
			ifs.read(reinterpret_cast<char*>(aabbMax45.data), sizeof(float) * 3);

			size_t numPts0 = 0;
			ifs.read(reinterpret_cast<char*>(&numPts0), sizeof(size_t));

			std::vector<Vector3f> pts0(numPts0), normals0(numPts0);
			std::vector<Vector3b> colors0(numPts0);
			ifs.read(reinterpret_cast<char*>(pts0.data()), sizeof(Vector3f) * numPts0);
			ifs.read(reinterpret_cast<char*>(normals0.data()), sizeof(Vector3f) * numPts0);
			ifs.read(reinterpret_cast<char*>(colors0.data()), sizeof(Vector3b) * numPts0);

			size_t numPts45 = 0;
			ifs.read(reinterpret_cast<char*>(&numPts45), sizeof(size_t));
			std::vector<Vector3f> pts45(numPts45), normals45(numPts45);
			std::vector<Vector3b> colors45(numPts45);
			ifs.read(reinterpret_cast<char*>(pts45.data()), sizeof(Vector3f) * numPts45);
			ifs.read(reinterpret_cast<char*>(normals45.data()), sizeof(Vector3f) * numPts45);
			//ifs.read(reinterpret_cast<char*>(colors45.data), sizeof(Vector3b) * numPts45);

			Vector3f sensorPos = rt0.block<3, 1>(0, 3);
			Matrix3f rot = rt0.block<3, 3>(0, 0);

			Matrix4f gpuMatrix0 = rt0;

			if (numberOfPoints0 < numPts0)
			{
				if (d_points0) cudaFree(d_points0);
				if (d_normals0) cudaFree(d_normals0);
				if (d_colors0) cudaFree(d_colors0);
				numberOfPoints0 = (unsigned int)numPts0 * 2;
				cudaMalloc(&d_points0, sizeof(Vector3f) * numberOfPoints0);
				cudaMalloc(&d_normals0, sizeof(Vector3f) * numberOfPoints0);
				cudaMalloc(&d_colors0, sizeof(Vector3b) * numberOfPoints0);

				printf("Allocated GPU 0 buffers for %u points.\n", numberOfPoints0);
			}

			cudaMemcpy(d_points0, pts0.data(), sizeof(Vector3f) * numPts0, cudaMemcpyHostToDevice);
			cudaMemcpy(d_normals0, normals0.data(), sizeof(Vector3f) * numPts0, cudaMemcpyHostToDevice);
			cudaMemcpy(d_colors0, colors0.data(), sizeof(Vector3b) * numPts0, cudaMemcpyHostToDevice);

			voxelDb.IntegrateTSDF(
				gpuMatrix0,
				d_points0,
				d_normals0,
				d_colors0,
				(uint32_t)numPts0,
				0.8f,
				(unsigned int)i);

			Matrix4f gpuMatrix45 = rt45;

			if (numberOfPoints45 < numPts45)
			{
				if (d_points45) cudaFree(d_points45);
				if (d_normals45) cudaFree(d_normals45);
				if (d_colors45) cudaFree(d_colors45);
				numberOfPoints45 = (unsigned int)numPts45 * 2;
				cudaMalloc(&d_points45, sizeof(Vector3f) * numberOfPoints45);
				cudaMalloc(&d_normals45, sizeof(Vector3f) * numberOfPoints45);
				cudaMalloc(&d_colors45, sizeof(Vector3b) * numberOfPoints45);

				printf("Allocated GPU 45 buffers for %u points.\n", numberOfPoints45);
			}

			cudaMemcpy(d_points45, pts45.data(), sizeof(Vector3f) * numPts45, cudaMemcpyHostToDevice);
			cudaMemcpy(d_normals45, normals45.data(), sizeof(Vector3f) * numPts45, cudaMemcpyHostToDevice);
			cudaMemcpy(d_colors45, colors45.data(), sizeof(Vector3b) * numPts45, cudaMemcpyHostToDevice);

			voxelDb.IntegrateTSDF(
				gpuMatrix45,
				d_points45,
				d_normals45,
				d_colors45,
				(uint32_t)numPts45,
				0.8f,
				(unsigned int)i);

			printf("[%5zd] =-=-= ", i);
			CUDA_TE(patch);
		}

		printf("numberOfPatches: %zu\n", numberOfPatches);
		
		CUDA_TE(PatchTotal);

		if (d_points0) cudaFree(d_points0);
		if (d_normals0) cudaFree(d_normals0);
		if (d_colors0) cudaFree(d_colors0);

		if (d_points45) cudaFree(d_points45);
		if (d_normals45) cudaFree(d_normals45);
		if (d_colors45) cudaFree(d_colors45);

		uint32_t activeBlocksCount = 0;
		cudaMemcpy(&activeBlocksCount, voxelDb.d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		//VisualizeVoxelsBatch_Surface_Clustering_Filtered(voxelDb, maxBlocks, 0.8f, activeBlocksCount);
		voxelDb.VisualizeVoxelsBatch_Surface(voxelDb, 160000, 0.8f, activeBlocksCount);

		voxelDb.Free();
	}

	/*
	void VisualizeVoxelsBatch_Surface_Clustering_Filtered(VoxelDataBase<Voxel>& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
	{
		uint32_t maxOut = 70000000;
		std::vector<ExtractedVoxel> hostOut(maxOut);

		// 1. Zero-crossing 포인트 추출
		uint32_t finalCnt = voxelDb.ExtractZeroCrossingVoxelsToHost(blockSize, hostOut.data(), maxOut);

		if (finalCnt > 0)
		{
			uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;

			std::vector<Vector3f> surfacePoints;
			std::vector<Vector3f> surfaceNormals;
			std::vector<Vector4f> surfaceColors;
			surfacePoints.reserve(limit);
			surfaceNormals.reserve(limit);
			surfaceColors.reserve(limit);

			float3 aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
			float3 aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

			for (uint32_t i = 0; i < limit; i++)
			{
				surfacePoints.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);
				surfaceNormals.emplace_back(hostOut[i].normal.x, hostOut[i].normal.y, hostOut[i].normal.z);
				surfaceColors.emplace_back(
					hostOut[i].color[0] / 255.0f,
					hostOut[i].color[1] / 255.0f,
					hostOut[i].color[2] / 255.0f,
					1.0f
				);

				aabbMin.x = std::min(aabbMin.x, hostOut[i].position.x);
				aabbMin.y = std::min(aabbMin.y, hostOut[i].position.y);
				aabbMin.z = std::min(aabbMin.z, hostOut[i].position.z);
				aabbMax.x = std::max(aabbMax.x, hostOut[i].position.x);
				aabbMax.y = std::max(aabbMax.y, hostOut[i].position.y);
				aabbMax.z = std::max(aabbMax.z, hostOut[i].position.z);
			}

			size_t rawCount = surfacePoints.size();

			// 2. GPU PointCloud 생성 및 클러스터링
			CuPointCloud cloud;
			cloud.FromHostPointers(
				(float3*)surfacePoints.data(),
				(float3*)surfaceNormals.data(),
				(float4*)surfaceColors.data(),
				(uint32_t)rawCount,
				aabbMin,
				aabbMax);

			CuSparseCells cellGrid;
			cellGrid.cellSize = 0.3f;
			cellGrid.Build(&cloud, cellGrid.cellSize);

			unsigned int* labelsDevice = nullptr;
			cudaMalloc(&labelsDevice, rawCount * sizeof(unsigned int));

			// Execute 함수와 동일한 임계값 적용
			cellGrid.ApplyClustering(&cloud, labelsDevice, 0.125f);

			// 3. 결과 다운로드 및 통계 처리 (Execute 로직 반영)
			std::vector<unsigned int> labelsHost(rawCount);
			std::vector<float3> pointsHost(rawCount);
			std::vector<float3> normalsHost(rawCount);
			std::vector<uchar3> colorsHost(rawCount);

			cudaMemcpy(labelsHost.data(), labelsDevice, rawCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaMemcpy(pointsHost.data(), (const float3*)thrust::raw_pointer_cast(cloud.points.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(normalsHost.data(), (const float3*)thrust::raw_pointer_cast(cloud.normals.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(colorsHost.data(), (const uchar3*)thrust::raw_pointer_cast(cloud.colors.data()), rawCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

			struct ClusterStats
			{
				int count = 0;
				unsigned int originalLabel = 0;
			};
			std::map<unsigned int, ClusterStats> clusterMap;

			for (size_t i = 0; i < rawCount; ++i)
			{
				auto& stats = clusterMap[labelsHost[i]];
				stats.count++;
				stats.originalLabel = labelsHost[i];
			}

			// 4. 크기순 정렬 및 랭킹 부여
			std::vector<ClusterStats> sortedClusters;
			sortedClusters.reserve(clusterMap.size());
			for (auto const& [label, stats] : clusterMap)
			{
				if (stats.count >= 3) sortedClusters.push_back(stats);
			}

			std::sort(sortedClusters.begin(), sortedClusters.end(),
				[](const ClusterStats& a, const ClusterStats& b) { return a.count > b.count; });

			unsigned int maxClusterId = sortedClusters.empty() ? 0xFFFFFFFF : sortedClusters[0].originalLabel;

			// 5. 가장 큰 클러스터(Rank 0)만 선별하여 시각화
			std::vector<Eigen::Vector3f> mainPos;
			std::vector<Eigen::Vector3f> mainNorm;
			std::vector<Eigen::Vector4f> mainCol;

			aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
			aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

			for (size_t i = 0; i < rawCount; ++i)
			{
				if (labelsHost[i] == maxClusterId && maxClusterId != 0xFFFFFFFF)
				{
					mainPos.emplace_back(pointsHost[i].x, pointsHost[i].y, pointsHost[i].z);

					Eigen::Vector3f n(normalsHost[i].x, normalsHost[i].y, normalsHost[i].z);
					if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
					mainNorm.push_back(n);

					mainCol.emplace_back(
						(float)colorsHost[i].x / 255.0f,
						(float)colorsHost[i].y / 255.0f,
						(float)colorsHost[i].z / 255.0f,
						1.0f
					);

					aabbMin.x = std::min(aabbMin.x, pointsHost[i].x);
					aabbMin.y = std::min(aabbMin.y, pointsHost[i].y);
					aabbMin.z = std::min(aabbMin.z, pointsHost[i].z);
					aabbMax.x = std::max(aabbMax.x, pointsHost[i].x);
					aabbMax.y = std::max(aabbMax.y, pointsHost[i].y);
					aabbMax.z = std::max(aabbMax.z, pointsHost[i].z);
				}
			}

			if (!mainPos.empty())
			{
				//VD::AddSphereBatch("PointCloud", mainPos, mainNorm, 0.05f, mainCol);
			}

			cudaFree(labelsDevice);

			CuPointCloud filterd;
			filterd.FromHostPointers(
				(float3*)mainPos.data(),
				(float3*)mainNorm.data(),
				(float4*)mainCol.data(),
				(uint32_t)mainPos.size(),
				aabbMin,
				aabbMax);

			CuSparseCells filterCellGrid;
			filterCellGrid.cellSize = 0.3f;
			filterCellGrid.Build(&filterd, filterCellGrid.cellSize);

			//        filterCellGrid.ApplyEdgePreservingSmoothing(
			//            &filterd,
			//            0.5f,   // radius: 주변 이웃 탐색 반경
			//            0.7f,   // factor: 스무딩 강도 (0.0 ~ 1.0)
			//            0.15f,  // edgeThreshold: 엣지 보존 임계값 (0.0 ~ 1.0)
			//			  30);    // iterations: 반복 횟수 

			filterCellGrid.ApplyEnergySmoothing(
				&filterd,
				0.5f,   // radius: 주변 이웃 탐색 반경
				0.1f,   // dataWeight: 데이터 적합도 가중치
				0.9f,   // smoothWeight: 스무딩 가중치
				30);    // iterations: 반복 횟수


			std::vector<float3> smoothPoints(mainPos.size());
			std::vector<float3> smoothNormals(mainPos.size());
			std::vector<uchar3> smoothColors(mainPos.size());
			cudaMemcpy(smoothPoints.data(), (const float3*)thrust::raw_pointer_cast(filterd.points.data()), mainPos.size() * sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(smoothNormals.data(), (const float3*)thrust::raw_pointer_cast(filterd.normals.data()), mainPos.size() * sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(smoothColors.data(), (const uchar3*)thrust::raw_pointer_cast(filterd.colors.data()), mainPos.size() * sizeof(uchar3), cudaMemcpyDeviceToHost);
			std::vector<Eigen::Vector3f> smoothPos;
			std::vector<Eigen::Vector3f> smoothNorm;
			std::vector<Eigen::Vector4f> smoothCol;
			smoothPos.reserve(mainPos.size());
			smoothNorm.reserve(mainPos.size());
			smoothCol.reserve(mainPos.size());
			for (size_t i = 0; i < mainPos.size(); ++i)
			{
				smoothPos.emplace_back(smoothPoints[i].x, smoothPoints[i].y, smoothPoints[i].z);
				Eigen::Vector3f n(smoothNormals[i].x, smoothNormals[i].y, smoothNormals[i].z);
				if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
				smoothNorm.push_back(n);
				smoothCol.emplace_back(
					(float)smoothColors[i].x / 255.0f,
					(float)smoothColors[i].y / 255.0f,
					(float)smoothColors[i].z / 255.0f,
					1.0f
				);
			}
			VD::AddSphereBatch("SmoothedPointCloud", smoothPos, smoothNorm, 0.05f, smoothCol);
		}
	}
	*/
}

#include "Apps.h"
class AppVVV : public App
{
public:
	virtual void Execute() override
	{
		VDB::ExecuteAppVVV();
	}
};
REGISTER_APP(AppVVV, "AppVVV");
