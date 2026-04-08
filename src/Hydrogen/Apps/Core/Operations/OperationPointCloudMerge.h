#pragma once

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/DataStructures/CudaHashMap.cuh>

struct cached_allocator;

namespace Huvitz
{
	class OperationPointCloudMerge
	{
	public:
		OperationPointCloudMerge(uint64_t capacityhint = 1024, cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);
		~OperationPointCloudMerge();

		OperationPointCloudMerge(const OperationPointCloudMerge&) = delete;
		OperationPointCloudMerge& operator=(const OperationPointCloudMerge&) = delete;

		void Resize(uint64_t capacityhint, cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);

		void Clear(cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);

		void Merge(
			const float3* points,
			const float3* normals,
			const float4* colors,
			unsigned int numberOfPoints,
			cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);

		void Merge(
			const float3* points,
			const float3* normals,
			const float4* colors,
			unsigned int numberOfPoints,
			cuAABB aabb,
			cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);

		void Extract(
			float3* outPoints,
			float3* outNormals,
			float4* outColors,
			unsigned int& outNumberOfPoints,
			cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);

		struct Point
		{
			float3 position{ 0.0f, 0.0f, 0.0f };
			float3 normal{ 0.0f, 0.0f, 0.0f };
			float4 color{ 0.0f, 0.0f, 0.0f, 0.0f };
			unsigned int count = 0;
		};

	protected:
		CudaHashMap<uint64_t, Point> pointHashMap;
		unsigned int allocatedSizeHint = 0;
	};
}
