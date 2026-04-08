#pragma once

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/Common/CUDAMath.h>

#include <Core/DataStructures/VolumeBase.h>

#include <Helium/Serialization.hpp>

#include <DataFrameIO/DataFrameIO.hpp>

namespace Huvitz
{
	using BlockID = uint32_t;
	static constexpr BlockID INVALID_BLOCK = 0xFFFFFFFF;

	static constexpr float TSDF_TRUNC_DIST = 1.0f;
	static constexpr float MAX_WEIGHT = 100.0f;

	struct DirectionalVoxel
	{
		float    value;
		float    weight;
		Vector3f normal;
		Vector3b color;
		uint8_t  validMask;
		__device__ __host__ inline bool HasDirection(int d) const
		{
			return (validMask >> d) & 1u;
		}
		__device__ __host__ inline float GetValue(int d) const
		{
			return HasDirection(d) ? value : FLT_MAX;
		}
	};

	class VoxelCache
	{

	};
}