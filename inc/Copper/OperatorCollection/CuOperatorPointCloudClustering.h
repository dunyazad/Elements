#pragma once

#include <vector>

#include <Copper/CopperCommon.h>
#include <Copper/OperatorCollection/CuOperatorCommon.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

struct COPPER_API CuOperatorPointCloudClustering
{
	void Execute(const CuOperatorParameters& params, std::vector<uint64_t>& result);

#ifdef __CUDACC__
	thrust::device_vector<uint64_t> ExecuteDevice(const CuOperatorParameters& params);
#endif
};
