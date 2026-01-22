#pragma once

#include <vector>

#include <Copper/CopperCommon.h>
#include <Copper/OperatorCollection/CuOperatorCommon.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

struct COPPER_API CuOperatorPointCloudLDE
{
	void Execute(const CuOperatorParameters& params, std::vector<float>& result);

#ifdef __CUDACC__
	thrust::device_vector<float> ExecuteDevice(const CuOperatorParameters& params);
#endif
};
