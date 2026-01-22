#pragma once

#include <vector>

#include <Copper/CopperCommon.h>
#include <Copper/OperatorCollection/CuOperatorCommon.h>

struct COPPER_API CuOperatorPointCloudKDE
{
	void Execute(const CuOperatorParameters& params, std::vector<float>& result);
};
