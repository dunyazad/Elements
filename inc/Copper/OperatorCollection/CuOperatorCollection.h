#pragma once

#include <Copper/CopperCommon.h>

#include <Copper/OperatorCollection/CuOperatorPointCloudKDE.h>
#include <Copper/OperatorCollection/CuOperatorPointCloudLDE.h>
#include <Copper/OperatorCollection/CuOperatorPointCloudPFOR.h>
#include <Copper/OperatorCollection/CuOperatorPointCloudROR.h>
#include <Copper/OperatorCollection/CuOperatorPointCloudSOR.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

struct COPPER_API CuOperatorCollection
{
};