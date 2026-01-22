#include <Copper/OperatorCollection/CuOperatorPointCloudKDE.h>
#include <Copper/OperatorCollection/CuOperatorCommonDevice.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuSparseDataBlock.h>
#include <Copper/CuTransferFunction.h>

//#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
//#include <thrust/fill.h>
//#include <thrust/iterator/zip_iterator.h>
//#include <thrust/tuple.h>
#include <thrust/pair.h>

void CuOperatorPointCloudKDE::Execute(const CuOperatorParameters& params, std::vector<float>& result)
{
	thrust::device_vector<float> deviceResult;

	CuPointCloud* pointCloud = params.GetParameter("pointCloud", static_cast<CuPointCloud*>(nullptr));
	float bandwidth = params.GetParameter("bandwidth", 1.0f);

	if (!pointCloud)
	{
		printf("CuOperatorPointCloudKDE: pointCloud parameter is missing.\n");

		return;
	}

	CuSparseDataBlock* sparseBlock = params.GetParameter("sparseDataBlock", static_cast<CuSparseDataBlock*>(nullptr));
	if (!sparseBlock)
	{
		printf("CuOperatorPointCloudKDE: sparseDataBlock parameter is missing.\n");

		return;
	}

    if (pointCloud->size() == 0)
    {
        return;
    }

    int numPoints = (int)pointCloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    deviceResult.resize(numPoints);
    float searchRadius = bandwidth * 3.0f;

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(pointCloud->points.data()),
        thrust::raw_pointer_cast(sparseBlock->cellStartIndices.data()),
        thrust::raw_pointer_cast(sparseBlock->cellEndIndices.data()),
        thrust::raw_pointer_cast(deviceResult.data()),
        numPoints, searchRadius, 1, sparseBlock->cellSize, sparseBlock->gridSize, sparseBlock->worldOrigin);
    cudaDeviceSynchronize();

    auto minmax = thrust::minmax_element(deviceResult.begin(), deviceResult.end());

    //// Transfer Function »ý¼º
    //CuTransferFunction tf(*minmax.first, *minmax.second);
    //tf.SetJet();
    //tf.invert = true; // ¹Ðµµ ³·À½=Low=»¡°­

    //applyTransferFunctionKernel << <numBlocks, blockSize >> > (
    //    (uchar3*)thrust::raw_pointer_cast(pointCloud->colors.data()),
    //    thrust::raw_pointer_cast(deviceResult.data()),
    //    numPoints, tf);
    //cudaDeviceSynchronize();

    result.resize(deviceResult.size());
    thrust::copy(deviceResult.begin(), deviceResult.end(), result.begin());

    return;
}
