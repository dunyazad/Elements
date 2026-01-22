#include <Copper/OperatorCollection/CuOperatorPointCloudLDE.h>
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

void CuOperatorPointCloudLDE::Execute(const CuOperatorParameters& params, std::vector<float>& result)
{
    thrust::device_vector<float> deviceResult;

    CuPointCloud* pointCloud = params.GetParameter("pointCloud", static_cast<CuPointCloud*>(nullptr));
    float radius = params.GetParameter("radius", 0.5f);

    if (!pointCloud)
    {
        printf("CuOperatorPointCloudLDE: pointCloud parameter is missing.\n");

        return;
    }

    CuSparseDataBlock* sparseBlock = params.GetParameter("sparseDataBlock", static_cast<CuSparseDataBlock*>(nullptr));
    if (!sparseBlock)
    {
        printf("CuOperatorPointCloudLDE: sparseDataBlock parameter is missing.\n");

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

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(pointCloud->points.data()),
        thrust::raw_pointer_cast(sparseBlock->cellStartIndices.data()),
        thrust::raw_pointer_cast(sparseBlock->cellEndIndices.data()),
        thrust::raw_pointer_cast(deviceResult.data()),
        numPoints, radius, 0, sparseBlock->cellSize, sparseBlock->gridSize, sparseBlock->worldOrigin);

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
}

thrust::device_vector<float> CuOperatorPointCloudLDE::ExecuteDevice(const CuOperatorParameters& params)
{
    thrust::device_vector<float> deviceResult;

    CuPointCloud* pointCloud = params.GetParameter("pointCloud", static_cast<CuPointCloud*>(nullptr));
    float radius = params.GetParameter("radius", 0.5f);

    if (!pointCloud)
    {
        printf("CuOperatorPointCloudLDE: pointCloud parameter is missing.\n");

        return deviceResult;
    }

    CuSparseDataBlock* sparseBlock = params.GetParameter("sparseDataBlock", static_cast<CuSparseDataBlock*>(nullptr));
    if (!sparseBlock)
    {
        printf("CuOperatorPointCloudLDE: sparseDataBlock parameter is missing.\n");

        return deviceResult;
    }

    if (pointCloud->size() == 0)
    {
        return deviceResult;
    }

    int numPoints = (int)pointCloud->size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    deviceResult.resize(numPoints);

    computeDensityKernel << <numBlocks, blockSize >> > (
        thrust::raw_pointer_cast(pointCloud->points.data()),
        thrust::raw_pointer_cast(sparseBlock->cellStartIndices.data()),
        thrust::raw_pointer_cast(sparseBlock->cellEndIndices.data()),
        thrust::raw_pointer_cast(deviceResult.data()),
        numPoints, radius, 0, sparseBlock->cellSize, sparseBlock->gridSize, sparseBlock->worldOrigin);

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

    return deviceResult;
}
