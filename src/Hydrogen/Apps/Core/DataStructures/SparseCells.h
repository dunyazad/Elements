#pragma once

#include <Core/Common/DeviceCommon.h>

namespace Huvitz
{
	class PCD;

	class SparseCells
	{
	public:
		SparseCells();

		void Build(PCD* cloud, CUstream_st* stream);
		void Build(PCD* cloud, float cellSize, CUstream_st* stream);
		void Build(float3* d_points, size_t numberOfPoints, float cellSize, CUstream_st* stream);

		void ApplyClustering(PCD* cloud, unsigned int* d_outLabels, float clusterDistance, CUstream_st* stream);
		void ApplyClustering(float3* d_points, size_t numberOfPoints, unsigned int* d_outLabels, float clusterDistance = 0.125f, CUstream_st* stream = nullptr);

		thrust::device_vector<int> hashCodes;
		thrust::device_vector<int> cellStartIndices;
		thrust::device_vector<int> cellEndIndices;

	private:
		int3 gridSize;
		int numberOfCells = 0;
		float cellSize = 0.0f;
		float3 worldOrigin;

		float computeAutoCellSize(float3* positions, size_t size, float multiplier, CUstream_st* stream);
		float computeAutoCellSize(const thrust::device_vector<float3>& points, float multiplier, CUstream_st* stream);
	};
}
