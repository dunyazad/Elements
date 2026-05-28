#pragma once

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/DataStructures/CudaHashMap.cuh>

struct cached_allocator;

namespace Huvitz
{
	namespace Core
	{
		class OperationPointCloudClustering
		{
		public:
			OperationPointCloudClustering(uint64_t capacityhint = 1024, cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);
			~OperationPointCloudClustering();

			OperationPointCloudClustering(const OperationPointCloudClustering&) = delete;
			OperationPointCloudClustering& operator=(const OperationPointCloudClustering&) = delete;
			void Resize(uint64_t capacityhint, cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr);

		private:
			unsigned int allocatedSizeHint = 0;
		};
	}
}
