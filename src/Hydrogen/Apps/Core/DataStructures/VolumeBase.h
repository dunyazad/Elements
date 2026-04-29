#pragma once

#include <Core/Common/DeviceCommon.h>
#include <Core/Common/DevicePrimitiveTypes.h>
#include <Core/Common/CUDAMath.h>

struct cached_allocator;

namespace Huvitz
{
	namespace Core
	{
		class IntegrationParameters
		{
		public:
			virtual ~IntegrationParameters()
			{
			}

			//float blockSize = 0.8f;
			//float truncationDistance = 1.0f;
			//float maxWeight = 100.0f;
		};

		class ExtractionParameters
		{
		public:
			virtual ~ExtractionParameters()
			{
			}
			//float blockSize = 0.8f;
			//float truncationDistance = 1.0f;
		};

		class VolumeBase
		{
		public:
			VolumeBase() {}
			virtual ~VolumeBase() {}
			virtual void Clear(cached_allocator* allocator, CUstream_st* stream) = 0;

			virtual void Integrate(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream) = 0;
			virtual void Extract(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream) = 0;
		};
	}
}
