#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

namespace PointCloudProcessing
{
	class NormalDeviation : public PointCloudProcessor
	{
	public:
		NormalDeviation();
		virtual ~NormalDeviation() override = default;
		virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
	};
}
