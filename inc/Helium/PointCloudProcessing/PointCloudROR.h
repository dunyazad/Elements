#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

namespace PointCloudProcessing
{
	class ROR : public PointCloudProcessor
	{
	public:
		ROR();
		virtual ~ROR() override = default;
		virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
	};
}
