#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

class NormalDeviation : public PointCloudProcessor
{
public:
	NormalDeviation();
	virtual ~NormalDeviation() override = default;
	virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
};
