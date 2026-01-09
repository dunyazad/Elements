#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

class SOR : public PointCloudProcessor
{
public:
	SOR();
	virtual ~SOR() override = default;
	virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
};
