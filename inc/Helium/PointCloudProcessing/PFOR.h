#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

class PFOR : public PointCloudProcessor
{
public:
	PFOR();
	virtual ~PFOR() override = default;
	virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
};
