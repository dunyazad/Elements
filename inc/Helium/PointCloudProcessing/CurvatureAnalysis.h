#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

class CurvatureAnalysis : public PointCloudProcessor
{
public:
	CurvatureAnalysis();
	virtual ~CurvatureAnalysis() override = default;
	virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
};
