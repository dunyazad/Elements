#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class CurvatureAnalysis : public PointProcessor
	{
	public:
		CurvatureAnalysis();
		virtual ~CurvatureAnalysis() override = default;
		virtual std::vector<uint8_t> Process(const PointProcessorParameters& parameters) override;
	};
}
