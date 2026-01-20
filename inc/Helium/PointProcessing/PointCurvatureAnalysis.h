#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class CurvatureAnalysis : public PointProcessor
	{
	public:
		CurvatureAnalysis();
		virtual ~CurvatureAnalysis() override = default;
		virtual void Process(const PointProcessorParameters& parameters) override;
	};
}
