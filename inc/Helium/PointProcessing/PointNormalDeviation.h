#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class NormalDeviation : public PointProcessor
	{
	public:
		NormalDeviation();
		virtual ~NormalDeviation() override = default;
		virtual void Process(const PointProcessorParameters& parameters) override;
	};
}