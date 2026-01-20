#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class ROR : public PointProcessor
	{
	public:
		ROR();
		virtual ~ROR() override = default;
		virtual void Process(const PointProcessorParameters& parameters) override;
	};
}