#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class SOR : public PointProcessor
	{
	public:
		SOR();
		virtual ~SOR() override = default;
		virtual void Process(const PointProcessorParameters& parameters) override;
	};
}
