#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class NormalDeviation : public PointProcessor
	{
	public:
		NormalDeviation();
		virtual ~NormalDeviation() override = default;
		virtual std::vector<uint8_t> Process(const PointProcessorParameters& parameters) override;
	};
}