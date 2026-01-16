#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class ROR : public PointProcessor
	{
	public:
		ROR();
		virtual ~ROR() override = default;
		virtual std::vector<uint8_t> Process(const PointProcessorParameters& parameters) override;
	};
}