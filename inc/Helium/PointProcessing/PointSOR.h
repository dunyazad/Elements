#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class SOR : public PointProcessor
	{
	public:
		SOR();
		virtual ~SOR() override = default;
		virtual std::vector<uint8_t> Process(const PointProcessorParameters& parameters) override;
	};
}
