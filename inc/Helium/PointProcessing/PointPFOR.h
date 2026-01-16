#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class PFOR : public PointProcessor
	{
	public:
		PFOR();
		virtual ~PFOR() override = default;
		virtual std::vector<uint8_t> Process(const PointProcessorParameters& parameters) override;
	};
}