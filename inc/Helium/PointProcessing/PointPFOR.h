#pragma once

#include <Helium/PointProcessing/PointProcessor.h>

namespace PointProcessing
{
	class PFOR : public PointProcessor
	{
	public:
		PFOR();
		virtual ~PFOR() override = default;
		virtual void Process(const PointProcessorParameters& parameters) override;
	};
}