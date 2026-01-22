#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

namespace PointCloudProcessing
{
	class KDE : public PointCloudProcessor
	{
	public:
		KDE();
		virtual ~KDE() override = default;
		virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;
	};
}
