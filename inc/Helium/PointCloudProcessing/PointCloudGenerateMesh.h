#pragma once

#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

#include <Eigen/Dense>

namespace PointCloudProcessing
{
	class Triangle
	{
	public:
		Eigen::Vector3f v[3];
		Eigen::Vector3f n[3];
		Eigen::Vector3f c[3];
	};

	class PointCloudGenerateMesh : public PointCloudProcessor
	{
	public:
		PointCloudGenerateMesh();
		virtual ~PointCloudGenerateMesh() override = default;
		virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) override;

		struct GridKey
		{
			int x = 0;
			int y = 0;
			int z = 0;

			bool operator==(const GridKey& o) const
			{
				return x == o.x && y == o.y && z == o.z;
			}
		};

		struct GridKeyHash
		{
			size_t operator()(const GridKey& k) const
			{
				return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
			}
		};

		struct SNVertex
		{
			Eigen::Vector3f pos;
			Eigen::Vector3f normal;
			Eigen::Vector3f color;
		};

	private:
		std::vector<Triangle> triangles;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> holeEdges;
	};
}
