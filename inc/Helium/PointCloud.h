#pragma once

#include <future>
#include <string>
#include <vector>

#include <entt/entt.hpp>

#include <Eigen/Dense>

class Renderable;

using Entity = entt::entity;
#define InvalidEntity ((Entity)UINT32_MAX)

struct ProcessedInstanceData
{
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector4f> colors;
	std::vector<Eigen::Matrix4f> transforms;
	size_t pointCount = 0;
};

class PointCloud
{
public:
	PointCloud();
	~PointCloud();

	bool LoadFromPLY(const std::string& fileName);

	void UpdateLoading();

	bool SetVisible(bool isVisible);

	int Pick(const Eigen::Vector3f& rayOrigin, const Eigen::Vector3f& rayDirection) const;

	inline const std::string& GetFileName() const { return fileName; }

protected:
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector4f> colors;

	Entity entity = InvalidEntity;
	std::string entityName;
	Renderable* renderable = nullptr;
	bool isLoading = false;
	std::future<ProcessedInstanceData> loadingFuture;
	std::string fileName;
};
