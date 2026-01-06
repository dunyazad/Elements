#pragma once

#include <future>
#include <string>
#include <vector>

#include <entt/entt.hpp>

#include <Eigen/Dense>

#include <Helium/TypeDefinitions.h>

class Renderable;

using Entity = entt::entity;
#define InvalidEntity ((Entity)UINT32_MAX)

struct ProcessedInstanceData
{
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector4f> colors;
	std::vector<Eigen::Matrix4f> transforms;
	AABB aabb;
	size_t pointCount = 0;
};

class PointCloud
{
public:
	PointCloud();
	~PointCloud();

	bool LoadFromPLY(const std::string& fileName, const std::string& name);

	void UpdateLoading();

	PointCloud* Clone();

	bool SetVisible(bool isVisible);

	int Pick(const Eigen::Vector3f& rayOrigin, const Eigen::Vector3f& rayDirection) const;

	inline int GetID() const { return id; }
	inline const std::string& GetName() const { return name; }
	inline const std::string& GetFileName() const { return fileName; }

	inline size_t Size() const { return positions.size(); }
	
	inline const Eigen::Vector3f& GetPosition(size_t index) const { return positions[index]; }
	inline const Eigen::Vector3f& GetNormal(size_t index) const { return normals[index]; }
	inline const Eigen::Vector4f& GetColor(size_t index) const { return colors[index]; }
	inline uint64_t GetPointFlag(size_t index) const { return pointFlags[index]; }
	inline int GetClusterId(size_t index) const { return clusterIds[index]; }

	inline const std::vector<Eigen::Vector3f>& GetPositions() const { return positions; }
	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }
	inline const std::vector<uint64_t>& GetPointFlags() const { return pointFlags; }
	inline const std::vector<int>& GetClusterIds() const { return clusterIds; }

	inline const AABB& GetAABB() const { return aabb; }

protected:
	static int nextID;

	int id = -1;
	std::string name;
	std::string fileName;

	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector4f> colors;
	std::vector<int> clusterIds;
	std::vector<uint64_t> pointFlags;

	AABB aabb;

	Entity entity = InvalidEntity;
	std::string entityName;
	Renderable* renderable = nullptr;
	bool isLoading = false;
	std::future<ProcessedInstanceData> loadingFuture;
};
