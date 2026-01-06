#pragma once

#include <future>
#include <functional>
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

	using OnPLYLoadedCallback = std::function<void(PointCloud*)>;
	void SetOnPLYLoadedCallback(const OnPLYLoadedCallback& callback) { onPLYLoadedCallback = callback; }

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
	inline int GetClusterId(size_t index) const { return clusterIDs[index]; }

	inline void SetPosition(size_t index, const Eigen::Vector3f& position) { positions[index] = position; }
	inline void SetNormal(size_t index, const Eigen::Vector3f& normal) { normals[index] = normal; }
	inline void SetColor(size_t index, const Eigen::Vector4f& color) { colors[index] = color; }
	inline void SetPointFlag(size_t index, uint64_t flag) { pointFlags[index] = flag; }
	inline void SetClusterID(size_t index, int clusterID) { clusterIDs[index] = clusterID; }

	inline const std::vector<Eigen::Vector3f>& GetPositions() const { return positions; }
	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }
	inline const std::vector<uint64_t>& GetPointFlags() const { return pointFlags; }
	inline const std::vector<int>& GetClusterIDs() const { return clusterIDs; }

	inline void SetPositions(const std::vector<Eigen::Vector3f>& positions) { this->positions = positions; }
	inline void SetNormals(const std::vector<Eigen::Vector3f>& normals) { this->normals = normals; }
	inline void SetColors(const std::vector<Eigen::Vector4f>& colors) { this->colors = colors; }
	inline void SetPointFlags(const std::vector<uint64_t>& pointFlags) { this->pointFlags = pointFlags; }
	inline void SetClusterIDs(const std::vector<int>& clusterIDs) { this->clusterIDs = clusterIDs; }

	inline const AABB& GetAABB() const { return aabb; }

	const std::vector<std::pair<int, int>>& GetSortedClusters() const { return sortedClusters; }
	std::vector<std::pair<int, int>>& GetSortedClusters() { return sortedClusters; }

protected:
	static int nextID;

	int id = -1;
	std::string name;
	std::string fileName;

	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector4f> colors;
	std::vector<uint64_t> pointFlags;
	std::vector<int> clusterIDs;

	AABB aabb;

	Entity entity = InvalidEntity;
	std::string entityName;
	Renderable* renderable = nullptr;
	bool isLoading = false;
	std::future<ProcessedInstanceData> loadingFuture;
	OnPLYLoadedCallback onPLYLoadedCallback = nullptr;

	std::vector<std::pair<int, int>> sortedClusters;
};
