#pragma once

#include <any>
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

	using OnPLYLoadedCallback = std::function<void(PointCloud*)>;
	void SetOnPLYLoadedCallback(const OnPLYLoadedCallback& callback) { onPLYLoadedCallback = callback; }
	bool LoadFromPLY(const std::string& fileName, const std::string& name, OnPLYLoadedCallback callback);

	void SetupEntity(ProcessedInstanceData& data);

	void UpdateLoading();

	void UpdateRenderable();

	using OnClonedCallback = std::function<void(PointCloud*)>;
	void SetOnClonedCallback(const OnClonedCallback& callback) { onClonedCallback = callback; }
	PointCloud* Clone(OnClonedCallback callback);

	bool SetVisible(bool isVisible);

	int Pick(const Eigen::Vector3f& rayOrigin, const Eigen::Vector3f& rayDirection) const;

	inline int GetID() const { return id; }
	inline const std::string& GetName() const { return name; }
	inline void SetName(const std::string& name) { this->name = name; }
	inline const std::string& GetFileName() const { return fileName; }
	inline void SetFileName(const std::string& fileName) { this->fileName = fileName; }

	inline size_t Size() const { return positions.size(); }

	inline const Eigen::Vector3f& GetPosition(size_t index) const { return positions[index]; }
	inline const Eigen::Vector3f& GetNormal(size_t index) const { return normals[index]; }
	inline const Eigen::Vector4f& GetColor(size_t index) const { return colors[index]; }

	inline void SetPosition(size_t index, const Eigen::Vector3f& position) { positions[index] = position; }
	inline void SetNormal(size_t index, const Eigen::Vector3f& normal) { normals[index] = normal; }
	inline void SetColor(size_t index, const Eigen::Vector4f& color) { colors[index] = color; }

	inline const std::vector<Eigen::Vector3f>& GetPositions() const { return positions; }
	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	inline void SetPositions(const std::vector<Eigen::Vector3f>& positions) { this->positions = positions; UpdateRenderable(); }
	inline void SetNormals(const std::vector<Eigen::Vector3f>& normals) { this->normals = normals; UpdateRenderable(); }
	inline void SetColors(const std::vector<Eigen::Vector4f>& colors) { this->colors = colors; UpdateRenderable(); }

	inline const AABB& GetAABB() const { return aabb; }

	template <typename T>
	void SetAttribute(const std::string& key, const T& value)
	{
		if(attributes.find(key) != attributes.end())
		{
			if (attributes[key].type() != typeid(T))
			{
				ErrorLog("", "SetAttribute: Overwriting attribute '%s' with different type.", key.c_str());
			}

			RemoveAttribute(key);
		}

		attributes[key] = value;
	}

	void RemoveAttribute(const std::string& key)
	{
		auto it = attributes.find(key);
		if (it != attributes.end())
		{
			attributes.erase(it);
		}
	}

	template <typename T>
	T& GetAttribute(const std::string& key)
	{
		auto it = attributes.find(key);
		if (it == attributes.end())
		{
			attributes[key] = T();
		}

		return std::any_cast<T&>(attributes[key]);
	}

	template <typename T>
	const T& GetAttribute(const std::string& key) const
	{
		auto it = attributes.find(key);
		if (it != attributes.end())
		{
			if (it->second.type() == typeid(T))
			{
				return std::any_cast<const T&>(it->second);
			}
			else
			{
				ErrorLog("", "GetAttribute: Bad any_cast for key '%s'. Returning empty value.", key.c_str());
			}
		}

		static const T empty{};
		return empty;
	}

	template <typename T>
	bool HasAttribute(const std::string& key) const
	{
		auto it = attributes.find(key);
		if (it == attributes.end())
			return false;

		return it->second.type() == typeid(T);
	}

protected:
	static int nextID;

	int id = -1;
	std::string name;
	std::string fileName;

	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector4f> colors;

	AABB aabb;

	Entity entity = InvalidEntity;
	std::string entityName;
	Renderable* renderable = nullptr;
	bool isLoading = false;
	std::future<ProcessedInstanceData> loadingFuture;
	bool isCloning = false;
	std::future<ProcessedInstanceData> cloningFuture;
	OnPLYLoadedCallback onPLYLoadedCallback = nullptr;
	OnClonedCallback onClonedCallback = nullptr;

	std::map<std::string, std::any> attributes;
};
