#pragma once

#include <string>

#include <entt/entt.hpp>

class Renderable;

using Entity = entt::entity;
#define InvalidEntity ((Entity)UINT32_MAX)

class PointCloud
{
public:
	PointCloud();
	~PointCloud();

	bool LoadFromPLY(const std::string& filename);

protected:
	Entity entity = InvalidEntity;
	std::string entityName;
	Renderable* renderable = nullptr;
};
