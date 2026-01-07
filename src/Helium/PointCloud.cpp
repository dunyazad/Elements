#include "pch.h"

#include <Helium/PointCloud.h>
#include <Helium/HeliumCore.h>
#include <Helium/GeometryBuilder.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
#include <Helium/Components/CameraManipulator.h>

#include <limits>
#include <cmath>
#include <iostream>

using VD = VisualDebugging;

int PointCloud::nextID = -1;

PointCloud::PointCloud()
{
	id = ++nextID;
}

PointCloud::~PointCloud()
{
	Helium.RemoveEntity(entity);
}

bool PointCloud::LoadFromPLY(const std::string& fileName, const std::string& name, OnPLYLoadedCallback callback)
{
	this->fileName = fileName;
	this->name = name;
	this->onPLYLoadedCallback = callback;

	if (isLoading) return false;

	isLoading = true;

	loadingFuture = std::async(std::launch::async, [this]() -> ProcessedInstanceData {
		ProcessedInstanceData data;
		PLYFormat ply;

		ply.Deserialize(this->fileName);

		data.pointCount = ply.GetPoints().size();

		data.positions.reserve(data.pointCount);
		data.normals.reserve(data.pointCount);
		data.colors.reserve(data.pointCount);
		data.transforms.reserve(data.pointCount);

		for (size_t i = 0; i < data.pointCount; i++)
		{
			const Eigen::Vector3f& p = ply.GetPoints()[i];
			Eigen::Vector3f n = (i < ply.GetNormals().size()) ? ply.GetNormals()[i] : Eigen::Vector3f::Zero();
			Eigen::Vector4f c = (i < ply.GetColors().size()) ? ply.GetColors()[i] : Eigen::Vector4f::Ones();

			data.positions.push_back(p);
			data.normals.push_back(n);
			data.colors.push_back(c);
			data.aabb.Expand(p);

			Eigen::Affine3f tm = Eigen::Affine3f::Identity();
			Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();

			if (n.norm() > 0.0001f)
			{
				Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
				Eigen::Vector3f normalDir = n.normalized();

				Eigen::Vector3f axis = up.cross(normalDir);
				float dot = up.dot(normalDir);

				if (dot > 1.0f) dot = 1.0f;
				else if (dot < -1.0f) dot = -1.0f;

				float angle = std::acos(dot);

				if (axis.norm() > 0.0001f)
				{
					axis.normalize();
					rot = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
				}
				else if (dot < -0.9f)
				{
					rot = Eigen::AngleAxisf(3.1415926f, Eigen::Vector3f::UnitX()).toRotationMatrix();
				}
			}

			tm.translate(p);
			tm.rotate(rot);
			tm.scale(0.1f);

			data.transforms.push_back(tm.matrix());
		}

		return data;
		});

	return true;
}

void PointCloud::SetupEntity(ProcessedInstanceData& data)
{
	if (InvalidEntity == entity)
	{
		entity = Helium.CreateEntity(entityName);
	}

	Helium.CreateEventCallback<KeyEvent>(entity, [this](Entity entity, const KeyEvent& event) {
		auto renderable = Helium.GetComponent<Renderable>(entity);
		if (nullptr == renderable) return;
		if (false == renderable->IsVisible()) return;
		if (this != Helium.GetSelectedPointCloud()) return;

		if (0 == event.action)
		{
			if (192 == event.keyCode) renderable->NextDrawingMode();
			else if ('1' == event.keyCode) renderable->SetActiveShaderIndex(0);
			else if ('2' == event.keyCode) renderable->SetActiveShaderIndex(1);
		}
		});

	Helium.CreateEventCallback<MouseButtonEvent>(entity, [this](Entity entity, const MouseButtonEvent& event) {
		auto renderable = Helium.GetComponent<Renderable>(entity);
		if (nullptr == renderable) return;
		if (false == renderable->IsVisible()) return;
		if (this != Helium.GetSelectedPointCloud()) return;

		if (0 == event.action && 0 == event.button)
		{
			auto cameraEntity = Helium.GetEntityByName("MainCamera");
			auto camera = Helium.GetComponent<Camera>(cameraEntity);
			if (nullptr == camera) return;

			Ray ray = camera->ScreenPointToRay(
				(float)event.xpos,
				(float)event.ypos,
				Helium.GetWidth(),
				Helium.GetHeight()
			);

			int pickedIndex = this->Pick(ray.origin, ray.direction);

			if (pickedIndex != -1)
			{
				json j;
				j["PointPicked"];
				j["PointPicked"]["PointCloudID"] = GetID();
				j["PointPicked"]["PickedIndex"] = pickedIndex;
				j["PointPicked"]["IsCtrlPressed"] = event.IsCtrlPressed();

				CustomEvent customEvent(j.dump());
				Helium.EnqueueEvent<CustomEvent>(customEvent);
			}

			json j;
			j["EventType"] = "PointSelected";
			j["Parameters"]["PointCloudID"] = GetID();
			j["Parameters"]["PickedIndex"] = pickedIndex;
			j["Parameters"]["IsCtrlPressed"] = event.IsCtrlPressed();

			Helium.NativeToManaged(j.dump().c_str());
		}
		});

	renderable = Helium.CreateComponent<Renderable>(entity);
	renderable->Initialize(Renderable::GeometryMode::Triangles);

	renderable->AddShader(Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
	renderable->AddShader(Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
	renderable->SetActiveShaderIndex(1);

	GeometryBuilder::BuildSphere(renderable, { 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6, { 1.0f, 1.0f, 1.0f, 1.0f });

	for (size_t i = 0; i < data.pointCount; i++)
	{
		renderable->AddInstanceNormal(data.normals[i]);
		renderable->AddInstanceColor(data.colors[i]);
		renderable->AddInstanceTransform(data.transforms[i]);
		renderable->IncreaseNumberOfInstances();
	}

	renderable->EnableInstancing();

	this->positions = std::move(data.positions);
	this->normals = std::move(data.normals);
	this->colors = std::move(data.colors);
	this->aabb = data.aabb;
}

void PointCloud::UpdateLoading()
{
	if (isLoading && loadingFuture.valid())
	{
		if (loadingFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			ProcessedInstanceData data = loadingFuture.get();

			if (InvalidEntity == entity)
			{
				entityName = "PointCloud";
			}

			SetupEntity(data);

			isLoading = false;

			if (onPLYLoadedCallback)
			{
				onPLYLoadedCallback(this);

				json j;
				j["EventType"] = "PointCloudLoaded";
				j["Parameters"]["PointCloudID"] = id;
				j["Parameters"]["FileName"] = fileName;
				j["Parameters"]["Name"] = name;

				Helium.NativeToManaged(j.dump().c_str());
			}
		}
	}

	if (isCloning && cloningFuture.valid())
	{
		if (cloningFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			ProcessedInstanceData data = cloningFuture.get();

			SetupEntity(data);

			isCloning = false;

			if (onClonedCallback)
			{
				onClonedCallback(this);
			}

			json j;
			j["EventType"] = "PointCloudCloned";
			j["Parameters"]["PointCloudID"] = id;
			j["Parameters"]["FileName"] = fileName;
			j["Parameters"]["Name"] = name;

			Helium.NativeToManaged(j.dump().c_str());
		}
	}
}

PointCloud* PointCloud::Clone(OnClonedCallback callback)
{
	if (isLoading || isCloning)
	{
		std::cout << "[PointCloud] Cannot clone while loading or cloning." << std::endl;
		return nullptr;
	}

	PointCloud* newPC = new PointCloud();

	newPC->name = this->name + "_Clone";
	newPC->fileName = this->fileName;
	newPC->entityName = this->entityName + "_Clone";
	newPC->onClonedCallback = callback;

	newPC->isCloning = true;

	newPC->cloningFuture = std::async(std::launch::async, [this]() -> ProcessedInstanceData {
		ProcessedInstanceData data;

		data.positions = this->positions;
		data.normals = this->normals;
		data.colors = this->colors;
		data.aabb = this->aabb;
		data.pointCount = this->positions.size();

		data.transforms.reserve(data.pointCount);

		for (size_t i = 0; i < data.pointCount; i++)
		{
			const Eigen::Vector3f& p = data.positions[i];
			const Eigen::Vector3f& n = data.normals[i];

			Eigen::Affine3f tm = Eigen::Affine3f::Identity();
			Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();

			if (n.norm() > 0.0001f)
			{
				Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
				Eigen::Vector3f normalDir = n.normalized();
				Eigen::Vector3f axis = up.cross(normalDir);
				float dot = up.dot(normalDir);

				if (dot > 1.0f) dot = 1.0f; else if (dot < -1.0f) dot = -1.0f;
				float angle = std::acos(dot);

				if (axis.norm() > 0.0001f)
				{
					axis.normalize();
					rot = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
				}
				else if (dot < -0.9f)
				{
					rot = Eigen::AngleAxisf(3.1415926f, Eigen::Vector3f::UnitX()).toRotationMatrix();
				}
			}

			tm.translate(p);
			tm.rotate(rot);
			tm.scale(0.1f);

			data.transforms.push_back(tm.matrix());
		}

		return data;
		});

	return newPC;
}

bool PointCloud::SetVisible(bool isVisible)
{
	if (renderable)
	{
		renderable->SetVisible(isVisible);
		return true;
	}
	return false;
}

int PointCloud::Pick(const Eigen::Vector3f& rayOrigin, const Eigen::Vector3f& rayDirection) const
{
	if (positions.empty())
	{
		return -1;
	}

	int closestIndex = -1;
	float minDistance = std::numeric_limits<float>::max();

	const float radius = 0.05f;
	const float radiusSq = radius * radius;

	Eigen::Vector3f D = rayDirection.normalized();

	for (size_t i = 0; i < positions.size(); ++i)
	{
		const Eigen::Vector3f& C = positions[i];
		Eigen::Vector3f L = C - rayOrigin;

		float tca = L.dot(D);

		if (tca < 0) continue;

		float d2 = L.dot(L) - tca * tca;

		if (d2 > radiusSq) continue;

		float thc = std::sqrt(radiusSq - d2);

		float t0 = tca - thc;

		if (t0 < minDistance)
		{
			minDistance = t0;
			closestIndex = static_cast<int>(i);
		}
	}

	return closestIndex;
}
