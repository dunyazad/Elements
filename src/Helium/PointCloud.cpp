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

extern void OnPointCloudCreated(int id, const std::string& fileName, const std::string& name);
extern void OnPointCloudDeleted(int ID);

int PointCloud::nextID = -1;

PointCloud::PointCloud()
{
	id = ++nextID;
}

PointCloud::~PointCloud()
{
	Helium.RemoveEntity(entity);

	OnPointCloudDeleted(GetID());
}

bool PointCloud::LoadFromPLY(const std::string& fileName, const std::string& name)
{
	this->fileName = fileName;
	this->name = name;

	if (isLoading) return false; // Already loading

	isLoading = true;

	// Launch async task for IO and CPU heavy math
	loadingFuture = std::async(std::launch::async, [this]() -> ProcessedInstanceData {	
		ProcessedInstanceData data;
		PLYFormat ply;

		// IO Operation
		ply.Deserialize(this->fileName);
		ply.SwapAxisYZ();

		data.pointCount = ply.GetPoints().size();

		// Pre-allocate memory to avoid reallocation overhead
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

void PointCloud::UpdateLoading()
{
	if (isLoading && loadingFuture.valid())
	{
		if (loadingFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			ProcessedInstanceData data = loadingFuture.get();

			if (InvalidEntity == entity)
			{
				entity = Helium.CreateEntity("PointCloud");
				entityName = "PointCloud";
			}

			Helium.CreateEventCallback<KeyEvent>(entity, [this](Entity entity, const KeyEvent& event) {
				auto renderable = Helium.GetComponent<Renderable>(entity);
				if (nullptr == renderable) return;
				if (false == renderable->IsVisible()) return;
				if (this != Helium.GetSelectedPointCloud()) return;

				if (0 == event.action)
				{
					if (192 == event.keyCode)
					{
						renderable->NextDrawingMode();
					}
					else if ('1' == event.keyCode)
					{
						renderable->SetActiveShaderIndex(0);
					}
					else if ('2' == event.keyCode)
					{
						renderable->SetActiveShaderIndex(1);
					}
				}
				});

			Helium.CreateEventCallback<MouseButtonEvent>(entity, [this](Entity entity, const MouseButtonEvent& event) {
				auto renderable = Helium.GetComponent<Renderable>(entity);
				if (nullptr == renderable) return;
				if (false == renderable->IsVisible()) return;
				if (this != Helium.GetSelectedPointCloud()) return;

				if (0 == event.action && 0 == event.button)
				{
					VD::Clear("Selected Point");

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
						auto position = this->positions[pickedIndex];
						auto normal = this->normals[pickedIndex];

						VD::Clear("Selected Point");
						VD::AddSphere("Selected Point", position, normal, 0.051f, {1.0f, 0.0f, 0.0f, 1.0f});

						if (event.IsCtrlPressed())
						{
							auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
							if (cameraManipulator)
							{
								cameraManipulator->SetCenter(position);
							}
						}
					}
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

			isLoading = false;

			OnPointCloudCreated(this->id, this->fileName, this->name);
		}
	}
}

PointCloud* PointCloud::Clone()
{
	if (isLoading)
	{
		std::cout << "[PointCloud] Cannot clone while loading." << std::endl;
		return nullptr;
	}

	PointCloud* newPC = new PointCloud();

	newPC->name = this->name + "_Clone";
	newPC->fileName = this->fileName;

	newPC->positions.resize(this->positions.size());
	newPC->normals.resize(this->normals.size());
	newPC->colors.resize(this->colors.size());
	memcpy(newPC->positions.data(), this->positions.data(), sizeof(Eigen::Vector3f) * this->positions.size());
	memcpy(newPC->normals.data(), this->normals.data(), sizeof(Eigen::Vector3f) * this->normals.size());
	memcpy(newPC->colors.data(), this->colors.data(), sizeof(Eigen::Vector4f) * this->colors.size());

	newPC->entityName = this->entityName + "_Clone";
	newPC->entity = Helium.CreateEntity(newPC->entityName);

	Helium.CreateEventCallback<KeyEvent>(newPC->entity, [newPC](Entity entity, const KeyEvent& event) {
		auto renderable = Helium.GetComponent<Renderable>(entity);
		if (nullptr == renderable) return;
		if (false == renderable->IsVisible()) return;
		if (newPC != Helium.GetSelectedPointCloud()) return;

		if (0 == event.action)
		{
			if (192 == event.keyCode) renderable->NextDrawingMode();
			else if ('1' == event.keyCode) renderable->SetActiveShaderIndex(0);
			else if ('2' == event.keyCode) renderable->SetActiveShaderIndex(1);
		}
		});

	Helium.CreateEventCallback<MouseButtonEvent>(newPC->entity, [newPC](Entity entity, const MouseButtonEvent& event) {
		auto renderable = Helium.GetComponent<Renderable>(entity);
		if (nullptr == renderable) return;
		if (false == renderable->IsVisible()) return;
		if (newPC != Helium.GetSelectedPointCloud()) return;

		if (0 == event.action && 0 == event.button)
		{
			VD::Clear("Selected Point");

			auto cameraEntity = Helium.GetEntityByName("MainCamera");
			auto camera = Helium.GetComponent<Camera>(cameraEntity);
			if (nullptr == camera) return;

			Ray ray = camera->ScreenPointToRay(
				(float)event.xpos,
				(float)event.ypos,
				Helium.GetWidth(),
				Helium.GetHeight()
			);

			int pickedIndex = newPC->Pick(ray.origin, ray.direction);

			if (pickedIndex != -1)
			{
				auto position = newPC->positions[pickedIndex];
				auto normal = newPC->normals[pickedIndex];

				VD::Clear("Selected Point");
				VD::AddSphere("Selected Point", position, normal, 0.051f, { 1.0f, 0.0f, 0.0f, 1.0f });

				if (event.IsCtrlPressed())
				{
					auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
					if (cameraManipulator)
					{
						cameraManipulator->SetCenter(position);
					}
				}
			}
		}
		});

	newPC->renderable = Helium.CreateComponent<Renderable>(newPC->entity);
	newPC->renderable->Initialize(Renderable::GeometryMode::Triangles);

	newPC->renderable->AddShader(Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
	newPC->renderable->AddShader(Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
	newPC->renderable->SetActiveShaderIndex(1);

	GeometryBuilder::BuildSphere(newPC->renderable, { 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6, { 1.0f, 1.0f, 1.0f, 1.0f });

	for (size_t i = 0; i < newPC->positions.size(); i++)
	{
		const Eigen::Vector3f& p = newPC->positions[i];
		const Eigen::Vector3f& n = newPC->normals[i];
		const Eigen::Vector4f& c = newPC->colors[i];

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
		tm.scale(0.1f); // Scale

		newPC->renderable->AddInstanceNormal(n);
		newPC->renderable->AddInstanceColor(c);
		newPC->renderable->AddInstanceTransform(tm.matrix());
		newPC->renderable->IncreaseNumberOfInstances();
	}

	newPC->renderable->EnableInstancing();

	newPC->isLoading = false;

	OnPointCloudCreated(newPC->id, newPC->fileName, newPC->name);

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

	// 시각적 구체의 반지름: 기본 반지름(0.5) * 스케일(0.1)
	const float radius = 0.05f;
	const float radiusSq = radius * radius;

	Eigen::Vector3f D = rayDirection.normalized();

	for (size_t i = 0; i < positions.size(); ++i)
	{
		const Eigen::Vector3f& C = positions[i]; // Sphere Center
		Eigen::Vector3f L = C - rayOrigin;

		// tca: 광선 방향으로 구의 중심까지 투영된 거리
		float tca = L.dot(D);

		// 구가 광선의 시작점 뒤에 있으면 무시
		if (tca < 0) continue;

		// d2: 구의 중심에서 광선까지의 수직 거리의 제곱
		float d2 = L.dot(L) - tca * tca;

		// 광선이 구의 반지름 범위를 벗어나면 충돌 없음
		if (d2 > radiusSq) continue;

		// thc: 교차점까지의 거리 보정값
		float thc = std::sqrt(radiusSq - d2);

		// t0: 첫 번째 교차점 거리
		float t0 = tca - thc;

		// 가장 가까운 충돌 지점 갱신
		if (t0 < minDistance)
		{
			minDistance = t0;
			closestIndex = static_cast<int>(i);
		}
	}

	return closestIndex;
}
