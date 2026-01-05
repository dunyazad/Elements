#include "pch.h"

#include <Helium/PointCloud.h>
#include <Helium/HeliumCore.h>
#include <Helium/GeometryBuilder.h>
#include <Helium/Serialization.hpp>
#include <Helium/Components/CameraManipulator.h>

#include <limits>
#include <cmath>
#include <iostream>

PointCloud::PointCloud()
{
}

PointCloud::~PointCloud()
{
}

bool PointCloud::LoadFromPLY(const std::string& fileName)
{
	this->fileName = fileName;

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
					auto cameraEntity = Helium.GetEntityByName("MainCamera");
					auto camera = Helium.GetComponent<Camera>(cameraEntity);
					if (nullptr == camera) return;

					float ndcX = (2.0f * event.xpos) / (float)Helium.GetWidth() - 1.0f;
					float ndcY = 1.0f - (2.0f * event.ypos) / (float)Helium.GetHeight();

					Eigen::Matrix4f view = camera->GetViewMatrix();
					Eigen::Matrix4f proj = camera->GetProjectionMatrix();
					Eigen::Matrix4f invVP = (proj * view).inverse();

					Eigen::Vector4f screenPos(ndcX, ndcY, 1.0f, 1.0f);
					Eigen::Vector4f worldPos = invVP * screenPos;

					if (std::abs(worldPos.w()) > 1e-6f)
					{
						worldPos /= worldPos.w();
					}

					Eigen::Vector3f rayOrigin = camera->GetEye();
					Eigen::Vector3f rayDir = (worldPos.head<3>() - rayOrigin).normalized();

					int pickedIndex = this->Pick(rayOrigin, rayDir);

					if (pickedIndex != -1)
					{
						std::cout << "Picked Instance Index: " << pickedIndex << std::endl;

						//renderable->SetInstanceColor(pickedIndex, { 1.0f, 0.0f, 0.0f, 1.0f });
						auto position = this->positions[pickedIndex];

						auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
						if (cameraManipulator)
						{
							cameraManipulator->SetCenter(position);
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
		}
	}
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
