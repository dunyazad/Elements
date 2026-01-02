#include "pch.h"

#include <Helium/Color.hpp>
#include <Helium/HeliumCore.h>
#include <Helium/VisualDebugging.h>
#include <Helium/GeometryBuilder.h>

#include <algorithm>

bool VisualDebugging::initialized = false;
std::map<std::string, Entity> VisualDebugging::entities;
std::map<std::string, DebuggingRenderable*> VisualDebugging::debuggingRenderables;
std::map<std::string, TextBlock*> VisualDebugging::textBlocks;
std::vector<std::string> VisualDebugging::selectionRenderables;
unsigned int VisualDebugging::selectionIndex = 0;

std::mutex VisualDebugging::commandMutex;
std::vector<std::function<void()>> VisualDebugging::commandQueue;
std::vector<std::function<void()>> VisualDebugging::pendingCommands;

inline Eigen::Matrix4f Translate(const Eigen::Vector3f& t)
{
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m.block<3, 1>(0, 3) = t;
	return m;
}

inline Eigen::Matrix4f Scale(const Eigen::Vector3f& s)
{
	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m(0, 0) = s.x();
	m(1, 1) = s.y();
	m(2, 2) = s.z();
	return m;
}

inline Eigen::Matrix4f Rotate(float angle, const Eigen::Vector3f& axis)
{
	Eigen::Matrix3f r =
		Eigen::AngleAxisf(angle, axis.normalized()).toRotationMatrix();

	Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
	m.block<3, 3>(0, 0) = r;
	return m;
}

void VisualDebugging::Initialize()
{
	if (false == initialized)
	{
		initialized = true;
		pendingCommands.reserve(1000);
	}
}

void VisualDebugging::Terminate()
{
	if (true == initialized)
	{
		pendingCommands.clear();
	}
}

void VisualDebugging::DispatchCommands()
{
	{
		std::lock_guard<std::mutex> lock(commandMutex);
		if (!commandQueue.empty())
		{
			pendingCommands.insert(
				pendingCommands.end(),
				std::make_move_iterator(commandQueue.begin()),
				std::make_move_iterator(commandQueue.end())
			);
			commandQueue.clear();
		}
	}

	if (pendingCommands.empty()) return;

	constexpr long long kMaxExecutionTimeMicros = 2000; // 2ms
	auto startTime = std::chrono::high_resolution_clock::now();

	size_t processedCount = 0;
	for (const auto& command : pendingCommands)
	{
		command();
		processedCount++;

		auto currentTime = std::chrono::high_resolution_clock::now();
		auto elapsedMicros = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count();

		if (elapsedMicros > kMaxExecutionTimeMicros)
		{
			break;
		}
	}

	if (processedCount > 0)
	{
		if (processedCount == pendingCommands.size())
		{
			pendingCommands.clear();
		}
		else
		{
			pendingCommands.erase(pendingCommands.begin(), pendingCommands.begin() + processedCount);
		}
	}
}

void VisualDebugging::CreateLineEntity(const std::string& tag)
{
	auto entity = Helium.CreateEntity(tag);
	entities[tag] = entity;

	auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
	renderable->Initialize(Renderable::GeometryMode::Lines);
	debuggingRenderables[tag] = renderable;

	renderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));
}

void VisualDebugging::CreateTriangleEntity(const std::string& tag)
{
	auto entity = Helium.CreateEntity(tag);
	entities[tag] = entity;

	auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
	renderable->Initialize(Renderable::GeometryMode::Triangles);
	debuggingRenderables[tag] = renderable;

	renderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
}

void VisualDebugging::CreateBoxEntity(const std::string& tag)
{
	auto entity = Helium.CreateEntity(tag);
	entities[tag] = entity;

	auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
	renderable->Initialize(Renderable::GeometryMode::Triangles);
	debuggingRenderables[tag] = renderable;

	{
		auto shader = Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs"));
		renderable->AddShader(shader);
	}
	{
		auto shader = Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs"));
		renderable->AddShader(shader);
	}
	renderable->SetActiveShaderIndex(1);

	GeometryBuilder::BuildBox(renderable, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, Color::white());
}

void VisualDebugging::CreateWiredBoxEntity(const std::string& tag)
{
	auto entity = Helium.CreateEntity(tag);
	entities[tag] = entity;

	auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
	renderable->Initialize(Renderable::GeometryMode::Lines);
	debuggingRenderables[tag] = renderable;

	auto shader = Helium.CreateShader("InstancingWithoutLighting", File("../../res/Shaders/InstancingWithoutLighting.vs"), File("../../res/Shaders/InstancingWithoutLighting.fs"));
	renderable->AddShader(shader);

	GeometryBuilder::BuildBox(renderable, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, Color::white(), true);
}

void VisualDebugging::CreateSphereEntity(const std::string& tag)
{
	auto entity = Helium.CreateEntity(tag);
	entities[tag] = entity;

	auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
	renderable->Initialize(Renderable::GeometryMode::Triangles);
	debuggingRenderables[tag] = renderable;

	{
		auto shader = Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs"));
		renderable->AddShader(shader);
	}
	{
		auto shader = Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs"));
		renderable->AddShader(shader);
	}
	renderable->SetActiveShaderIndex(1);

	GeometryBuilder::BuildSphere(renderable, { 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6, Color::white());
}

void VisualDebugging::Clear(const std::string& tag)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();

			if (debuggingRenderables.end() != debuggingRenderables.find(tag))
			{
				auto& renderable = debuggingRenderables[tag];
				if (renderable->IsInstancingEnabled())
				{
					renderable->ClearInstancingData();
				}
				else
				{
					renderable->Clear();
				}
				return;
			}
		});
}

void VisualDebugging::ClearAll()
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([]()
		{
			for (auto& kvp : debuggingRenderables)
			{
				if (kvp.second->IsInstancingEnabled())
				{
					kvp.second->ClearInstancingData();
				}
				else
				{
					kvp.second->Clear();
				}
			}
		});
}

void VisualDebugging::SetVisibility(bool visible, const std::string& tag)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();

			if (debuggingRenderables.end() != debuggingRenderables.find(tag))
			{
				auto& renderable = debuggingRenderables[tag];
				renderable->SetVisible(visible);
				return;
			}
		});
}

void VisualDebugging::SetVisibilityAll(bool visible)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();

			for (auto& kvp : debuggingRenderables)
			{
				kvp.second->SetVisible(visible);
			}
		});
}

void VisualDebugging::ToggleVisibility(const std::string& tag)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();

			if (debuggingRenderables.end() != debuggingRenderables.find(tag))
			{
				auto& renderable = debuggingRenderables[tag];
				renderable->SetVisible(!renderable->IsVisible());
				return;
			}
		});
}

void VisualDebugging::ToggleVisibilityAll()
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([]()
		{
			if (false == initialized) Initialize();

			for (auto& kvp : debuggingRenderables)
			{
				kvp.second->SetVisible(!kvp.second->IsVisible());
			}
		});
}

void VisualDebugging::AddLine(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector4f& c)
{
	AddLine(tag, v0, v1, c, c);
}

void VisualDebugging::AddLine(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector4f& c0, const Eigen::Vector4f& c1)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();
			if (entities.end() == entities.find(tag)) CreateLineEntity(tag);

			auto& renderable = debuggingRenderables[tag];
			renderable->AddVertex(v0);
			renderable->AddVertex(v1);
			renderable->AddColor4(c0);
			renderable->AddColor4(c1);
		});
}

void VisualDebugging::AddTriangle(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector4f& c)
{
	AddTriangle(tag, v0, v1, v2, c, c, c);
}

void VisualDebugging::AddTriangle(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector4f& c0, const Eigen::Vector4f& c1, const Eigen::Vector4f& c2)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();
			if (entities.end() == entities.find(tag)) CreateTriangleEntity(tag);

			auto& renderable = debuggingRenderables[tag];
			auto i0 = renderable->AddVertex(v0);
			auto i1 = renderable->AddVertex(v1);
			auto i2 = renderable->AddVertex(v2);

			auto normal = (v1 - v0).cross(v2 - v0).normalized();

			renderable->AddNormal(normal);
			renderable->AddNormal(normal);
			renderable->AddNormal(normal);

			renderable->AddColor4(c0);
			renderable->AddColor4(c1);
			renderable->AddColor4(c2);

			renderable->AddIndex((unsigned int)i0);
			renderable->AddIndex((unsigned int)i1);
			renderable->AddIndex((unsigned int)i2);
		});
}

void VisualDebugging::AddBox(const std::string& tag, const AABB& aabb, const Eigen::Vector4f& color)
{
	auto center = (aabb.min + aabb.max) * 0.5f;
	auto dimensions = aabb.max - aabb.min;
	AddBox(tag, center, { 0.0f, 0.0f, 1.0f }, dimensions, color);
}

void VisualDebugging::AddBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
	AddBox(tag, center, Eigen::Vector3f(0.0f, 1.0f, 0.0f), dimensions, color);
}

void VisualDebugging::AddBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();
			if (entities.end() == entities.find(tag)) CreateBoxEntity(tag);

			auto& renderable = debuggingRenderables[tag];

			renderable->AddInstanceColor(color);
			renderable->AddInstanceNormal(normal);

			Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
			Eigen::Matrix4f rot = Eigen::Matrix4f::Identity();
			if (normal.norm() > 0.0001f)
			{
				Eigen::Vector3f z(0.0f, 0.0f, 1.0f);
				Eigen::Vector3f axis = z.cross(normal);

				if (axis.norm() > 0.0001f)
				{
					axis.normalize();
					float angle = acos(normal.normalized().dot(z));
					rot = Rotate(angle, axis);
				}
			}
			tm = Translate(center) * rot * Scale(dimensions);
			renderable->AddInstanceTransform(tm);

			renderable->IncreaseNumberOfInstances();
		});
}

void VisualDebugging::AddWiredBox(const std::string& tag, const AABB& aabb, const Eigen::Vector4f& color)
{
	auto center = (aabb.min + aabb.max) * 0.5f;
	auto dimensions = aabb.max - aabb.min;
	AddWiredBox(tag, center, { 0.0f, 0.0f, 1.0f }, dimensions, color);
}

void VisualDebugging::AddWiredBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
	AddWiredBox(tag, center, Eigen::Vector3f(0.0f, 1.0f, 0.0f), dimensions, color);
}

void VisualDebugging::AddWiredBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();
			if (entities.end() == entities.find(tag)) CreateWiredBoxEntity(tag);

			auto& renderable = debuggingRenderables[tag];

			renderable->AddInstanceColor(color);
			renderable->AddInstanceNormal(normal);

			Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
			Eigen::Matrix4f rot = Eigen::Matrix4f::Identity();
			if (normal.norm() > 0.0001f)
			{
				Eigen::Vector3f z(0.0f, 0.0f, 1.0f);
				Eigen::Vector3f axis = z.cross(normal);

				if (axis.norm() > 0.0001f)
				{
					axis.normalize();
					float angle = acos(normal.normalized().dot(z));
					rot = Rotate(angle, axis);
				}
			}
			tm = Translate(center) * rot * Scale(dimensions);
			renderable->AddInstanceTransform(tm);

			renderable->IncreaseNumberOfInstances();
		});
}

void VisualDebugging::AddSphere(const std::string& tag, const Eigen::Vector3f& center, float radius, const Eigen::Vector4f& color)
{
	AddSphere(tag, center, Eigen::Vector3f(0.0f, 1.0f, 0.0f), radius, color);
}

void VisualDebugging::AddSphere(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, const Eigen::Vector4f& color)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();
			if (entities.end() == entities.find(tag)) CreateSphereEntity(tag);

			auto& renderable = debuggingRenderables[tag];

			renderable->AddInstanceColor(color);
			renderable->AddInstanceNormal({ 0.0f, 0.1f, 0.0f });

			Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
			Eigen::Matrix4f rot = Eigen::Matrix4f::Identity();
			if (normal.norm() > 0.0001f)
			{
				Eigen::Vector3f z(0.0f, 0.0f, 1.0f);
				Eigen::Vector3f axis = z.cross(normal);

				if (axis.norm() > 0.0001f)
				{
					axis.normalize();
					float angle = acos(normal.normalized().dot(z));
					rot = Rotate(angle, axis);
				}
			}
			tm = Translate(center)
				* rot
				* Scale(Eigen::Vector3f(radius * 2.0f, radius * 2.0f, radius * 2.0f));
			renderable->AddInstanceTransform(tm);

			renderable->IncreaseNumberOfInstances();
		});
}

void VisualDebugging::ClearSelectionList()
{
	selectionRenderables.clear();
	selectionIndex = 0;
}

void VisualDebugging::AddToSelectionList(const std::string& tag)
{
	std::lock_guard<std::mutex> lock(commandMutex);
	commandQueue.emplace_back([=]()
		{
			if (false == initialized) Initialize();

			if (debuggingRenderables.end() != debuggingRenderables.find(tag))
			{
				//if (std::find(selectionRenderables.begin(), selectionRenderables.end(), tag) == selectionRenderables.end())

				selectionRenderables.push_back(tag);
			}
			else
			{
				printf("Warning: Tag not found %s\n", tag.c_str());
			}
		});
}

unsigned int VisualDebugging::ShowNextSelection()
{
	for (auto& tag : selectionRenderables)
	{
		SetVisibility(false, tag);
	}

	selectionIndex++;
	selectionIndex = selectionIndex % selectionRenderables.size();

	auto& tag = selectionRenderables[selectionIndex];
	SetVisibility(true, tag);

	return selectionIndex;
}

unsigned int VisualDebugging::ShowPreviousSelection()
{
	for (auto& tag : selectionRenderables)
	{
		SetVisibility(false, tag);
	}

	if (0 == selectionIndex) selectionIndex += (unsigned int)selectionRenderables.size();
	selectionIndex--;

	auto& tag = selectionRenderables[selectionIndex];
	SetVisibility(true, tag);

	return selectionIndex;
}
