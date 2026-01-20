#include "pch.h"

#include <execution>

#include <glad/glad.h>
#include <nlohmann/json.hpp>
#include <robin_hood/robin_hood.h>

#include <Helium/Color.hpp>

#include <Helium/HeliumCore.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>
#include <Helium/HeliumEvents.h>

#include <Helium/Systems/EventSystem.h>
#include <Helium/Systems/GUISystem.h>
#include <Helium/Systems/InputSystem.h>
#include <Helium/Systems/RenderSystem.h>
#include <Helium/Systems/ImmediateModeRenderSystem.h>

#include <Helium/Components/Components.h>
#include <Helium/Components/GUI/GUIComponent.h>

#include <Helium/GeometryBuilder.h>
#include <Helium/Serialization.hpp>

#include <Helium/VisualDebugging.h>

#include <Helium/PointCloud.h>

using VD = VisualDebugging;

extern void He_LogInternal(HeliumLogLevel level, const char* key, char* message);

void InitializePrimitives()
{
	{
		auto entity = Helium.CreateEntity("Plane");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);

		GeometryBuilder::BuildPlane(
			renderable,
			10.0f, 10.0f, 10, 10,
			Eigen::Vector3f(0.0f, 0.0f, 0.0f),
			Eigen::Vector3f(0.0f, 0.0f, 1.0f),
			Eigen::Vector4f(0.7f, 0.2f, 0.2f, 1.0f)
		);
	
		Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(8.0f, 0.0f, -5.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("PlaneWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);

		GeometryBuilder::BuildPlane(
			wireRenderable,
			10.0f, 10.0f, 10, 10,
			Eigen::Vector3f(0.0f, 0.0f, 0.0f),
			Eigen::Vector3f(0.0f, 0.0f, 1.0f),
			Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f),
			true
		);
		Helium.CreateComponent<Transform>(wireEntity)->SetLocalPosition(Eigen::Vector3f(8.0f, 0.0f, -5.0f));
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));
	}

	float xStride = 2.5f;
	float currentX = 0.0f;

	// --- Box ---
	{
		auto entity = Helium.CreateEntity("Box");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildBox(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f), Eigen::Vector4f(1.0f, 0.5f, 0.5f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		// Wireframe
		auto wireEntity = Helium.CreateEntity("BoxWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildBox(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(1.01f, 1.01f, 1.01f), Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Sphere ---
	{
		auto entity = Helium.CreateEntity("Sphere");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildSphere(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.6f, 24, 24, Eigen::Vector4f(0.5f, 1.0f, 0.5f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("SphereWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildSphere(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.61f, 12, 12, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Disk ---
	{
		auto entity = Helium.CreateEntity("Disk");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildDisk(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.7f, 32, Eigen::Vector4f(0.5f, 0.5f, 1.0f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("DiskWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildDisk(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.71f, 16, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Cylinder ---
	{
		auto entity = Helium.CreateEntity("Cylinder");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildCylinder(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.5f, 1.5f, 24, Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("CylinderWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildCylinder(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.51f, 1.51f, 12, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Cone ---
	{
		auto entity = Helium.CreateEntity("Cone");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildCone(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.6f, 1.5f, 24, Eigen::Vector4f(0.0f, 1.0f, 1.0f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("ConeWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildCone(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.61f, 1.51f, 12, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Capsule ---
	{
		auto entity = Helium.CreateEntity("Capsule");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildCapsule(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.4f, 1.8f, 16, 8, Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("CapsuleWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildCapsule(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.41f, 1.81f, 16, 8, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Torus ---
	{
		auto entity = Helium.CreateEntity("Torus");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildTorus(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.6f, 0.2f, 32, 16, Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("TorusWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildTorus(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.61f, 0.21f, 16, 8, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Arrow ---
	{
		auto entity = Helium.CreateEntity("Arrow");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		GeometryBuilder::BuildArrow(renderable, Eigen::Vector3f(currentX, -0.5f, 0.0f), Eigen::Vector3f(currentX, 1.0f, 0.5f), 0.1f, 0.25f, 0.4f, Eigen::Vector4f(0.8f, 0.1f, 0.1f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("ArrowWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildArrow(wireRenderable, Eigen::Vector3f(currentX, -0.5f, 0.0f), Eigen::Vector3f(currentX, 1.0f, 0.5f), 0.11f, 0.26f, 0.41f, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Frustum ---
	{
		auto entity = Helium.CreateEntity("Frustum");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Lines);

		// 임의의 ViewProjection 역행렬 생성 (시각화를 위해)
		float fov = 45.0f * 3.14159f / 180.0f;
		float aspect = 16.0f / 9.0f;
		float zNear = 0.5f;
		float zFar = 2.0f;
		float tanHalfFovy = tan(fov / 2.0f);

		Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
		projection(0, 0) = 1.0f / (aspect * tanHalfFovy);
		projection(1, 1) = 1.0f / (tanHalfFovy);
		projection(2, 2) = -(zFar + zNear) / (zFar - zNear);
		projection(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
		projection(3, 2) = -1.0f;

		Eigen::Affine3f viewTransform = Eigen::Affine3f::Identity();
		viewTransform.translate(Eigen::Vector3f(-currentX, 0.0f, -2.5f));

		Eigen::Matrix4f viewProj = projection * viewTransform.matrix();
		Eigen::Matrix4f invViewProj = viewProj.inverse();

		GeometryBuilder::BuildFrustum(renderable, invViewProj, Eigen::Vector4f(0.0f, 1.0f, 1.0f, 1.0f));
		renderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride;
	}

	// --- Tube ---
	{
		std::vector<Eigen::Vector3f> tubePoints;
		tubePoints.push_back(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
		tubePoints.push_back(Eigen::Vector3f(0.5f, 1.0f, 0.0f));
		tubePoints.push_back(Eigen::Vector3f(1.5f, -1.0f, 0.5f));
		tubePoints.push_back(Eigen::Vector3f(2.5f, 0.5f, -0.5f));
		tubePoints.push_back(Eigen::Vector3f(3.0f, 0.0f, 0.0f));

		auto entity = Helium.CreateEntity("Tube");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);

		GeometryBuilder::BuildTube(
			renderable,
			tubePoints,
			0.2f,
			15,
			16,
			Eigen::Vector4f(0.2f, 0.8f, 1.0f, 1.0f)
		);

		Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(currentX, 0.0f, 0.0f));
		renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = Helium.CreateEntity("TubeWire");
		auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);

		GeometryBuilder::BuildTube(
			wireRenderable,
			tubePoints,
			0.205f,
			15,
			8,
			Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f),
			true
		);

		Helium.CreateComponent<Transform>(wireEntity)->SetLocalPosition(Eigen::Vector3f(currentX, 0.0f, 0.0f));
		wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

		currentX += xStride * 1.5f;
	}
}

const Eigen::Vector4f RED = { 1.0f, 0.0f, 0.0f, 1.0f };
const Eigen::Vector4f GREEN = { 0.0f, 1.0f, 0.0f, 1.0f };
const Eigen::Vector4f BLUE = { 0.0f, 0.0f, 1.0f, 1.0f };
const Eigen::Vector4f WHITE = { 1.0f, 1.0f, 1.0f, 1.0f };
const Eigen::Vector4f YELLOW = { 1.0f, 1.0f, 0.0f, 1.0f };
const Eigen::Vector4f CYAN = { 0.0f, 1.0f, 1.0f, 1.0f };
const Eigen::Vector4f MAGENTA = { 1.0f, 0.0f, 1.0f, 1.0f };

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

void InitializeVisualDebugging()
{
	float x = -10.0f;
	float z = 0.0f;
	float spacing = 3.0f;

	VisualDebugging::AddBox("Test_Box", { x, 1.0f, z }, { 1.0f, 1.0f, 1.0f }, RED);
	x += spacing;

	VisualDebugging::AddSphere("Test_Sphere", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, GREEN);
	x += spacing;

	VisualDebugging::AddCylinder("Test_Cylinder", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 2.0f, BLUE);
	x += spacing;

	VisualDebugging::AddCone("Test_Cone", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 2.0f, YELLOW);
	x += spacing;

	VisualDebugging::AddCapsule("Test_Capsule", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 1.0f, 16, CYAN);
	x += spacing;

	VisualDebugging::AddDisk("Test_Disk", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 1.0f, MAGENTA);
	x += spacing;

	VisualDebugging::AddTorus("Test_Torus", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 1.0f, 0.3f, 32, 16, WHITE);

	x = -10.0f;
	z = 3.0f;

	VisualDebugging::AddWiredBox("Test_WiredBox", { x, 1.0f, z }, { 1.2f, 1.2f, 1.2f }, WHITE);
	x += spacing;

	VisualDebugging::AddTube("Test_Tube", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 16, 16, CYAN);
	x += spacing;

	VisualDebugging::AddArrow("Test_Arrow_Up", { x, 0.0f, z }, { 0.0f, 1.0f, 0.0f }, 2.0f, GREEN);
	x += spacing;

	VisualDebugging::AddArrow("Test_Arrow_Right", { x, 1.0f, z }, { 1.0f, 0.0f, 0.0f }, 2.0f, RED);
	x += spacing;

	VisualDebugging::AddArrow("Test_Arrow_Forward", { x, 1.0f, z }, { 0.0f, 0.0f, 1.0f }, 2.0f, BLUE);
	x += spacing;

	Eigen::Vector3f diag = Eigen::Vector3f(1.0f, 1.0f, 1.0f).normalized();
	VisualDebugging::AddArrow("Test_Arrow_Diag", { x, 1.0f, z }, diag, 2.0f, YELLOW);

	x = -10.0f;
	z = 6.0f;

	VisualDebugging::AddLine("Test_Line", { x - 1.0f, 0.0f, z }, { x + 1.0f, 2.0f, z }, RED, BLUE);
	x += spacing;

	VisualDebugging::AddTriangle("Test_Triangle",
		{ x, 0.0f, z }, { x + 1.0f, 2.0f, z }, { x - 1.0f, 2.0f, z },
		YELLOW);
	x += spacing;

	Eigen::Matrix4f fakeInvViewProj = Eigen::Matrix4f::Identity();
	fakeInvViewProj = Translate({ x, 1.0f, z }) * Scale({ 1.0f, 1.0f, 1.0f });
	VisualDebugging::AddFrustum("Test_Frustum", fakeInvViewProj, MAGENTA);
	x += spacing;

	VisualDebugging::AddGrid("Test_Grid", 20, 1.0f, { 0.3f, 0.3f, 0.3f, 1.0f });

	VisualDebugging::AddArrow("Axis_X", { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, 1.0f, RED);
	VisualDebugging::AddArrow("Axis_Y", { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 1.0f, GREEN);
	VisualDebugging::AddArrow("Axis_Z", { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 1.0f, BLUE);
}

void HeliumCore::InitializeScene()
{
	{
		auto cameraEntity = CreateEntity("MainCamera");
		auto camera = CreateComponent<Camera>(cameraEntity);
		camera->SetProjectionMode(Camera::Perspective);
		Helium.GetComponent<Camera>(cameraEntity)->SetEye(Eigen::Vector3f(0.0f, 50.0f, 50.0f));
		Helium.GetComponent<Camera>(cameraEntity)->SetTarget(Eigen::Vector3f(0.0f, 0.0f, 0.0f));

		auto cameraManipulator = CreateComponent<CameraManipulatorTrackball>(cameraEntity);
		cameraManipulator->SetCamera(camera);

		auto eventSystem = GetEventSystem();
		if (eventSystem)
		{
			eventSystem->Subscribe<MousePositionEvent>("3D", [cameraEntity](const MousePositionEvent& e) {
				auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
				if (cameraManipulator) cameraManipulator->OnMousePosition(e);
				});

			eventSystem->Subscribe<MouseButtonEvent>("3D", [cameraEntity](const MouseButtonEvent& e) {
				auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
				if (cameraManipulator) cameraManipulator->OnMouseButton(e);
				});

			eventSystem->Subscribe<MouseWheelEvent>("3D", [cameraEntity](const MouseWheelEvent& e) {
				auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
				if (cameraManipulator) cameraManipulator->OnMouseWheel(e);
				});

			eventSystem->Subscribe<KeyEvent>("3D", [cameraEntity](const KeyEvent& e) {
				auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
				if (cameraManipulator) cameraManipulator->OnKey(e);
				});
		}
	}

	{
		auto entity = CreateEntity("Grid");
		auto renderable = CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Lines);

		GeometryBuilder::BuildGrid(
			renderable,
			500.0f,
			50,
			Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f)
		);

		Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
		renderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));
	}

	//InitializePrimitives();
	//InitializeVisualDebugging();

	//{
	//	VD::AddDisk("PFOR", { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 5.0f, Color::yellow());
	//}

	{
		auto entity = CreateEntity("Main");
		Helium.CreateEventCallback<CustomEvent>(entity, "3D", [](Entity e, const CustomEvent& event) {
			json j = json::parse(event.jsonString);
			if (j.contains("PointPicked"))
			{
				auto pointCloudID = j["PointPicked"]["PointCloudID"].get<int>();
				auto pickedIndex = j["PointPicked"]["PickedIndex"].get<int>();
				auto isCtrlPressed = j["PointPicked"]["IsCtrlPressed"].get<bool>();
				auto isShiftPressed = j["PointPicked"]["IsShiftPressed"].get<bool>();

				auto pointCloud = Helium.GetPointCloud(pointCloudID);
				if (nullptr != pointCloud)
				{
					Eigen::Vector3f pickedPosition = pointCloud->GetPosition(pickedIndex);
					Eigen::Vector3f pickedNormal = pointCloud->GetNormal(pickedIndex);

					if (isCtrlPressed)
					{
						auto cameraEntity = Helium.GetEntityByName("MainCamera");
						auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cameraEntity);
						if (cameraManipulator)
						{
							cameraManipulator->SetCenter(pickedPosition);
						}
					}

					if (isShiftPressed)
					{
						VD::Clear("Selected Point");
						VD::AddSphere("Selected Point", pickedPosition, pickedNormal, 1.0f, { 0.0f, 0.0f, 1.0f, 0.5f });
					}
				}
			}
			});

		Helium.CreateEventCallback<KeyEvent>(entity, "3D", [](Entity e, const KeyEvent& event) {
			if (event.action == 0 && KeyCode::Enter == event.keyCode)
			{
				Helium.NotifyMessage("Enter key pressed!");
			}
			else if (event.action == 0 && KeyCode::F1 <= event.keyCode && KeyCode::F8 >= event.keyCode)
			{
				json j;
				j["EventType"] = "TogglePointCloud";
				j["Parameters"]["Order"] = (int)event.keyCode - (int)KeyCode::F1;
				if (event.IsShiftPressed())
				{
					j["Parameters"]["Exclusive"] = true;
				}
				else
				{
					j["Parameters"]["Exclusive"] = false;
				}

				Helium.NativeToManaged(j.dump().c_str());
			}
			});
	}
}
