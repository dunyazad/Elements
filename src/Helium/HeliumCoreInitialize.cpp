#include "pch.h"

#include <glad/glad.h>

#include <Helium/HeliumCore.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>
#include <Helium/HeliumEvents.h>

#include <Helium/Systems/EventSystem.h>
#include <Helium/Systems/InputSystem.h>
#include <Helium/Systems/RenderSystem.h>
#include <Helium/Systems/ImmediateModeRenderSystem.h>

#include <Helium/Components/Components.h>

#include <Helium/GeometryBuilder.h>

extern void He_LogInternal(HeliumLogLevel level, const char* key, char* message);

void HeliumCore::InitializeScene()
{
    // 1. Camera Setup
    {
        auto cameraEntity = CreateEntity("MainCamera");
        auto camera = CreateComponent<Camera>(cameraEntity);
        camera->SetProjectionMode(Camera::Perspective);
        // 카메라 위치를 좀 더 뒤로 이동하여 전체 씬이 잘 보이게 조정
        Helium.GetComponent<Camera>(cameraEntity)->SetEye(Eigen::Vector3f(8.0f, 5.0f, 15.0f));
        Helium.GetComponent<Camera>(cameraEntity)->SetTarget(Eigen::Vector3f(8.0f, 0.0f, 0.0f));


        auto cameraManipulator = CreateComponent<CameraManipulatorTrackball>(cameraEntity);
        cameraManipulator->SetCamera(camera);

        auto eventSystem = GetEventSystem();
        if (eventSystem)
        {
            CreateEventCallback<MousePositionEvent>(cameraEntity, [](Entity entity, const MousePositionEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnMousePosition(e);
                });

            CreateEventCallback<MouseButtonEvent>(cameraEntity, [](Entity entity, const MouseButtonEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnMouseButton(e);
                });
            CreateEventCallback<MouseWheelEvent>(cameraEntity, [](Entity entity, const MouseWheelEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnMouseWheel(e);
                });
            CreateEventCallback<KeyEvent>(cameraEntity, [](Entity entity, const KeyEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnKey(e);
                });
        }
    }

    // 2. Grid (Floor)
    {
        auto entity = CreateEntity("Grid");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Lines);

        GeometryBuilder::BuildGrid(
            renderable,
            40.0f, // Size
            20,    // Divisions
            Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f)
        );

        // 바닥에 깔리도록 Y를 낮춤
        Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(0.0f, -2.0f, 0.0f));
        renderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));
    }

    // 3. Plane
    {
        auto entity = CreateEntity("Plane");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);

        GeometryBuilder::BuildPlane(
            renderable,
            10.0f, 10.0f, 10, 10,
            Eigen::Vector3f(0.0f, 0.0f, 0.0f),
            Eigen::Vector3f(0.0f, 0.0f, 1.0f),
            Eigen::Vector4f(0.7f, 0.2f, 0.2f, 1.0f)
        );
        // Plane을 뒤로 밀어서 배경처럼 사용
        Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(8.0f, 0.0f, -5.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = CreateEntity("PlaneWire");
		auto wireRenderable = CreateComponent<Renderable>(wireEntity);
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
		wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));
    }

    // 4. Primitives
    // 배치 간격
    float xStride = 2.5f;
    float currentX = 0.0f;

    // --- Box ---
    {
        auto entity = CreateEntity("Box");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildBox(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f), Eigen::Vector4f(1.0f, 0.5f, 0.5f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        // Wireframe
        auto wireEntity = CreateEntity("BoxWire");
        auto wireRenderable = CreateComponent<Renderable>(wireEntity);
        wireRenderable->Initialize(Renderable::Lines);
        GeometryBuilder::BuildBox(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(1.01f, 1.01f, 1.01f), Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
        wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Sphere ---
    {
        auto entity = CreateEntity("Sphere");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildSphere(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.6f, 24, 24, Eigen::Vector4f(0.5f, 1.0f, 0.5f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        auto wireEntity = CreateEntity("SphereWire");
        auto wireRenderable = CreateComponent<Renderable>(wireEntity);
        wireRenderable->Initialize(Renderable::Lines);
        GeometryBuilder::BuildSphere(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.61f, 12, 12, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
        wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Disk ---
    {
        auto entity = CreateEntity("Disk");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildDisk(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.7f, 32, Eigen::Vector4f(0.5f, 0.5f, 1.0f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = CreateEntity("DiskWire");
		auto wireRenderable = CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildDisk(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.71f, 16, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Cylinder ---
    {
        auto entity = CreateEntity("Cylinder");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildCylinder(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.5f, 1.5f, 24, Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        auto wireEntity = CreateEntity("CylinderWire");
        auto wireRenderable = CreateComponent<Renderable>(wireEntity);
        wireRenderable->Initialize(Renderable::Lines);
        GeometryBuilder::BuildCylinder(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.51f, 1.51f, 12, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
        wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Cone ---
    {
        auto entity = CreateEntity("Cone");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildCone(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.6f, 1.5f, 24, Eigen::Vector4f(0.0f, 1.0f, 1.0f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        auto wireEntity = CreateEntity("ConeWire");
        auto wireRenderable = CreateComponent<Renderable>(wireEntity);
        wireRenderable->Initialize(Renderable::Lines);
        GeometryBuilder::BuildCone(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.61f, 1.51f, 12, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
        wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Capsule ---
    {
        auto entity = CreateEntity("Capsule");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildCapsule(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.4f, 1.8f, 16, 8, Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        auto wireEntity = CreateEntity("CapsuleWire");
        auto wireRenderable = CreateComponent<Renderable>(wireEntity);
        wireRenderable->Initialize(Renderable::Lines);
        GeometryBuilder::BuildCapsule(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.41f, 1.81f, 16, 8, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
        wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Torus ---
    {
        auto entity = CreateEntity("Torus");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildTorus(renderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.6f, 0.2f, 32, 16, Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = CreateEntity("TorusWire");
		auto wireRenderable = CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildTorus(wireRenderable, Eigen::Vector3f(currentX, 0.0f, 0.0f), 0.61f, 0.21f, 16, 8, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Arrow ---
    {
        auto entity = CreateEntity("Arrow");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);
        GeometryBuilder::BuildArrow(renderable, Eigen::Vector3f(currentX, -0.5f, 0.0f), Eigen::Vector3f(currentX, 1.0f, 0.5f), 0.1f, 0.25f, 0.4f, Eigen::Vector4f(0.8f, 0.1f, 0.1f, 1.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

		auto wireEntity = CreateEntity("ArrowWire");
		auto wireRenderable = CreateComponent<Renderable>(wireEntity);
		wireRenderable->Initialize(Renderable::Lines);
		GeometryBuilder::BuildArrow(wireRenderable, Eigen::Vector3f(currentX, -0.5f, 0.0f), Eigen::Vector3f(currentX, 1.0f, 0.5f), 0.11f, 0.26f, 0.41f, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f), true);
		wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Frustum ---
    {
        auto entity = CreateEntity("Frustum");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Lines);

        // 임의의 ViewProjection 역행렬 생성 (시각화를 위해)
        // 1. Projection (Perspective)
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

        // 2. View (Camera at origin looking -Z)
        // 시각화할 Frustum을 현재 위치(currentX)로 이동시키기 위해 View 행렬을 조정
        Eigen::Affine3f viewTransform = Eigen::Affine3f::Identity();
        viewTransform.translate(Eigen::Vector3f(-currentX, 0.0f, -2.5f)); // 카메라가 오브젝트를 바라보는 역변환

        Eigen::Matrix4f viewProj = projection * viewTransform.matrix();
        Eigen::Matrix4f invViewProj = viewProj.inverse();

        GeometryBuilder::BuildFrustum(renderable, invViewProj, Eigen::Vector4f(0.0f, 1.0f, 1.0f, 1.0f));
        renderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    {
        // 1. 제어점(Control Points) 생성 - S자 형태의 곡선
        std::vector<Eigen::Vector3f> tubePoints;
        tubePoints.push_back(Eigen::Vector3f(0.0f, 0.0f, 0.0f));   // 시작
        tubePoints.push_back(Eigen::Vector3f(0.5f, 1.0f, 0.0f));   // 위로
        tubePoints.push_back(Eigen::Vector3f(1.5f, -1.0f, 0.5f));  // 아래로 + 깊이 변화
        tubePoints.push_back(Eigen::Vector3f(2.5f, 0.5f, -0.5f));  // 다시 위로
        tubePoints.push_back(Eigen::Vector3f(3.0f, 0.0f, 0.0f));   // 끝

        // Solid Tube
        auto entity = CreateEntity("Tube");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Triangles);

        // radius: 0.2, curveSegments: 15(부드럽게), radialSegments: 16(둥글게)
        GeometryBuilder::BuildTube(
            renderable,
            tubePoints,
            0.2f,
            15,
            16,
            Eigen::Vector4f(0.2f, 0.8f, 1.0f, 1.0f) // Cyan Color
        );

        Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(currentX, 0.0f, 0.0f));
        renderable->AddShader(CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        // Wireframe Tube
        auto wireEntity = CreateEntity("TubeWire");
        auto wireRenderable = CreateComponent<Renderable>(wireEntity);
        wireRenderable->Initialize(Renderable::Lines);

        // Wireframe은 반지름을 아주 살짝 키워서(0.21f) 겹침 방지
        GeometryBuilder::BuildTube(
            wireRenderable,
            tubePoints,
            0.205f,
            15,
            8, // 와이어프레임은 가로줄을 좀 적게(8) 해서 보기 편하게
            Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f),
            true
        );

        Helium.CreateComponent<Transform>(wireEntity)->SetLocalPosition(Eigen::Vector3f(currentX, 0.0f, 0.0f));
        wireRenderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride * 1.5f; // 튜브 길이가 있으므로 간격을 좀 더 벌림
    }
}
