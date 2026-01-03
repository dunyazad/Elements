#include "pch.h"

#include <thread>

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
#include <Helium/Serialization.hpp>

#include <Helium/VisualDebugging.h>

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
        // Plane을 뒤로 밀어서 배경처럼 사용
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
        renderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride;
    }

    // --- Tube ---
    {
        // 1. 제어점(Control Points) 생성 - S자 형태의 곡선
        std::vector<Eigen::Vector3f> tubePoints;
        tubePoints.push_back(Eigen::Vector3f(0.0f, 0.0f, 0.0f));   // 시작
        tubePoints.push_back(Eigen::Vector3f(0.5f, 1.0f, 0.0f));   // 위로
        tubePoints.push_back(Eigen::Vector3f(1.5f, -1.0f, 0.5f));  // 아래로 + 깊이 변화
        tubePoints.push_back(Eigen::Vector3f(2.5f, 0.5f, -0.5f));  // 다시 위로
        tubePoints.push_back(Eigen::Vector3f(3.0f, 0.0f, 0.0f));   // 끝

        // Solid Tube
        auto entity = Helium.CreateEntity("Tube");
        auto renderable = Helium.CreateComponent<Renderable>(entity);
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
        renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

        // Wireframe Tube
        auto wireEntity = Helium.CreateEntity("TubeWire");
        auto wireRenderable = Helium.CreateComponent<Renderable>(wireEntity);
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
        wireRenderable->AddShader(Helium.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

        currentX += xStride * 1.5f; // 튜브 길이가 있으므로 간격을 좀 더 벌림
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
    // 1. 기본 도형 테스트 (Row 0: Z = 0)
    // ---------------------------------------------------------
    float x = -10.0f;
    float z = 0.0f;
    float spacing = 3.0f;

    // Box
    VisualDebugging::AddBox("Test_Box", { x, 1.0f, z }, { 1.0f, 1.0f, 1.0f }, RED);
    x += spacing;

    // Sphere
    VisualDebugging::AddSphere("Test_Sphere", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f}, 0.5f, GREEN);
    x += spacing;

    // Cylinder (기본 Up 방향)
    VisualDebugging::AddCylinder("Test_Cylinder", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 2.0f, BLUE);
    x += spacing;

    // Cone
    VisualDebugging::AddCone("Test_Cone", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 2.0f, YELLOW);
    x += spacing;

    // Capsule
    VisualDebugging::AddCapsule("Test_Capsule", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 1.0f, 16, CYAN);
    x += spacing;

    // Disk
    VisualDebugging::AddDisk("Test_Disk", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 1.0f, MAGENTA);
    x += spacing;

    // Torus
    VisualDebugging::AddTorus("Test_Torus", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 1.0f, 0.3f, 32, 16, WHITE);

    // ---------------------------------------------------------
    // 2. 특수 및 복합 도형 (Row 1: Z = 3)
    // ---------------------------------------------------------
    x = -10.0f;
    z = 3.0f;

    // Wired Box
    VisualDebugging::AddWiredBox("Test_WiredBox", { x, 1.0f, z }, { 1.2f, 1.2f, 1.2f }, WHITE);
    x += spacing;

    // Tube (수정된 로직 검증: ControlPoints 기반)
    // 반지름 0.5, 곡선분할 16, 원형분할 16
    VisualDebugging::AddTube("Test_Tube", { x, 1.0f, z }, { 0.0f, 1.0f, 0.0f }, 0.5f, 16, 16, CYAN);
    x += spacing;

    // Arrow (Up)
    VisualDebugging::AddArrow("Test_Arrow_Up", { x, 0.0f, z }, { 0.0f, 1.0f, 0.0f }, 2.0f, GREEN);
    x += spacing;

    // Arrow (Right) - 회전 로직 검증
    VisualDebugging::AddArrow("Test_Arrow_Right", { x, 1.0f, z }, { 1.0f, 0.0f, 0.0f }, 2.0f, RED);
    x += spacing;

    // Arrow (Forward) - 회전 로직 검증
    VisualDebugging::AddArrow("Test_Arrow_Forward", { x, 1.0f, z }, { 0.0f, 0.0f, 1.0f }, 2.0f, BLUE);
    x += spacing;

    // Arrow (Diagonal) - 임의의 방향 회전 검증
    Eigen::Vector3f diag = Eigen::Vector3f(1.0f, 1.0f, 1.0f).normalized();
    VisualDebugging::AddArrow("Test_Arrow_Diag", { x, 1.0f, z }, diag, 2.0f, YELLOW);

    // ---------------------------------------------------------
    // 3. 라인 및 기타 (Row 2: Z = 6)
    // ---------------------------------------------------------
    x = -10.0f;
    z = 6.0f;

    // Line
    VisualDebugging::AddLine("Test_Line", { x - 1.0f, 0.0f, z }, { x + 1.0f, 2.0f, z }, RED, BLUE);
    x += spacing;

    // Triangle
    VisualDebugging::AddTriangle("Test_Triangle",
        { x, 0.0f, z }, { x - 1.0f, 2.0f, z }, { x + 1.0f, 2.0f, z },
        YELLOW);
    x += spacing;

    // Frustum (가상의 ViewProj 역행렬 생성)
    // 간단한 박스 형태의 Frustum 테스트
    Eigen::Matrix4f fakeInvViewProj = Eigen::Matrix4f::Identity();
    fakeInvViewProj = Translate({ x, 1.0f, z }) * Scale({ 1.0f, 1.0f, 1.0f });
    VisualDebugging::AddFrustum("Test_Frustum", fakeInvViewProj, MAGENTA);
    x += spacing;

    // ---------------------------------------------------------
    // 4. 바닥 그리드
    // ---------------------------------------------------------
    // 원점(0,0,0) 바닥에 20x20 그리드
    VisualDebugging::AddGrid("Test_Grid", 20, 1.0f, { 0.3f, 0.3f, 0.3f, 1.0f });

    // 좌표축 표시 (원점)
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

    {
        auto entity = CreateEntity("Grid");
        auto renderable = CreateComponent<Renderable>(entity);
        renderable->Initialize(Renderable::Lines);

        GeometryBuilder::BuildGrid(
            renderable,
            500.0f, // Size
            50,    // Divisions
            Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f)
        );

        Helium.CreateComponent<Transform>(entity)->SetLocalPosition(Eigen::Vector3f(0.0f, -2.0f, 0.0f));
        renderable->AddShader(CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));
    }

    //InitializePrimitives();

    InitializeVisualDebugging();

    return;

    {
        std::string plyFilename = "../../res/PLY/Compound.ply";

        auto entity = Helium.CreateEntity("PointCloud");

        Helium.CreateEventCallback<KeyEvent>(entity, [](Entity entity, const KeyEvent& event) {
            auto renderable = Helium.GetComponent<Renderable>(entity);
            if (nullptr == renderable) return;

            if (0 == event.action)
            {
                if (GLFW_KEY_GRAVE_ACCENT == event.keyCode)
                {
                    renderable->NextDrawingMode();
                }
                else if (GLFW_KEY_1 == event.keyCode)
                {
                    renderable->SetActiveShaderIndex(0);
                }
                else if (GLFW_KEY_2 == event.keyCode)
                {
                    renderable->SetActiveShaderIndex(1);
                }
            }
            });

        auto renderable = Helium.CreateComponent<Renderable>(entity);

        PLYFormat ply;
        ply.Deserialize(plyFilename);
        ply.SwapAxisYZ();

        renderable->Initialize(Renderable::GeometryMode::Triangles);

        renderable->AddShader(Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
        renderable->AddShader(Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
        renderable->SetActiveShaderIndex(1);

        GeometryBuilder::BuildSphere(renderable, { 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6, { 1.0f, 1.0f, 1.0f, 1.0f });

        // Eigen 기반 구조에서는 size()가 점의 개수를 의미합니다. (더 이상 / 3 불필요)
        size_t pointCount = ply.GetPoints().size();
        //pointCloud.Resize(pointCount);

        for (size_t i = 0; i < pointCount; i++)
        {
            const Eigen::Vector3f& p = ply.GetPoints()[i];

            // Normals (데이터가 없을 경우 예외 처리)
            Eigen::Vector3f n = (i < ply.GetNormals().size()) ? ply.GetNormals()[i] : Eigen::Vector3f::Zero();

            renderable->AddInstanceNormal(n);

            // Colors (PLYFormat이 내부적으로 Vector4f로 색상을 관리한다고 가정)
            // 만약 색상이 없다면 기본 흰색 사용
            Eigen::Vector4f c = (i < ply.GetColors().size()) ? ply.GetColors()[i] : Eigen::Vector4f::Ones();

            renderable->AddInstanceColor(c);
            //pointCloud.SetColor(i, c.x(), c.y(), c.z());
            //
            //pointCloud.SetPosition(i, p.x(), p.y(), p.z());
            //pointCloud.SetNormal(i, n.x(), n.y(), n.z());

            // Deep Learning Classes (만약 PLYFormat에 해당 기능이 유지되었다면 사용)
            // if (!ply.GetDeepLearningClasses().empty())
            // {
            // 	pointCloud.SetPointDeepLearningClassID(i, ply.GetDeepLearningClasses()[i]);
            // }

            // --- Transformation Matrix (Eigen) ---
            Eigen::Affine3f tm = Eigen::Affine3f::Identity();
            Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();

            // 법선 벡터가 유효하다면 회전 계산 (Z축을 법선 방향으로 정렬)
            if (n.norm() > 0.0001f)
            {
                Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
                Eigen::Vector3f normalDir = n.normalized();

                // GLM의 axis/angle 로직을 Eigen으로 변환
                // 혹은 더 간단하게: Eigen::Quaternionf::FromTwoVectors(up, normalDir).toRotationMatrix();

                Eigen::Vector3f axis = up.cross(normalDir);
                float dot = up.dot(normalDir);

                // acos 범위 안전장치
                if (dot > 1.0f) dot = 1.0f;
                else if (dot < -1.0f) dot = -1.0f;

                float angle = std::acos(dot);

                if (axis.norm() > 0.0001f)
                {
                    axis.normalize();
                    rot = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
                }
                else if (dot < -0.9f) // 정반대 방향일 경우 (180도 회전)
                {
                    rot = Eigen::AngleAxisf(3.1415926f, Eigen::Vector3f::UnitX()).toRotationMatrix();
                }
            }

            // 변환 적용 순서: Translate * Rotate * Scale
            tm.translate(p);
            tm.rotate(rot);
            tm.scale(0.1f);

            renderable->AddInstanceTransform(tm.matrix());
            renderable->IncreaseNumberOfInstances();
        }

        renderable->EnableInstancing();
    }
}
