#include "pch.h"
#include <Helium/VisualDebugging.h>
#include <Helium/GeometryBuilder.h>
#include <Helium/HeliumCore.h>

#include <entt/entt.hpp>

// ============================================================================
// Static Member Initialization
// ============================================================================

bool VisualDebugging::initialized = false;
std::map<std::string, Entity> VisualDebugging::entities;
std::map<std::string, DebuggingRenderable*> VisualDebugging::debuggingRenderables;
std::map<std::string, TextBlock*> VisualDebugging::textBlocks;
std::vector<std::string> VisualDebugging::selectionRenderables;
unsigned int VisualDebugging::selectionIndex = 0;

std::mutex VisualDebugging::commandMutex;
std::vector<std::function<void()>> VisualDebugging::commandQueue;
std::vector<std::function<void()>> VisualDebugging::pendingCommands;

// ============================================================================
// Internal Helper Functions
// ============================================================================

namespace
{
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

    void SetupStandardShaders(DebuggingRenderable* renderable)
    {
        auto shaderInstancing = Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs"));
        renderable->AddShader(shaderInstancing);

        auto shaderNoNormal = Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs"));
        renderable->AddShader(shaderNoNormal);

        renderable->SetActiveShaderIndex(1);
    }
}

// ============================================================================
// Core System (Lifecycle & Dispatch)
// ============================================================================

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

    constexpr long long kMaxExecutionTimeMicros = 2000; // 2ms limit
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

// ============================================================================
// Entity Creation Functions (Specific Geometry Builders)
// ============================================================================

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
    SetupStandardShaders(renderable);
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

void VisualDebugging::CreateSphereEntity(const std::string& tag, float radius, unsigned int slices, unsigned int stacks)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;
    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;
    SetupStandardShaders(renderable);

    GeometryBuilder::BuildSphere(renderable, { 0.0f, 0.0f, 0.0f }, radius, slices, stacks, Color::white());
}

void VisualDebugging::CreateDiskEntity(const std::string& tag, float radius, unsigned int slices, bool isBillboard)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;
    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;

    if (isBillboard)
    {
        // Requires a shader that handles billboard rotation (Spherical or Cylindrical)
        auto shaderBillboard = Helium.CreateShader(
            "InstancingBillboard",
            File("../../res/Shaders/InstancingBillboard.vs"),
            File("../../res/Shaders/InstancingBillboard.fs")
        );
        int shaderIndex = renderable->AddShader(shaderBillboard);
        renderable->SetActiveShaderIndex(shaderIndex);
    }
    else
    {
        SetupStandardShaders(renderable);
    }

    GeometryBuilder::BuildDisk(renderable, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, radius, slices, Color::white());
}

void VisualDebugging::CreateCylinderEntity(const std::string& tag, float radius, float height, unsigned int slices)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;
    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;
    SetupStandardShaders(renderable);

    GeometryBuilder::BuildCylinder(renderable, { 0.0f, 0.0f, 0.0f }, radius, height, slices, Color::white());
}

void VisualDebugging::CreateConeEntity(const std::string& tag, float radius, float height, unsigned int slices)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;
    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;
    SetupStandardShaders(renderable);

    GeometryBuilder::BuildCone(renderable, { 0.0f, 0.0f, 0.0f }, radius, height, slices, Color::white());
}

void VisualDebugging::CreateCapsuleEntity(const std::string& tag, float radius, float height, unsigned int rings)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;
    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;
    SetupStandardShaders(renderable);
    GeometryBuilder::BuildCapsule(renderable, { 0.0f, 0.0f, 0.0f }, radius, height, rings, 16, Color::white());
}

void VisualDebugging::CreateTorusEntity(const std::string& tag, float majorRadius, float minorRadius, unsigned int majorSegments, unsigned int minorSegments)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;
    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;
    SetupStandardShaders(renderable);
    GeometryBuilder::BuildTorus(renderable, { 0.0f, 0.0f, 0.0f }, majorRadius, minorRadius, majorSegments, minorSegments, Color::white());
}

void VisualDebugging::CreateTubeEntity(const std::string& tag, float radius, unsigned int curveSegments, unsigned int radialSegments)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;

    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;

    SetupStandardShaders(renderable);

    // 직선 튜브를 위한 경로점 생성 (Y축 기준, 중심점 0,0,0, 높이 1.0)
    // 시작점: (0, -0.5, 0), 끝점: (0, 0.5, 0)
    std::vector<Eigen::Vector3f> controlPoints;
    controlPoints.push_back(Eigen::Vector3f(0.0f, -0.5f, 0.0f));
    controlPoints.push_back(Eigen::Vector3f(0.0f, 0.5f, 0.0f));

    GeometryBuilder::BuildTube(
        renderable,
        controlPoints,   // 경로점 벡터
        radius,          // 반지름
        curveSegments,   // 곡선 분할 수 (직선이므로 1 이상이면 됨)
        radialSegments,  // 원형 분할 수
        Color::white(),  // 색상
        false            // 와이어프레임 여부
    );
}

void VisualDebugging::CreateArrowEntity(const std::string& tag)
{
    auto entity = Helium.CreateEntity(tag);
    entities[tag] = entity;

    auto renderable = Helium.CreateComponent<DebuggingRenderable>(entity);
    renderable->Initialize(Renderable::GeometryMode::Triangles);
    debuggingRenderables[tag] = renderable;

    SetupStandardShaders(renderable);

    // [중요] 인스턴싱을 위한 "Base Model" 생성
    // Y축 방향(Up), 길이 1.0, 두께는 적절한 비율로 설정
    // 이렇게 만들어두면 AddArrow에서 Scale을 통해 길이를 조절합니다.
    Eigen::Vector3f start(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f end(0.0f, 1.0f, 0.0f);

    // 비율 예시: 기둥 반지름 0.05, 헤드 반지름 0.1, 헤드 길이 0.2 (전체 길이 1.0 기준)
    GeometryBuilder::BuildArrow(
        renderable,
        start,
        end,
        0.05f,  // Stem Radius
        0.1f,   // Head Radius
        0.2f,   // Head Length
        Color::white(),
        false
    );
}

// ============================================================================
// Universal Helper (AddGeometryInstance)
// ============================================================================

void VisualDebugging::AddGeometryInstance(
    const std::string& tag,
    std::function<void(const std::string&)> createEntityFunc,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& normal,
    const Eigen::Vector3f& scale,
    const Eigen::Vector4f& color)
{
    std::lock_guard<std::mutex> lock(commandMutex);
    commandQueue.emplace_back([=]()
        {
            if (false == initialized) Initialize();

            // 1. Check if Entity exists, if not, create it using the provided lambda
            if (entities.end() == entities.find(tag))
            {
                createEntityFunc(tag);
            }

            // 2. Safety check in case creation failed
            if (debuggingRenderables.find(tag) == debuggingRenderables.end()) return;

            auto& renderable = debuggingRenderables[tag];

            // 3. Add Instance Data
            renderable->AddInstanceColor(color);
            renderable->AddInstanceNormal(normal);

            Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f rot = Eigen::Matrix4f::Identity();

            // 4. Calculate Rotation (Align Y-up model to Normal)
            if (normal.norm() > 0.0001f)
            {
                Eigen::Quaternionf q;
                q.setFromTwoVectors(Eigen::Vector3f::UnitY(), normal.normalized());
                rot.block<3, 3>(0, 0) = q.toRotationMatrix();
            }

            // 5. Calculate Final Transform (T * R * S)
            tm = Translate(center) * rot * Scale(scale);
            renderable->AddInstanceTransform(tm);
            if (1.0f > color.w())
            {
				renderable->SetUseAlpha(true);
            }

            renderable->EnableInstancing();
        });
}

// ============================================================================
// Public Interface Implementation
// ============================================================================

void VisualDebugging::AddBox(const std::string& tag, const AABB& aabb, const Eigen::Vector4f& color)
{
    auto center = (aabb.min + aabb.max) * 0.5f;
    auto dimensions = aabb.max - aabb.min;
    AddBox(tag, center, { 0.0f, 1.0f, 0.0f }, dimensions, color);
}

void VisualDebugging::AddBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
    AddBox(tag, center, { 0.0f, 1.0f, 0.0f }, dimensions, color);
}

void VisualDebugging::AddBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
    AddGeometryInstance(tag, CreateBoxEntity, center, normal, dimensions, color);
}

void VisualDebugging::AddWiredBox(const std::string& tag, const AABB& aabb, const Eigen::Vector4f& color)
{
    auto center = (aabb.min + aabb.max) * 0.5f;
    auto dimensions = aabb.max - aabb.min;
    AddWiredBox(tag, center, { 0.0f, 1.0f, 0.0f }, dimensions, color);
}

void VisualDebugging::AddWiredBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
    AddWiredBox(tag, center, { 0.0f, 1.0f, 0.0f }, dimensions, color);
}

void VisualDebugging::AddWiredBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color)
{
    AddGeometryInstance(tag, CreateWiredBoxEntity, center, normal, dimensions, color);
}

void VisualDebugging::AddSphere(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, const Eigen::Vector4f& color)
{
    // 기본 품질(slices=16, stacks=16) 사용
    unsigned int slices = 6;
    unsigned int stacks = 6;

    // Scale 계산: Unit Sphere(0.5) * 2 * radius = radius
    Eigen::Vector3f scale(radius * 2.0f, radius * 2.0f, radius * 2.0f);

    AddGeometryInstance(tag, [=](const std::string& t) {
        // [중요] 인스턴싱을 위해 Base Model은 Unit Size(0.5)로 생성
        CreateSphereEntity(t, 0.5f, slices, stacks);
        }, center, normal, scale, color);
}

void VisualDebugging::AddDisk(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, unsigned int slices, const Eigen::Vector4f& color, bool isBillboard)
{
    Eigen::Vector3f scale(radius * 2.0f, 1.0f, radius * 2.0f);

    AddGeometryInstance(tag, [=](const std::string& t) {
        CreateDiskEntity(t, 0.5f, slices, isBillboard);
        }, center, normal, scale, color);
}

void VisualDebugging::AddCylinder(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, float height, const Eigen::Vector4f& color)
{
    unsigned int slices = 16;
    Eigen::Vector3f scale(radius * 2.0f, height, radius * 2.0f);

    AddGeometryInstance(tag, [=](const std::string& t) {
        // Base Model: Radius 0.5, Height 1.0
        CreateCylinderEntity(t, 0.5f, 1.0f, slices);
        }, center, normal, scale, color);
}

void VisualDebugging::AddCone(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, float height, const Eigen::Vector4f& color)
{
    unsigned int slices = 16;
    Eigen::Vector3f scale(radius * 2.0f, height, radius * 2.0f);

    AddGeometryInstance(tag, [=](const std::string& t) {
        // Base Model: Radius 0.5, Height 1.0
        CreateConeEntity(t, 0.5f, 1.0f, slices);
        }, center, normal, scale, color);
}

void VisualDebugging::AddCapsule(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, float height, unsigned int rings, const Eigen::Vector4f& color)
{
    Eigen::Vector3f scale(radius * 2.0f, height, radius * 2.0f);

    AddGeometryInstance(tag, [=](const std::string& t) {
        // Capsule은 Scale이 비균등(반원 부분 유지)해야 하므로
        // 정확한 표현을 위해서는 Scale대신 Mesh 자체를 해당 크기로 만들어야 할 수도 있습니다.
        // 하지만 여기서는 Debug용이므로 Scale 근사 방식을 사용하고, Base는 Unit으로 생성합니다.
        CreateCapsuleEntity(t, 0.5f, 1.0f, rings);
        }, center, normal, scale, color);
}

void VisualDebugging::AddTorus(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float majorRadius, float minorRadius, unsigned int majorSegments, unsigned int minorSegments, const Eigen::Vector4f& color)
{
    Eigen::Vector3f scale(1.0f, 1.0f, 1.0f);

    AddGeometryInstance(tag, [=](const std::string& t)
        {
            CreateTorusEntity(t, majorRadius, minorRadius, majorSegments, minorSegments);
        }, center, normal, scale, color);
}

void VisualDebugging::AddTube(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, unsigned int curveSegments, unsigned int radialSegments, const Eigen::Vector4f& color)
{
    // Scale 계산
    // 기본 모델의 높이가 1.0이므로, 높이를 조절하려면 Y scale을 변경해야 합니다.
    // 현재 함수 인자에는 height가 없으므로 기본 1.0 높이로 그려지거나,
    // 필요하다면 AddTube 인자에 height를 추가해야 합니다.
    // 여기서는 일단 높이 1.0을 유지합니다.
    Eigen::Vector3f scale(radius * 2.0f, 1.0f, radius * 2.0f);

    AddGeometryInstance(tag, [=](const std::string& t) {
        // Base Model: Radius 0.5f (Unit Size), Segments 전달
        CreateTubeEntity(t, 0.5f, curveSegments, radialSegments);
        }, center, normal, scale, color);
}

void VisualDebugging::AddArrow(const std::string& tag, const Eigen::Vector3f& start, const Eigen::Vector3f& direction, float length, const Eigen::Vector4f& color)
{
    if (length < 0.0001f) return;

    // Scale 계산
    // X, Z축(두께)은 길이와 무관하게 고정하거나, 길이에 비례하게 할 수 있습니다.
    // 여기서는 길이에 비례하여 두께가 커지는 방식(Uniform Scale 느낌)을 적용합니다.
    // 만약 두께를 고정하고 싶다면 scale.x, scale.z에 1.0f 등을 넣으시면 됩니다.

    // 1. 길이(Y축) 적용
    Eigen::Vector3f scale(length, length, length);

    // 2. 방향 벡터 정규화
    Eigen::Vector3f dirNormalized = direction.normalized();

    // 3. 인스턴스 추가 (CreateArrowEntity 람다 전달)
    AddGeometryInstance(tag, CreateArrowEntity, start, dirNormalized, scale, color);
}

void VisualDebugging::AddFrustum(const std::string& tag, const Eigen::Matrix4f& invViewProj, const Eigen::Vector4f& color)
{
    // NDC Corners
    std::vector<Eigen::Vector4f> corners = {
        {-1, -1, -1, 1}, { 1, -1, -1, 1}, { 1,  1, -1, 1}, {-1,  1, -1, 1}, // Near
        {-1, -1,  1, 1}, { 1, -1,  1, 1}, { 1,  1,  1, 1}, {-1,  1,  1, 1}  // Far
    };

    std::vector<Eigen::Vector3f> worldCorners;
    worldCorners.reserve(8);

    for (const auto& c : corners)
    {
        Eigen::Vector4f worldPos = invViewProj * c;
        worldCorners.push_back(worldPos.head<3>() / worldPos.w());
    }

    // Draw Near Plane
    AddLine(tag, worldCorners[0], worldCorners[1], color);
    AddLine(tag, worldCorners[1], worldCorners[2], color);
    AddLine(tag, worldCorners[2], worldCorners[3], color);
    AddLine(tag, worldCorners[3], worldCorners[0], color);

    // Draw Far Plane
    AddLine(tag, worldCorners[4], worldCorners[5], color);
    AddLine(tag, worldCorners[5], worldCorners[6], color);
    AddLine(tag, worldCorners[6], worldCorners[7], color);
    AddLine(tag, worldCorners[7], worldCorners[4], color);

    // Draw Connections
    AddLine(tag, worldCorners[0], worldCorners[4], color);
    AddLine(tag, worldCorners[1], worldCorners[5], color);
    AddLine(tag, worldCorners[2], worldCorners[6], color);
    AddLine(tag, worldCorners[3], worldCorners[7], color);
}

void VisualDebugging::AddGrid(const std::string& tag, int divisions, float spacing, const Eigen::Vector4f& color)
{
    float halfSize = (divisions * spacing) * 0.5f;

    for (int i = 0; i <= divisions; ++i)
    {
        float offset = -halfSize + (i * spacing);

        // X Axis Lines
        AddLine(tag, { -halfSize, 0.0f, offset }, { halfSize, 0.0f, offset }, color);

        // Z Axis Lines
        AddLine(tag, { offset, 0.0f, -halfSize }, { offset, 0.0f, halfSize }, color);
    }
}

// ============================================================================
// Low-Level Primitives (No Instancing)
// ============================================================================

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

// ============================================================================
// Visibility & Selection Management
// ============================================================================

void VisualDebugging::Clear(const std::string& tag)
{
    std::lock_guard<std::mutex> lock(commandMutex);
    commandQueue.emplace_back([=]()
        {
            //if (false == initialized) Initialize();
            if (debuggingRenderables.find(tag) != debuggingRenderables.end())
            {
                auto& renderable = debuggingRenderables[tag];
                if (renderable->IsInstancingEnabled())
                    renderable->ClearInstancingData();
                else
                    renderable->Clear();
            }
        });
}

void VisualDebugging::ClearAll()
{
    std::lock_guard<std::mutex> lock(commandMutex);
    commandQueue.emplace_back([=]()
        {
            for (auto& kvp : debuggingRenderables)
            {
                if (kvp.second->IsInstancingEnabled())
                    kvp.second->ClearInstancingData();
                else
                    kvp.second->Clear();
            }
        });
}

void VisualDebugging::SetVisibility(bool visible, const std::string& tag)
{
    std::lock_guard<std::mutex> lock(commandMutex);
    commandQueue.emplace_back([=]()
        {
            if (false == initialized) Initialize();
            if (debuggingRenderables.find(tag) != debuggingRenderables.end())
            {
                debuggingRenderables[tag]->SetVisible(visible);
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
            if (debuggingRenderables.find(tag) != debuggingRenderables.end())
            {
                auto& renderable = debuggingRenderables[tag];
                renderable->SetVisible(!renderable->IsVisible());
            }
        });
}

void VisualDebugging::ToggleVisibilityAll()
{
    std::lock_guard<std::mutex> lock(commandMutex);
    commandQueue.emplace_back([=]()
        {
            if (false == initialized) Initialize();
            for (auto& kvp : debuggingRenderables)
            {
                kvp.second->SetVisible(!kvp.second->IsVisible());
            }
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

            if (debuggingRenderables.find(tag) != debuggingRenderables.end())
            {
                // Optional: Check duplication before adding
                // if (std::find(selectionRenderables.begin(), selectionRenderables.end(), tag) == selectionRenderables.end())
                selectionRenderables.push_back(tag);
            }
        });
}

unsigned int VisualDebugging::ShowNextSelection()
{
    if (selectionRenderables.empty()) return 0;

    for (auto& tag : selectionRenderables) SetVisibility(false, tag);

    selectionIndex++;
    selectionIndex = selectionIndex % selectionRenderables.size();

    SetVisibility(true, selectionRenderables[selectionIndex]);
    return selectionIndex;
}

unsigned int VisualDebugging::ShowPreviousSelection()
{
    if (selectionRenderables.empty()) return 0;

    for (auto& tag : selectionRenderables) SetVisibility(false, tag);

    if (0 == selectionIndex) selectionIndex = (unsigned int)selectionRenderables.size();
    selectionIndex--;

    SetVisibility(true, selectionRenderables[selectionIndex]);
    return selectionIndex;
}
