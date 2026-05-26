#if 0
#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include "HalfEdgeMesh.hpp"

namespace Eigen
{
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

void AddMesh(const std::string& name, const HEM::Mesh& mesh)
{
    auto entity = Helium.CreateEntity(name);
    auto renderable = Helium.CreateComponent<Renderable>(entity);
    renderable->Initialize(Renderable::Triangles);
    renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
    renderable->SetFaceCullingMode(Renderable::BackFace);

    Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [renderable](Entity e, const KeyEvent& event)
        {
            if (event.action == 1 && KeyCode::D1 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::Solid);
            }
            else if (event.action == 1 && KeyCode::D2 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::WireFrame);
            }
            else if (event.action == 1 && KeyCode::D3 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::WireFrameOverSolid);
            }
            else if (event.action == 1 && KeyCode::D4 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::Point);
            }
        });

    Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [renderable](Entity e, const MouseButtonEvent& event)
        {
            if (event.action == 1 && event.button == MouseButton::Left)
            {
                auto cameraEntity = Helium.GetEntityByName("MainCamera");
                auto camera = Helium.GetComponent<Camera>(cameraEntity);
                if (nullptr == camera)
                {
                    return;
                }

                Ray ray = camera->ScreenPointToRay(
                    (float)event.xpos,
                    (float)event.ypos,
                    Helium.GetWidth(),
                    Helium.GetHeight()
                );

                int pickedTriangleIndex = -1;
                float pickedDistance = std::numeric_limits<float>::max();

                for (size_t i = 0; i < renderable->GetNumberOfIndices() / 3; i++)
                {
                    auto i0 = renderable->GetIndex(i * 3);
                    auto i1 = renderable->GetIndex(i * 3 + 1);
                    auto i2 = renderable->GetIndex(i * 3 + 2);
                    Eigen::Vector3f v0 = renderable->GetVertex(i0);
                    Eigen::Vector3f v1 = renderable->GetVertex(i1);
                    Eigen::Vector3f v2 = renderable->GetVertex(i2);

                    auto normal = (v1 - v0).cross(v2 - v0).normalized();

                    float t = 0.0f;
                    if (ray.IntersectTriangle(v0, v1, v2, t))
                    {
                        if (t < pickedDistance)
                        {
                            pickedDistance = t;
                            pickedTriangleIndex = (int)i;
                        }
                    }
                }

                if (-1 != pickedTriangleIndex)
                {
                    VD::Clear("HitPoint");
                    VD::Clear("HitTriangle");

                    auto i0 = renderable->GetIndex(pickedTriangleIndex * 3);
                    auto i1 = renderable->GetIndex(pickedTriangleIndex * 3 + 1);
                    auto i2 = renderable->GetIndex(pickedTriangleIndex * 3 + 2);
                    Eigen::Vector3f v0 = renderable->GetVertex(i0);
                    Eigen::Vector3f v1 = renderable->GetVertex(i1);
                    Eigen::Vector3f v2 = renderable->GetVertex(i2);
                    auto normal = (v1 - v0).cross(v2 - v0).normalized();

                    Eigen::Vector3f hitPoint = ray.origin + ray.direction * pickedDistance;
                    VD::AddSphere("HitPoint", hitPoint, normal, 0.01f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

                    VD::AddTriangle("HitTriangle", v0, v1, v2, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

                    if (event.modifiers & static_cast<int>(KeyModifiers::Control))
                    {
                        camera->SetTarget(hitPoint);
                    }
                }
            }
        });

    const auto& points = mesh.GetPoints();
    const auto& vertices = mesh.GetVertices();
    const auto& edges = mesh.GetEdges();
    const auto& faces = mesh.GetFaces();

    std::vector<Eigen::Vector3f> positions(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
    {
        positions[i] = points[vertices[i].pid];
    }

    std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        Eigen::Vector3f normal = (positions[v1] - positions[v0]).cross(positions[v2] - positions[v0]).normalized();
        normals[v0] += normal;
        normals[v1] += normal;
        normals[v2] += normal;
    }

    for (size_t i = 0; i < normals.size(); i++)
    {
        normals[i].normalize();
    }

    std::vector<unsigned int> indices;
    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        indices.push_back((unsigned int)v0);
        indices.push_back((unsigned int)v1);
        indices.push_back((unsigned int)v2);
    }

    renderable->AddVertices(positions);
    renderable->AddNormals(normals);
    renderable->AddColors4(std::vector<Eigen::Vector4f>(positions.size(), Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f)));
    renderable->AddIndices(indices);
}

class AppHalfEdgeMesh : public App
{
public:
    void Execute_Intersection()
    {
        TS(MeshA);
        HEM::Mesh mesh_A;
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");

            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
            {
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            }

            mesh_A.Build(stl.GetPoints(), indices);
        }
        TE(MeshA);

        TS(MeshB);
        HEM::Mesh mesh_B;
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");

            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
            {
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            }

            mesh_B.Build(stl.GetPoints(), indices);
        }
        TE(MeshB);

        TS(IntersectionLines);
        auto intersectionResult = mesh_A.FindIntersectionLines(mesh_B);
        for (auto& result : intersectionResult)
        {
            VD::AddLine("Intersection", result.first, result.second, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
        }
        TE(IntersectionLines);

        TS(SplitFaces);
        mesh_A.SplitIntersectingFaces(mesh_B);
        TE(SplitFaces);

        AddMesh("Mesh_A_Split", mesh_A);
        //AddMesh("Mesh_B", mesh_B);
    }

    virtual void Execute() override
    {
        Execute_Intersection();
    }
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");

#endif // 0
















#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include "HalfEdgeMesh.hpp"

namespace Eigen
{
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

void AddMesh(const std::string& name, const HEM::Mesh& mesh, const Eigen::Vector4f& color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f), const Eigen::Vector3f& offset = Eigen::Vector3f::Zero())
{
    auto entity = Helium.CreateEntity(name);
    auto renderable = Helium.CreateComponent<Renderable>(entity);
    renderable->Initialize(Renderable::Triangles);
    renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
    renderable->SetFaceCullingMode(Renderable::NoCulling);

    Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [renderable](Entity e, const KeyEvent& event)
        {
            if (event.action == 1 && KeyCode::D1 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::Solid);
            }
            else if (event.action == 1 && KeyCode::D2 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::WireFrame);
            }
            else if (event.action == 1 && KeyCode::D3 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::WireFrameOverSolid);
            }
            else if (event.action == 1 && KeyCode::D4 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::Point);
            }
        });

    Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [renderable](Entity e, const MouseButtonEvent& event)
        {
            if (event.action == 1 && event.button == MouseButton::Left)
            {
                auto cameraEntity = Helium.GetEntityByName("MainCamera");
                auto camera = Helium.GetComponent<Camera>(cameraEntity);
                if (nullptr == camera)
                {
                    return;
                }

                Ray ray = camera->ScreenPointToRay(
                    (float)event.xpos,
                    (float)event.ypos,
                    Helium.GetWidth(),
                    Helium.GetHeight()
                );

                int pickedTriangleIndex = -1;
                float pickedDistance = std::numeric_limits<float>::max();

                for (size_t i = 0; i < renderable->GetNumberOfIndices() / 3; i++)
                {
                    auto i0 = renderable->GetIndex(i * 3);
                    auto i1 = renderable->GetIndex(i * 3 + 1);
                    auto i2 = renderable->GetIndex(i * 3 + 2);
                    Eigen::Vector3f v0 = renderable->GetVertex(i0);
                    Eigen::Vector3f v1 = renderable->GetVertex(i1);
                    Eigen::Vector3f v2 = renderable->GetVertex(i2);

                    auto normal = (v1 - v0).cross(v2 - v0).normalized();

                    float t = 0.0f;
                    if (ray.IntersectTriangle(v0, v1, v2, t))
                    {
                        if (t < pickedDistance)
                        {
                            pickedDistance = t;
                            pickedTriangleIndex = (int)i;
                        }
                    }
                }

                if (-1 != pickedTriangleIndex)
                {
                    VD::Clear("HitPoint");
                    VD::Clear("HitTriangle");

                    auto i0 = renderable->GetIndex(pickedTriangleIndex * 3);
                    auto i1 = renderable->GetIndex(pickedTriangleIndex * 3 + 1);
                    auto i2 = renderable->GetIndex(pickedTriangleIndex * 3 + 2);
                    Eigen::Vector3f v0 = renderable->GetVertex(i0);
                    Eigen::Vector3f v1 = renderable->GetVertex(i1);
                    Eigen::Vector3f v2 = renderable->GetVertex(i2);
                    auto normal = (v1 - v0).cross(v2 - v0).normalized();

                    Eigen::Vector3f hitPoint = ray.origin + ray.direction * pickedDistance;
                    VD::AddSphere("HitPoint", hitPoint, normal, 0.01f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

                    VD::AddTriangle("HitTriangle", v0, v1, v2, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

                    if (event.modifiers & static_cast<int>(KeyModifiers::Control))
                    {
                        camera->SetTarget(hitPoint);
                    }
                }
            }
        });

    const auto& points = mesh.GetPoints();
    const auto& vertices = mesh.GetVertices();
    const auto& edges = mesh.GetEdges();
    const auto& faces = mesh.GetFaces();

    std::vector<Eigen::Vector3f> positions(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
    {
        positions[i] = points[vertices[i].pid] + offset;
    }

    std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        Eigen::Vector3f normal = (positions[v1] - positions[v0]).cross(positions[v2] - positions[v0]).normalized();
        normals[v0] += normal;
        normals[v1] += normal;
        normals[v2] += normal;
    }

    for (size_t i = 0; i < normals.size(); i++)
    {
        normals[i].normalize();
    }

    std::vector<unsigned int> indices;
    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        indices.push_back((unsigned int)v0);
        indices.push_back((unsigned int)v1);
        indices.push_back((unsigned int)v2);
    }

    renderable->AddVertices(positions);
    renderable->AddNormals(normals);
    renderable->AddColors4(std::vector<Eigen::Vector4f>(positions.size(), color));
    renderable->AddIndices(indices);
}

class HeliumMesh : public HEM::Mesh
{
public:
    HeliumMesh() : HEM::Mesh()
    {
    }

    virtual ~HeliumMesh()
    {
    }

    Entity entity = InvalidEntity;

    bool isDirty = true;

    Eigen::Vector3f offset = Eigen::Vector3f::Zero();
    Eigen::Vector4f color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

    void Update()
    {
        if (false == isDirty)
        {
            return;
        }

        if (InvalidEntity == entity)
        {
            entity = Helium.CreateEntity("HeliumMesh");
            auto renderable = Helium.CreateComponent<Renderable>(entity);
            renderable->Initialize(Renderable::Triangles);
            renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
            renderable->SetFaceCullingMode(Renderable::NoCulling);

            Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [](Entity e, const KeyEvent& event)
                {
                    auto currentRenderable = Helium.GetComponent<Renderable>(e);
                    if (nullptr == currentRenderable)
                    {
                        return;
                    }

                    if (event.action == 1 && KeyCode::D1 == event.keyCode)
                    {
                        currentRenderable->SetDrawingMode(Renderable::Solid);
                    }
                    else if (event.action == 1 && KeyCode::D2 == event.keyCode)
                    {
                        currentRenderable->SetDrawingMode(Renderable::WireFrame);
                    }
                    else if (event.action == 1 && KeyCode::D3 == event.keyCode)
                    {
                        currentRenderable->SetDrawingMode(Renderable::WireFrameOverSolid);
                    }
                    else if (event.action == 1 && KeyCode::D4 == event.keyCode)
                    {
                        currentRenderable->SetDrawingMode(Renderable::Point);
                    }
                });

            Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [this](Entity e, const MouseButtonEvent& event)
                {
                    if (event.action == 1 && event.button == MouseButton::Left)
                    {
                        auto currentRenderable = Helium.GetComponent<Renderable>(e);
                        if (nullptr == currentRenderable)
                        {
                            return;
                        }

                        auto cameraEntity = Helium.GetEntityByName("MainCamera");
                        auto camera = Helium.GetComponent<Camera>(cameraEntity);
                        if (nullptr == camera)
                        {
                            return;
                        }

                        Ray ray = camera->ScreenPointToRay(
                            (float)event.xpos,
                            (float)event.ypos,
                            Helium.GetWidth(),
                            Helium.GetHeight()
                        );

                        int pickedTriangleIndex = -1;
                        float pickedDistance = std::numeric_limits<float>::max();

                        for (size_t i = 0; i < currentRenderable->GetNumberOfIndices() / 3; i++)
                        {
                            auto i0 = currentRenderable->GetIndex(i * 3);
                            auto i1 = currentRenderable->GetIndex(i * 3 + 1);
                            auto i2 = currentRenderable->GetIndex(i * 3 + 2);
                            Eigen::Vector3f v0 = currentRenderable->GetVertex(i0);
                            Eigen::Vector3f v1 = currentRenderable->GetVertex(i1);
                            Eigen::Vector3f v2 = currentRenderable->GetVertex(i2);

                            auto normal = (v1 - v0).cross(v2 - v0).normalized();

                            float t = 0.0f;
                            if (ray.IntersectTriangle(v0, v1, v2, t))
                            {
                                if (t < pickedDistance)
                                {
                                    pickedDistance = t;
                                    pickedTriangleIndex = (int)i;
                                }
                            }
                        }

                        if (-1 != pickedTriangleIndex)
                        {
                            VD::Clear("HitPoint");
                            VD::Clear("HitTriangle");

                            auto i0 = currentRenderable->GetIndex(pickedTriangleIndex * 3);
                            auto i1 = currentRenderable->GetIndex(pickedTriangleIndex * 3 + 1);
                            auto i2 = currentRenderable->GetIndex(pickedTriangleIndex * 3 + 2);
                            Eigen::Vector3f v0 = currentRenderable->GetVertex(i0);
                            Eigen::Vector3f v1 = currentRenderable->GetVertex(i1);
                            Eigen::Vector3f v2 = currentRenderable->GetVertex(i2);
                            auto normal = (v1 - v0).cross(v2 - v0).normalized();

                            Eigen::Vector3f hitPoint = ray.origin + ray.direction * pickedDistance;
                            //VD::AddSphere("HitPoint", hitPoint, normal, 0.01f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

                            //VD::AddTriangle("HitTriangle", v0, v1, v2, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

							//color = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f);

							SplitFaceByPoint(pickedTriangleIndex, hitPoint);

							isDirty = true;

                            Update();

                            if (event.modifiers & static_cast<int>(KeyModifiers::Control))
                            {
                                camera->SetTarget(hitPoint);
                            }
                        }
                    }
                });

            std::vector<Eigen::Vector3f> positions(vertices.size());
            for (size_t i = 0; i < vertices.size(); i++)
            {
                positions[i] = points[vertices[i].pid] + offset;
            }

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

            for (const auto& face : faces)
            {
                if (face.eid == HEM::INVALID_ID)
                {
                    continue;
                }

                HEM::EID e0 = face.eid;
                HEM::EID e1 = edges[e0].neid;
                HEM::EID e2 = edges[e0].peid;

                HEM::VID v0 = edges[e2].vid;
                HEM::VID v1 = edges[e0].vid;
                HEM::VID v2 = edges[e1].vid;

                Eigen::Vector3f normal = (positions[v1] - positions[v0]).cross(positions[v2] - positions[v0]).normalized();
                normals[v0] += normal;
                normals[v1] += normal;
                normals[v2] += normal;
            }

            for (size_t i = 0; i < normals.size(); i++)
            {
                normals[i].normalize();
            }

            std::vector<unsigned int> indices;
            for (const auto& face : faces)
            {
                if (face.eid == HEM::INVALID_ID)
                {
                    continue;
                }

                HEM::EID e0 = face.eid;
                HEM::EID e1 = edges[e0].neid;
                HEM::EID e2 = edges[e0].peid;

                HEM::VID v0 = edges[e2].vid;
                HEM::VID v1 = edges[e0].vid;
                HEM::VID v2 = edges[e1].vid;

                indices.push_back((unsigned int)v0);
                indices.push_back((unsigned int)v1);
                indices.push_back((unsigned int)v2);
            }

            renderable->AddVertices(positions);
            renderable->AddNormals(normals);
            renderable->AddColors4(std::vector<Eigen::Vector4f>(positions.size(), color));
            renderable->AddIndices(indices);
            renderable->Update();
        }
        else
        {
            std::vector<Eigen::Vector3f> positions(vertices.size());
            for (size_t i = 0; i < vertices.size(); i++)
            {
                positions[i] = points[vertices[i].pid] + offset;
            }

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

            for (const auto& face : faces)
            {
                if (face.eid == HEM::INVALID_ID)
                {
                    continue;
                }

                HEM::EID e0 = face.eid;
                HEM::EID e1 = edges[e0].neid;
                HEM::EID e2 = edges[e0].peid;

                HEM::VID v0 = edges[e2].vid;
                HEM::VID v1 = edges[e0].vid;
                HEM::VID v2 = edges[e1].vid;

                Eigen::Vector3f normal = (positions[v1] - positions[v0]).cross(positions[v2] - positions[v0]).normalized();
                normals[v0] += normal;
                normals[v1] += normal;
                normals[v2] += normal;
            }

            for (size_t i = 0; i < normals.size(); i++)
            {
                normals[i].normalize();
            }

            std::vector<unsigned int> indices;
            for (const auto& face : faces)
            {
                if (face.eid == HEM::INVALID_ID)
                {
                    continue;
                }

                HEM::EID e0 = face.eid;
                HEM::EID e1 = edges[e0].neid;
                HEM::EID e2 = edges[e0].peid;

                HEM::VID v0 = edges[e2].vid;
                HEM::VID v1 = edges[e0].vid;
                HEM::VID v2 = edges[e1].vid;

                indices.push_back((unsigned int)v0);
                indices.push_back((unsigned int)v1);
                indices.push_back((unsigned int)v2);
            }

            auto renderable = Helium.GetComponent<Renderable>(entity);
            renderable->SetVertices(positions);
            renderable->SetNormals(normals);
            renderable->SetColors4(std::vector<Eigen::Vector4f>(positions.size(), color));
            renderable->SetIndices(indices);
            renderable->Update();
        }

        isDirty = false;
    }
};

class AppHalfEdgeMesh : public App
{
public:
    void Execute_Intersection()
    {
        TS(MeshA);
        HEM::Mesh mesh_A;
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
            {
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            }
            mesh_A.Build(stl.GetPoints(), indices);
        }
        TE(MeshA);

        TS(MeshB);
        HEM::Mesh mesh_B;
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
            {
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            }
            mesh_B.Build(stl.GetPoints(), indices);
        }
        TE(MeshB);

        HEM::Mesh mesh_A_orig = mesh_A;
        HEM::Mesh mesh_B_orig = mesh_B;

        TS(SplitFaces);
        mesh_A.SplitIntersectingFaces(mesh_B_orig);
        mesh_B.SplitIntersectingFaces(mesh_A_orig);
        TE(SplitFaces);

        TS(Separate);
        HEM::Mesh mesh_A_Inside, mesh_A_Outside;
        mesh_A.SeparateByInsideOutside(mesh_B_orig, mesh_A_Inside, mesh_A_Outside);

        HEM::Mesh mesh_B_Inside, mesh_B_Outside;
        mesh_B.SeparateByInsideOutside(mesh_A_orig, mesh_B_Inside, mesh_B_Outside);
        TE(Separate);

        TS(ExtractComponents);
        auto A_insideChunks = mesh_A_Inside.GetConnectedComponents();
        auto A_outsideChunks = mesh_A_Outside.GetConnectedComponents();
        auto B_insideChunks = mesh_B_Inside.GetConnectedComponents();
        auto B_outsideChunks = mesh_B_Outside.GetConnectedComponents();
        TE(ExtractComponents);

        std::vector<HEM::Mesh*> allChunks;
        for (auto& m : A_outsideChunks)
        {
            allChunks.push_back(&m);
        }
        for (auto& m : A_insideChunks)
        {
            allChunks.push_back(&m);
        }
        for (auto& m : B_outsideChunks)
        {
            allChunks.push_back(&m);
        }
        for (auto& m : B_insideChunks)
        {
            allChunks.push_back(&m);
        }

        Eigen::Vector3f globalCenter = Eigen::Vector3f::Zero();
        for (auto* m : allChunks)
        {
            globalCenter += m->GetCentroid();
        }
        globalCenter /= (float)allChunks.size();

        int colorIdx = 0;
        float spreadScale = 0.0f;

        auto RenderChunks = [&](const std::vector<HEM::Mesh>& chunks, const std::string& prefix)
            {
                for (size_t i = 0; i < chunks.size(); ++i)
                {
                    std::string name = prefix + std::to_string(i);
                    float r = ((colorIdx * 123 + 50) % 255) / 255.0f;
                    float g = ((colorIdx * 321 + 100) % 255) / 255.0f;
                    float b = ((colorIdx * 213 + 150) % 255) / 255.0f;

                    Eigen::Vector3f chunkCenter = chunks[i].GetCentroid();
                    Eigen::Vector3f dir = chunkCenter - globalCenter;

                    if (dir.norm() < 1e-3f)
                    {
                        dir = chunkCenter.norm() > 1e-6f ? chunkCenter : Eigen::Vector3f(0, 1, 0);
                    }

                    dir.normalize();

                    Eigen::Vector3f offset = dir * spreadScale;

                    AddMesh(name, chunks[i], Eigen::Vector4f(r, g, b, 1.0f), offset);

                    std::string stlPath = "D:\\temp\\" + name + ".stl";
                    auto mesh = chunks[i];
                    mesh.ToSTL(stlPath);

                    ++colorIdx;
                }
            };

        RenderChunks(A_outsideChunks, "A_Outside_");
        RenderChunks(B_insideChunks, "B_Inside_");
    }

    void Execute_Basic()
    {
        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh = *meshes.back();
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
            {
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            }
            mesh.Build(stl.GetPoints(), indices);
        }

        Helium.AddOnUpdateCallback([this](float timeDelta)
            {
                for (auto& m : this->meshes)
                {
                    m->Update();
                }
            });
    }

    virtual void Execute() override
    {
        //Execute_Intersection();

		Execute_Basic();
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
