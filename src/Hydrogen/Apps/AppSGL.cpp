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

#include "SimpleGeometryLibrary.hpp"

class HeliumMesh : public SGL::Mesh
{
public:
    HeliumMesh() : SGL::Mesh()
    {
    }

    virtual ~HeliumMesh()
    {
    }

    Entity entity = InvalidEntity;

    bool isDirty = true;

    Eigen::Vector3f offset;
    Eigen::Vector4f color = { 0.8f, 0.8f, 0.8f, 1.0f };

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

                            auto normal = (v1 - v0).normalized().cross((v2 - v0).normalized());
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
                            auto normal = (v1 - v0).normalized().cross((v2 - v0).normalized());

                            Eigen::Vector3f hitPoint = ray.origin + ray.direction * pickedDistance;
                            //VD::AddSphere("HitPoint", hitPoint, normal, 0.01f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

                            //VD::AddTriangle("HitTriangle", v0, v1, v2, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

                            //color = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f);

                            SplitFaceByPoint(pickedTriangleIndex, hitPoint);

                            isDirty = true;

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

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f());

            for (const auto& face : faces)
            {
                if (face.eid == SGL::INVALID_ID)
                {
                    continue;
                }

                SGL::EID e0 = face.eid;
                SGL::EID e1 = halfEdges[e0].neid;
                SGL::EID e2 = halfEdges[e0].peid;

                SGL::VID v0 = halfEdges[e2].vid;
                SGL::VID v1 = halfEdges[e0].vid;
                SGL::VID v2 = halfEdges[e1].vid;
                Eigen::Vector3f normal = (positions[v1] - positions[v0]).normalized().cross((positions[v2] - positions[v0]).normalized());
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
                if (face.eid == SGL::INVALID_ID)
                {
                    continue;
                }

                SGL::EID e0 = face.eid;
                SGL::EID e1 = halfEdges[e0].neid;
                SGL::EID e2 = halfEdges[e0].peid;

                SGL::VID v0 = halfEdges[e2].vid;
                SGL::VID v1 = halfEdges[e0].vid;
                SGL::VID v2 = halfEdges[e1].vid;

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

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f());

            for (const auto& face : faces)
            {
                if (face.eid == SGL::INVALID_ID)
                {
                    continue;
                }

                SGL::EID e0 = face.eid;
                SGL::EID e1 = halfEdges[e0].neid;
                SGL::EID e2 = halfEdges[e0].peid;

                SGL::VID v0 = halfEdges[e2].vid;
                SGL::VID v1 = halfEdges[e0].vid;
                SGL::VID v2 = halfEdges[e1].vid;

                Eigen::Vector3f normal = (positions[v1] - positions[v0]).normalized().cross((positions[v2] - positions[v0]).normalized());
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
                if (face.eid == SGL::INVALID_ID)
                {
                    continue;
                }

                SGL::EID e0 = face.eid;
                SGL::EID e1 = halfEdges[e0].neid;
                SGL::EID e2 = halfEdges[e0].peid;

                SGL::VID v0 = halfEdges[e2].vid;
                SGL::VID v1 = halfEdges[e0].vid;
                SGL::VID v2 = halfEdges[e1].vid;

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

class AppSGL : public App
{
public:
    virtual void Execute() override
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

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppSGL, "AppSGL");
