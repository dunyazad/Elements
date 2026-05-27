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

namespace Eigen
{
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

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

                        if (!bvh.nodes.empty())
                        {
                            Eigen::Vector3f localOrigin = ray.origin - offset;
                            Eigen::Vector3f invDir(1.0f / ray.direction.x(), 1.0f / ray.direction.y(), 1.0f / ray.direction.z());

                            std::vector<int> stack;
                            stack.reserve(64);
                            stack.push_back(0);

                            while (!stack.empty())
                            {
                                int nodeIdx = stack.back();
                                stack.pop_back();

                                const SGL::BVHNode& node = bvh.nodes[nodeIdx];

                                if (!IntersectRayAABB(localOrigin, invDir, node.bounds))
                                {
                                    continue;
                                }

                                if (node.IsLeaf())
                                {
                                    for (int i = 0; i < node.faceCount; ++i)
                                    {
                                        SGL::FID fid = bvh.faceIds[node.faceOffset + i];
                                        Eigen::Vector3f v0, v1, v2;
                                        GetFaceVertices(fid, v0, v1, v2);

                                        float t = 0.0f;
                                        if (IntersectRayTriangle(localOrigin, ray.direction, v0, v1, v2, t))
                                        {
                                            if (t > 0.0f && t < pickedDistance)
                                            {
                                                pickedDistance = t;
                                                pickedTriangleIndex = (int)fid;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    stack.push_back(node.leftChild);
                                    stack.push_back(node.rightChild);
                                }
                            }
                        }

                        if (-1 != pickedTriangleIndex)
                        {
                            VD::Clear("HitPoint");
                            VD::Clear("HitTriangle");

                            Eigen::Vector3f localHitPoint = (ray.origin - offset) + ray.direction * pickedDistance;

                            SplitFaceByPoint(pickedTriangleIndex, localHitPoint);
                            RebuildBVH();

                            isDirty = true;

                            if (event.modifiers & static_cast<int>(KeyModifiers::Control))
                            {
                                Eigen::Vector3f worldHitPoint = localHitPoint + offset;
                                camera->SetTarget(worldHitPoint);
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

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

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

class AppHalfEdgeMesh : public App
{
public:
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
		Execute_Basic();
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
