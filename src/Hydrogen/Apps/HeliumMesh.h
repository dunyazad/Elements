#pragma once

#include <vector>
#include <limits>
#include <iostream>

#include <Eigen/Dense>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/VisualDebugging.h>

#include "SimpleGeometryLibrary.hpp"

// Renderable mesh shared by all apps. Inherits the geometry kernel
// (SGL::Mesh) and adds Helium rendering plus interactive picking.
// Class-body member definitions are implicitly inline, so including this
// header in multiple translation units does not violate ODR.
class HeliumMesh : public SGL::Mesh
{
public:
    HeliumMesh() : SGL::Mesh() {}
    virtual ~HeliumMesh() {}

    Entity entity = InvalidEntity;
    bool is_dirty = true;
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();

    Eigen::Vector4f mesh_color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

    enum OperationMode
    {
        None,
        FlipEdge,
        SplitFace,
    } current_mode = None;

    void Update()
    {
        using VD = VisualDebugging;

        if (false == is_dirty) return;

        Renderable* renderable = nullptr;

        if (InvalidEntity == entity)
        {
            entity = Helium.CreateEntity("HeliumMesh");
            renderable = Helium.CreateComponent<Renderable>(entity);
            renderable->Initialize(Renderable::Triangles);
            renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
            renderable->SetFaceCullingMode(Renderable::NoCulling);

            Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [this](Entity e, const KeyEvent& event)
                {
                    auto current_renderable = Helium.GetComponent<Renderable>(e);
                    if (nullptr == current_renderable) return;

                    if (event.action == 1 && KeyCode::D1 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::Solid);
                    else if (event.action == 1 && KeyCode::D2 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::WireFrame);
                    else if (event.action == 1 && KeyCode::D3 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::WireFrameOverSolid);
                    else if (event.action == 1 && KeyCode::D4 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::Point);
                    else if (event.action == 1 && KeyCode::D5 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::None);
                    else if (event.action == 1 && KeyCode::F1 == event.keyCode)
                        current_mode = OperationMode::None;
                    else if (event.action == 1 && KeyCode::F2 == event.keyCode)
                        current_mode = OperationMode::FlipEdge;
                    else if (event.action == 1 && KeyCode::F3 == event.keyCode)
                        current_mode = OperationMode::SplitFace;
                });

            Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [this](Entity e, const MouseButtonEvent& event)
                {
                    if (event.action == 1 && event.button == MouseButton::Left)
                    {
                        auto current_renderable = Helium.GetComponent<Renderable>(e);
                        if (nullptr == current_renderable) return;

                        auto camera_entity = Helium.GetEntityByName("MainCamera");
                        auto camera = Helium.GetComponent<Camera>(camera_entity);
                        if (nullptr == camera) return;

                        TS(Picking);
                        Ray ray = camera->ScreenPointToRay(
                            (float)event.xpos,
                            (float)event.ypos,
                            Helium.GetWidth(),
                            Helium.GetHeight()
                        );

                        Eigen::Vector3f local_origin = ray.origin - offset;

                        SGL::IntersectionResult hit_result;
                        bool is_hit = IntersectGridRay(local_origin, ray.direction, hit_result);
                        TE(Picking);

                        if (is_hit)
                        {
                            auto world_picked_point = hit_result.hit_point + offset;

                            if (event.IsCtrlPressed())
                            {
                                auto cameraEntity = Helium.GetEntityByName("MainCamera");
                                Helium.GetComponent<Camera>(cameraEntity)->SetTarget(world_picked_point);
                            }

                            if (OperationMode::None == current_mode)
                            {
                                VD::Clear("PickedTriangle");
                                VD::Clear("PickedTrianglePoints");
                            }
                            else if (OperationMode::FlipEdge == current_mode)
                            {
                                OpenMesh::EdgeHandle target_edge;
                                float min_dist = std::numeric_limits<float>::max();
                                Eigen::Vector3f target_v0, target_v1;

                                for (auto fh_it = fh_iter(hit_result.fh); fh_it.is_valid(); ++fh_it)
                                {
                                    auto v0_handle = from_vertex_handle(*fh_it);
                                    auto v1_handle = to_vertex_handle(*fh_it);

                                    Eigen::Vector3f v0(point(v0_handle).data());
                                    Eigen::Vector3f v1(point(v1_handle).data());

                                    float dist = DistanceToSegment(hit_result.hit_point, v0, v1);

                                    if (dist < min_dist)
                                    {
                                        min_dist = dist;
                                        target_edge = edge_handle(*fh_it);
                                        target_v0 = v0;
                                        target_v1 = v1;
                                    }
                                }

                                if (target_edge.is_valid())
                                {
                                    VD::Clear("Picked edge");

                                    if (is_flip_ok(target_edge) && IsConvexQuadrilateral(target_edge))
                                    {
                                        flip(target_edge);

                                        TS(BuildSpatialHashMap);
                                        BuildSpatialHashMap();
                                        TE(BuildSpatialHashMap);

                                        is_dirty = true;
                                    }
                                    else
                                    {
                                        std::cout << "[Warning] Cannot flip this edge (Boundary or Non-manifold)." << std::endl;
                                    }
                                }
                            }
                            else if (OperationMode::SplitFace == current_mode)
                            {
                                if (hit_result.type != SGL::IntersectionType::Vertex)
                                {
                                    OpenMesh::VertexHandle new_v = add_vertex(Point(hit_result.hit_point.x(), hit_result.hit_point.y(), hit_result.hit_point.z()));

                                    if (hit_result.type == SGL::IntersectionType::Face)
                                    {
                                        split(hit_result.fh, new_v);
                                        is_dirty = true;
                                    }
                                    else if (hit_result.type == SGL::IntersectionType::Edge)
                                    {
                                        split(hit_result.eh, new_v);
                                        is_dirty = true;
                                    }

                                    if (is_dirty)
                                    {
                                        TS(BuildSpatialHashMap);
                                        BuildSpatialHashMap();
                                        TE(BuildSpatialHashMap);
                                    }
                                }
                                else
                                {
                                    std::cout << "[Info] Snapped to existing vertex. Split aborted to prevent degenerate faces." << std::endl;
                                }
                            }

                            VD::Clear("Picked");
                            VD::AddSphere("Picked", world_picked_point, { 0.0f, 0.0f, 1.0f }, 0.005f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
                        }
                    }
                });
        }
        else
        {
            renderable = Helium.GetComponent<Renderable>(entity);
        }

        if (renderable)
        {
            std::vector<Eigen::Vector3f> positions(n_vertices());
            for (auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
            {
                auto p = point(*v_it);
                positions[v_it->idx()] = Eigen::Vector3f(p[0], p[1], p[2]) + offset;
            }

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());
            std::vector<unsigned int> indices;
            indices.reserve(n_faces() * 3);

            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                auto fv_it = cfv_iter(*f_it);
                int idx0 = fv_it->idx(); ++fv_it;
                int idx1 = fv_it->idx(); ++fv_it;
                int idx2 = fv_it->idx();

                indices.push_back(static_cast<unsigned int>(idx0));
                indices.push_back(static_cast<unsigned int>(idx1));
                indices.push_back(static_cast<unsigned int>(idx2));

                Eigen::Vector3f normal_vector = (positions[idx1] - positions[idx0]).cross(positions[idx2] - positions[idx0]).normalized();
                normals[idx0] += normal_vector;
                normals[idx1] += normal_vector;
                normals[idx2] += normal_vector;
            }

            for (size_t i = 0; i < normals.size(); i++) normals[i].normalize();

            renderable->SetVertices(positions);
            renderable->SetNormals(normals);

            renderable->SetColors4(std::vector<Eigen::Vector4f>(positions.size(), mesh_color));
            renderable->SetIndices(indices);
            renderable->Update();
        }

        is_dirty = false;
    }
};
