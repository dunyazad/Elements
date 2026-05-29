#define _USE_MATH_DEFINES
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
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
#include <OpenMesh/Core/IO/MeshIO.hh>

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
    HeliumMesh() : SGL::Mesh() {}
    virtual ~HeliumMesh() {}

    Entity entity = InvalidEntity;
    bool is_dirty = true;
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();

    // [버그 수정]: OpenMesh의 내부 color() 메서드와 이름 충돌(Shadowing)이 
    // 발생하지 않도록 변수명을 mesh_color로 변경합니다!
    Eigen::Vector4f mesh_color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

    enum OperationMode
    {
        None,
        FlipEdge,
        SplitFace,
    } current_mode = None;

    void Update()
    {
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

                            if (OperationMode::FlipEdge == current_mode)
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

            // [버그 수정]: 변경된 변수명 mesh_color를 적용합니다.
            renderable->SetColors4(std::vector<Eigen::Vector4f>(positions.size(), mesh_color));
            renderable->SetIndices(indices);
            renderable->Update();
        }

        is_dirty = false;
    }
};

class AppHalfEdgeMesh : public App
{
public:
    void ExecuteBasic_visualize_border()
    {
        {
            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mesh = *meshes.back();
            {
                STLFormat stl;
                stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");

                const std::vector<Eigen::Vector3f>& stl_points = stl.GetPoints();
                std::vector<Eigen::Vector3f> welded_points;
                std::vector<Eigen::Vector3i> welded_indices;

                robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map;

                for (size_t i = 0; i < stl_points.size(); i += 3)
                {
                    Eigen::Vector3i face_indices;

                    for (int j = 0; j < 3; ++j)
                    {
                        const Eigen::Vector3f& p = stl_points[i + j];
                        auto it = vertex_map.find(p);

                        if (it != vertex_map.end())
                        {
                            face_indices[j] = it->second;
                        }
                        else
                        {
                            int new_idx = static_cast<int>(welded_points.size());
                            welded_points.push_back(p);
                            vertex_map[p] = new_idx;
                            face_indices[j] = new_idx;
                        }
                    }

                    if (face_indices[0] != face_indices[1] &&
                        face_indices[1] != face_indices[2] &&
                        face_indices[2] != face_indices[0])
                    {
                        welded_indices.push_back(face_indices);
                    }
                }

                mesh.Build(welded_points, welded_indices);
            }
        }

        {
            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mesh = *meshes.back();
            {
                STLFormat stl;
                stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");

                const std::vector<Eigen::Vector3f>& stl_points = stl.GetPoints();
                std::vector<Eigen::Vector3f> welded_points;
                std::vector<Eigen::Vector3i> welded_indices;

                robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map;

                for (size_t i = 0; i < stl_points.size(); i += 3)
                {
                    Eigen::Vector3i face_indices;

                    for (int j = 0; j < 3; ++j)
                    {
                        const Eigen::Vector3f& p = stl_points[i + j];
                        auto it = vertex_map.find(p);

                        if (it != vertex_map.end())
                        {
                            face_indices[j] = it->second;
                        }
                        else
                        {
                            int new_idx = static_cast<int>(welded_points.size());
                            welded_points.push_back(p);
                            vertex_map[p] = new_idx;
                            face_indices[j] = new_idx;
                        }
                    }

                    if (face_indices[0] != face_indices[1] &&
                        face_indices[1] != face_indices[2] &&
                        face_indices[2] != face_indices[0])
                    {
                        welded_indices.push_back(face_indices);
                    }
                }

                mesh.Build(welded_points, welded_indices);
            }
        }

        TS(ExtractIntersectionSegments);
        std::vector<char> cut_faces_mask;
        auto raw_segments = meshes[0]->ExtractIntersectionSegments(*meshes[1], &cut_faces_mask);
        auto welded_segments = meshes[0]->WeldSegments(raw_segments, 5e-3f);
        auto linked_segments = meshes[0]->LinkSegments(welded_segments, 5e-3f);

        for (auto& ring : linked_segments)
        {
            if (ring.size() > 2 && (ring.front() - ring.back()).norm() > 1e-6f)
            {
                ring.push_back(ring.front());
            }
        }
        TE(ExtractIntersectionSegments);

        meshes[0]->DeleteMarkedFaces(cut_faces_mask);
        meshes[0]->is_dirty = true;

        auto separated_parts = meshes[0]->SeparateDisconnectedMeshes();
        std::cout << "[Info] Mesh separated into " << separated_parts.size() << " parts." << std::endl;

        VD::Clear("CutLines");
        VD::Clear("MatchedBoundaries");
        VD::Clear("StartPoints");

        for (size_t r = 0; r < linked_segments.size(); ++r)
        {
            const auto& ring = linked_segments[r];
            for (size_t i = 0; i < ring.size() - 1; ++i)
            {
                VD::AddLine("CutLines", ring[i], ring[i + 1], Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
            }
        }

        std::vector<Eigen::Vector4f> part_colors = {
            Eigen::Vector4f(0.9f, 0.5f, 0.5f, 1.0f),
            Eigen::Vector4f(0.5f, 0.9f, 0.5f, 1.0f),
            Eigen::Vector4f(0.5f, 0.6f, 0.9f, 1.0f),
            Eigen::Vector4f(0.8f, 0.6f, 0.9f, 1.0f)
        };

        std::vector<Eigen::Vector4f> border_colors = {
            Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f),
            Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f),
            Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f),
            Eigen::Vector4f(0.6f, 0.0f, 1.0f, 1.0f)
        };

        if (separated_parts.size() >= 2)
        {
            for (size_t i = 0; i < separated_parts.size(); ++i)
            {
                auto& part = separated_parts[i];

                std::string filename = "D:\\Temp\\separated_part_" + std::to_string(i) + ".stl";
                OpenMesh::IO::write_mesh(*part, filename);

                auto boundaries = part->ExtractBoundaryLoops();

                Eigen::Vector4f mesh_color = part_colors[i % part_colors.size()];
                Eigen::Vector4f border_color = border_colors[i % border_colors.size()];

                std::map<int, std::vector<OpenMesh::VertexHandle>> best_boundary_for_ring;

                for (const auto& boundary : boundaries)
                {
                    int matched_ring_idx = part->MatchBoundaryToRing(boundary, linked_segments);

                    if (matched_ring_idx != -1)
                    {
                        if (best_boundary_for_ring.find(matched_ring_idx) == best_boundary_for_ring.end() ||
                            boundary.size() > best_boundary_for_ring[matched_ring_idx].size())
                        {
                            best_boundary_for_ring[matched_ring_idx] = boundary;
                        }
                    }
                }

                for (const auto& pair : best_boundary_for_ring)
                {
                    const auto& boundary = pair.second;
                    int ring_idx = pair.first;
                    const auto& ring = linked_segments[ring_idx];

                    for (size_t b = 0; b < boundary.size(); ++b)
                    {
                        Eigen::Vector3f p1(part->point(boundary[b]).data());
                        Eigen::Vector3f p2(part->point(boundary[(b + 1) % boundary.size()]).data());
                        VD::AddLine("MatchedBoundaries", p1, p2, border_color);
                    }

                    int start_a = 0;
                    int start_b = 0;
                    float min_dist = std::numeric_limits<float>::max();

                    int num_a = static_cast<int>(boundary.size());
                    int num_b_original = static_cast<int>(ring.size());
                    int num_b = (ring.front() - ring.back()).norm() < 1e-6f ? num_b_original - 1 : num_b_original;

                    for (int n = 0; n < num_a; ++n)
                    {
                        Eigen::Vector3f p_a(part->point(boundary[n]).data());
                        for (int m = 0; m < num_b; ++m)
                        {
                            float d = (p_a - ring[m]).squaredNorm();
                            if (d < min_dist)
                            {
                                min_dist = d;
                                start_a = n;
                                start_b = m;
                            }
                        }
                    }

                    Eigen::Vector3f p_a0(part->point(boundary[start_a]).data());
                    Eigen::Vector3f p_b0 = ring[start_b];

                    VD::AddSphere("StartPoints", p_a0, { 0.0f, 0.0f, 1.0f }, 0.05f, mesh_color);
                    VD::AddSphere("StartPoints", p_b0, { 0.0f, 0.0f, 1.0f }, 0.05f, border_color);
                    VD::AddLine("StartPoints", p_a0, p_b0, mesh_color);
                }

                auto new_helium_mesh = std::make_unique<HeliumMesh>();
                new_helium_mesh->assign(*part);
                new_helium_mesh->BuildSpatialHashMap();
                new_helium_mesh->mesh_color = mesh_color;
                new_helium_mesh->is_dirty = true;
                meshes.push_back(std::move(new_helium_mesh));
            }
        }

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes)
                {
                    m->Update();
                }

                if (meshes.size() >= 3)
                {
                    auto renderable_a = Helium.GetComponent<Renderable>(meshes[0]->entity);
                    auto renderable_b = Helium.GetComponent<Renderable>(meshes[1]->entity);
                    if (nullptr != renderable_a)
                    {
                        renderable_a->SetVisible(false);
                    }
                    if (nullptr != renderable_b)
                    {
                        renderable_b->SetVisible(false);
                    }
                }
            });
    }

    void ExecuteBasic()
    {
        {
            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mesh = *meshes.back();
            {
                STLFormat stl;
                stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");

                const std::vector<Eigen::Vector3f>& stl_points = stl.GetPoints();
                std::vector<Eigen::Vector3f> welded_points;
                std::vector<Eigen::Vector3i> welded_indices;

                robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map;

                for (size_t i = 0; i < stl_points.size(); i += 3)
                {
                    Eigen::Vector3i face_indices;

                    for (int j = 0; j < 3; ++j)
                    {
                        const Eigen::Vector3f& p = stl_points[i + j];
                        auto it = vertex_map.find(p);

                        if (it != vertex_map.end())
                        {
                            face_indices[j] = it->second;
                        }
                        else
                        {
                            int new_idx = static_cast<int>(welded_points.size());
                            welded_points.push_back(p);
                            vertex_map[p] = new_idx;
                            face_indices[j] = new_idx;
                        }
                    }

                    if (face_indices[0] != face_indices[1] &&
                        face_indices[1] != face_indices[2] &&
                        face_indices[2] != face_indices[0])
                    {
                        welded_indices.push_back(face_indices);
                    }
                }

                mesh.Build(welded_points, welded_indices);
            }
        }

        {
            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mesh = *meshes.back();
            {
                STLFormat stl;
                stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");

                const std::vector<Eigen::Vector3f>& stl_points = stl.GetPoints();
                std::vector<Eigen::Vector3f> welded_points;
                std::vector<Eigen::Vector3i> welded_indices;

                robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map;

                for (size_t i = 0; i < stl_points.size(); i += 3)
                {
                    Eigen::Vector3i face_indices;

                    for (int j = 0; j < 3; ++j)
                    {
                        const Eigen::Vector3f& p = stl_points[i + j];
                        auto it = vertex_map.find(p);

                        if (it != vertex_map.end())
                        {
                            face_indices[j] = it->second;
                        }
                        else
                        {
                            int new_idx = static_cast<int>(welded_points.size());
                            welded_points.push_back(p);
                            vertex_map[p] = new_idx;
                            face_indices[j] = new_idx;
                        }
                    }

                    if (face_indices[0] != face_indices[1] &&
                        face_indices[1] != face_indices[2] &&
                        face_indices[2] != face_indices[0])
                    {
                        welded_indices.push_back(face_indices);
                    }
                }

                mesh.Build(welded_points, welded_indices);
            }
        }

        TS(ExtractIntersectionSegments);
        std::vector<char> cut_faces_mask;
        auto raw_segments = meshes[0]->ExtractIntersectionSegments(*meshes[1], &cut_faces_mask);
        auto welded_segments = meshes[0]->WeldSegments(raw_segments, 5e-3f);
        auto linked_segments = meshes[0]->LinkSegments(welded_segments, 5e-3f);

        for (auto& ring : linked_segments)
        {
            if (ring.size() > 2 && (ring.front() - ring.back()).norm() > 1e-6f)
            {
                ring.push_back(ring.front());
            }
        }
        TE(ExtractIntersectionSegments);

        meshes[0]->DeleteMarkedFaces(cut_faces_mask);
        meshes[0]->is_dirty = true;

        auto separated_parts = meshes[0]->SeparateDisconnectedMeshes();

        VD::Clear("CutLines");

        for (size_t r = 0; r < linked_segments.size(); ++r)
        {
            const auto& ring = linked_segments[r];
            for (size_t i = 0; i < ring.size() - 1; ++i)
            {
                VD::AddLine("CutLines", ring[i], ring[i + 1], Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
            }
        }

        std::vector<Eigen::Vector4f> part_colors = {
            Eigen::Vector4f(0.9f, 0.5f, 0.5f, 1.0f),
            Eigen::Vector4f(0.5f, 0.9f, 0.5f, 1.0f),
            Eigen::Vector4f(0.5f, 0.6f, 0.9f, 1.0f),
            Eigen::Vector4f(0.8f, 0.6f, 0.9f, 1.0f)
        };

        if (separated_parts.size() >= 2)
        {
            for (size_t i = 0; i < separated_parts.size(); ++i)
            {
                auto& part = separated_parts[i];

                if (part->n_faces() <= 1)
                {
                    continue;
                }

                auto boundaries = part->ExtractBoundaryLoops();

                //std::cout << "\n=============================================" << std::endl;
                //std::cout << "[StitchLog] Starting Mesh Processing for Part " << i << std::endl;
                //std::cout << "=============================================" << std::endl;

                for (const auto& boundary : boundaries)
                {
                    if (boundary.size() < 3)
                    {
                        continue;
                    }

                    int best_ring_index = part->FindBestMatchingRing(boundary, linked_segments);
                    if (best_ring_index == -1)
                    {
                        continue;
                    }

                    const auto& original_ring = linked_segments[best_ring_index];
                    int number_b_original = static_cast<int>(original_ring.size());
                    int number_b = (original_ring.front() - original_ring.back()).norm() < 1e-6f ? number_b_original - 1 : number_b_original;

                    std::vector<Eigen::Vector3f> unique_ring;
                    for (int k = 0; k < number_b; ++k)
                    {
                        unique_ring.push_back(original_ring[k]);
                    }

                    std::vector<OpenMesh::VertexHandle> loop_a = boundary;
                    int number_a = static_cast<int>(loop_a.size());

                    int best_a = 0;
                    int best_b = 0;
                    part->FindBestStartPoints(loop_a, unique_ring, best_a, best_b);

                    std::rotate(loop_a.begin(), loop_a.begin() + best_a, loop_a.end());

                    std::vector<Eigen::Vector3f> ring_forward;
                    std::vector<Eigen::Vector3f> ring_reverse;

                    for (int k = 0; k < number_b; ++k)
                    {
                        ring_forward.push_back(unique_ring[(best_b + k) % number_b]);
                        ring_reverse.push_back(unique_ring[(best_b - k + number_b) % number_b]);
                    }

                    //std::cout << "[Step 3 & 4] Calculating DP Paths for Forward and Reverse Directions..." << std::endl;

                    std::vector<int> path_forward;
                    float cost_forward = part->CalculateDPCost(loop_a, ring_forward, path_forward);

                    std::vector<int> path_reverse;
                    float cost_reverse = part->CalculateDPCost(loop_a, ring_reverse, path_reverse);

                    std::vector<Eigen::Vector3f> optimal_ring;
                    std::vector<int> optimal_path;

                    if (cost_forward <= cost_reverse)
                    {
                        //std::cout << "[StitchLog] Selected Direction: Forward (Cost: " << cost_forward << " vs " << cost_reverse << ")" << std::endl;
                        optimal_ring = ring_forward;
                        optimal_path = path_forward;
                    }
                    else
                    {
                        //std::cout << "[StitchLog] Selected Direction: Reverse (Cost: " << cost_reverse << " vs " << cost_forward << ")" << std::endl;
                        optimal_ring = ring_reverse;
                        optimal_path = path_reverse;
                    }

                    std::vector<OpenMesh::VertexHandle> loop_b;
                    loop_b.reserve(number_b);
                    for (const auto& pt : optimal_ring)
                    {
                        loop_b.push_back(part->add_vertex(SGL::OMMesh::Point(pt.x(), pt.y(), pt.z())));
                    }

                    if (!part->ValidateLoopData(loop_a, loop_b))
                    {
                        //std::cout << "[StitchLog] Face creation aborted due to validation failure." << std::endl;
                        continue;
                    }

                    //std::cout << "[Step 5] Creating Faces from DP Path..." << std::endl;
                    int index_a = 0;
                    int index_b = 0;

                    for (int step : optimal_path)
                    {
                        if (step == 1)
                        {
                            int next_a = (index_a + 1) % number_a;
                            part->AddFaceWithLog(loop_a[next_a], loop_a[index_a], loop_b[index_b % number_b]);
                            index_a++;
                        }
                        else
                        {
                            int next_b = (index_b + 1) % number_b;
                            part->AddFaceWithLog(loop_b[next_b], loop_b[index_b], loop_a[index_a % number_a]);
                            index_b++;
                        }
                    }
                    //std::cout << "[StitchLog] Zippering completed for current ring." << std::endl;
                }

                part->garbage_collection();
                part->BuildSpatialHashMap();

                std::string filename = "D:\\Temp\\separated_part_" + std::to_string(i) + ".stl";
                OpenMesh::IO::write_mesh(*part, filename);

                Eigen::Vector4f mesh_color = part_colors[i % part_colors.size()];

                auto new_helium_mesh = std::make_unique<HeliumMesh>();
                new_helium_mesh->assign(*part);
                new_helium_mesh->BuildSpatialHashMap();
                new_helium_mesh->mesh_color = mesh_color;
                new_helium_mesh->is_dirty = true;
                meshes.push_back(std::move(new_helium_mesh));
            }
        }

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes)
                {
                    m->Update();
                }

                if (meshes.size() >= 3)
                {
                    auto renderable_a = Helium.GetComponent<Renderable>(meshes[0]->entity);
                    auto renderable_b = Helium.GetComponent<Renderable>(meshes[1]->entity);
                    if (nullptr != renderable_a)
                    {
                        renderable_a->SetVisible(false);
                    }
                    if (nullptr != renderable_b)
                    {
                        renderable_b->SetVisible(false);
                    }
                }
            });
    }

    enum class BooleanOperation
    {
        Union,        // 합집합 (A ∪ B)
        Intersection, // 교집합 (A ∩ B)
        Difference    // 차집합 (A - B)
    };

    void ExecuteBooleanOperation()
    {
        BooleanOperation current_op = BooleanOperation::Union;

        // ---------------------------------------------------------
        // 1. Mesh A, Mesh B 로드
        // ---------------------------------------------------------
        {
            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mesh = *meshes.back();
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\Gadget.stl");

            const std::vector<Eigen::Vector3f>& stl_points = stl.GetPoints();
            std::vector<Eigen::Vector3f> welded_points;
            std::vector<Eigen::Vector3i> welded_indices;
            robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map;

            for (size_t i = 0; i < stl_points.size(); i += 3)
            {
                Eigen::Vector3i face_indices;
                for (int j = 0; j < 3; ++j)
                {
                    const Eigen::Vector3f& p = stl_points[i + j];
                    auto it = vertex_map.find(p);
                    if (it != vertex_map.end()) {
                        face_indices[j] = it->second;
                    }
                    else {
                        int new_idx = static_cast<int>(welded_points.size());
                        welded_points.push_back(p);
                        vertex_map[p] = new_idx;
                        face_indices[j] = new_idx;
                    }
                }
                if (face_indices[0] != face_indices[1] && face_indices[1] != face_indices[2] && face_indices[2] != face_indices[0]) {
                    welded_indices.push_back(face_indices);
                }
            }
            mesh.Build(welded_points, welded_indices);
        }

        {
            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mesh = *meshes.back();
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\Doughnut.stl");

            const std::vector<Eigen::Vector3f>& stl_points = stl.GetPoints();
            std::vector<Eigen::Vector3f> welded_points;
            std::vector<Eigen::Vector3i> welded_indices;
            robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map;

            for (size_t i = 0; i < stl_points.size(); i += 3)
            {
                Eigen::Vector3i face_indices;
                for (int j = 0; j < 3; ++j)
                {
                    const Eigen::Vector3f& p = stl_points[i + j];
                    auto it = vertex_map.find(p);
                    if (it != vertex_map.end()) {
                        face_indices[j] = it->second;
                    }
                    else {
                        int new_idx = static_cast<int>(welded_points.size());
                        welded_points.push_back(p);
                        vertex_map[p] = new_idx;
                        face_indices[j] = new_idx;
                    }
                }
                if (face_indices[0] != face_indices[1] && face_indices[1] != face_indices[2] && face_indices[2] != face_indices[0]) {
                    welded_indices.push_back(face_indices);
                }
            }
            mesh.Build(welded_points, welded_indices);
        }

        // ---------------------------------------------------------
        // 2. 원본 메쉬 백업 (Ray-Casting Inside/Outside 판별용)
        // ---------------------------------------------------------
        HeliumMesh originalA; originalA.assign(*meshes[0]); originalA.BuildSpatialHashMap();
        HeliumMesh originalB; originalB.assign(*meshes[1]); originalB.BuildSpatialHashMap();

        // ---------------------------------------------------------
        // 3. 교차선 추출 및 양쪽 마스크 생성
        // ---------------------------------------------------------
        TS(ExtractIntersectionSegments);
        std::vector<char> maskA, maskB;
        auto raw_segments = meshes[0]->ExtractIntersectionSegments(*meshes[1], &maskA);
        meshes[1]->ExtractIntersectionSegments(*meshes[0], &maskB);

        auto welded_segments = meshes[0]->WeldSegments(raw_segments, 5e-3f);
        auto linked_segments = meshes[0]->LinkSegments(welded_segments, 5e-3f);

        for (auto& ring : linked_segments)
        {
            if (ring.size() > 2 && (ring.front() - ring.back()).norm() > 1e-6f) {
                ring.push_back(ring.front());
            }
        }
        TE(ExtractIntersectionSegments);

        // ---------------------------------------------------------
        // 4. 물리적 절단 및 분리
        // ---------------------------------------------------------
        meshes[0]->DeleteMarkedFaces(maskA);
        meshes[1]->DeleteMarkedFaces(maskB);

        auto partsA = meshes[0]->SeparateDisconnectedMeshes();
        auto partsB = meshes[1]->SeparateDisconnectedMeshes();

        // ---------------------------------------------------------
        // 5. 구멍 메우기 + Boolean 분류기 (람다 캡처에 current_op 추가)
        // ---------------------------------------------------------
        auto ProcessAndBooleanParts = [&](std::vector<std::unique_ptr<SGL::Mesh>>& parts,
            const HeliumMesh& original_other,
            bool is_mesh_A)
            {
                std::vector<std::unique_ptr<HeliumMesh>> result_meshes;

                for (size_t i = 0; i < parts.size(); ++i)
                {
                    auto& part = parts[i];
                    if (part->n_faces() <= 1) continue;

                    // [Zippering 로직]
                    auto boundaries = part->ExtractBoundaryLoops();
                    for (const auto& boundary : boundaries)
                    {
                        if (boundary.size() < 3) continue;
                        int best_ring_index = part->FindBestMatchingRing(boundary, linked_segments);
                        if (best_ring_index == -1) continue;

                        const auto& original_ring = linked_segments[best_ring_index];
                        int number_b_original = static_cast<int>(original_ring.size());
                        int number_b = (original_ring.front() - original_ring.back()).norm() < 1e-6f ? number_b_original - 1 : number_b_original;

                        std::vector<Eigen::Vector3f> unique_ring;
                        for (int k = 0; k < number_b; ++k) unique_ring.push_back(original_ring[k]);

                        std::vector<OpenMesh::VertexHandle> loop_a = boundary;
                        int number_a = static_cast<int>(loop_a.size());

                        int best_a = 0, best_b = 0;
                        part->FindBestStartPoints(loop_a, unique_ring, best_a, best_b);
                        std::rotate(loop_a.begin(), loop_a.begin() + best_a, loop_a.end());

                        std::vector<Eigen::Vector3f> ring_forward, ring_reverse;
                        for (int k = 0; k < number_b; ++k) {
                            ring_forward.push_back(unique_ring[(best_b + k) % number_b]);
                            ring_reverse.push_back(unique_ring[(best_b - k + number_b) % number_b]);
                        }

                        std::vector<int> path_forward, path_reverse;
                        float cost_forward = part->CalculateDPCost(loop_a, ring_forward, path_forward);
                        float cost_reverse = part->CalculateDPCost(loop_a, ring_reverse, path_reverse);

                        std::vector<Eigen::Vector3f> optimal_ring = (cost_forward <= cost_reverse) ? ring_forward : ring_reverse;
                        std::vector<int> optimal_path = (cost_forward <= cost_reverse) ? path_forward : path_reverse;

                        std::vector<OpenMesh::VertexHandle> loop_b;
                        loop_b.reserve(number_b);
                        for (const auto& pt : optimal_ring) {
                            loop_b.push_back(part->add_vertex(SGL::OMMesh::Point(pt.x(), pt.y(), pt.z())));
                        }

                        if (!part->ValidateLoopData(loop_a, loop_b)) continue;

                        int index_a = 0, index_b = 0;
                        for (int step : optimal_path) {
                            if (step == 1) {
                                part->AddFaceWithLog(loop_a[(index_a + 1) % number_a], loop_a[index_a], loop_b[index_b % number_b]);
                                index_a++;
                            }
                            else {
                                part->AddFaceWithLog(loop_b[(index_b + 1) % number_b], loop_b[index_b], loop_a[index_a % number_a]);
                                index_b++;
                            }
                        }
                    }

                    part->garbage_collection();
                    part->BuildSpatialHashMap();

                    // ---------------------------------------------------------
                    // [Boolean 필터링 로직]
                    // ---------------------------------------------------------
                    Eigen::Vector3f sample_pt = part->GetSafeSamplePoint();
                    bool is_inside = original_other.IsPointInside(sample_pt);

                    bool should_keep = false;
                    bool should_flip = false;

                    switch (current_op)
                    {
                    case BooleanOperation::Union: // 합집합 (A ∪ B)
                        if (is_mesh_A) {
                            if (!is_inside) should_keep = true; // A 외부 유지
                        }
                        else {
                            if (!is_inside) should_keep = true; // B 외부 유지
                        }
                        break;

                    case BooleanOperation::Intersection: // 교집합 (A ∩ B)
                        if (is_mesh_A) {
                            if (is_inside) should_keep = true; // A 내부 유지
                        }
                        else {
                            if (is_inside) should_keep = true; // B 내부 유지
                        }
                        break;

                    case BooleanOperation::Difference: // 차집합 (A - B)
                        if (is_mesh_A) {
                            if (!is_inside) should_keep = true; // A 외부 유지
                        }
                        else {
                            if (is_inside) {
                                should_keep = true;             // B 내부 유지
                                should_flip = true;             // 파인 면이므로 노멀 뒤집기
                            }
                        }
                        break;
                    }

                    // 조건에 맞는 조각만 최종 결과에 추가
                    if (should_keep)
                    {
                        if (should_flip) part->FlipAllFaces();

                        auto final_mesh = std::make_unique<HeliumMesh>();
                        final_mesh->assign(*part);
                        final_mesh->BuildSpatialHashMap();

                        // Mesh A 출신은 연두색, Mesh B 출신은 붉은색으로 시각화하여 어디서 왔는지 구별
                        final_mesh->mesh_color = is_mesh_A ? Eigen::Vector4f(0.5f, 0.9f, 0.5f, 1.0f) : Eigen::Vector4f(0.9f, 0.5f, 0.5f, 1.0f);
                        final_mesh->is_dirty = true;

                        result_meshes.push_back(std::move(final_mesh));
                    }
                }

                return result_meshes;
            };

        // ---------------------------------------------------------
        // 6. 결과 취합 및 렌더링 세팅
        // ---------------------------------------------------------
        auto final_A_parts = ProcessAndBooleanParts(partsA, originalB, true);
        auto final_B_parts = ProcessAndBooleanParts(partsB, originalA, false);

        meshes.clear(); // 화면에 원본이 남지 않도록 기존 배열 비우기

        for (auto& m : final_A_parts) meshes.push_back(std::move(m));
        for (auto& m : final_B_parts) meshes.push_back(std::move(m));

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes)
                {
                    m->Update();
                }
            });

        std::cout << "[Boolean] Operation Complete! Assembled " << meshes.size() << " parts." << std::endl;

        HeliumMesh merged_result;

        for (const auto& m : meshes)
        {
            // 원본 파편 메쉬의 Vertex ID -> 병합된 메쉬의 Vertex ID 매핑 배열
            std::vector<OpenMesh::VertexHandle> vmap(m->n_vertices(), OpenMesh::VertexHandle(-1));

            for (auto f_it = m->faces_begin(); f_it != m->faces_end(); ++f_it)
            {
                if (m->status(*f_it).deleted()) continue;

                std::vector<OpenMesh::VertexHandle> face_vhandles;
                for (auto fv_it = m->cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
                {
                    int old_idx = fv_it->idx();

                    // 아직 병합된 메쉬에 추가되지 않은 정점이라면 새로 추가
                    if (!vmap[old_idx].is_valid())
                    {
                        Eigen::Vector3f pos(m->point(*fv_it).data());
                        vmap[old_idx] = merged_result.add_vertex(SGL::OMMesh::Point(pos.x(), pos.y(), pos.z()));
                    }
                    face_vhandles.push_back(vmap[old_idx]);
                }
                merged_result.add_face(face_vhandles);
            }
        }

        // 현재 진행한 연산 종류에 따라 파일 이름 동적 지정
        std::string filename = "D:\\Temp\\boolean_result_";
        switch (current_op)
        {
        case BooleanOperation::Union:        filename += "union.stl"; break;
        case BooleanOperation::Intersection: filename += "intersection.stl"; break;
        case BooleanOperation::Difference:   filename += "difference.stl"; break;
        }

        // 단일 메쉬로 병합된 결과를 파일로 출력
        if (OpenMesh::IO::write_mesh(merged_result, filename))
        {
            std::cout << "[Info] Successfully saved boolean result to: " << filename << std::endl;
        }
        else
        {
            std::cout << "[Error] Failed to save boolean result." << std::endl;
        }
    }

    void ExecuteSplitting()
    {
        // 1. Mesh A 로드 및 용접
        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_a = *meshes.back();

        STLFormat STL_A;
        STL_A.Deserialize("D:\\Resources\\3D\\STL\\Cube.stl");

        const std::vector<Eigen::Vector3f>& points_a = STL_A.GetPoints();
        std::vector<Eigen::Vector3f> welded_points_a;
        std::vector<Eigen::Vector3i> welded_indices_a;
        robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map_a;

        for (size_t i = 0; i < points_a.size(); i += 3)
        {
            Eigen::Vector3i face_indices;
            for (int j = 0; j < 3; ++j)
            {
                const Eigen::Vector3f& p = points_a[i + j];
                auto it = vertex_map_a.find(p);
                if (it != vertex_map_a.end())
                {
                    face_indices[j] = it->second;
                }
                else
                {
                    int new_idx = static_cast<int>(welded_points_a.size());
                    welded_points_a.push_back(p);
                    vertex_map_a[p] = new_idx;
                    face_indices[j] = new_idx;
                }
            }
            if (face_indices[0] != face_indices[1] && face_indices[1] != face_indices[2] && face_indices[2] != face_indices[0])
            {
                welded_indices_a.push_back(face_indices);
            }
        }
        mesh_a.Build(welded_points_a, welded_indices_a);

        // 2. Mesh B 로드 및 용접
        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_b = *meshes.back();

        STLFormat STL_B;
        STL_B.Deserialize("D:\\Resources\\3D\\STL\\Cylinder.stl");

        const std::vector<Eigen::Vector3f>& points_b = STL_B.GetPoints();
        std::vector<Eigen::Vector3f> welded_points_b;
        std::vector<Eigen::Vector3i> welded_indices_b;
        robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vertex_map_b;

        for (size_t i = 0; i < points_b.size(); i += 3)
        {
            Eigen::Vector3i face_indices;
            for (int j = 0; j < 3; ++j)
            {
                const Eigen::Vector3f& p = points_b[i + j];
                auto it = vertex_map_b.find(p);
                if (it != vertex_map_b.end())
                {
                    face_indices[j] = it->second;
                }
                else
                {
                    int new_idx = static_cast<int>(welded_points_b.size());
                    welded_points_b.push_back(p);
                    vertex_map_b[p] = new_idx;
                    face_indices[j] = new_idx;
                }
            }
            if (face_indices[0] != face_indices[1] && face_indices[1] != face_indices[2] && face_indices[2] != face_indices[0])
            {
                welded_indices_b.push_back(face_indices);
            }
        }
        mesh_b.Build(welded_points_b, welded_indices_b);

        // 공간 해시맵 빌드 (양쪽 모두 필요)
        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        std::cout << "[Split] Extracting exact intersection segments..." << std::endl;

        std::vector<char> mask_a;
        // Face 간의 삼각형 교차로 만들어진 원시 교차선 추출
        auto raw_segments = mesh_a.ExtractIntersectionSegments(mesh_b, &mask_a);
        auto welded_segments = mesh_a.WeldSegments(raw_segments, 1e-4f);

        VD::Clear("IntersectionSegments");
        VD::Clear("IntersectionPoints");

        // --- 핵심 로직: 이중 분할 (Dual Splitting) ---

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> refined_segments_a;
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> final_refined_segments;

        // 1차 분할: Mesh A의 내부 엣지들을 기준으로 선분들을 쪼갭니다.
        for (const auto& segment : welded_segments)
        {
            std::vector<Eigen::Vector3f> split_points_a = mesh_a.SplitSegmentWithMeshEdges(segment.first, segment.second);
            for (size_t i = 0; i < split_points_a.size() - 1; ++i)
            {
                refined_segments_a.push_back({ split_points_a[i], split_points_a[i + 1] });
            }
        }

        // 2차 분할: Mesh A에 의해 쪼개진 하위 선분들을 다시 Mesh B의 내부 엣지들을 기준으로 쪼갭니다.
        for (const auto& segment : refined_segments_a)
        {
            std::vector<Eigen::Vector3f> split_points_b = mesh_b.SplitSegmentWithMeshEdges(segment.first, segment.second);
            for (size_t i = 0; i < split_points_b.size() - 1; ++i)
            {
                final_refined_segments.push_back({ split_points_b[i], split_points_b[i + 1] });
            }
        }

        // 양쪽 메쉬의 엣지를 모두 통과하며 쪼개진 최종 교차점들을 수집합니다.
        std::vector<Eigen::Vector3f> intersection_points;
        for (const auto& segment : final_refined_segments)
        {
            intersection_points.push_back(segment.first);
            intersection_points.push_back(segment.second);
        }

        // 중복 정점 제거
        auto erase_iterator = std::unique(intersection_points.begin(), intersection_points.end(), [](const Eigen::Vector3f& p1, const Eigen::Vector3f& p2)
            {
                return (p1 - p2).squaredNorm() < 1e-8f;
            });
        intersection_points.erase(erase_iterator, intersection_points.end());

        std::cout << "[Split] Found " << intersection_points.size() << " exact intersection points on BOTH meshes. Visualizing..." << std::endl;

        // 1. 쪼개진 최종 교차선들 그리기 (노란색 선)
        for (const auto& seg : final_refined_segments)
        {
            VD::AddLine("IntersectionSegments", seg.first, seg.second, Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
        }

        // 2. 쪼개진 지점들 그리기 (빨간색 구)
        // 이 점들은 이제 Mesh A의 엣지 위에도 있고, Mesh B의 엣지 위에도 존재합니다!
        for (const auto& pt : intersection_points)
        {
            VD::AddSphere("IntersectionPoints", pt, { 0.0f, 0.0f, 1.0f }, 0.05f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
        }

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes)
                {
                    m->Update();
                }
            });
    }

    virtual void Execute() override
    {
        //ExecuteBasic();

		//ExecuteBooleanOperation();

        ExecuteSplitting();
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");