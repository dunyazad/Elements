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

struct Vector3fHash {
    size_t operator()(const Eigen::Vector3f& v) const {
        long long x = static_cast<long long>(std::round(v.x() * 10000.0f));
        long long y = static_cast<long long>(std::round(v.y() * 10000.0f));
        long long z = static_cast<long long>(std::round(v.z() * 10000.0f));

        size_t h1 = std::hash<long long>{}(x);
        size_t h2 = std::hash<long long>{}(y);
        size_t h3 = std::hash<long long>{}(z);

        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct Vector3fEqual {
    bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
        return (a - b).squaredNorm() < 1e-8f;
    }
};

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

                robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;

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

                robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;

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

                robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;

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

                robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;

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

    virtual void Execute() override
    {
        ExecuteBasic();
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");