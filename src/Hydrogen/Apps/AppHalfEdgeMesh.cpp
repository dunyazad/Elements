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
        
        // ====================================================================
        // [1] 시각화용 절단선(linked_segments) 유지 + 동시에 충돌 면 마킹!
        // ====================================================================
        TS(ExtractIntersectionSegments);
        std::vector<char> cut_faces_mask;

        // 추출하면서 마스크 포인터를 넘기면 부딪힌 면들이 cut_faces_mask에 기록됩니다.
        auto raw_segments = meshes[0]->ExtractIntersectionSegments(*meshes[1], &cut_faces_mask);
        auto welded_segments = meshes[0]->WeldSegments(raw_segments, 5e-3f);
        auto linked_segments = meshes[0]->LinkSegments(welded_segments, 5e-3f);

        // 시각화를 위해 선 닫기 처리
        for (auto& ring : linked_segments) {
            if (ring.size() > 2 && (ring.front() - ring.back()).norm() > 1e-6f) {
                ring.push_back(ring.front());
            }
        }
        TE(ExtractIntersectionSegments);

        // ====================================================================
        // [2] 마킹된 면을 즉시 삭제 (Trench Cut)
        // ====================================================================
        TS(TrenchCut);
        meshes[0]->DeleteMarkedFaces(cut_faces_mask);
        meshes[0]->is_dirty = true;
        TE(TrenchCut);

        // ====================================================================
        // [3] 순수 Flood Fill 로 파편 분리 (이전에 만드신 SeparateDisconnectedMeshes)
        // ====================================================================
        TS(SeparateAndExport);
        auto separated_parts = meshes[0]->SeparateDisconnectedMeshes();

        std::cout << "[Info] Mesh separated into " << separated_parts.size() << " parts." << std::endl;

        if (separated_parts.size() >= 2)
        {
            std::vector<Eigen::Vector4f> colors = {
                Eigen::Vector4f(0.9f, 0.4f, 0.4f, 1.0f),
                Eigen::Vector4f(0.4f, 0.8f, 0.4f, 1.0f),
                Eigen::Vector4f(0.4f, 0.5f, 0.9f, 1.0f),
                Eigen::Vector4f(0.8f, 0.8f, 0.3f, 1.0f)
            };

            for (size_t i = 0; i < separated_parts.size(); ++i)
            {
                std::string filename_stl = "D:\\Temp\\Separated_Part_" + std::to_string(i) + ".stl";
                if (OpenMesh::IO::write_mesh(*separated_parts[i], filename_stl)) {
                    std::cout << "[Export Success] " << filename_stl << " 저장 완료!" << std::endl;
                }

                auto new_helium_mesh = std::make_unique<HeliumMesh>();
                new_helium_mesh->assign(*separated_parts[i]);
                new_helium_mesh->BuildSpatialHashMap();
                new_helium_mesh->mesh_color = colors[i % colors.size()];
                new_helium_mesh->is_dirty = true;
                meshes.push_back(std::move(new_helium_mesh));
            }
        }
        TE(SeparateAndExport);

        // ====================================================================
        // [4] 보존된 linked_segments 로 노란색 절단선 시각화
        // ====================================================================
        VD::Clear("CutLines");
        for (const auto& ring : linked_segments)
        {
            for (size_t i = 0; i < ring.size() - 1; ++i)
            {
                VD::AddLine("CutLines", ring[i], ring[i + 1], Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
            }
        }

        // 렌더링 업데이트 시 원본 메쉬 숨기기
        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes) m->Update();

                if (meshes.size() >= 3)
                {
                    auto renderable_A = Helium.GetComponent<Renderable>(meshes[0]->entity);
                    auto renderable_B = Helium.GetComponent<Renderable>(meshes[1]->entity);
                    if (nullptr != renderable_A) renderable_A->SetVisible(false);
                    if (nullptr != renderable_B) renderable_B->SetVisible(false);
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