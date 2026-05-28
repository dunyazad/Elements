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

// 방금 덮어씌운 헤더를 인클루드 합니다!
#include "SimpleGeometryLibrary.hpp"

namespace Eigen
{
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

// ---------------------------------------------------------
// 해시 맵을 위한 Eigen::Vector3f 해시 및 동등 비교 연산자
// ---------------------------------------------------------
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
// ---------------------------------------------------------

class HeliumMesh : public SGL::Mesh
{
public:
    HeliumMesh() : SGL::Mesh() {}
    virtual ~HeliumMesh() {}

    Entity entity = InvalidEntity;
    bool is_dirty = true;
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();
    Eigen::Vector4f color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

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

            Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [](Entity e, const KeyEvent& event)
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

                        OpenMesh::FaceHandle picked_triangle(-1);
                        float picked_distance = std::numeric_limits<float>::max();

                        Eigen::Vector3f local_origin = ray.origin - offset;

                        int f_idx = -1;
                        if (IntersectGridRay(local_origin, ray.direction, picked_distance, f_idx))
                        {
                            picked_triangle = face_handle(f_idx);
                        }
                        TE(Picking);

                        if (picked_triangle.is_valid())
                        {
                            auto picked_point = ray.origin + ray.direction * picked_distance;
							if (event.IsCtrlPressed())
							{
								auto cameraEntity = Helium.GetEntityByName("MainCamera");
								Helium.GetComponent<Camera>(cameraEntity)->SetTarget(picked_point);
							}

                            OpenMesh::EdgeHandle target_edge;
                            float min_dist = std::numeric_limits<float>::max();
                            Eigen::Vector3f target_v0, target_v1; // 디버깅 선을 그리기 위한 변수

                            // 2. 면의 하프엣지(Halfedge)를 순회하며 엣지 선분과 교차점 사이의 거리를 정확히 측정
                            for (auto fh_it = fh_iter(picked_triangle); fh_it.is_valid(); ++fh_it)
                            {
                                // 해당 엣지의 진짜 양 끝점(Vertex)을 직접 가져옴 (순서 꼬임 원천 차단!)
                                auto v0_handle = from_vertex_handle(*fh_it);
                                auto v1_handle = to_vertex_handle(*fh_it);

                                Eigen::Vector3f v0(point(v0_handle).data());
                                Eigen::Vector3f v1(point(v1_handle).data());

                                // 클릭한 교차점(hit_point)에서 엣지 선분(v0 ~ v1)까지의 거리 계산
                                float dist = DistanceToSegment(picked_point, v0, v1);

                                // 가장 가까운 엣지 찾기
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
                                // 시각화할 때는 offset을 더해 다시 월드 좌표로 변환해 줍니다.
                                //VD::AddLine("Picked edge", target_v0 + offset, target_v1 + offset, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

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
                                    // 외곽선(Boundary)이거나 위상 구조가 망가지는 경우 플립 방지
                                    std::cout << "[Warning] Cannot flip this edge (Boundary or Non-manifold)." << std::endl;
                                }
                            }

                            VD::Clear("Picked");
                            //VD::AddTriangle("Picked", v0 + offset, v1 + offset, v2 + offset, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
                            VD::AddSphere("Picked", picked_point, {0.0f, 0.0f, 1.0f}, 0.005f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
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
            renderable->SetColors4(std::vector<Eigen::Vector4f>(positions.size(), color));
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
                stl.Deserialize("D:\\Resources\\3D\\STL\\maxilla_fixed.stl");

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

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes) m->Update();
            });
    }

    virtual void Execute() override
    {
        ExecuteBasic();
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
