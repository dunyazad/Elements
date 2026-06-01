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

class HeliumMesh;

std::vector<Eigen::Vector3f> intersection_points;
std::map<HeliumMesh*, std::map<SGL::Mesh::FaceHandle, std::vector<Eigen::Vector3f>>> face_point_map;

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
#if 0
                                auto points = face_point_map[this][hit_result.fh];

                                Eigen::Vector3f v0, v1, v2;
                                GetFaceVertices(hit_result.fh, v0, v1, v2);

                                VD::AddTriangle("PickedTriangle", v0, v1, v2, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 0.5f));
                                for (size_t i = 0; i < points.size(); i++)
                                {
                                    auto& p = points[i];
                                    VD::AddSphere("PickedTrianglePoints", p, { 0.0f, 0.0f, 1.0f }, 0.005f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

                                    printf("[Info] Intersection Point %zu: (%.4f, %.4f, %.4f)\n", i, p.x(), p.y(), p.z());
                                }
#endif // 0

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

class AppHalfEdgeMesh : public App
{
public:
    void ExecuteSplitting()
    {
        float snap_eps = 1e-3f;

        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_a = *meshes.back();

        STLFormat STL_A;
        STL_A.Deserialize("D:\\Resources\\3D\\STL\\Cube.stl");
        //STL_A.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");
        {
            const std::vector<Eigen::Vector3f>& points_a = STL_A.GetPoints();
            std::vector<Eigen::Vector3f> wp;
            std::vector<Eigen::Vector3i> wi;
            robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vm;
            for (size_t i = 0; i < points_a.size(); i += 3)
            {
                Eigen::Vector3i fi;
                for (int j = 0; j < 3; ++j)
                {
                    const Eigen::Vector3f& p = points_a[i + j];
                    auto it = vm.find(p);
                    if (it != vm.end()) fi[j] = it->second;
                    else { int ni = (int)wp.size(); wp.push_back(p); vm[p] = ni; fi[j] = ni; }
                }
                if (fi[0] != fi[1] && fi[1] != fi[2] && fi[2] != fi[0]) wi.push_back(fi);
            }
            mesh_a.Build(wp, wi);
        }

        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_b = *meshes.back();

        STLFormat STL_B;
        STL_B.Deserialize("D:\\Resources\\3D\\STL\\Cylinder.stl");
        //STL_B.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");
        {
            const std::vector<Eigen::Vector3f>& points_b = STL_B.GetPoints();
            std::vector<Eigen::Vector3f> wp;
            std::vector<Eigen::Vector3i> wi;
            robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vm;
            for (size_t i = 0; i < points_b.size(); i += 3)
            {
                Eigen::Vector3i fi;
                for (int j = 0; j < 3; ++j)
                {
                    const Eigen::Vector3f& p = points_b[i + j];
                    auto it = vm.find(p);
                    if (it != vm.end()) fi[j] = it->second;
                    else { int ni = (int)wp.size(); wp.push_back(p); vm[p] = ni; fi[j] = ni; }
                }
                if (fi[0] != fi[1] && fi[1] != fi[2] && fi[2] != fi[0]) wi.push_back(fi);
            }
            mesh_b.Build(wp, wi);
        }

        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        std::cout << "[Split] Extracting intersection segments with face ids..." << std::endl;

        auto seg_a = mesh_a.ExtractIntersectionSegmentsWithFace(mesh_b);
        auto seg_b = mesh_b.ExtractIntersectionSegmentsWithFace(mesh_a);
        std::cout << "[Split] seg_a=" << seg_a.size() << " seg_b=" << seg_b.size() << std::endl;

        auto fpm_a = mesh_a.BuildFacePointsFromSegments(seg_a, snap_eps);
        auto fpm_b = mesh_b.BuildFacePointsFromSegments(seg_b, snap_eps);
        std::cout << "[Split] faces_a=" << fpm_a.size() << " faces_b=" << fpm_b.size() << std::endl;

        // 두 번째 인자로 교차 선분 배열(Constraints)을 넘겨줍니다!
        mesh_a.SplitFaces(fpm_a, seg_a);
        mesh_b.SplitFaces(fpm_b, seg_b);

        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        int ba = 0, bb = 0;
        for (auto e_it = mesh_a.edges_begin(); e_it != mesh_a.edges_end(); ++e_it) if (mesh_a.is_boundary(*e_it)) ba++;
        for (auto e_it = mesh_b.edges_begin(); e_it != mesh_b.edges_end(); ++e_it) if (mesh_b.is_boundary(*e_it)) bb++;
        std::cout << "[Split] boundary edges A=" << ba << " B=" << bb << std::endl;

        mesh_a.SaveSTL("D:\\Temp\\mesh_a_split.stl");
        mesh_b.SaveSTL("D:\\Temp\\mesh_b_split.stl");

        VD::Clear("IntersectionSegments");
        for (const auto& s : seg_a)
            VD::AddLine("IntersectionSegments", std::get<0>(s), std::get<1>(s), Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));

        mesh_a.is_dirty = true;
        mesh_b.is_dirty = true;

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes) m->Update();
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