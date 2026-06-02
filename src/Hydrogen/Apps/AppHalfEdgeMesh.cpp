#define _USE_MATH_DEFINES
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
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
    // load an STL into a welded mesh (deduplicate identical vertices first)
    void LoadWelded(HeliumMesh& mesh, const std::string& path)
    {
        STLFormat stl;
        stl.Deserialize(path);

        const std::vector<Eigen::Vector3f>& pts = stl.GetPoints();
        std::vector<Eigen::Vector3f> wp;
        std::vector<Eigen::Vector3i> wi;
        robin_hood::unordered_map<Eigen::Vector3f, int, SGL::Vector3fHash, SGL::Vector3fEqual> vm;

        for (size_t i = 0; i < pts.size(); i += 3)
        {
            Eigen::Vector3i fi;
            for (int j = 0; j < 3; ++j)
            {
                const Eigen::Vector3f& p = pts[i + j];
                auto it = vm.find(p);
                if (it != vm.end()) fi[j] = it->second;
                else { int ni = (int)wp.size(); wp.push_back(p); vm[p] = ni; fi[j] = ni; }
            }
            if (fi[0] != fi[1] && fi[1] != fi[2] && fi[2] != fi[0]) wi.push_back(fi);
        }
        mesh.Build(wp, wi);
    }

    // shared-intersection split: A and B share one canonical vertex pool,
    // so cut-curve vertices are bit-identical on both meshes.
    void ExecuteSplittingShared()
    {
        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_a = *meshes.back();
        LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\A_Maxillar.stl");

        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_b = *meshes.back();
        LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\A_Tooth.stl");

        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        {
            int pa = 0, pb = 0;
            for (auto e = mesh_a.edges_begin(); e != mesh_a.edges_end(); ++e) if (mesh_a.is_boundary(*e)) pa++;
            for (auto e = mesh_b.edges_begin(); e != mesh_b.edges_end(); ++e) if (mesh_b.is_boundary(*e)) pb++;
            std::cout << "[PreSplit] original boundary edges A=" << pa << " B=" << pb << std::endl;
        }

        // Mark intentional openings BEFORE any repair. Boundary loops with many
        // edges (spout, handle openings) are kept open; small loops are defects
        // to be filled. Threshold chosen above teapot defect-hole sizes.
        const size_t opening_min_edges = 13;
        mesh_a.MarkLargeOpeningsAsProtected(opening_min_edges);
        mesh_b.MarkLargeOpeningsAsProtected(opening_min_edges);

        auto print_diag = [](const char* name, const SGL::Mesh::MeshDiagnostics& d)
            {
                std::cout << "[Diagnose " << name << "] V=" << d.num_vertices
                    << " F=" << d.num_faces << " E=" << d.num_edges << std::endl;
                std::cout << "    boundary_edges=" << d.boundary_edges
                    << " boundary_loops=" << d.boundary_loops << std::endl;
                std::cout << "    non_manifold_verts=" << d.non_manifold_vertices
                    << " isolated_verts=" << d.isolated_vertices << std::endl;
                std::cout << "    degenerate_faces=" << d.degenerate_faces
                    << " duplicate_faces=" << d.duplicate_faces << std::endl;
                std::cout << "    watertight=" << (d.is_watertight ? "YES" : "NO") << std::endl;

                std::cout << "    signed_volume=" << d.signed_volume << std::endl;

                // loop size histogram: small loops likely real holes,
                // large loops may be open structure (e.g. lattice gaps)
                int small_loops = 0, large_loops = 0;
                float max_perim = 0.0f;
                for (size_t i = 0; i < d.loop_sizes.size(); ++i)
                {
                    if (d.loop_sizes[i] <= 12) small_loops++;
                    else large_loops++;
                    if (d.loop_perimeters[i] > max_perim) max_perim = d.loop_perimeters[i];
                }
                std::cout << "    loops small(<=12)=" << small_loops
                    << " large(>12)=" << large_loops
                    << " max_perimeter=" << max_perim << std::endl;
            };

        if (!mesh_a.Diagnose().is_watertight)
        {
            std::cout << "[Repair] mesh A is not watertight, repairing..." << std::endl;
            auto da = mesh_a.Repair(30);
            print_diag("A after repair", da);
            mesh_a.BuildSpatialHashMap();
        }

        if (!mesh_b.Diagnose().is_watertight)
        {
            std::cout << "[Repair] mesh B is not watertight, repairing..." << std::endl;
            auto db = mesh_b.Repair(30);
            print_diag("B after repair", db);
            mesh_b.BuildSpatialHashMap();
        }

        print_diag("A", mesh_a.Diagnose());
        print_diag("B", mesh_b.Diagnose());

        std::cout << "[SplitShared] computing intersection once..." << std::endl;

        // compute the intersection a single time from A's perspective;
        // each segment knows both its A-face and its B-face.
        auto shared = mesh_a.ExtractSharedIntersection(mesh_b);
        std::cout << "[SplitShared] segments=" << shared.size() << std::endl;

        // canonical pool cell tied to model scale; both meshes use the same pool
        Eigen::Vector3f bb_min(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        Eigen::Vector3f bb_max(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
        mesh_a.ForeachVertices([&](int, const Eigen::Vector3f& p) { bb_min = bb_min.cwiseMin(p); bb_max = bb_max.cwiseMax(p); });
        float diag = (bb_max - bb_min).norm();
        if (diag < 1e-6f) diag = 1.0f;

        SGL::CanonicalPool pool(diag * 1e-6f);

        auto distribute = [&](HeliumMesh& mesh, int owner_face, int cid,
            std::map<int, std::vector<Eigen::Vector3f>>& fp)
            {
                const Eigen::Vector3f& p = pool.Point(cid);
                OpenMesh::FaceHandle fh = mesh.face_handle(owner_face);
                if (!fh.is_valid() || mesh.status(fh).deleted()) return;

                float edge_eps = diag * 1e-5f;
                OpenMesh::VertexHandle vh;
                OpenMesh::EdgeHandle eh;

                if (mesh.IsOnVertex(p, fh, vh, edge_eps))
                {
                    return;
                }
                else if (mesh.IsOnEdge(p, fh, eh, edge_eps))
                {
                    fp[mesh.face_handle(mesh.halfedge_handle(eh, 0)).idx()].push_back(p);
                    fp[mesh.face_handle(mesh.halfedge_handle(eh, 1)).idx()].push_back(p);
                }
                else
                {
                    fp[owner_face].push_back(p);
                }
            };

        // build per-face data for both meshes, all referencing the SAME ids
        // build per-face data, all referencing the SAME canonical ids
        std::map<int, std::vector<int>> fpA, fpB;
        std::map<int, std::vector<std::pair<int, int>>> fcA, fcB;

        for (const auto& s : shared)
        {
            int id1 = pool.GetID(s.p1);
            int id2 = pool.GetID(s.p2);

            fpA[s.faceA].push_back(id1);
            fpA[s.faceA].push_back(id2);
            fpB[s.faceB].push_back(id1);
            fpB[s.faceB].push_back(id2);

            if (id1 != id2)
            {
                fcA[s.faceA].push_back({ id1, id2 });
                fcB[s.faceB].push_back({ id1, id2 });
            }
        }

        std::cout << "[SplitShared] cut faces A=" << fpA.size() << " B=" << fpB.size()
            << " pool size=" << pool.Size() << std::endl;

        mesh_a.SplitFacesShared(fpA, fcA, pool);
        mesh_b.SplitFacesShared(fpB, fcB, pool);

        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        // ==== 진단: A와 B의 절단점이 split 후 일치하는지 ====
        {
            float test_eps = diag * 1e-5f;
            int matched = 0, unmatched = 0;
            for (const auto& s : shared)
            {
                // is s.p1 present as a vertex in BOTH meshes?
                bool inA = (mesh_a.FindFaceAtPosition(s.p1).is_valid());
                bool inB = (mesh_b.FindFaceAtPosition(s.p1).is_valid());
                // closest vertex distance check
                float dA = 1e9f, dB = 1e9f;
                for (auto v = mesh_a.vertices_begin(); v != mesh_a.vertices_end(); ++v)
                {
                    float d = (Eigen::Vector3f(mesh_a.point(*v).data()) - s.p1).squaredNorm();
                    if (d < dA) dA = d;
                }
                for (auto v = mesh_b.vertices_begin(); v != mesh_b.vertices_end(); ++v)
                {
                    float d = (Eigen::Vector3f(mesh_b.point(*v).data()) - s.p1).squaredNorm();
                    if (d < dB) dB = d;
                }
                if (dA < test_eps * test_eps && dB < test_eps * test_eps) matched++;
                else unmatched++;
                if (matched + unmatched >= 50) break;   // sample first 50
            }
            std::cout << "[Diag] cut-point match A&B: matched=" << matched
                << " unmatched=" << unmatched << std::endl;
        }
        // ==== 진단 끝 ====

        // diagnostics: boundary should now be just the shared cut curve(s)
        int ba = 0, bb = 0;
        for (auto e_it = mesh_a.edges_begin(); e_it != mesh_a.edges_end(); ++e_it) if (mesh_a.is_boundary(*e_it)) ba++;
        for (auto e_it = mesh_b.edges_begin(); e_it != mesh_b.edges_end(); ++e_it) if (mesh_b.is_boundary(*e_it)) bb++;
        std::cout << "[SplitShared] boundary edges A=" << ba << " B=" << bb << std::endl;

        auto loops_b = mesh_b.ExtractBoundaryLoops();
        std::cout << "[SplitShared] B boundary loops=" << loops_b.size() << std::endl;
        for (size_t li = 0; li < loops_b.size(); ++li)
        {
            std::cout << "  loop " << li << " size=" << loops_b[li].size() << std::endl;

            // perimeter and coords of this loop
            float perim = 0.0f;
            std::vector<Eigen::Vector3f> pts;
            for (auto vh : loops_b[li]) pts.push_back(Eigen::Vector3f(mesh_b.point(vh).data()));
            for (size_t k = 0; k < pts.size(); ++k)
                perim += (pts[(k + 1) % pts.size()] - pts[k]).norm();
            std::cout << "    perimeter=" << perim << std::endl;
            for (auto& p : pts)
                std::cout << "    (" << p.x() << ", " << p.y() << ", " << p.z() << ")" << std::endl;
        }

        // ==== 분류 단계 (검증용) ====
        // build a cut-curve pool from the shared intersection points
        SGL::CanonicalPool cutPool(diag * 1e-5f);
        for (const auto& s : shared)
        {
            cutPool.GetID(s.p1);
            cutPool.GetID(s.p2);
        }

        float cut_eps = diag * 1e-4f;

        auto cutA = mesh_a.MarkCutEdges(cutPool, cut_eps);
        auto cutB = mesh_b.MarkCutEdges(cutPool, cut_eps);

        int ngA = 0, ngB = 0;
        auto grpA = mesh_a.ClassifyFaceGroups(cutA, ngA);
        auto grpB = mesh_b.ClassifyFaceGroups(cutB, ngB);

        std::cout << "[Classify] A groups=" << ngA << " B groups=" << ngB << std::endl;

        // color each group differently so we can eyeball the split
        auto color_by_group = [](HeliumMesh& m, const std::vector<int>& grp, int ng)
            {
                // per-face color is not stored; instead recolor whole mesh per group
                // here we just report group sizes
                std::vector<int> sz(ng, 0);
                for (auto f_it = m.faces_begin(); f_it != m.faces_end(); ++f_it)
                    if (!m.status(*f_it).deleted()) sz[grp[f_it->idx()]]++;
                for (int g = 0; g < ng; ++g)
                    std::cout << "    group " << g << " faces=" << sz[g] << std::endl;
            };
        std::cout << "  A:" << std::endl; color_by_group(mesh_a, grpA, ngA);
        std::cout << "  B:" << std::endl; color_by_group(mesh_b, grpB, ngB);
        // ==== 분류 단계 끝 ====

        // ==== 그룹별 안/밖 판정 (다수결 레이캐스팅) ====
        // for each group, sample several faces; a face is "inside other" if a
        // ray from its centroid crosses the other mesh an odd number of times.
        auto classify_inside = [](HeliumMesh& self, const std::vector<int>& grp, int ng,
            HeliumMesh& other) -> std::vector<int>
            {
                // collect a few sample faces per group
                std::vector<std::vector<OpenMesh::FaceHandle>> samples(ng);
                for (auto f_it = self.faces_begin(); f_it != self.faces_end(); ++f_it)
                {
                    if (self.status(*f_it).deleted()) continue;
                    int g = grp[f_it->idx()];
                    if (samples[g].size() < 15) samples[g].push_back(*f_it);
                }

                Eigen::Vector3f dir(0.5773f, 0.5773f, 0.5773f);   // arbitrary fixed ray

                std::vector<int> inside(ng, 0);   // 1=inside other, 0=outside, -1=unknown
                for (int g = 0; g < ng; ++g)
                {
                    if (samples[g].empty()) { inside[g] = -1; continue; }
                    int votes_in = 0, votes_out = 0;
                    for (auto fh : samples[g])
                    {
                        Eigen::Vector3f c = self.FaceCentroid(fh);
                        if (other.IsInsideByNearestFace(c)) votes_in++;
                        else votes_out++;
                    }
                    inside[g] = (votes_in > votes_out) ? 1 : 0;
                }
                return inside;
            };

        auto insideA = classify_inside(mesh_a, grpA, ngA, mesh_b);
        auto insideB = classify_inside(mesh_b, grpB, ngB, mesh_a);

        std::cout << "[InOut] A big groups:" << std::endl;
        for (int g = 0; g < ngA; ++g)
        {
            int cnt = 0;
            for (auto f = mesh_a.faces_begin(); f != mesh_a.faces_end(); ++f)
                if (!mesh_a.status(*f).deleted() && grpA[f->idx()] == g) cnt++;
            if (cnt > 100)
                std::cout << "    group " << g << " faces=" << cnt
                << " inside_B=" << insideA[g] << std::endl;
        }
        std::cout << "[InOut] B big groups:" << std::endl;
        for (int g = 0; g < ngB; ++g)
        {
            int cnt = 0;
            for (auto f = mesh_b.faces_begin(); f != mesh_b.faces_end(); ++f)
                if (!mesh_b.status(*f).deleted() && grpB[f->idx()] == g) cnt++;
            if (cnt > 100)
                std::cout << "    group " << g << " faces=" << cnt
                << " inside_A=" << insideB[g] << std::endl;
        }
        // ==== 안/밖 판정 끝 ====

        // ==== boolean 결과 조립 (A - B 예시) ====
        // group selection by inside/outside flags:
        //   union     : A outside + B outside
        //   intersect : A inside  + B inside
        //   diff A-B  : A outside + B inside (B flipped)
        auto build_selection = [&](bool a_inside_wanted, bool b_inside_wanted, bool flip_b)
            {
                std::set<int> keepA, keepB;
                for (int g = 0; g < ngA; ++g)
                    if ((insideA[g] == 1) == a_inside_wanted) keepA.insert(g);
                for (int g = 0; g < ngB; ++g)
                    if ((insideB[g] == 1) == b_inside_wanted) keepB.insert(g);

                std::vector<Eigen::Vector3f> pts;
                std::vector<Eigen::Vector3i> idx;
                mesh_a.CollectGroupFaces(grpA, keepA, false, pts, idx);
                mesh_b.CollectGroupFaces(grpB, keepB, flip_b, pts, idx);
                return std::make_pair(pts, idx);
            };

        struct BoolOp { const char* name; bool aIn; bool bIn; bool flipB; float ox; };
        BoolOp ops[] = {
            { "diff",  false, true,  true,  200.0f },
            { "union", false, false, false, 400.0f },
            { "inter", true,  true,  false, 600.0f },
        };

        for (const auto& op : ops)
        {
            auto sel = build_selection(op.aIn, op.bIn, op.flipB);

            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& mr = *meshes.back();
            mr.Build(sel.first, sel.second);
            mr.WeldVerticesByPosition(diag * 1e-5f);
            int holes_filled = mr.FillSmallBoundaryHoles(8);
            int removed = mr.RemoveSmallBoundaryFaces(12);
            if (removed > 0) mr.FillSmallBoundaryHoles(12);
            mr.offset = Eigen::Vector3f(op.ox, 0.0f, 0.0f);
            mr.mesh_color = Eigen::Vector4f(1.0f, 0.6f, 0.2f, 1.0f);

            int rb = 0;
            for (auto e = mr.edges_begin(); e != mr.edges_end(); ++e)
                if (mr.is_boundary(*e)) rb++;
            int rb2 = 0;
            for (auto e = mr.edges_begin(); e != mr.edges_end(); ++e)
                if (mr.is_boundary(*e)) rb2++;
            std::cout << "[Boolean] " << op.name << " V=" << mr.n_vertices()
                << " F=" << mr.n_faces()
                << " boundary " << rb << "->" << rb2 << std::endl;
                //<< " (filled " << holes_filled << ")" << std::endl;

            std::string path = std::string("D:\\Temp\\boolean_") + op.name + ".stl";
            mr.SaveSTL(path);
            mr.is_dirty = true;

            if (std::string(op.name) == "diff")
            {
                auto rl = mr.ExtractBoundaryLoops();
                std::cout << "[DiffLoops] count=" << rl.size() << std::endl;
                for (size_t i = 0; i < rl.size() && i < 12; ++i)
                {
                    Eigen::Vector3f p(mr.point(rl[i][0]).data());
                    std::cout << "  loop " << i << " size=" << rl[i].size()
                        << " first=(" << p.x() << "," << p.y() << "," << p.z() << ")"
                        << std::endl;
                }
            }
        }

        mesh_a.SaveSTL("D:\\Temp\\mesh_a_shared.stl");
        mesh_b.SaveSTL("D:\\Temp\\mesh_b_shared.stl");

        VD::Clear("IntersectionSegments");
        for (const auto& s : shared)
            VD::AddLine("IntersectionSegments", s.p1, s.p2, Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));

        mesh_a.is_dirty = true;
        mesh_b.is_dirty = true;

        Helium.AddOnUpdateCallback([this](float time_delta)
            {
                for (auto& m : this->meshes) m->Update();
            });
    }

    // original independent-split path, kept for reference
    void ExecuteSplitting()
    {
        float snap_eps = 1e-3f;

        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_a = *meshes.back();
        LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\Cube.stl");

        meshes.emplace_back(std::make_unique<HeliumMesh>());
        HeliumMesh& mesh_b = *meshes.back();
        LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\Cylinder.stl");

        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        std::cout << "[Split] Extracting intersection segments with face ids..." << std::endl;

        auto seg_a = mesh_a.ExtractIntersectionSegmentsWithFace(mesh_b);
        auto seg_b = mesh_b.ExtractIntersectionSegmentsWithFace(mesh_a);
        std::cout << "[Split] seg_a=" << seg_a.size() << " seg_b=" << seg_b.size() << std::endl;

        auto fpm_a = mesh_a.BuildFacePointsFromSegments(seg_a, snap_eps);
        auto fpm_b = mesh_b.BuildFacePointsFromSegments(seg_b, snap_eps);
        std::cout << "[Split] faces_a=" << fpm_a.size() << " faces_b=" << fpm_b.size() << std::endl;

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
        ExecuteSplittingShared();
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");