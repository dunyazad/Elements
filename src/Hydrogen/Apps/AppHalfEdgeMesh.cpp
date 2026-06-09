#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include "SimpleGeometryLibrary.hpp"
#include "RGO/HeliumMesh.h"
#include <OpenMesh/Core/IO/MeshIO.hh>


class AppHalfEdgeMesh : public App
{
public:
    virtual void Initialize() override
    {
    }

    // load an STL into a welded mesh (deduplicate identical vertices first)
    void LoadWelded(SGLHeliumMesh& mesh, const std::string& path)
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
        meshes.emplace_back(std::make_unique<SGLHeliumMesh>());
        SGLHeliumMesh& mesh_a = *meshes.back();
        LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\A_Maxillar.stl");

        meshes.emplace_back(std::make_unique<SGLHeliumMesh>());
        SGLHeliumMesh& mesh_b = *meshes.back();
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

        // build per-face data for both meshes, all referencing the SAME
        // canonical ids
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

        // diagnostics: cut points should coincide on both meshes after split
        {
            float test_eps = diag * 1e-5f;
            int matched = 0, unmatched = 0;
            for (const auto& s : shared)
            {
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

        // report group sizes so we can eyeball the split
        auto report_groups = [](SGLHeliumMesh& m, const std::vector<int>& grp, int ng)
            {
                std::vector<int> sz(ng, 0);
                for (auto f_it = m.faces_begin(); f_it != m.faces_end(); ++f_it)
                    if (!m.status(*f_it).deleted()) sz[grp[f_it->idx()]]++;
                for (int g = 0; g < ng; ++g)
                    std::cout << "    group " << g << " faces=" << sz[g] << std::endl;
            };
        std::cout << "  A:" << std::endl; report_groups(mesh_a, grpA, ngA);
        std::cout << "  B:" << std::endl; report_groups(mesh_b, grpB, ngB);

        // per-group inside/outside by nearest-face majority vote. nearest-face
        // is robust on open meshes (the jaw has a protected opening), unlike a
        // ray-crossing test that would miscount through the opening.
        auto classify_inside = [](SGLHeliumMesh& self, const std::vector<int>& grp, int ng,
            SGLHeliumMesh& other) -> std::vector<int>
            {
                // collect a few sample faces per group
                std::vector<std::vector<OpenMesh::FaceHandle>> samples(ng);
                for (auto f_it = self.faces_begin(); f_it != self.faces_end(); ++f_it)
                {
                    if (self.status(*f_it).deleted()) continue;
                    int g = grp[f_it->idx()];
                    if (samples[g].size() < 15) samples[g].push_back(*f_it);
                }

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

        // boolean result assembly. group selection by inside/outside flags:
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

            meshes.emplace_back(std::make_unique<SGLHeliumMesh>());
            SGLHeliumMesh& mr = *meshes.back();
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
            std::cout << "[Boolean] " << op.name << " V=" << mr.n_vertices()
                << " F=" << mr.n_faces()
                << " boundary " << rb << "->" << rb << std::endl;

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

    virtual void Execute() override
    {
        ExecuteSplittingShared();
    }

    std::vector<std::unique_ptr<SGLHeliumMesh>> meshes;
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
