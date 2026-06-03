#define _USE_MATH_DEFINES
#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define NOMINMAX

#include "Apps.h"

#include <execution>
#include <algorithm>
#include <vector>
#include <set>
#include <cmath>
#include <iostream>

#include <robin_hood/robin_hood.h>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include "SimpleGeometryLibrary.hpp"
#include "HeliumMesh.h"
#include "TextMeshGenerator.hpp"

#include <OpenMesh/Core/IO/MeshIO.hh>

class AppTextMesh : public App
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

    // Step 1/2 shared verification: is a mesh valid boolean input?
    // Checks watertightness, z-range, manifold-ness. tag names the mesh.
    void VerifyInputMesh(HeliumMesh& m, const char* tag, float z0, float z1)
    {
        auto d = m.Diagnose();

        Eigen::Vector3f bmin(1e30f, 1e30f, 1e30f), bmax(-1e30f, -1e30f, -1e30f);
        m.ForeachVertices([&](int, const Eigen::Vector3f& p)
            { bmin = bmin.cwiseMin(p); bmax = bmax.cwiseMax(p); });

        std::cout << "=== [verify] " << tag << " ===" << std::endl;
        std::cout << "  V=" << d.num_vertices << " F=" << d.num_faces
            << " E=" << d.num_edges << std::endl;
        std::cout << "  boundary_edges=" << d.boundary_edges
            << " non_manifold_verts=" << d.non_manifold_vertices
            << " degenerate_faces=" << d.degenerate_faces
            << " duplicate_faces=" << d.duplicate_faces << std::endl;
        std::cout << "  z-range=[" << bmin.z() << "," << bmax.z()
            << "] expected=[" << z0 << "," << z1 << "]" << std::endl;
        std::cout << "  signed_volume=" << d.signed_volume << std::endl;

        bool z_ok = std::abs(bmin.z() - z0) < 1e-3f && std::abs(bmax.z() - z1) < 1e-3f;
        bool watertight = (d.boundary_edges == 0 &&
            d.non_manifold_vertices == 0 &&
            d.degenerate_faces == 0);

        std::cout << "  => watertight=" << (watertight ? "YES" : "NO")
            << " z_range_ok=" << (z_ok ? "YES" : "NO")
            << " :: " << (watertight && z_ok ? "PASS" : "FAIL")
            << std::endl;
    }

    // Step 3-1 verification: did ExtractSharedIntersection capture the cut curve
    // correctly? PASS requires no degenerate segments and no invalid face refs.
    // Points with degree != 2 are reported as warnings: these are glyph-outline
    // self-intersections (strokes meeting at a point), which are valid input and
    // are handled fine by B; they do not fail this step.
    void VerifyIntersection(const std::vector<SGL::SharedSegment>& shared, float diag)
    {
        std::cout << "=== [Step3-1 verify] intersection curve ===" << std::endl;
        std::cout << "  segments=" << shared.size() << std::endl;
        if (shared.empty()) { std::cout << "  => STEP3-1 FAIL (no segments)" << std::endl; return; }

        int degenerate = 0, bad_face = 0;
        for (const auto& s : shared)
        {
            if ((s.p1 - s.p2).norm() < diag * 1e-6f) degenerate++;
            if (s.faceA < 0 || s.faceB < 0) bad_face++;
        }

        float w = diag * 1e-4f, w_sq = w * w;
        std::vector<Eigen::Vector3f> ends;
        ends.reserve(shared.size() * 2);
        for (const auto& s : shared) { ends.push_back(s.p1); ends.push_back(s.p2); }

        std::vector<char> visited(ends.size(), 0);
        int unique_pts = 0, branch_pts = 0;
        for (size_t i = 0; i < ends.size(); ++i)
        {
            if (visited[i]) continue;
            int shared_here = 1;
            for (size_t j = i + 1; j < ends.size(); ++j)
                if (!visited[j] && (ends[i] - ends[j]).squaredNorm() < w_sq)
                {
                    visited[j] = 1; shared_here++;
                }
            visited[i] = 1;
            unique_pts++;
            if (shared_here != 2) branch_pts++;
        }

        std::cout << "  degenerate_segments=" << degenerate
            << " invalid_face_refs=" << bad_face << std::endl;
        std::cout << "  unique_curve_points=" << unique_pts
            << " branch_points(self-intersections)=" << branch_pts << std::endl;

        bool pass = (degenerate == 0 && bad_face == 0);
        std::cout << "  => STEP3-1 " << (pass ? "PASS" : "FAIL")
            << " (branch points are valid glyph self-intersections)" << std::endl;
    }

    // Step 3-2 verification: do A and B share the same canonical id for the same
    // physical intersection point? Each segment endpoint is registered into both
    // fpA[faceA] and fpB[faceB] using one shared pool, so the ids MUST match.
    // This confirms the cut curve uses identical vertex ids on both meshes.
    void VerifyCanonicalSharing(
        const std::vector<SGL::SharedSegment>& shared,
        SGL::CanonicalPool& pool,
        const std::map<int, std::vector<int>>& fpA,
        const std::map<int, std::vector<int>>& fpB)
    {
        std::cout << "=== [Step3-2 verify] canonical id sharing ===" << std::endl;

        int checked = 0, mismatched = 0, missing_in_A = 0, missing_in_B = 0;

        auto id_in_face = [](const std::map<int, std::vector<int>>& fp, int face, int id) -> bool
            {
                auto it = fp.find(face);
                if (it == fp.end()) return false;
                for (int c : it->second) if (c == id) return true;
                return false;
            };

        for (const auto& s : shared)
        {
            int id1 = pool.GetID(s.p1);   // same pool -> returns the existing id
            int id2 = pool.GetID(s.p2);
            checked += 2;

            // id1, id2 must appear in BOTH fpA[faceA] and fpB[faceB]
            if (!id_in_face(fpA, s.faceA, id1)) missing_in_A++;
            if (!id_in_face(fpA, s.faceA, id2)) missing_in_A++;
            if (!id_in_face(fpB, s.faceB, id1)) missing_in_B++;
            if (!id_in_face(fpB, s.faceB, id2)) missing_in_B++;
        }

        // also confirm: no two different ids map to the same physical position
        float eps = (pool.Size() > 0) ? 0.0f : 0.0f;
        (void)eps;

        std::cout << "  endpoints_checked=" << checked << std::endl;
        std::cout << "  missing_in_A=" << missing_in_A
            << " missing_in_B=" << missing_in_B << std::endl;

        bool pass = (missing_in_A == 0 && missing_in_B == 0);
        std::cout << "  => STEP3-2 " << (pass ? "PASS" : "FAIL")
            << " (same point -> same id in both A and B)" << std::endl;
    }

    // Step 3-3 helper: collect the set of cut edges in a split mesh. A cut edge
    // is one whose BOTH endpoints coincide with intersection-curve points (pool
    // points). Returns them as unordered position-key pairs so A and B can be
    // compared directly.
    std::set<std::pair<std::array<int, 3>, std::array<int, 3>>> CollectCutEdges(
        HeliumMesh& m, SGL::CanonicalPool& pool, float diag)
    {
        float match = diag * 1e-4f;
        float quant = diag * 1e-4f;   // position quantization for the key

        auto key_of = [&](const Eigen::Vector3f& p) -> std::array<int, 3>
            {
                return { (int)std::lround(p.x() / quant),
                         (int)std::lround(p.y() / quant),
                         (int)std::lround(p.z() / quant) };
            };

        std::set<std::pair<std::array<int, 3>, std::array<int, 3>>> edges;

        for (auto e = m.edges_begin(); e != m.edges_end(); ++e)
        {
            auto h = m.halfedge_handle(*e, 0);
            Eigen::Vector3f a(m.point(m.from_vertex_handle(h)).data());
            Eigen::Vector3f b(m.point(m.to_vertex_handle(h)).data());
            if (pool.Contains(a, match) && pool.Contains(b, match))
            {
                auto ka = key_of(a), kb = key_of(b);
                if (kb < ka) std::swap(ka, kb);
                edges.insert({ ka, kb });
            }
        }
        return edges;
    }

    // Step 3-3 diagnosis: where are a split mesh's boundary edges? Buckets each
    // boundary edge by its z-range so we can tell if the split broke the cut
    // plane (z=1 outline) or some other region. Reports counts per z-character.
    void ReportBoundaryByZ(HeliumMesh& m, const char* tag, float cut_z)
    {
        int at_cut = 0, below = 0, above = 0, spanning = 0, total = 0;
        for (auto e = m.edges_begin(); e != m.edges_end(); ++e)
        {
            if (!m.is_boundary(*e)) continue;
            total++;
            auto h = m.halfedge_handle(*e, 0);
            float za = m.point(m.from_vertex_handle(h)).data()[2];
            float zb = m.point(m.to_vertex_handle(h)).data()[2];
            bool a_cut = std::abs(za - cut_z) < 1e-3f;
            bool b_cut = std::abs(zb - cut_z) < 1e-3f;
            if (a_cut && b_cut) at_cut++;
            else if (za < cut_z - 1e-3f && zb < cut_z - 1e-3f) below++;
            else if (za > cut_z + 1e-3f && zb > cut_z + 1e-3f) above++;
            else spanning++;
        }
        std::cout << "  [bndZ " << tag << "] total=" << total
            << " at_cut_plane(z=" << cut_z << ")=" << at_cut
            << " below=" << below << " above=" << above
            << " spanning=" << spanning << std::endl;
    }

    // Step 3-3 diagnosis: is the weld tolerance larger than the spacing between
    // cut-curve points on the cut plane? If two distinct outline points are
    // closer than snap_eps, WeldVerticesByPosition merges them, which tears the
    // triangles that referenced them apart -> boundary edges on the cut plane.
    void ReportMinCutPointSpacing(SGL::CanonicalPool& pool, float diag, float cut_z)
    {
        float snap_eps = diag * 1e-5f;

        // gather pool points lying on the cut plane
        std::vector<Eigen::Vector3f> pts;
        for (int i = 0; i < (int)pool.Size(); ++i)
        {
            const Eigen::Vector3f& p = pool.Point(i);
            if (std::abs(p.z() - cut_z) < 1e-3f) pts.push_back(p);
        }

        // brute-force nearest-neighbor min distance (O(n^2); diagnostic only)
        float min_d = 1e30f;
        Eigen::Vector3f pa, pb;
        for (size_t i = 0; i < pts.size(); ++i)
            for (size_t j = i + 1; j < pts.size(); ++j)
            {
                float d = (pts[i] - pts[j]).norm();
                if (d < min_d) { min_d = d; pa = pts[i]; pb = pts[j]; }
            }

        int closer_than_eps = 0;
        for (size_t i = 0; i < pts.size(); ++i)
            for (size_t j = i + 1; j < pts.size(); ++j)
                if ((pts[i] - pts[j]).norm() < snap_eps) closer_than_eps++;

        std::cout << "  [spacing] cut-plane points=" << pts.size()
            << " snap_eps=" << snap_eps
            << " min_spacing=" << min_d << std::endl;
        std::cout << "  [spacing] point pairs closer than snap_eps=" << closer_than_eps
            << "  => " << (min_d < snap_eps ? "WELD WILL MERGE DISTINCT POINTS"
                : "spacing safe") << std::endl;
    }

    // Step 3-3 diagnosis: which A faces receive cut constraints, and what is
    // their z-extent? If the cut plane (z=cut_z) is split across MULTIPLE A
    // faces that each run their own CDT, those faces subdivide the shared
    // outline independently and their triangles won't meet -> boundary on the
    // cut plane. Reports each constraint-bearing face's z-range and seg count.
    void ReportCutFaces(HeliumMesh& mesh_a,
        const std::map<int, std::vector<std::pair<int, int>>>& fcA, float cut_z)
    {
        std::cout << "  [cutfaces] A faces receiving constraints:" << std::endl;
        int on_cut_plane = 0;
        for (const auto& kv : fcA)
        {
            OpenMesh::FaceHandle fh = mesh_a.face_handle(kv.first);
            if (!fh.is_valid() || mesh_a.status(fh).deleted()) continue;
            Eigen::Vector3f v0, v1, v2;
            mesh_a.GetFaceVertices(fh, v0, v1, v2);
            float zmn = std::min({ v0.z(), v1.z(), v2.z() });
            float zmx = std::max({ v0.z(), v1.z(), v2.z() });
            bool is_cut_plane = (std::abs(zmn - cut_z) < 1e-3f && std::abs(zmx - cut_z) < 1e-3f);
            if (is_cut_plane) on_cut_plane++;
            std::cout << "    faceA=" << kv.first << " segs=" << kv.second.size()
                << " z[" << zmn << "," << zmx << "]"
                << (is_cut_plane ? " <== ON CUT PLANE" : "") << std::endl;
        }
        std::cout << "  [cutfaces] faces lying exactly on cut plane z=" << cut_z
            << " = " << on_cut_plane
            << (on_cut_plane > 1 ? "  (MULTIPLE -> independent CDTs tear the seam)" : "")
            << std::endl;
    }

    // Step 3-3 fix verification: do the coplanar cut faces group correctly?
    // The two plate-top triangles (face2,face3) should merge into one group of
    // size 2 so they share a single CDT instead of two independent ones.
    void ReportCoplanarGroups(HeliumMesh& mesh_a,
        const std::map<int, std::vector<std::pair<int, int>>>& fcA, float diag)
    {
        std::set<int> cut_faces;
        for (const auto& kv : fcA) cut_faces.insert(kv.first);

        auto groups = mesh_a.GroupCoplanarCutFaces(cut_faces, 1e-4f, diag * 1e-4f);

        std::cout << "  [coplanar] cut faces=" << cut_faces.size()
            << " grouped into " << groups.size() << " groups:" << std::endl;
        for (const auto& g : groups)
        {
            std::cout << "    group size=" << g.size() << " faces:";
            for (int fi : g) std::cout << " " << fi;
            std::cout << std::endl;
        }
    }

    // Step 3-3 verification: after splitting, is each split mesh still watertight
    // and do A and B carry the same cut-line edges? Buckets boundary edges by z,
    // reports which A faces carry constraints on the cut plane.
    void VerifySplitConsistency(HeliumMesh& mesh_a, HeliumMesh& mesh_b,
        SGL::CanonicalPool& pool, float diag, float cut_z,
        const std::map<int, std::vector<std::pair<int, int>>>& fcA_before_split)
    {
        std::cout << "=== [Step3-3 verify] split consistency ===" << std::endl;

        auto da = mesh_a.Diagnose();
        auto db = mesh_b.Diagnose();
        std::cout << "  A: V=" << da.num_vertices << " F=" << da.num_faces
            << " boundary=" << da.boundary_edges
            << " non_manifold=" << da.non_manifold_vertices << std::endl;
        ReportBoundaryByZ(mesh_a, "A", cut_z);
        std::cout << "  B: V=" << db.num_vertices << " F=" << db.num_faces
            << " boundary=" << db.boundary_edges
            << " non_manifold=" << db.non_manifold_vertices << std::endl;
        ReportBoundaryByZ(mesh_b, "B", cut_z);

        bool a_wt = (da.boundary_edges == 0 && da.non_manifold_vertices == 0);
        bool b_wt = (db.boundary_edges == 0 && db.non_manifold_vertices == 0);

        std::cout << "  => STEP3-3 "
            << (a_wt && b_wt ? "PASS" : "FAIL")
            << " (A_watertight=" << (a_wt ? "Y" : "N")
            << " B_watertight=" << (b_wt ? "Y" : "N") << ")" << std::endl;
    }

    // Step 3-4 verification: at each final-boundary hole, find the faces in A
    // and B that touch that location and report their inside/outside test and
    // keep decision. A hole means a face that SHOULD have been kept was dropped
    // (or vice versa) by the per-face centroid classification.
    void VerifyClassificationAtHoles(
        HeliumMesh& mesh_a, HeliumMesh& mesh_b,
        const std::vector<char>& keepA, const std::vector<char>& keepB,
        const std::vector<char>& inFaceA, const std::vector<char>& inFaceB,
        const std::vector<Eigen::Vector3f>& hole_points)
    {
        std::cout << "=== [Step3-4 verify] classification at holes ===" << std::endl;

        auto probe = [&](HeliumMesh& m, const std::vector<char>& keep,
            const std::vector<char>& inside, const char* tag, const Eigen::Vector3f& h)
            {
                for (auto f = m.faces_begin(); f != m.faces_end(); ++f)
                {
                    if (m.status(*f).deleted()) continue;
                    Eigen::Vector3f v0, v1, v2;
                    m.GetFaceVertices(*f, v0, v1, v2);
                    bool touches = (v0 - h).norm() < 0.05f || (v1 - h).norm() < 0.05f || (v2 - h).norm() < 0.05f;
                    if (!touches) continue;
                    float zmn = std::min({ v0.z(), v1.z(), v2.z() });
                    float zmx = std::max({ v0.z(), v1.z(), v2.z() });
                    Eigen::Vector3f ctr = (v0 + v1 + v2) / 3.0f;
                    std::cout << "    [" << tag << "] face=" << f->idx()
                        << " z[" << zmn << "," << zmx << "]"
                        << " ctr=(" << ctr.x() << "," << ctr.y() << "," << ctr.z() << ")"
                        << " inside=" << (int)inside[f->idx()]
                        << " keep=" << (int)keep[f->idx()] << std::endl;
                }
            };

        for (const auto& h : hole_points)
        {
            std::cout << "  hole at (" << h.x() << "," << h.y() << "," << h.z() << "):" << std::endl;
            probe(mesh_a, keepA, inFaceA, "A", h);
            probe(mesh_b, keepB, inFaceB, "B", h);
        }
    }

    // Step 3-4 verification: for each exact hole vertex, find which faces in A
    // and B contain that vertex, and report each face's inside test and keep
    // decision. Tells us whether the hole triangle was wrongly dropped (kept=0
    // when it should be 1) or never existed in either mesh.
    void ProbeHoleVertex(HeliumMesh& mesh_a, HeliumMesh& mesh_b,
        const std::vector<char>& keepA, const std::vector<char>& keepB,
        const std::vector<char>& inFaceA, const std::vector<char>& inFaceB,
        const Eigen::Vector3f& hv)
    {
        std::cout << "  vertex (" << hv.x() << "," << hv.y() << "," << hv.z() << "):" << std::endl;

        auto probe = [&](HeliumMesh& m, const std::vector<char>& keep,
            const std::vector<char>& inside, const char* tag)
            {
                int found = 0;
                for (auto f = m.faces_begin(); f != m.faces_end(); ++f)
                {
                    if (m.status(*f).deleted()) continue;
                    Eigen::Vector3f v0, v1, v2;
                    m.GetFaceVertices(*f, v0, v1, v2);
                    bool has = (v0 - hv).norm() < 0.01f || (v1 - hv).norm() < 0.01f || (v2 - hv).norm() < 0.01f;
                    if (!has) continue;
                    found++;
                    Eigen::Vector3f ctr = (v0 + v1 + v2) / 3.0f;
                    float zmn = std::min({ v0.z(), v1.z(), v2.z() });
                    float zmx = std::max({ v0.z(), v1.z(), v2.z() });
                    std::cout << "    [" << tag << "] face=" << f->idx()
                        << " z[" << zmn << "," << zmx << "]"
                        << " ctr=(" << ctr.x() << "," << ctr.y() << "," << ctr.z() << ")"
                        << " inside=" << (int)inside[f->idx()]
                        << " keep=" << (int)keep[f->idx()] << std::endl;
                }
                if (found == 0) std::cout << "    [" << tag << "] no face has this vertex" << std::endl;
            };
        probe(mesh_a, keepA, inFaceA, "A");
        probe(mesh_b, keepB, inFaceB, "B");
    }

    // Step 3-4 diagnosis: compare two inside-tests for specific A top faces.
    // (1) IsInsideByNearestFace (current, suspect) vs
    // (2) 2D ray-crossing in the z=cut plane against B's geometry: cast a ray
    //     in +x from the point and count B edges crossed at this z; odd = inside.
    // If the two disagree on the misclassified faces, the nearest-face test is
    // the culprit and a planar test fixes it.
    void CompareInsideTests(HeliumMesh& mesh_a, HeliumMesh& mesh_b,
        const std::vector<Eigen::Vector3f>& centroids, float cut_z)
    {
        std::cout << "=== [Step3-4 compare] inside tests on suspect A faces ===" << std::endl;

        for (const auto& c : centroids)
        {
            bool nf = mesh_b.IsInsideByNearestFace(c);

            // 2D ray crossing: ray from c in +x at z=cut_z, count B triangle
            // edges (projected) that the ray crosses on the cut plane
            int crossings = 0;
            for (auto f = mesh_b.faces_begin(); f != mesh_b.faces_end(); ++f)
            {
                if (mesh_b.status(*f).deleted()) continue;
                Eigen::Vector3f v0, v1, v2;
                mesh_b.GetFaceVertices(*f, v0, v1, v2);
                // only faces straddling the cut plane contribute a cross-section edge
                float zmn = std::min({ v0.z(), v1.z(), v2.z() });
                float zmx = std::max({ v0.z(), v1.z(), v2.z() });
                if (zmn > cut_z || zmx < cut_z) continue;

                // intersect each triangle edge with plane z=cut_z, gather 2 pts
                Eigen::Vector3f tv[3] = { v0, v1, v2 };
                std::vector<Eigen::Vector2f> hits;
                for (int e = 0; e < 3; ++e)
                {
                    Eigen::Vector3f a = tv[e], b = tv[(e + 1) % 3];
                    float za = a.z() - cut_z, zb = b.z() - cut_z;
                    if ((za <= 0 && zb > 0) || (za > 0 && zb <= 0))
                    {
                        float t = za / (za - zb);
                        Eigen::Vector3f p = a + t * (b - a);
                        hits.push_back(Eigen::Vector2f(p.x(), p.y()));
                    }
                }
                if (hits.size() != 2) continue;
                // ray from c in +x crosses segment hits[0]-hits[1]?
                Eigen::Vector2f p1 = hits[0], p2 = hits[1];
                float cy = c.y();
                if ((p1.y() > cy) != (p2.y() > cy))
                {
                    float xint = p1.x() + (cy - p1.y()) / (p2.y() - p1.y()) * (p2.x() - p1.x());
                    if (xint > c.x()) crossings++;
                }
            }
            bool planar_inside = (crossings % 2) == 1;

            std::cout << "  c=(" << c.x() << "," << c.y() << "," << c.z() << ")"
                << " nearest_face_inside=" << (int)nf
                << " planar_crossings=" << crossings
                << " planar_inside=" << (int)planar_inside
                << (nf != planar_inside ? "  <== DISAGREE" : "") << std::endl;
        }
    }

    // run one boolean op between target (A) and text (B) using the shared
    // split + classify pipeline. op: "diff" (engrave) or "union" (emboss).
    // Result is built into 'out'.
    void BooleanInto(HeliumMesh& out, HeliumMesh& mesh_a, HeliumMesh& mesh_b,
        const char* op)
    {
        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        Eigen::Vector3f bb_min(1e30f, 1e30f, 1e30f), bb_max(-1e30f, -1e30f, -1e30f);
        mesh_a.ForeachVertices([&](int, const Eigen::Vector3f& p) { bb_min = bb_min.cwiseMin(p); bb_max = bb_max.cwiseMax(p); });
        float diag = (bb_max - bb_min).norm();
        if (diag < 1e-6f) diag = 1.0f;

        auto shared = mesh_a.ExtractSharedIntersection(mesh_b);
        std::cout << "[Bool:" << op << "] segments=" << shared.size() << std::endl;
        if (shared.empty())
        {
            std::cout << "[Bool:" << op << "] no intersection; meshes do not overlap" << std::endl;
            return;
        }

        SGL::CanonicalPool pool(diag * 1e-6f);
        std::map<int, std::vector<int>> fpA, fpB;
        std::map<int, std::vector<std::pair<int, int>>> fcA, fcB;

        for (const auto& s : shared)
        {
            int id1 = pool.GetID(s.p1);
            int id2 = pool.GetID(s.p2);
            fpA[s.faceA].push_back(id1); fpA[s.faceA].push_back(id2);
            fpB[s.faceB].push_back(id1); fpB[s.faceB].push_back(id2);
            if (id1 != id2)
            {
                fcA[s.faceA].push_back({ id1, id2 });
                fcB[s.faceB].push_back({ id1, id2 });
            }
        }

        mesh_a.SplitFacesShared(fpA, fcA, pool);
        mesh_b.SplitFacesShared(fpB, fcB, pool);
        mesh_a.BuildSpatialHashMap();
        mesh_b.BuildSpatialHashMap();

        float cut_z = 1.0f;

        // Inside test, by face type:
        //  - A face flat ON the cut plane: SplitFacesShared already ran
        //    eraseOuterTrianglesAndHoles, which removed the glyph-interior
        //    region from the plate top. So any cut-plane face that SURVIVED the
        //    split is, by construction, outside the glyph (plate material). We
        //    classify it as outside directly; point-in-test (nearest-face OR 2D
        //    ray) is unreliable for these thin slivers right on the outline.
        //  - every other face: open-mesh-safe nearest-face test.
        auto classify_faces = [&](HeliumMesh& self, HeliumMesh& other, bool self_is_cut_target) -> std::vector<char>
            {
                std::vector<char> inside(self.n_faces(), 0);
                for (auto f_it = self.faces_begin(); f_it != self.faces_end(); ++f_it)
                {
                    if (self.status(*f_it).deleted()) continue;
                    Eigen::Vector3f v0, v1, v2;
                    self.GetFaceVertices(*f_it, v0, v1, v2);
                    Eigen::Vector3f c = (v0 + v1 + v2) / 3.0f;

                    Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
                    bool flat_on_cut = false;
                    if (n.squaredNorm() > 1e-20f)
                    {
                        n.normalize();
                        flat_on_cut = std::abs(n.z()) > 0.99f && std::abs(c.z() - cut_z) < 1e-3f;
                    }

                    if (flat_on_cut && self_is_cut_target)
                        inside[f_it->idx()] = 0;   // survived erase -> outside glyph
                    else
                        inside[f_it->idx()] = other.IsInsideByNearestFace(c) ? 1 : 0;
                }
                return inside;
            };

        // mesh_a is the plate (cut target whose top lies on the cut plane);
        // mesh_b is the glyph solid.
        auto inFaceA = classify_faces(mesh_a, mesh_b, true);
        auto inFaceB = classify_faces(mesh_b, mesh_a, false);

        bool aIn, bIn, flipB;
        if (std::string(op) == "diff") { aIn = false; bIn = true;  flipB = true; }
        else { aIn = false; bIn = false; flipB = false; }

        std::vector<char> keepA(mesh_a.n_faces(), 0), keepB(mesh_b.n_faces(), 0);
        for (auto f = mesh_a.faces_begin(); f != mesh_a.faces_end(); ++f)
            if (!mesh_a.status(*f).deleted())
                keepA[f->idx()] = ((inFaceA[f->idx()] == 1) == aIn) ? 1 : 0;
        for (auto f = mesh_b.faces_begin(); f != mesh_b.faces_end(); ++f)
            if (!mesh_b.status(*f).deleted())
                keepB[f->idx()] = ((inFaceB[f->idx()] == 1) == bIn) ? 1 : 0;

        std::vector<Eigen::Vector3f> pts;
        std::vector<Eigen::Vector3i> idx;
        mesh_a.CollectFacesByMask(keepA, false, pts, idx);
        mesh_b.CollectFacesByMask(keepB, flipB, pts, idx);

        out.clear();
        out.Build(pts, idx);
        out.WeldVerticesByPosition(diag * 1e-5f);

        int rb = 0;
        for (auto e = out.edges_begin(); e != out.edges_end(); ++e)
            if (out.is_boundary(*e)) rb++;
        std::cout << "[Bool:" << op << "] result V=" << out.n_vertices()
            << " F=" << out.n_faces() << " boundary=" << rb << std::endl;
    }

    // Build an axis-aligned box plate covering [cx-hx, cx+hx] x [cy-hy, cy+hy]
    // in x/y, spanning z0..z1. 12 triangles, closed. Used as an engrave/emboss
    // target when no STL is supplied.
    void BuildPlate(HeliumMesh& m,
        float cx, float cy, float hx, float hy, float z0, float z1)
    {
        std::vector<Eigen::Vector3f> p = {
            {cx - hx, cy - hy, z0}, {cx + hx, cy - hy, z0},
            {cx + hx, cy + hy, z0}, {cx - hx, cy + hy, z0},
            {cx - hx, cy - hy, z1}, {cx + hx, cy - hy, z1},
            {cx + hx, cy + hy, z1}, {cx - hx, cy + hy, z1},
        };
        // outward-facing winding for a closed box
        std::vector<Eigen::Vector3i> f = {
            {0,2,1},{0,3,2},   // bottom (z0), normal -Z
            {4,5,6},{4,6,7},   // top    (z1), normal +Z
            {0,1,5},{0,5,4},   // -Y
            {1,2,6},{1,6,5},   // +X
            {2,3,7},{2,7,6},   // +Y
            {3,0,4},{3,4,7},   // -X
        };
        m.clear();
        m.Build(p, f);
        m.BuildSpatialHashMap();
    }

    void Execute() override
    {
        // ---- 1. load font ----
        TextMeshGenerator gen;
        {
            auto file = File("D:\\Workspace\\Elements\\res\\Fonts\\NanumGothic\\NanumGothic.ttf", true);
            auto data = file.ReadAllBytes();
            if (data.empty() || !gen.LoadFontFromMemory(data.data(), data.size()))
            {
                std::cout << "!!! [Text] font load failed" << std::endl;
                return;
            }
        }

        // ---- 2. build the text solid (input only, NOT rendered) ----
        HeliumMesh text;

        float scale = 0.02f;     // font-unit -> model-unit
        float depth = 2.0f;      // extrusion thickness
        float flatness = 0.05f;  // bezier chord error in model units
        gen.Create3DText(text, u8"ABC 한글 123 漢字", depth, scale, flatness);

        if (text.n_faces() == 0)
        {
            std::cout << "!!! [Text] empty text mesh" << std::endl;
            return;
        }

        // Step 1: text solid input must be a watertight solid, z in [0,depth]
        VerifyInputMesh(text, "Step1 TextMesh", 0.0f, depth);

        // ---- 3. build target (input only, NOT rendered) ----
        float plate_z0 = -3.0f, plate_z1 = 1.0f;
        HeliumMesh target;
        BuildPlate(target, 100.0f, 0.0f, 110.0f, 20.0f, plate_z0, plate_z1);

        // Step 2-1: plate must be a watertight box, z in [z0,z1]
        VerifyInputMesh(target, "Step2-1 BuildPlate", plate_z0, plate_z1);

        // ---- 4. engrave (diff) and emboss (union) ----
        // each needs its own fresh copies because the boolean splits inputs.
        auto fresh = [&](HeliumMesh& dst, HeliumMesh& src)
            {
                std::vector<Eigen::Vector3f> p;
                std::vector<Eigen::Vector3i> idx;
                for (auto f = src.faces_begin(); f != src.faces_end(); ++f)
                {
                    if (src.status(*f).deleted()) continue;
                    Eigen::Vector3f v[3]; int k = 0;
                    for (auto fv = src.cfv_iter(*f); fv.is_valid() && k < 3; ++fv, ++k)
                        v[k] = Eigen::Vector3f(src.point(*fv).data());
                    if (k != 3) continue;
                    int base = (int)p.size();
                    p.push_back(v[0]); p.push_back(v[1]); p.push_back(v[2]);
                    idx.push_back(Eigen::Vector3i(base, base + 1, base + 2));
                }
                dst.clear();
                dst.Build(p, idx);
                {
                    Eigen::Vector3f bmin(1e30f, 1e30f, 1e30f), bmax(-1e30f, -1e30f, -1e30f);
                    for (const auto& q : p) { bmin = bmin.cwiseMin(q); bmax = bmax.cwiseMax(q); }
                    float dd = (bmax - bmin).norm();
                    if (dd < 1e-6f) dd = 1.0f;
                    dst.WeldVerticesByPosition(dd * 1e-5f);
                }
                dst.BuildSpatialHashMap();
            };

        // engrave
        {
            HeliumMesh a, b;
            fresh(a, target); fresh(b, text);

            // Step 2-2: fresh copies must stay watertight (manifold, not soup)
            VerifyInputMesh(a, "Step2-2 fresh A (plate)", plate_z0, plate_z1);
            VerifyInputMesh(b, "Step2-2 fresh B (text)", 0.0f, depth);

            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& engraved = *meshes.back();
            BooleanInto(engraved, a, b, "diff");
            engraved.offset = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            engraved.mesh_color = Eigen::Vector4f(1.0f, 0.6f, 0.2f, 1.0f);
            engraved.SaveSTL("D:\\Temp\\text_engrave.stl");
            engraved.is_dirty = true;
        }

        // emboss
        {
            HeliumMesh a, b;
            fresh(a, target); fresh(b, text);

            meshes.emplace_back(std::make_unique<HeliumMesh>());
            HeliumMesh& embossed = *meshes.back();
            BooleanInto(embossed, a, b, "union");
            embossed.offset = Eigen::Vector3f(0.0f, 50.0f, 0.0f);
            embossed.mesh_color = Eigen::Vector4f(0.4f, 1.0f, 0.4f, 1.0f);
            embossed.SaveSTL("D:\\Temp\\text_emboss.stl");
            embossed.is_dirty = true;
        }

        text.SaveSTL("D:\\Temp\\text_solid.stl");

        Helium.AddOnUpdateCallback([this](float)
            {
                for (auto& m : this->meshes) m->Update();
            });
    }

    std::vector<std::unique_ptr<HeliumMesh>> meshes;
};

REGISTER_APP(AppTextMesh, "AppTextMesh");