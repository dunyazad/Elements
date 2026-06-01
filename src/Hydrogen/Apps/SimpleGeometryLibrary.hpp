#pragma once

#include <vector>
#include <algorithm>
#include <limits>
#include <execution>
#include <numeric>
#include <functional>
#include <mutex>
#include <deque>
#include <Eigen/Dense>
#include <robin_hood/robin_hood.h>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <CDT.h>

inline float DistanceToSegmentSquared(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    Eigen::Vector3f ab = b - a;
    float l2 = ab.squaredNorm();

    if (l2 < 1e-8f)
    {
        return (p - a).squaredNorm();
    }

    float t = (p - a).dot(ab) / l2;
    t = std::max(0.0f, std::min(1.0f, t));
    Eigen::Vector3f projection = a + t * ab;

    return (p - projection).squaredNorm();
}

inline float DistanceToSegment(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    return std::sqrt(DistanceToSegmentSquared(p, a, b));
}

namespace SGL
{
    using OMMesh = OpenMesh::TriMesh_ArrayKernelT<>;

    struct Int3Hash {
        size_t operator()(const Eigen::Vector3i& v) const {
            size_t h1 = v.x() * 73856093;
            size_t h2 = v.y() * 19349663;
            size_t h3 = v.z() * 83492791;
            return h1 ^ h2 ^ h3;
        }
    };

    struct Int3Equal {
        bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
            return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
        }
    };

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

    enum class IntersectionType
    {
        None,
        Vertex,
        Edge,
        Face
    };

    struct IntersectionResult {
        IntersectionType type = IntersectionType::None;
        float t = std::numeric_limits<float>::max();
        OpenMesh::VertexHandle vh;
        OpenMesh::EdgeHandle eh;
        OpenMesh::FaceHandle fh;
        Eigen::Vector3f hit_point;
    };

    class Mesh : public OMMesh
    {
    public:
        Mesh()
        {
            request_vertex_status();
            request_edge_status();
            request_halfedge_status();
            request_face_status();
        }

        struct CutNode
        {
            Eigen::Vector3f pos;
            OpenMesh::FaceHandle fh;
            OpenMesh::EdgeHandle eh;
            OpenMesh::VertexHandle vh;

            CutNode(const Eigen::Vector3f& p) : pos(p) {}
        };

        void Build(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector3i>& indices)
        {
            std::vector<VertexHandle> v_handles;
            v_handles.reserve(points.size());
            for (const auto& p : points)
            {
                v_handles.push_back(add_vertex(OMMesh::Point(p[0], p[1], p[2])));
            }

            for (const auto& idx : indices)
            {
                std::vector<VertexHandle> face_v_handles;
                face_v_handles.push_back(v_handles[idx[0]]);
                face_v_handles.push_back(v_handles[idx[1]]);
                face_v_handles.push_back(v_handles[idx[2]]);
                add_face(face_v_handles);
            }

            BuildSpatialHashMap();
        }

        void BuildSpatialHashMap()
        {
            size_t num_faces = n_faces();
            if (num_faces == 0) return;

            grid_min.setConstant(std::numeric_limits<float>::max());
            grid_max.setConstant(-std::numeric_limits<float>::max());

            struct FaceAABB {
                Eigen::Vector3f min;
                Eigen::Vector3f max;
            };
            std::vector<FaceAABB> face_bounds(num_faces);
            std::vector<int> face_indices(num_faces);
            std::iota(face_indices.begin(), face_indices.end(), 0);

            std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), [&](int i) {
                Eigen::Vector3f v0, v1, v2;
                GetFaceVertices(face_handle(i), v0, v1, v2);

                face_bounds[i].min = v0.cwiseMin(v1).cwiseMin(v2);
                face_bounds[i].max = v0.cwiseMax(v1).cwiseMax(v2);
                });

            for (const auto& bounds : face_bounds)
            {
                grid_min = grid_min.cwiseMin(bounds.min);
                grid_max = grid_max.cwiseMax(bounds.max);
            }

            Eigen::Vector3f extent = grid_max - grid_min;
            grid_min -= extent * 0.01f;
            grid_max += extent * 0.01f;
            extent = grid_max - grid_min;

            float volume = extent.x() * extent.y() * extent.z();
            float ideal_cell_vol = volume / (num_faces * 0.5f);
            float c_size = std::max(0.1f, std::cbrt(ideal_cell_vol));
            grid_cell_size = Eigen::Vector3f(c_size, c_size, c_size);

            hash_map.clear();
            hash_map.reserve(num_faces * 2);

            for (int i = 0; i < num_faces; ++i)
            {
                Eigen::Vector3f local_min = face_bounds[i].min - grid_min;
                Eigen::Vector3f local_max = face_bounds[i].max - grid_min;

                int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
                int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
                int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));

                int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
                int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
                int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

                for (int cz = min_z; cz <= max_z; ++cz) {
                    for (int cy = min_y; cy <= max_y; ++cy) {
                        for (int cx = min_x; cx <= max_x; ++cx) {
                            hash_map[Eigen::Vector3i(cx, cy, cz)].push_back(i);
                        }
                    }
                }
            }
        }

        bool IsOnVertex(const Eigen::Vector3f& pt, OpenMesh::FaceHandle fh, OpenMesh::VertexHandle& out_vh, float eps = 1e-4f) const
        {
            out_vh = OpenMesh::VertexHandle(-1);
            if (!fh.is_valid() || status(fh).deleted()) return false;

            float eps_sq = eps * eps;
            for (auto fv_it = cfv_iter(fh); fv_it.is_valid(); ++fv_it)
            {
                Eigen::Vector3f v(point(*fv_it).data());
                if ((pt - v).squaredNorm() < eps_sq)
                {
                    out_vh = *fv_it;
                    return true;
                }
            }
            return false;
        }

        bool IsOnEdge(const Eigen::Vector3f& pt, OpenMesh::FaceHandle fh, OpenMesh::EdgeHandle& out_eh, float eps = 1e-4f) const
        {
            out_eh = OpenMesh::EdgeHandle(-1);
            if (!fh.is_valid() || status(fh).deleted()) return false;

            float eps_sq = eps * eps;
            float min_dist_sq = eps_sq;

            for (auto fh_it = cfh_iter(fh); fh_it.is_valid(); ++fh_it)
            {
                Eigen::Vector3f a(point(from_vertex_handle(*fh_it)).data());
                Eigen::Vector3f b(point(to_vertex_handle(*fh_it)).data());

                // 선분까지의 정확한 수직 거리 계산
                float dist_sq = DistanceToSegmentSquared(pt, a, b);

                if (dist_sq < min_dist_sq)
                {
                    // 점이 꼭짓점(Vertex) 반경 내부에 있다면 IsOnVertex에서 처리하도록 엣지 판정에서는 제외
                    if ((pt - a).squaredNorm() > eps_sq && (pt - b).squaredNorm() > eps_sq)
                    {
                        min_dist_sq = dist_sq;
                        out_eh = edge_handle(*fh_it);
                    }
                }
            }
            return out_eh.is_valid();
        }

        bool IsOnTriangle(const Eigen::Vector3f& point, OpenMesh::FaceHandle fh, float eps = 1e-4f) const
        {
            if (!fh.is_valid() || status(fh).deleted()) return false;

            Eigen::Vector3f v0, v1, v2;
            GetFaceVertices(fh, v0, v1, v2);

            Eigen::Vector3f e1 = v1 - v0;
            Eigen::Vector3f e2 = v2 - v0;
            Eigen::Vector3f normal = e1.cross(e2);
            float area2 = normal.norm();
            if (area2 < 1e-12f) return false;
            normal /= area2;

            Eigen::Vector3f w = point - v0;
            if (std::abs(normal.dot(w)) > eps) return false;

            float d00 = e1.dot(e1);
            float d01 = e1.dot(e2);
            float d11 = e2.dot(e2);
            float d20 = w.dot(e1);
            float d21 = w.dot(e2);
            float denom = d00 * d11 - d01 * d01;
            if (std::abs(denom) < 1e-12f) return false;

            float v = (d11 * d20 - d01 * d21) / denom;
            float u = (d00 * d21 - d01 * d20) / denom;

            float margin = eps;
            return (u >= -margin) && (v >= -margin) && (u + v <= 1.0f + margin);
        }

        IntersectionResult IntersectRayFaceWithSnap(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction, OpenMesh::FaceHandle fh) const
        {
            IntersectionResult result;
            const float EPS = 1e-4f;

            auto fv_it = cfv_iter(fh);
            auto vh0 = *fv_it++;
            auto vh1 = *fv_it++;
            auto vh2 = *fv_it;

            Eigen::Vector3f v0(point(vh0).data());
            Eigen::Vector3f v1(point(vh1).data());
            Eigen::Vector3f v2(point(vh2).data());

            Eigen::Vector3f edge1 = v1 - v0;
            Eigen::Vector3f edge2 = v2 - v0;
            Eigen::Vector3f pvec = direction.cross(edge2);

            float det = edge1.dot(pvec);
            if (std::abs(det) < 1e-8f) return result;

            float inv_det = 1.0f / det;
            Eigen::Vector3f tvec = origin - v0;

            float u = tvec.dot(pvec) * inv_det;
            if (u < -EPS || u > 1.0f + EPS) return result;

            Eigen::Vector3f qvec = tvec.cross(edge1);
            float v = direction.dot(qvec) * inv_det;
            if (v < -EPS || u + v > 1.0f + EPS) return result;

            float t = edge2.dot(qvec) * inv_det;
            if (t <= 1e-6f) return result;

            float w = 1.0f - u - v;
            result.t = t;
            result.hit_point = origin + direction * t;

            if (w >= 1.0f - EPS) { result.type = IntersectionType::Vertex; result.vh = vh0; result.hit_point = v0; return result; }
            if (u >= 1.0f - EPS) { result.type = IntersectionType::Vertex; result.vh = vh1; result.hit_point = v1; return result; }
            if (v >= 1.0f - EPS) { result.type = IntersectionType::Vertex; result.vh = vh2; result.hit_point = v2; return result; }

            auto fh_it = cfh_iter(fh);
            auto he0 = *fh_it++;
            auto he1 = *fh_it++;
            auto he2 = *fh_it;
            if (w <= EPS) { result.type = IntersectionType::Edge; result.eh = edge_handle(he1); return result; }
            if (u <= EPS) { result.type = IntersectionType::Edge; result.eh = edge_handle(he2); return result; }
            if (v <= EPS) { result.type = IntersectionType::Edge; result.eh = edge_handle(he0); return result; }

            result.type = IntersectionType::Face;
            result.fh = fh;

            return result;
        }

        bool IntersectGridRay(const Eigen::Vector3f& origin, const Eigen::Vector3f& dir, float& out_t, int& out_face) const
        {
            IntersectionResult res;
            if (IntersectGridRay(origin, dir, res))
            {
                out_t = res.t;
                out_face = res.fh.is_valid() ? res.fh.idx() : -1;
                if (out_face == -1 && res.eh.is_valid()) {
                    out_face = face_handle(halfedge_handle(res.eh, 0)).idx();
                }
                return true;
            }
            return false;
        }

        bool IntersectGridRay(const Eigen::Vector3f& origin, const Eigen::Vector3f& dir, IntersectionResult& out_hit) const
        {
            out_hit.t = std::numeric_limits<float>::max();
            out_hit.type = IntersectionType::None;

            if (hash_map.empty()) return false;

            Eigen::Vector3f inv_dir;
            inv_dir.x() = std::abs(dir.x()) < 1e-8f ? (dir.x() < 0.0f ? -1e8f : 1e8f) : 1.0f / dir.x();
            inv_dir.y() = std::abs(dir.y()) < 1e-8f ? (dir.y() < 0.0f ? -1e8f : 1e8f) : 1.0f / dir.y();
            inv_dir.z() = std::abs(dir.z()) < 1e-8f ? (dir.z() < 0.0f ? -1e8f : 1e8f) : 1.0f / dir.z();

            Eigen::Vector3f t0 = (grid_min - origin).cwiseProduct(inv_dir);
            Eigen::Vector3f t1 = (grid_max - origin).cwiseProduct(inv_dir);
            Eigen::Vector3f tmin = t0.cwiseMin(t1);
            Eigen::Vector3f tmax = t0.cwiseMax(t1);

            float t_enter = tmin.maxCoeff();
            float t_exit = tmax.minCoeff();

            if (t_enter > t_exit || t_exit < 0.0f) return false;

            t_enter = std::max(0.0f, t_enter);
            Eigen::Vector3f start_pos = origin + dir * t_enter;
            Eigen::Vector3f local_pos = start_pos - grid_min;

            int cx = static_cast<int>(std::floor(local_pos.x() / grid_cell_size.x()));
            int cy = static_cast<int>(std::floor(local_pos.y() / grid_cell_size.y()));
            int cz = static_cast<int>(std::floor(local_pos.z() / grid_cell_size.z()));

            int stepX = (dir.x() > 0.0f) ? 1 : -1;
            int stepY = (dir.y() > 0.0f) ? 1 : -1;
            int stepZ = (dir.z() > 0.0f) ? 1 : -1;

            float tDeltaX = std::abs(grid_cell_size.x() * inv_dir.x());
            float tDeltaY = std::abs(grid_cell_size.y() * inv_dir.y());
            float tDeltaZ = std::abs(grid_cell_size.z() * inv_dir.z());

            float tMaxX = t_enter + ((stepX > 0) ? ((cx + 1) * grid_cell_size.x() - local_pos.x()) * std::abs(inv_dir.x()) : (local_pos.x() - cx * grid_cell_size.x()) * std::abs(inv_dir.x()));
            float tMaxY = t_enter + ((stepY > 0) ? ((cy + 1) * grid_cell_size.y() - local_pos.y()) * std::abs(inv_dir.y()) : (local_pos.y() - cy * grid_cell_size.y()) * std::abs(inv_dir.y()));
            float tMaxZ = t_enter + ((stepZ > 0) ? ((cz + 1) * grid_cell_size.z() - local_pos.z()) * std::abs(inv_dir.z()) : (local_pos.z() - cz * grid_cell_size.z()) * std::abs(inv_dir.z()));

            bool hit = false;
            float max_t_search = t_exit + 0.1f;
            float current_t = t_enter;

            while (current_t <= max_t_search)
            {
                if (hit && out_hit.t < current_t)
                {
                    break;
                }

                Eigen::Vector3i cell_pos(cx, cy, cz);
                auto it = hash_map.find(cell_pos);
                if (it != hash_map.end())
                {
                    const std::vector<int>& face_list = it->second;
                    for (int f_idx : face_list)
                    {
                        IntersectionResult res = IntersectRayFaceWithSnap(origin, dir, face_handle(f_idx));

                        if (res.type != IntersectionType::None && res.t < out_hit.t)
                        {
                            out_hit = res;
                            out_hit.fh = face_handle(f_idx);
                            hit = true;
                        }
                    }
                }

                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) { cx += stepX; current_t = tMaxX; tMaxX += tDeltaX; }
                    else { cz += stepZ; current_t = tMaxZ; tMaxZ += tDeltaZ; }
                }
                else {
                    if (tMaxY < tMaxZ) { cy += stepY; current_t = tMaxY; tMaxY += tDeltaY; }
                    else { cz += stepZ; current_t = tMaxZ; tMaxZ += tDeltaZ; }
                }
            }

            return hit;
        }

        bool IsConvexQuadrilateral(OpenMesh::EdgeHandle eh) const
        {
            if (is_boundary(eh)) return false;

            auto h0 = halfedge_handle(eh, 0);
            auto h1 = halfedge_handle(eh, 1);

            auto v_top = to_vertex_handle(next_halfedge_handle(h0));
            auto v_left = to_vertex_handle(h1);
            auto v_bottom = to_vertex_handle(next_halfedge_handle(h1));
            auto v_right = to_vertex_handle(h0);

            Eigen::Vector3f p0(point(v_top).data());
            Eigen::Vector3f p1(point(v_left).data());
            Eigen::Vector3f p2(point(v_bottom).data());
            Eigen::Vector3f p3(point(v_right).data());

            Eigen::Vector3f e0 = p1 - p0;
            Eigen::Vector3f e1 = p2 - p1;
            Eigen::Vector3f e2 = p3 - p2;
            Eigen::Vector3f e3 = p0 - p3;

            Eigen::Vector3f n = (e0.cross(e1) + e2.cross(e3)).normalized();

            float c0 = e0.cross(e1).dot(n);
            float c1 = e1.cross(e2).dot(n);
            float c2 = e2.cross(e3).dot(n);
            float c3 = e3.cross(e0).dot(n);

            const float EPSILON = -1e-4f;
            if (c0 < EPSILON || c1 < EPSILON || c2 < EPSILON || c3 < EPSILON)
            {
                return false;
            }

            return true;
        }

        bool IsPointInTriangle(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) const
        {
            Eigen::Vector3f v0 = c - a;
            Eigen::Vector3f v1 = b - a;
            Eigen::Vector3f v2 = p - a;

            float dot00 = v0.dot(v0);
            float dot01 = v0.dot(v1);
            float dot02 = v0.dot(v2);
            float dot11 = v1.dot(v1);
            float dot12 = v1.dot(v2);

            float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

            // 정점이 엣지 위에 있을 때를 대비해 -1e-4f 마진 허용
            return (u >= -1e-4f) && (v >= -1e-4f) && (u + v <= 1.0f + 1e-4f);
        }

        std::map<OpenMesh::FaceHandle, std::vector<Eigen::Vector3f>> MapFacePointIndices(
            const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& raw_segments, float eps = 1e-4f) const
        {
            std::map<OpenMesh::FaceHandle, std::vector<Eigen::Vector3f>> face_point_map;

            auto add_unique = [&](OpenMesh::FaceHandle fh, const Eigen::Vector3f& p)
                {
                    if (!fh.is_valid() || status(fh).deleted()) return;
                    auto& vec = face_point_map[fh];
                    for (const auto& q : vec)
                        if ((q - p).squaredNorm() < eps * eps) return;
                    vec.push_back(p);
                };

            auto assign_point = [&](const Eigen::Vector3f& p)
                {
                    OpenMesh::FaceHandle fh = FindFaceAtPosition(p);
                    if (!fh.is_valid()) return;

                    OpenMesh::VertexHandle vh;
                    OpenMesh::EdgeHandle eh;

                    if (IsOnVertex(p, fh, vh, eps))
                    {
                        for (auto vf_it = cvf_iter(vh); vf_it.is_valid(); ++vf_it)
                            add_unique(*vf_it, p);
                    }
                    else if (IsOnEdge(p, fh, eh, eps))
                    {
                        auto h0 = halfedge_handle(eh, 0);
                        auto h1 = halfedge_handle(eh, 1);
                        add_unique(face_handle(h0), p);
                        add_unique(face_handle(h1), p);
                    }
                    else if (IsOnTriangle(p, fh, eps))
                    {
                        add_unique(fh, p);
                    }
                };

            for (const auto& seg : raw_segments)
            {
                assign_point(seg.first);
                assign_point(seg.second);
            }

            return face_point_map;
        }

        void TriangulatePolygon(std::vector<OpenMesh::VertexHandle>& poly_vertices, const Eigen::Vector3f& normal)
        {
            if (poly_vertices.size() < 3) return;

            // 정점이 3개 남을 때까지 "귀(Ear)"를 찾아서 잘라내며 면(Face)을 생성합니다.
            while (poly_vertices.size() > 3)
            {
                bool ear_found = false;
                for (size_t i = 0; i < poly_vertices.size(); ++i)
                {
                    size_t prev = (i == 0) ? poly_vertices.size() - 1 : i - 1;
                    size_t next = (i == poly_vertices.size() - 1) ? 0 : i + 1;

                    Eigen::Vector3f p0 = Eigen::Vector3f(point(poly_vertices[prev]).data());
                    Eigen::Vector3f p1 = Eigen::Vector3f(point(poly_vertices[i]).data());
                    Eigen::Vector3f p2 = Eigen::Vector3f(point(poly_vertices[next]).data());

                    // 1. 해당 정점이 볼록(Convex)한지 검사 (외적과 법선 벡터의 내적 활용)
                    Eigen::Vector3f cross_prod = (p1 - p0).cross(p2 - p1);
                    if (cross_prod.dot(normal) > 0.0f)
                    {
                        // 2. 만들어질 삼각형 내부에 다른 폴리곤 정점이 들어있는지 검사
                        bool is_ear = true;
                        for (size_t j = 0; j < poly_vertices.size(); ++j)
                        {
                            if (j == prev || j == i || j == next) continue;
                            if (IsPointInTriangle(Eigen::Vector3f(point(poly_vertices[j]).data()), p0, p1, p2))
                            {
                                is_ear = false;
                                break;
                            }
                        }

                        // 3. 진정한 귀(Ear)로 판명되면 삼각형을 추가하고 해당 정점을 폴리곤에서 제거
                        if (is_ear)
                        {
                            add_face(poly_vertices[prev], poly_vertices[i], poly_vertices[next]);
                            poly_vertices.erase(poly_vertices.begin() + i);
                            ear_found = true;
                            break;
                        }
                    }
                }

                // 부동소수점 오차로 꼬여서 귀를 못 찾으면 무한루프 방지를 위해 강제 종료 (Fan Triangulation 폴백)
                if (!ear_found) break;
            }

            // 마지막 남은 3개의 정점으로 최종 삼각형 생성
            if (poly_vertices.size() == 3)
            {
                add_face(poly_vertices[0], poly_vertices[1], poly_vertices[2]);
            }
        }

        void ForeachVertices(std::function<void(int, const Eigen::Vector3f&)> func) const
        {
            for (auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
            {
                auto p = point(*v_it);
                func(v_it->idx(), Eigen::Vector3f(p[0], p[1], p[2]));
            }
        }

        void GetFaceVertices(OpenMesh::FaceHandle f_handle, Eigen::Vector3f& v0, Eigen::Vector3f& v1, Eigen::Vector3f& v2) const
        {
            auto fv_it = cfv_iter(f_handle);
            auto p0 = point(*fv_it++);
            v0 = Eigen::Vector3f(p0[0], p0[1], p0[2]);
            auto p1 = point(*fv_it++);
            v1 = Eigen::Vector3f(p1[0], p1[1], p1[2]);
            auto p2 = point(*fv_it++);
            v2 = Eigen::Vector3f(p2[0], p2[1], p2[2]);
        }

        bool IntersectRayTriangle(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction,
            const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2,
            float& t) const
        {
            Eigen::Vector3f edge1 = v1 - v0;
            Eigen::Vector3f edge2 = v2 - v0;
            Eigen::Vector3f pvec = direction.cross(edge2);

            float det = edge1.dot(pvec);
            if (std::abs(det) < 1e-8f) return false;

            float inv_det = 1.0f / det;
            Eigen::Vector3f tvec = origin - v0;

            float u = tvec.dot(pvec) * inv_det;
            if (u < -1e-5f || u > 1.0f + 1e-5f) return false;

            Eigen::Vector3f qvec = tvec.cross(edge1);
            float v = direction.dot(qvec) * inv_det;
            if (v < -1e-5f || u + v > 1.0f + 1e-5f) return false;

            t = edge2.dot(qvec) * inv_det;
            return t > 1e-6f;
        }

        bool IntersectTriangleTriangle(
            const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2,
            const Eigen::Vector3f& u0, const Eigen::Vector3f& u1, const Eigen::Vector3f& u2,
            Eigen::Vector3f& out_p1, Eigen::Vector3f& out_p2) const
        {
            std::vector<Eigen::Vector3f> hit_points;

            auto intersect_segment_tri = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b,
                const Eigen::Vector3f& t0, const Eigen::Vector3f& t1, const Eigen::Vector3f& t2) {
                    Eigen::Vector3f dir = b - a;
                    float max_t = dir.norm();
                    if (max_t < 1e-8f) return;
                    dir /= max_t;

                    float t;
                    if (IntersectRayTriangle(a, dir, t0, t1, t2, t)) {
                        if (t >= -1e-5f && t <= max_t + 1e-5f) {
                            hit_points.push_back(a + dir * t);
                        }
                    }
                };

            intersect_segment_tri(v0, v1, u0, u1, u2);
            intersect_segment_tri(v1, v2, u0, u1, u2);
            intersect_segment_tri(v2, v0, u0, u1, u2);

            intersect_segment_tri(u0, u1, v0, v1, v2);
            intersect_segment_tri(u1, u2, v0, v1, v2);
            intersect_segment_tri(u2, u0, v0, v1, v2);

            if (hit_points.size() < 2) return false;

            Eigen::Vector3f p1 = hit_points[0];
            Eigen::Vector3f p2 = p1;
            float max_dist_sq = 0.0f;

            for (size_t i = 0; i < hit_points.size(); ++i) {
                for (size_t j = i + 1; j < hit_points.size(); ++j) {
                    float dist_sq = (hit_points[i] - hit_points[j]).squaredNorm();
                    if (dist_sq > max_dist_sq) {
                        max_dist_sq = dist_sq;
                        p1 = hit_points[i];
                        p2 = hit_points[j];
                    }
                }
            }

            if (max_dist_sq < 1e-8f) return false;

            out_p1 = p1;
            out_p2 = p2;
            return true;
        }

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> ExtractIntersectionSegments(const Mesh& other_mesh, std::vector<char>* out_cut_faces = nullptr) const
        {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> segments;
            std::mutex mtx;

            size_t other_num_faces = other_mesh.n_faces();
            if (other_num_faces == 0 || hash_map.empty()) return segments;

            if (out_cut_faces) {
                out_cut_faces->assign(n_faces(), 0);
            }

            std::vector<int> other_face_indices(other_num_faces);
            std::iota(other_face_indices.begin(), other_face_indices.end(), 0);

            std::for_each(std::execution::par_unseq, other_face_indices.begin(), other_face_indices.end(), [&](int i)
                {
                    Eigen::Vector3f u0, u1, u2;
                    other_mesh.GetFaceVertices(other_mesh.face_handle(i), u0, u1, u2);

                    Eigen::Vector3f b_min = u0.cwiseMin(u1).cwiseMin(u2);
                    Eigen::Vector3f b_max = u0.cwiseMax(u1).cwiseMax(u2);

                    Eigen::Vector3f local_min = b_min - grid_min;
                    Eigen::Vector3f local_max = b_max - grid_min;

                    int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
                    int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
                    int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));

                    int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
                    int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
                    int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

                    std::vector<int> candidate_faces;
                    for (int cz = min_z; cz <= max_z; ++cz) {
                        for (int cy = min_y; cy <= max_y; ++cy) {
                            for (int cx = min_x; cx <= max_x; ++cx) {
                                auto it = hash_map.find(Eigen::Vector3i(cx, cy, cz));
                                if (it != hash_map.end()) {
                                    candidate_faces.insert(candidate_faces.end(), it->second.begin(), it->second.end());
                                }
                            }
                        }
                    }

                    if (candidate_faces.empty()) return;

                    std::sort(candidate_faces.begin(), candidate_faces.end());
                    candidate_faces.erase(std::unique(candidate_faces.begin(), candidate_faces.end()), candidate_faces.end());

                    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> local_segments;
                    std::vector<int> local_cut_faces;

                    for (int f_idx : candidate_faces)
                    {
                        Eigen::Vector3f v0, v1, v2;
                        GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                        Eigen::Vector3f p1, p2;
                        if (IntersectTriangleTriangle(v0, v1, v2, u0, u1, u2, p1, p2))
                        {
                            local_segments.push_back({ p1, p2 });
                            if (out_cut_faces) local_cut_faces.push_back(f_idx);
                        }
                    }

                    if (!local_segments.empty())
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        segments.insert(segments.end(), local_segments.begin(), local_segments.end());

                        if (out_cut_faces) {
                            for (int f_idx : local_cut_faces) {
                                (*out_cut_faces)[f_idx] = 1;
                            }
                        }
                    }
                });

            return segments;
        }

        std::map<OpenMesh::FaceHandle, std::vector<Eigen::Vector3f>> BuildFacePointsFromSegments(
            const std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, int>>& seg_with_face, float eps = 1e-4f) const
        {
            std::map<OpenMesh::FaceHandle, std::vector<Eigen::Vector3f>> face_point_map;

            auto add_unique = [&](OpenMesh::FaceHandle fh, const Eigen::Vector3f& p)
                {
                    if (!fh.is_valid() || status(fh).deleted()) return;
                    auto& vec = face_point_map[fh];
                    for (const auto& q : vec)
                        if ((q - p).squaredNorm() < eps * eps) return;
                    vec.push_back(p);
                };

            auto add_point_to_face = [&](OpenMesh::FaceHandle fh, const Eigen::Vector3f& p)
                {
                    if (!fh.is_valid() || status(fh).deleted()) return;

                    OpenMesh::VertexHandle vh;
                    OpenMesh::EdgeHandle eh;

                    if (IsOnVertex(p, fh, vh, eps))
                    {
                        for (auto vf_it = cvf_iter(vh); vf_it.is_valid(); ++vf_it)
                            add_unique(*vf_it, p);
                    }
                    else if (IsOnEdge(p, fh, eh, eps))
                    {
                        add_unique(face_handle(halfedge_handle(eh, 0)), p);
                        add_unique(face_handle(halfedge_handle(eh, 1)), p);
                    }
                    else
                    {
                        add_unique(fh, p);
                    }
                };

            for (const auto& s : seg_with_face)
            {
                OpenMesh::FaceHandle fh = face_handle(std::get<2>(s));
                add_point_to_face(fh, std::get<0>(s));
                add_point_to_face(fh, std::get<1>(s));
            }

            return face_point_map;
        }

        std::vector<Eigen::Vector3f> SplitSegmentWithMeshEdges(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float snap_eps = 1e-4f) const
        {
            std::vector<Eigen::Vector3f> internal_points;

            if (hash_map.empty())
            {
                internal_points.push_back(p1);
                internal_points.push_back(p2);
                return internal_points;
            }

            Eigen::Vector3f u = p2 - p1;
            float u_len_sq = u.squaredNorm();
            if (u_len_sq < 1e-8f)
            {
                internal_points.push_back(p1);
                internal_points.push_back(p2);
                return internal_points;
            }

            Eigen::Vector3f b_min = p1.cwiseMin(p2) - Eigen::Vector3f::Constant(grid_cell_size.maxCoeff());
            Eigen::Vector3f b_max = p1.cwiseMax(p2) + Eigen::Vector3f::Constant(grid_cell_size.maxCoeff());

            Eigen::Vector3f local_min = b_min - grid_min;
            Eigen::Vector3f local_max = b_max - grid_min;

            int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
            int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
            int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));

            int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
            int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
            int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

            std::vector<OpenMesh::EdgeHandle> candidate_edges;
            std::vector<bool> visited_edges(n_edges(), false);

            for (int cz = min_z; cz <= max_z; ++cz)
            {
                for (int cy = min_y; cy <= max_y; ++cy)
                {
                    for (int cx = min_x; cx <= max_x; ++cx)
                    {
                        auto it = hash_map.find(Eigen::Vector3i(cx, cy, cz));
                        if (it != hash_map.end())
                        {
                            for (int f_idx : it->second)
                            {
                                auto fh = face_handle(f_idx);
                                if (status(fh).deleted()) continue;

                                for (auto fe_it = cfe_iter(fh); fe_it.is_valid(); ++fe_it)
                                {
                                    if (!visited_edges[fe_it->idx()])
                                    {
                                        visited_edges[fe_it->idx()] = true;
                                        candidate_edges.push_back(*fe_it);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            float param_eps = snap_eps;
            float dist_eps_sq = snap_eps * snap_eps;

            for (auto eh : candidate_edges)
            {
                auto h0 = halfedge_handle(eh, 0);
                Eigen::Vector3f v0(point(from_vertex_handle(h0)).data());
                Eigen::Vector3f v1(point(to_vertex_handle(h0)).data());

                Eigen::Vector3f v = v1 - v0;
                Eigen::Vector3f w = p1 - v0;

                float a = u_len_sq;
                float b = u.dot(v);
                float c = v.squaredNorm();
                float d = u.dot(w);
                float e = v.dot(w);

                float D = a * c - b * b;
                float sc, tc;

                if (D < 1e-8f)
                {
                    continue;
                }
                else
                {
                    sc = (b * e - c * d) / D;
                    tc = (a * e - b * d) / D;
                }

                if (sc > param_eps && sc < 1.0f - param_eps && tc > param_eps && tc < 1.0f - param_eps)
                {
                    Eigen::Vector3f dP = w + (sc * u) - (tc * v);
                    if (dP.squaredNorm() < dist_eps_sq)
                    {
                        internal_points.push_back(p1 + sc * u);
                    }
                }
            }

            internal_points.push_back(p1);
            internal_points.push_back(p2);

            std::sort(internal_points.begin(), internal_points.end(), [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b)
                {
                    return (a - p1).squaredNorm() < (b - p1).squaredNorm();
                });

            auto erase_it = std::unique(internal_points.begin(), internal_points.end(), [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b)
                {
                    return (a - b).squaredNorm() < dist_eps_sq;
                });
            internal_points.erase(erase_it, internal_points.end());

            return internal_points;
        }

        void SplitFaces(
            std::map<OpenMesh::FaceHandle, std::vector<Eigen::Vector3f>>& facePointsMapping,
            const std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, int>>& original_segments)
        {
            std::map<int, std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>> face_constraints;
            for (const auto& s : original_segments)
            {
                face_constraints[std::get<2>(s)].push_back({ std::get<0>(s), std::get<1>(s) });
            }

            // raw polygon soup: no per-call dedup here, topology rebuilt by weld later
            std::vector<Eigen::Vector3f> all_new_points;
            std::vector<Eigen::Vector3i> all_new_indices;

            // scale-relative epsilons based on the model bounding box
            float diag = (grid_max - grid_min).norm();
            if (diag < 1e-6f) diag = 1.0f;
            const float snap_eps = diag * 1e-5f;
            const float merge_eps = diag * 1e-6f;
            const float merge_eps_sq = merge_eps * merge_eps;

            auto push_point = [&](const Eigen::Vector3f& p) -> int
                {
                    int idx = static_cast<int>(all_new_points.size());
                    all_new_points.push_back(p);
                    return idx;
                };

            int total_dropped_constraints = 0;

            for (auto& mapping : facePointsMapping)
            {
                auto& fh = mapping.first;
                if (!fh.is_valid() || status(fh).deleted()) continue;

                auto& points = mapping.second;

                auto fv_it = cfv_iter(fh);
                OpenMesh::VertexHandle vh0 = *fv_it++;
                OpenMesh::VertexHandle vh1 = *fv_it++;
                OpenMesh::VertexHandle vh2 = *fv_it;

                Eigen::Vector3f v0(point(vh0).data());
                Eigen::Vector3f v1(point(vh1).data());
                Eigen::Vector3f v2(point(vh2).data());

                Eigen::Vector3f e_normal = (v1 - v0).cross(v2 - v0);
                if (e_normal.squaredNorm() < 1e-12f) continue; // degenerate
                e_normal.normalize();

                Eigen::Vector3f axis_u = (v1 - v0).normalized();
                Eigen::Vector3f axis_v = e_normal.cross(axis_u).normalized();
                if (axis_u.cross(axis_v).dot(e_normal) < 0.0f) axis_v = -axis_v;

                Eigen::Vector3f origin = v0;

                auto to_uv = [&](const Eigen::Vector3f& p) -> Eigen::Vector2f
                    {
                        Eigen::Vector3f d = p - origin;
                        return Eigen::Vector2f(d.dot(axis_u), d.dot(axis_v));
                    };

                Eigen::Vector2f uv0 = to_uv(v0);
                Eigen::Vector2f uv1 = to_uv(v1);
                Eigen::Vector2f uv2 = to_uv(v2);

                // map a planar UV point back to 3D via barycentric on the original triangle
                auto uv_to_3d = [&](float ux, float uy) -> Eigen::Vector3f
                    {
                        Eigen::Vector2f p(ux, uy);
                        Eigen::Vector2f a0 = uv1 - uv0;
                        Eigen::Vector2f a1 = uv2 - uv0;
                        Eigen::Vector2f a2 = p - uv0;
                        float d00 = a0.dot(a0);
                        float d01 = a0.dot(a1);
                        float d11 = a1.dot(a1);
                        float d20 = a2.dot(a0);
                        float d21 = a2.dot(a1);
                        float den = d00 * d11 - d01 * d01;
                        if (std::abs(den) < 1e-20f) return v0;
                        float vb = (d11 * d20 - d01 * d21) / den;
                        float wb = (d00 * d21 - d01 * d20) / den;
                        float ub = 1.0f - vb - wb;
                        return ub * v0 + vb * v1 + wb * v2;
                    };

                std::vector<CDT::V2d<float>> cdt_points;

                auto add_point = [&](const Eigen::Vector3f& p) -> CDT::VertInd
                    {
                        Eigen::Vector2f uv = to_uv(p);
                        for (size_t i = 0; i < cdt_points.size(); ++i)
                        {
                            float dx = cdt_points[i].x - uv.x();
                            float dy = cdt_points[i].y - uv.y();
                            if (dx * dx + dy * dy < merge_eps_sq) return static_cast<CDT::VertInd>(i);
                        }
                        CDT::V2d<float> pt; pt.x = uv.x(); pt.y = uv.y();
                        cdt_points.push_back(pt);
                        return static_cast<CDT::VertInd>(cdt_points.size() - 1);
                    };

                CDT::VertInd vidx0 = add_point(v0);
                CDT::VertInd vidx1 = add_point(v1);
                CDT::VertInd vidx2 = add_point(v2);

                for (const auto& p : points) add_point(p);

                // boundary edges: chain collinear points along each triangle side
                std::vector<std::pair<CDT::VertInd, CDT::VertInd>> boundary_edges;
                std::vector<std::pair<float, CDT::VertInd>> b0, b1, b2;

                b0.push_back({ 0.0f, vidx0 }); b0.push_back({ (v1 - v0).squaredNorm(), vidx1 });
                b1.push_back({ 0.0f, vidx1 }); b1.push_back({ (v2 - v1).squaredNorm(), vidx2 });
                b2.push_back({ 0.0f, vidx2 }); b2.push_back({ (v0 - v2).squaredNorm(), vidx0 });

                for (size_t i = 3; i < cdt_points.size(); ++i)
                {
                    Eigen::Vector3f p = uv_to_3d(cdt_points[i].x, cdt_points[i].y);
                    float d0 = DistanceToSegmentSquared(p, v0, v1);
                    float d1 = DistanceToSegmentSquared(p, v1, v2);
                    float d2 = DistanceToSegmentSquared(p, v2, v0);

                    if (d0 < merge_eps_sq && d0 <= d1 && d0 <= d2) b0.push_back({ (p - v0).dot(v1 - v0), static_cast<CDT::VertInd>(i) });
                    else if (d1 < merge_eps_sq && d1 <= d0 && d1 <= d2) b1.push_back({ (p - v1).dot(v2 - v1), static_cast<CDT::VertInd>(i) });
                    else if (d2 < merge_eps_sq) b2.push_back({ (p - v2).dot(v0 - v2), static_cast<CDT::VertInd>(i) });
                }

                auto add_boundary = [&](std::vector<std::pair<float, CDT::VertInd>>& b)
                    {
                        std::sort(b.begin(), b.end());
                        b.erase(std::unique(b.begin(), b.end(),
                            [](const auto& lhs, const auto& rhs) { return lhs.second == rhs.second; }), b.end());
                        for (size_t k = 0; k + 1 < b.size(); ++k) boundary_edges.push_back({ b[k].second, b[k + 1].second });
                    };
                add_boundary(b0); add_boundary(b1); add_boundary(b2);

                // internal constraints from intersection segments
                std::vector<std::pair<CDT::VertInd, CDT::VertInd>> internal_edges;
                auto fc_it = face_constraints.find(fh.idx());
                if (fc_it != face_constraints.end())
                {
                    int dropped = 0;
                    for (const auto& seg : fc_it->second)
                    {
                        CDT::VertInd idxA = add_point(seg.first);
                        CDT::VertInd idxB = add_point(seg.second);
                        if (idxA == idxB) { dropped++; continue; }
                        internal_edges.push_back({ idxA, idxB });
                    }
                    if (dropped > 0)
                    {
                        total_dropped_constraints += dropped;
                        std::cout << "[SplitFaces] Face " << fh.idx() << " dropped "
                            << dropped << " degenerate constraints" << std::endl;
                    }
                }

                std::vector<CDT::Edge> all_edges;
                for (const auto& e : boundary_edges) all_edges.push_back({ e.first, e.second });
                for (const auto& e : internal_edges) all_edges.push_back({ e.first, e.second });

                auto ccw = [](const Eigen::Vector2f& p1, const Eigen::Vector2f& p2, const Eigen::Vector2f& p3) -> float
                    {
                        return (p2.x() - p1.x()) * (p3.y() - p1.y()) - (p2.y() - p1.y()) * (p3.x() - p1.x());
                    };
                bool is_base_ccw = ccw(uv0, uv1, uv2) > 0.0f;

                try
                {
                    // let CDT resolve crossing constraints, then keep only the
                    // constrained interior region (removes outer fan slivers)
                    CDT::Triangulation<float> cdt(
                        CDT::VertexInsertionOrder::Auto,
                        CDT::IntersectingConstraintEdges::TryResolve,
                        0.0f);

                    cdt.insertVertices(cdt_points);
                    if (!all_edges.empty()) cdt.insertEdges(all_edges);
                    cdt.eraseOuterTrianglesAndHoles();

                    delete_face(fh, false);

                    for (const auto& tri : cdt.triangles)
                    {
                        const auto& cv0 = cdt.vertices[tri.vertices[0]];
                        const auto& cv1 = cdt.vertices[tri.vertices[1]];
                        const auto& cv2 = cdt.vertices[tri.vertices[2]];

                        // reconstruct 3D directly from UV (no pos3d index dependency)
                        Eigen::Vector3f pa = uv_to_3d(cv0.x, cv0.y);
                        Eigen::Vector3f pb = uv_to_3d(cv1.x, cv1.y);
                        Eigen::Vector3f pc = uv_to_3d(cv2.x, cv2.y);

                        Eigen::Vector3i f_idx;
                        f_idx[0] = push_point(pa);
                        f_idx[1] = push_point(pb);
                        f_idx[2] = push_point(pc);

                        if (f_idx[0] == f_idx[1] || f_idx[1] == f_idx[2] || f_idx[2] == f_idx[0]) continue;

                        // align winding to the original face orientation
                        Eigen::Vector2f a(cv0.x, cv0.y);
                        Eigen::Vector2f b(cv1.x, cv1.y);
                        Eigen::Vector2f c(cv2.x, cv2.y);
                        bool cdt_ccw = ccw(a, b, c) > 0.0f;
                        if (cdt_ccw != is_base_ccw) std::swap(f_idx[1], f_idx[2]);

                        all_new_indices.push_back(f_idx);
                    }
                }
                catch (const std::exception& e)
                {
                    std::cout << "[ERROR] CDT Face " << fh.idx() << ": " << e.what() << std::endl;
                }
            }

            if (total_dropped_constraints > 0)
            {
                std::cout << "[SplitFaces] WARNING: total dropped constraints = "
                    << total_dropped_constraints << " (merge_eps may be too large)" << std::endl;
            }

            // optional: dump raw soup before weld to isolate CDT vs weld issues
            {
                Mesh soup;
                soup.Build(all_new_points, all_new_indices);
                soup.SaveSTL("D:\\Temp\\soup_preweld.stl");
            }

            // rebuild as raw soup, then weld with robust 3x3x3 neighbor search
            this->clear();
            this->Build(all_new_points, all_new_indices);
            this->WeldVerticesByPosition(snap_eps * 10.0f);

            std::cout << "[SplitFaces] Mesh rebuilt. Vertices: "
                << n_vertices() << ", Faces: " << n_faces() << std::endl;
        }

        void WeldVerticesByPosition(float snap_eps)
        {
            struct GridHash {
                size_t operator()(const Eigen::Vector3i& v) const {
                    size_t h1 = std::hash<int>{}(v.x());
                    size_t h2 = std::hash<int>{}(v.y());
                    size_t h3 = std::hash<int>{}(v.z());
                    return h1 ^ (h2 << 1) ^ (h3 << 2);
                }
            };
            struct GridEqual {
                bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
                    return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
                }
            };

            float cell = snap_eps;
            float eps_sq = cell * cell;

            robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, GridHash, GridEqual> grid;

            std::vector<Eigen::Vector3f> rep_points;

            auto cell_of = [&](const Eigen::Vector3f& p) -> Eigen::Vector3i {
                return Eigen::Vector3i(
                    static_cast<int>(std::floor(p.x() / cell)),
                    static_cast<int>(std::floor(p.y() / cell)),
                    static_cast<int>(std::floor(p.z() / cell)));
                };

            auto get_rep = [&](const Eigen::Vector3f& p) -> int {
                Eigen::Vector3i c = cell_of(p);
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            Eigen::Vector3i nc(c.x() + dx, c.y() + dy, c.z() + dz);
                            auto it = grid.find(nc);
                            if (it == grid.end()) continue;
                            for (int idx : it->second)
                                if ((rep_points[idx] - p).squaredNorm() < eps_sq)
                                    return idx;
                        }
                int new_idx = static_cast<int>(rep_points.size());
                rep_points.push_back(p);
                grid[c].push_back(new_idx);
                return new_idx;
                };

            std::vector<Eigen::Vector3i> new_indices;

            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                if (status(*f_it).deleted()) continue;

                std::vector<Eigen::Vector3f> pts;
                for (auto fv_it = cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
                    pts.push_back(Eigen::Vector3f(point(*fv_it).data()));

                if (pts.size() != 3) continue;

                Eigen::Vector3i fidx;
                for (int k = 0; k < 3; ++k)
                    fidx[k] = get_rep(pts[k]);

                if (fidx[0] == fidx[1] || fidx[1] == fidx[2] || fidx[2] == fidx[0])
                    continue;

                new_indices.push_back(fidx);
            }

            this->clear();
            this->Build(rep_points, new_indices);
        }

        bool SaveSTL(const std::string& filepath) const
        {
            std::ofstream ofs(filepath, std::ios::binary);
            if (!ofs.is_open()) return false;

            char header[80] = { 0 };
            ofs.write(header, 80);

            unsigned int tri_count = 0;
            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
                if (!status(*f_it).deleted()) tri_count++;
            ofs.write(reinterpret_cast<const char*>(&tri_count), sizeof(unsigned int));

            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                if (status(*f_it).deleted()) continue;

                Eigen::Vector3f v0, v1, v2;
                GetFaceVertices(*f_it, v0, v1, v2);

                Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
                if (n.squaredNorm() > 1e-20f) n.normalize();
                else n = Eigen::Vector3f::Zero();

                float buf[3];
                buf[0] = n.x(); buf[1] = n.y(); buf[2] = n.z();
                ofs.write(reinterpret_cast<const char*>(buf), sizeof(float) * 3);

                buf[0] = v0.x(); buf[1] = v0.y(); buf[2] = v0.z();
                ofs.write(reinterpret_cast<const char*>(buf), sizeof(float) * 3);

                buf[0] = v1.x(); buf[1] = v1.y(); buf[2] = v1.z();
                ofs.write(reinterpret_cast<const char*>(buf), sizeof(float) * 3);

                buf[0] = v2.x(); buf[1] = v2.y(); buf[2] = v2.z();
                ofs.write(reinterpret_cast<const char*>(buf), sizeof(float) * 3);

                unsigned short attr = 0;
                ofs.write(reinterpret_cast<const char*>(&attr), sizeof(unsigned short));
            }

            ofs.close();
            return true;
        }









        


        


        std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, int>> ExtractIntersectionSegmentsWithFace(const Mesh& other_mesh) const
        {
            std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, int>> segments;
            std::mutex mtx;

            size_t other_num_faces = other_mesh.n_faces();
            if (other_num_faces == 0 || hash_map.empty()) return segments;

            std::vector<int> other_face_indices(other_num_faces);
            std::iota(other_face_indices.begin(), other_face_indices.end(), 0);

            std::for_each(std::execution::par_unseq, other_face_indices.begin(), other_face_indices.end(), [&](int i)
                {
                    Eigen::Vector3f u0, u1, u2;
                    other_mesh.GetFaceVertices(other_mesh.face_handle(i), u0, u1, u2);

                    Eigen::Vector3f b_min = u0.cwiseMin(u1).cwiseMin(u2);
                    Eigen::Vector3f b_max = u0.cwiseMax(u1).cwiseMax(u2);
                    Eigen::Vector3f local_min = b_min - grid_min;
                    Eigen::Vector3f local_max = b_max - grid_min;

                    int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
                    int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
                    int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));
                    int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
                    int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
                    int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

                    std::vector<int> candidate_faces;
                    for (int cz = min_z; cz <= max_z; ++cz)
                        for (int cy = min_y; cy <= max_y; ++cy)
                            for (int cx = min_x; cx <= max_x; ++cx)
                            {
                                auto it = hash_map.find(Eigen::Vector3i(cx, cy, cz));
                                if (it != hash_map.end())
                                    candidate_faces.insert(candidate_faces.end(), it->second.begin(), it->second.end());
                            }

                    if (candidate_faces.empty()) return;
                    std::sort(candidate_faces.begin(), candidate_faces.end());
                    candidate_faces.erase(std::unique(candidate_faces.begin(), candidate_faces.end()), candidate_faces.end());

                    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, int>> local_segments;
                    for (int f_idx : candidate_faces)
                    {
                        Eigen::Vector3f v0, v1, v2;
                        GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                        Eigen::Vector3f p1, p2;
                        if (IntersectTriangleTriangle(v0, v1, v2, u0, u1, u2, p1, p2))
                        {
                            local_segments.push_back(std::make_tuple(p1, p2, f_idx));
                        }
                    }

                    if (!local_segments.empty())
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        segments.insert(segments.end(), local_segments.begin(), local_segments.end());
                    }
                });

            return segments;
        }

        std::vector<std::vector<Eigen::Vector3f>> LinkSegments(
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> segments,
            float link_tolerance = 1e-4f) const
        {
            std::vector<std::vector<Eigen::Vector3f>> polylines;
            float tol_sq = link_tolerance * link_tolerance;

            while (!segments.empty())
            {
                std::deque<Eigen::Vector3f> current_line;

                current_line.push_back(segments.back().first);
                current_line.push_back(segments.back().second);
                segments.pop_back();

                bool extended = true;
                while (extended)
                {
                    extended = false;
                    for (auto it = segments.begin(); it != segments.end(); ++it)
                    {
                        Eigen::Vector3f head = current_line.front();
                        Eigen::Vector3f tail = current_line.back();

                        if ((it->first - tail).squaredNorm() < tol_sq) {
                            current_line.push_back(it->second);
                            segments.erase(it); extended = true; break;
                        }
                        else if ((it->second - tail).squaredNorm() < tol_sq) {
                            current_line.push_back(it->first);
                            segments.erase(it); extended = true; break;
                        }
                        else if ((it->first - head).squaredNorm() < tol_sq) {
                            current_line.push_front(it->second);
                            segments.erase(it); extended = true; break;
                        }
                        else if ((it->second - head).squaredNorm() < tol_sq) {
                            current_line.push_front(it->first);
                            segments.erase(it); extended = true; break;
                        }
                    }
                }

                polylines.emplace_back(current_line.begin(), current_line.end());
            }

            return polylines;
        }

        void DeleteFacesAlongPolylines(const std::vector<std::vector<Eigen::Vector3f>>& polylines)
        {
            float cut_tol = 1.0e-3f; // 1mm 참호 두께
            float cut_tol_sq = cut_tol * cut_tol;

            std::vector<bool> faces_to_delete(n_faces(), false);

            for (const auto& ring : polylines)
            {
                for (size_t i = 0; i < ring.size() - 1; ++i)
                {
                    Eigen::Vector3f p0 = ring[i];
                    Eigen::Vector3f p1 = ring[i + 1];

                    Eigen::Vector3f min_pt = p0.cwiseMin(p1) - Eigen::Vector3f::Constant(cut_tol);
                    Eigen::Vector3f max_pt = p0.cwiseMax(p1) + Eigen::Vector3f::Constant(cut_tol);

                    Eigen::Vector3f local_min = min_pt - grid_min;
                    Eigen::Vector3f local_max = max_pt - grid_min;

                    int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
                    int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
                    int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));

                    int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
                    int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
                    int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

                    for (int cz = min_z; cz <= max_z; ++cz) {
                        for (int cy = min_y; cy <= max_y; ++cy) {
                            for (int cx = min_x; cx <= max_x; ++cx) {
                                auto it = hash_map.find(Eigen::Vector3i(cx, cy, cz));
                                if (it != hash_map.end()) {
                                    for (int f_idx : it->second) {
                                        if (faces_to_delete[f_idx]) continue;

                                        auto fh = face_handle(f_idx);
                                        if (status(fh).deleted()) continue;

                                        Eigen::Vector3f v0, v1, v2;
                                        GetFaceVertices(fh, v0, v1, v2);

                                        Eigen::Vector3f c = (v0 + v1 + v2) / 3.0f;
                                        Eigen::Vector3f m0 = (v0 + v1) * 0.5f;
                                        Eigen::Vector3f m1 = (v1 + v2) * 0.5f;
                                        Eigen::Vector3f m2 = (v2 + v0) * 0.5f;

                                        if (DistanceToSegmentSquared(c, p0, p1) < cut_tol_sq ||
                                            DistanceToSegmentSquared(v0, p0, p1) < cut_tol_sq ||
                                            DistanceToSegmentSquared(v1, p0, p1) < cut_tol_sq ||
                                            DistanceToSegmentSquared(v2, p0, p1) < cut_tol_sq ||
                                            DistanceToSegmentSquared(m0, p0, p1) < cut_tol_sq ||
                                            DistanceToSegmentSquared(m1, p0, p1) < cut_tol_sq ||
                                            DistanceToSegmentSquared(m2, p0, p1) < cut_tol_sq)
                                        {
                                            faces_to_delete[f_idx] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 마킹된 면들 삭제
            bool any_deleted = false;
            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                if (faces_to_delete[f_it->idx()] && !status(*f_it).deleted())
                {
                    delete_face(*f_it, false);
                    any_deleted = true;
                }
            }

            if (any_deleted) {
                garbage_collection();
                BuildSpatialHashMap();
            }
        }
        
        std::vector<std::unique_ptr<Mesh>> SeparateDisconnectedMeshes()
        {
            std::vector<int> face_partition(n_faces(), -1);
            int current_partition = 0;

            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                // 이미 참호로 지워진 면이거나 방문한 면이면 패스
                if (status(*f_it).deleted() || face_partition[f_it->idx()] != -1) continue;

                std::vector<OpenMesh::FaceHandle> queue;
                queue.push_back(*f_it);
                face_partition[f_it->idx()] = current_partition;

                size_t head = 0;
                while (head < queue.size())
                {
                    auto current_fh = queue[head++];

                    // 지워지지 않고 남아있는 "진짜 면"들끼리만 다리를 타고 넘어갑니다.
                    // 위상이 끊겨 있으므로 다른 파트로 넘어갈 확률은 0%입니다.
                    for (auto ff_it = cff_iter(current_fh); ff_it.is_valid(); ++ff_it)
                    {
                        if (!status(*ff_it).deleted() && face_partition[ff_it->idx()] == -1)
                        {
                            face_partition[ff_it->idx()] = current_partition;
                            queue.push_back(*ff_it);
                        }
                    }
                }
                current_partition++;
            }

            // 분리된 파티션 번호에 따라 새로운 Mesh 파편들을 생성
            std::vector<std::unique_ptr<Mesh>> result_meshes;
            for (int p = 0; p < current_partition; ++p)
            {
                auto new_mesh = std::make_unique<Mesh>();
                new_mesh->request_vertex_status();
                new_mesh->request_edge_status();
                new_mesh->request_halfedge_status();
                new_mesh->request_face_status();

                std::vector<OpenMesh::VertexHandle> vmap(n_vertices(), OpenMesh::VertexHandle(-1));
                int added_faces = 0;

                for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
                {
                    if (status(*f_it).deleted() || face_partition[f_it->idx()] != p) continue;

                    std::vector<OpenMesh::VertexHandle> face_vhandles;
                    for (auto fv_it = cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
                    {
                        int old_idx = fv_it->idx();
                        if (!vmap[old_idx].is_valid())
                        {
                            Eigen::Vector3f pos(point(*fv_it).data());
                            vmap[old_idx] = new_mesh->add_vertex(OMMesh::Point(pos.x(), pos.y(), pos.z()));
                        }
                        face_vhandles.push_back(vmap[old_idx]);
                    }
                    new_mesh->add_face(face_vhandles);
                    added_faces++;
                }

                // 면이 1개 이하인(삼각형 하나짜리) 찌꺼기 메쉬는 반환 리스트에서 무시합니다.
                if (added_faces > 1)
                {
                    new_mesh->BuildSpatialHashMap();
                    result_meshes.push_back(std::move(new_mesh));
                }
            }

            return result_meshes;
        }

        std::vector<std::vector<OpenMesh::VertexHandle>> ExtractBoundaryLoops() const
        {
            std::vector<std::vector<OpenMesh::VertexHandle>> boundaries;
            std::vector<bool> visited_he(n_halfedges(), false);

            for (auto h_it = halfedges_begin(); h_it != halfedges_end(); ++h_it)
            {
                // [버그 수정] status(*h_it) 대신 안전한 status(edge_handle(*h_it))를 사용합니다!
                if (status(edge_handle(*h_it)).deleted() || !is_boundary(*h_it) || visited_he[h_it->idx()]) continue;

                std::vector<OpenMesh::VertexHandle> loop;
                auto current_he = *h_it;

                do {
                    visited_he[current_he.idx()] = true;
                    loop.push_back(to_vertex_handle(current_he));
                    current_he = next_halfedge_handle(current_he);
                } while (current_he.is_valid() && current_he != *h_it && is_boundary(current_he));

                if (loop.size() >= 3) {
                    boundaries.push_back(loop);
                }
            }
            return boundaries;
        }

        OpenMesh::FaceHandle FindFaceAtPosition(const Eigen::Vector3f& pt) const
        {
            if (hash_map.empty()) return OpenMesh::FaceHandle(-1);

            Eigen::Vector3f local_pos = pt - grid_min;
            int cx = static_cast<int>(std::floor(local_pos.x() / grid_cell_size.x()));
            int cy = static_cast<int>(std::floor(local_pos.y() / grid_cell_size.y()));
            int cz = static_cast<int>(std::floor(local_pos.z() / grid_cell_size.z()));

            OpenMesh::FaceHandle best_face(-1);
            float min_dist_sq = std::numeric_limits<float>::max();

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto it = hash_map.find(Eigen::Vector3i(cx + dx, cy + dy, cz + dz));
                        if (it == hash_map.end()) continue;

                        for (int f_idx : it->second)
                        {
                            Eigen::Vector3f v0, v1, v2;
                            GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                            Eigen::Vector3f edge1 = v1 - v0;
                            Eigen::Vector3f edge2 = v2 - v0;
                            Eigen::Vector3f normal = edge1.cross(edge2);
                            if (normal.squaredNorm() < 1e-12f) continue;
                            normal.normalize();

                            Eigen::Vector3f v2p = pt - v0;
                            float d00 = edge1.dot(edge1);
                            float d01 = edge1.dot(edge2);
                            float d11 = edge2.dot(edge2);
                            float d20 = v2p.dot(edge1);
                            float d21 = v2p.dot(edge2);
                            float denom = d00 * d11 - d01 * d01;
                            if (std::abs(denom) < 1e-12f) continue;

                            float v = (d11 * d20 - d01 * d21) / denom;
                            float w = (d00 * d21 - d01 * d20) / denom;
                            float u = 1.0f - v - w;

                            float dist_sq;
                            if (v >= 0.0f && w >= 0.0f && u >= 0.0f)
                            {
                                // 삼각형 영역 안: 평면까지의 수직거리
                                float dp = normal.dot(v2p);
                                dist_sq = dp * dp;
                            }
                            else
                            {
                                // 삼각형 밖: 가장 가까운 변까지의 거리
                                float d_e0 = DistanceToSegmentSquared(pt, v0, v1);
                                float d_e1 = DistanceToSegmentSquared(pt, v1, v2);
                                float d_e2 = DistanceToSegmentSquared(pt, v2, v0);
                                dist_sq = std::min(d_e0, std::min(d_e1, d_e2));
                            }

                            if (dist_sq < min_dist_sq)
                            {
                                min_dist_sq = dist_sq;
                                best_face = face_handle(f_idx);
                            }
                        }
                    }
                }
            }
            return best_face;
        }

        void FlipAllFaces()
        {
            std::vector<Eigen::Vector3f> new_points;
            std::vector<Eigen::Vector3i> new_indices;

            // 기존 정점 좌표를 재활용하기 위한 해시맵
            robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;

            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                if (status(*f_it).deleted()) continue;

                std::vector<Eigen::Vector3f> pts;
                for (auto fv_it = cfv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
                    pts.push_back(Eigen::Vector3f(point(*fv_it).data()));
                }

                // 면이 3각형이 아니면 건너뜀 (안전 장치)
                if (pts.size() != 3) continue;

                Eigen::Vector3i face_indices;
                // 법선(Normal)을 뒤집기 위해 정점 순서를 역순(2, 1, 0)으로 삽입
                for (int k = 2; k >= 0; --k)
                {
                    const Eigen::Vector3f& p = pts[k];
                    auto it = vertex_map.find(p);
                    if (it != vertex_map.end()) {
                        face_indices[2 - k] = it->second;
                    }
                    else {
                        int new_idx = static_cast<int>(new_points.size());
                        new_points.push_back(p);
                        vertex_map[p] = new_idx;
                        face_indices[2 - k] = new_idx;
                    }
                }
                new_indices.push_back(face_indices);
            }

            // 기존 메쉬 데이터를 싹 날리고 안전하게 재조립
            this->clear();
            this->Build(new_points, new_indices);
        }

    protected:
        Eigen::Vector3f grid_min;
        Eigen::Vector3f grid_max;
        Eigen::Vector3f grid_cell_size;

        robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> hash_map;
    };
}
