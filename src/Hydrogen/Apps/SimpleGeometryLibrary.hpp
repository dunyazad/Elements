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
        // ---------------------------------------------------------
        // 위상 정보를 담은 절대 좌표 노드 (Co-refinement 용)
        // ---------------------------------------------------------
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
            request_vertex_status();
            request_edge_status();
            request_face_status();

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

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> ExtractIntersectionSegments(const Mesh& other_mesh) const
        {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> segments;
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

                    for (int f_idx : candidate_faces)
                    {
                        Eigen::Vector3f v0, v1, v2;
                        GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                        Eigen::Vector3f p1, p2;
                        if (IntersectTriangleTriangle(v0, v1, v2, u0, u1, u2, p1, p2))
                        {
                            local_segments.push_back({ p1, p2 });
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

        // 미세 오차 정점 병합 (Welding)
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> WeldSegments(
            const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& raw_segments,
            float weld_tolerance = 1e-4f) const
        {
            if (raw_segments.empty()) return {};

            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> welded_segments;
            std::vector<Eigen::Vector3f> unique_points;

            float tolerance_sq = weld_tolerance * weld_tolerance;

            auto get_welded_point = [&](const Eigen::Vector3f& p) -> Eigen::Vector3f {
                for (const auto& up : unique_points) {
                    if ((up - p).squaredNorm() < tolerance_sq) {
                        return up;
                    }
                }
                unique_points.push_back(p);
                return p;
                };

            for (const auto& seg : raw_segments)
            {
                Eigen::Vector3f p1 = get_welded_point(seg.first);
                Eigen::Vector3f p2 = get_welded_point(seg.second);

                if ((p1 - p2).squaredNorm() > 1e-8f)
                {
                    welded_segments.push_back({ p1, p2 });
                }
            }

            return welded_segments;
        }

        // 선분들을 폴리라인(루프)으로 연결 (Linking)
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

        // 스냅 후 위상 승격 (Topological Promotion)
        void PromoteTopology(CutNode& node, float snap_eps = 1e-4f) const
        {
            if (node.vh.is_valid()) return;

            if (node.eh.is_valid())
            {
                auto h0 = halfedge_handle(node.eh, 0);
                auto v0_handle = from_vertex_handle(h0);
                auto v1_handle = to_vertex_handle(h0);

                Eigen::Vector3f v0(point(v0_handle).data());
                Eigen::Vector3f v1(point(v1_handle).data());

                if ((node.pos - v0).squaredNorm() < snap_eps * snap_eps) {
                    node.eh = OpenMesh::EdgeHandle(-1);
                    node.vh = v0_handle;
                    node.pos = v0;
                }
                else if ((node.pos - v1).squaredNorm() < snap_eps * snap_eps) {
                    node.eh = OpenMesh::EdgeHandle(-1);
                    node.vh = v1_handle;
                    node.pos = v1;
                }
                return;
            }

            if (node.fh.is_valid())
            {
                for (auto fh_it = cfh_iter(node.fh); fh_it.is_valid(); ++fh_it)
                {
                    auto heh = *fh_it;
                    auto v0_handle = from_vertex_handle(heh);
                    auto v1_handle = to_vertex_handle(heh);

                    Eigen::Vector3f v0(point(v0_handle).data());
                    Eigen::Vector3f v1(point(v1_handle).data());

                    float dist = DistanceToSegment(node.pos, v0, v1);
                    if (dist < snap_eps)
                    {
                        node.fh = OpenMesh::FaceHandle(-1);
                        node.eh = edge_handle(heh);

                        if ((node.pos - v0).squaredNorm() < snap_eps * snap_eps) {
                            node.eh = OpenMesh::EdgeHandle(-1);
                            node.vh = v0_handle;
                            node.pos = v0;
                        }
                        else if ((node.pos - v1).squaredNorm() < snap_eps * snap_eps) {
                            node.eh = OpenMesh::EdgeHandle(-1);
                            node.vh = v1_handle;
                            node.pos = v1;
                        }
                        break;
                    }
                }
            }
        }

        // 결정론적 메쉬 강제 분할 (Deterministic Split)
        void SplitMeshDeterministic(const std::vector<CutNode>& polyline)
        {
            std::vector<OpenMesh::VertexHandle> path_vertices;
            std::vector<OpenMesh::FaceHandle> faces_to_delete;

            for (const auto& node : polyline)
            {
                OpenMesh::VertexHandle vh;
                if (node.vh.is_valid()) {
                    vh = node.vh;
                }
                else {
                    vh = add_vertex(OMMesh::Point(node.pos.x(), node.pos.y(), node.pos.z()));
                }
                path_vertices.push_back(vh);

                if (node.fh.is_valid()) faces_to_delete.push_back(node.fh);
                if (node.eh.is_valid()) {
                    faces_to_delete.push_back(face_handle(halfedge_handle(node.eh, 0)));
                    faces_to_delete.push_back(face_handle(halfedge_handle(node.eh, 1)));
                }
            }

            std::sort(faces_to_delete.begin(), faces_to_delete.end());
            faces_to_delete.erase(std::unique(faces_to_delete.begin(), faces_to_delete.end()), faces_to_delete.end());

            for (auto fh : faces_to_delete)
            {
                if (!fh.is_valid()) continue;

                Eigen::Vector3f v0, v1, v2;
                GetFaceVertices(fh, v0, v1, v2);
                Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();

                // [핵심 1] 원본 삼각형의 무게 중심과 기준 방향축 도출
                Eigen::Vector3f center = (v0 + v1 + v2) / 3.0f;
                Eigen::Vector3f ref_dir = (v0 - center).normalized();

                std::vector<OpenMesh::VertexHandle> new_polygon;
                for (auto fv_it = cfv_iter(fh); fv_it.is_valid(); ++fv_it) {
                    new_polygon.push_back(*fv_it);
                }

                delete_face(fh, false);

                for (auto vh : path_vertices) {
                    Eigen::Vector3f p(point(vh).data());
                    if (DistanceToSegment(p, v0, v1) < 1e-3f || DistanceToSegment(p, v1, v2) < 1e-3f || DistanceToSegment(p, v2, v0) < 1e-3f || IsPointInTriangle(p, v0, v1, v2)) {
                        if (std::find(new_polygon.begin(), new_polygon.end(), vh) == new_polygon.end()) {
                            new_polygon.push_back(vh);
                        }
                    }
                }

                // [핵심 2] 방사형 정렬 (Radial Sort) - 꼬인 폴리곤을 완벽하게 풀어줌!
                std::sort(new_polygon.begin(), new_polygon.end(), [&](OpenMesh::VertexHandle a, OpenMesh::VertexHandle b) {
                    Eigen::Vector3f pa = Eigen::Vector3f(point(a).data());
                    Eigen::Vector3f pb = Eigen::Vector3f(point(b).data());

                    // 너무 중앙에 가까운 점은 각도 계산이 터지므로 예외 처리
                    if ((pa - center).squaredNorm() < 1e-8f) return true;
                    if ((pb - center).squaredNorm() < 1e-8f) return false;

                    Eigen::Vector3f da = (pa - center).normalized();
                    Eigen::Vector3f db = (pb - center).normalized();

                    // 중심점을 기준으로 0~360도(2*PI) 각도 계산
                    float angle_a = std::atan2(normal.dot(ref_dir.cross(da)), ref_dir.dot(da));
                    float angle_b = std::atan2(normal.dot(ref_dir.cross(db)), ref_dir.dot(db));

                    if (angle_a < 0.0f) angle_a += 2.0f * static_cast<float>(M_PI);
                    if (angle_b < 0.0f) angle_b += 2.0f * static_cast<float>(M_PI);

                    // 각도가 거의 같다면 중심에서 더 가까운 놈을 먼저 (선형 겹침 방지)
                    if (std::abs(angle_a - angle_b) < 1e-4f) {
                        return (pa - center).squaredNorm() < (pb - center).squaredNorm();
                    }

                    return angle_a < angle_b;
                    });

                // 꼬임이 완벽히 풀린 다각형으로 다시 귀 자르기 실행
                TriangulatePolygon(new_polygon, normal);
            }

            garbage_collection();
            BuildSpatialHashMap();
        }

        // ---------------------------------------------------------
        // (NEW) 완벽한 동적 위상 추적 분할 (Dynamic Topology Tracking Split)
        // ---------------------------------------------------------
        void SplitMeshDynamic(const std::vector<Eigen::Vector3f>& polyline)
        {
            const float SNAP_EPS = 1e-4f;

            for (const auto& pt : polyline)
            {
                OpenMesh::FaceHandle target_fh(-1);
                OpenMesh::EdgeHandle target_eh(-1);
                OpenMesh::VertexHandle target_vh(-1);

                // 1. "미리 계산하지 않고", 매 점을 찌를 때마다 현재 업데이트된 메쉬에서 위치를 실시간으로 찾습니다.
                for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
                {
                    Eigen::Vector3f v0, v1, v2;
                    GetFaceVertices(*f_it, v0, v1, v2);

                    Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
                    if (n.squaredNorm() < 1e-8f) continue;
                    n.normalize();

                    // 평면에서 너무 멀면 패스
                    if (std::abs(n.dot(pt - v0)) > 1e-3f) continue;

                    // 이 점이 속한 "현재 살아있는" 면(Face)을 발견!
                    if (IsPointInTriangle(pt, v0, v1, v2))
                    {
                        target_fh = *f_it;

                        // 2. 엣지나 정점에 가까우면 즉시 위상 승격(Promotion)하여 슬리버 방지
                        for (auto fh_it = cfh_iter(*f_it); fh_it.is_valid(); ++fh_it)
                        {
                            auto heh = *fh_it;
                            auto vh0 = from_vertex_handle(heh);
                            auto vh1 = to_vertex_handle(heh);
                            Eigen::Vector3f ev0(point(vh0).data());
                            Eigen::Vector3f ev1(point(vh1).data());

                            if (DistanceToSegment(pt, ev0, ev1) < SNAP_EPS)
                            {
                                target_fh = OpenMesh::FaceHandle(-1); // 면 취소
                                target_eh = edge_handle(heh);         // 엣지 승격

                                if ((pt - ev0).squaredNorm() < SNAP_EPS * SNAP_EPS) {
                                    target_eh = OpenMesh::EdgeHandle(-1);
                                    target_vh = vh0; // 정점 승격
                                    break;
                                }
                                else if ((pt - ev1).squaredNorm() < SNAP_EPS * SNAP_EPS) {
                                    target_eh = OpenMesh::EdgeHandle(-1);
                                    target_vh = vh1; // 정점 승격
                                    break;
                                }
                            }
                        }
                        break; // 올바른 타겟을 찾았으므로 탐색 종료
                    }
                }

                // 3. 찾은 싱싱한(?) 위상을 바탕으로 즉시 쪼개기 (Stale Handle 문제 원천 차단)
                if (target_vh.is_valid()) continue; // 이미 있는 정점이면 통과

                if (target_eh.is_valid() || target_fh.is_valid())
                {
                    auto new_v = add_vertex(OMMesh::Point(pt.x(), pt.y(), pt.z()));
                    if (target_eh.is_valid()) {
                        split(target_eh, new_v);
                    }
                    else if (target_fh.is_valid()) {
                        split(target_fh, new_v); // 면이 쪼개지며 즉시 메쉬 토폴로지 업데이트 됨
                    }
                }
            }

            // 모든 분할이 끝난 후 공간 해시맵 1회 업데이트
            garbage_collection();
            BuildSpatialHashMap();
        }

        // ---------------------------------------------------------
        // (NEW) 해시 맵을 활용한 O(1) 초고속 점 소속 면(Face) 탐색
        // ---------------------------------------------------------
        OpenMesh::FaceHandle FindFaceAtPosition(const Eigen::Vector3f& pt) const
        {
            if (hash_map.empty()) return OpenMesh::FaceHandle(-1);

            Eigen::Vector3f local_pos = pt - grid_min;
            int cx = static_cast<int>(std::floor(local_pos.x() / grid_cell_size.x()));
            int cy = static_cast<int>(std::floor(local_pos.y() / grid_cell_size.y()));
            int cz = static_cast<int>(std::floor(local_pos.z() / grid_cell_size.z()));

            OpenMesh::FaceHandle best_face(-1);
            float min_dist = std::numeric_limits<float>::max();

            // 점이 위치한 복셀 및 인접 복셀(3x3x3=27칸)만 탐색 (전체 7만 개 탐색 -> 평균 10~20개 탐색으로 단축)
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        auto it = hash_map.find(Eigen::Vector3i(cx + dx, cy + dy, cz + dz));
                        if (it != hash_map.end()) {
                            for (int f_idx : it->second) {
                                Eigen::Vector3f v0, v1, v2;
                                GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                                Eigen::Vector3f edge1 = v1 - v0;
                                Eigen::Vector3f edge2 = v2 - v0;
                                Eigen::Vector3f normal = edge1.cross(edge2);
                                if (normal.squaredNorm() < 1e-8f) continue;
                                normal.normalize();

                                float dist_to_plane = std::abs(normal.dot(pt - v0));
                                if (dist_to_plane < min_dist) {
                                    Eigen::Vector3f v2p = pt - v0;
                                    float d00 = edge1.dot(edge1);
                                    float d01 = edge1.dot(edge2);
                                    float d11 = edge2.dot(edge2);
                                    float d20 = v2p.dot(edge1);
                                    float d21 = v2p.dot(edge2);
                                    float denom = d00 * d11 - d01 * d01;

                                    if (std::abs(denom) > 1e-8f) {
                                        float v = (d11 * d20 - d01 * d21) / denom;
                                        float w = (d00 * d21 - d01 * d20) / denom;
                                        float u = 1.0f - v - w;

                                        if (v >= -0.01f && w >= -0.01f && u >= -0.01f) {
                                            min_dist = dist_to_plane;
                                            best_face = face_handle(f_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return best_face;
        }

    protected:
        Eigen::Vector3f grid_min;
        Eigen::Vector3f grid_max;
        Eigen::Vector3f grid_cell_size;

        robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> hash_map;
    };
}