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

        // ---------------------------------------------------------
        // (1) 교차선 추출 + 동시에 충돌한 내 면(Face) 100% 정확하게 마킹
        // ---------------------------------------------------------
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> ExtractIntersectionSegments(const Mesh& other_mesh, std::vector<char>* out_cut_faces = nullptr) const
        {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> segments;
            std::mutex mtx;

            size_t other_num_faces = other_mesh.n_faces();
            if (other_num_faces == 0 || hash_map.empty()) return segments;

            // 마스크 배열이 넘어왔다면 내 면(Face) 개수만큼 0으로 초기화
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
                        // [핵심] 바로 여기서 충돌 판정이 일어납니다! 부딪힌 면(f_idx)을 기억해둡니다.
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

                        // 멀티스레드 안전하게 마스크에 1(충돌) 기록
                        if (out_cut_faces) {
                            for (int f_idx : local_cut_faces) {
                                (*out_cut_faces)[f_idx] = 1;
                            }
                        }
                    }
                });

            return segments;
        }

        // ---------------------------------------------------------
        // (2) 마킹된 면과 그 이웃(1-Ring)까지 통째로 날려버리는 물리적 단절 함수
        // ---------------------------------------------------------
        void DeleteMarkedFaces(const std::vector<char>& cut_faces_mask)
        {
            std::vector<OpenMesh::FaceHandle> to_delete;

            // 지울 면들을 1차적으로 수집
            for (size_t i = 0; i < cut_faces_mask.size(); ++i)
            {
                if (cut_faces_mask[i] == 1)
                {
                    auto fh = face_handle(static_cast<int>(i));
                    if (fh.is_valid() && !status(fh).deleted())
                    {
                        to_delete.push_back(fh);
                        // [필살기] 거대 삼각형으로 인한 누수를 막기 위해, 부딪힌 면의 이웃 면들까지 모조리 참호로 파버립니다!
                        for (auto ff_it = cff_iter(fh); ff_it.is_valid(); ++ff_it) {
                            if (!status(*ff_it).deleted()) {
                                to_delete.push_back(*ff_it);
                            }
                        }
                    }
                }
            }

            // 중복 제거
            std::sort(to_delete.begin(), to_delete.end());
            to_delete.erase(std::unique(to_delete.begin(), to_delete.end()), to_delete.end());

            int deleted_count = 0;
            for (auto fh : to_delete) {
                if (!status(fh).deleted()) {
                    delete_face(fh, false);
                    deleted_count++;
                }
            }

            if (deleted_count > 0)
            {
                garbage_collection(); // 영구 삭제! (위상 100% 끊어짐)
                BuildSpatialHashMap();
            }
            std::cout << "[Delete Log] 교차 판정된 " << deleted_count << "개의 면을 정확히 삭제했습니다." << std::endl;
        }

        // ---------------------------------------------------------
        // (REFACTOR) 교차선 추출과 동시에 충돌한 내 면(Face)을 다이렉트로 마킹
        // ---------------------------------------------------------
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> ExtractIntersectionSegmentsAndMarkFaces(
            const Mesh& other_mesh,
            std::vector<char>& out_my_intersected_faces) const
        {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> segments;
            std::mutex mtx;

            size_t num_faces = n_faces();
            out_my_intersected_faces.assign(num_faces, 0); // 0으로 초기화

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
                    std::vector<int> local_intersected_faces;

                    for (int f_idx : candidate_faces)
                    {
                        Eigen::Vector3f v0, v1, v2;
                        GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                        Eigen::Vector3f p1, p2;
                        if (IntersectTriangleTriangle(v0, v1, v2, u0, u1, u2, p1, p2))
                        {
                            local_segments.push_back({ p1, p2 });
                            local_intersected_faces.push_back(f_idx); // 충돌한 면 번호 기록
                        }
                    }

                    if (!local_segments.empty())
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        segments.insert(segments.end(), local_segments.begin(), local_segments.end());

                        // 충돌 마스크 마킹 (병렬 스레드 간 인덱스가 겹쳐도 동일한 값을 쓰므로 안전)
                        for (int f_idx : local_intersected_faces) {
                            out_my_intersected_faces[f_idx] = 1;
                        }
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

        // ---------------------------------------------------------
        // (TRENCH CUT) 교차선 주변의 면을 마킹하고 즉시 삭제하여 물리적 틈새를 만듦
        // ---------------------------------------------------------
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
        
        // ---------------------------------------------------------
        // 물리적으로 단절된 메쉬를 순수 Flood Fill로 그룹화하여 분리
        // (방어막 검사 같은 복잡한 연산이 일절 없는 초고속 로직)
        // ---------------------------------------------------------
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

        // ---------------------------------------------------------
        // 1. 메쉬에 뚫려있는 모든 경계선(Boundary Loops)을 추출
        // ---------------------------------------------------------
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

        // ---------------------------------------------------------
        // 2. 특정 경계선(Boundary)이 어떤 절단선(Ring)과 짝인지 찾아주는 함수
        // ---------------------------------------------------------
        int MatchBoundaryToRing(const std::vector<OpenMesh::VertexHandle>& boundary_loop,
            const std::vector<std::vector<Eigen::Vector3f>>& polylines) const
        {
            int best_ring_idx = -1;
            float min_avg_dist = std::numeric_limits<float>::max();

            for (size_t r = 0; r < polylines.size(); ++r)
            {
                const auto& ring = polylines[r];
                if (ring.empty()) continue;

                float total_dist = 0.0f;
                // 경계선 정점들이 현재 검사 중인 ring과 얼마나 떨어져 있는지 평균 거리를 계산
                for (auto vh : boundary_loop)
                {
                    Eigen::Vector3f p(point(vh).data());
                    float min_d_sq = std::numeric_limits<float>::max();

                    for (size_t i = 0; i < ring.size() - 1; ++i) {
                        float d_sq = DistanceToSegmentSquared(p, ring[i], ring[i + 1]);
                        if (d_sq < min_d_sq) min_d_sq = d_sq;
                    }
                    total_dist += std::sqrt(min_d_sq);
                }

                float avg_dist = total_dist / boundary_loop.size();

                // 가장 평균 거리가 짧은(가장 찰딱 붙어있는) Ring을 이 구멍의 주인으로 판정!
                if (avg_dist < min_avg_dist) {
                    min_avg_dist = avg_dist;
                    best_ring_idx = static_cast<int>(r);
                }
            }
            return best_ring_idx;
        }

        void FillHolesWithEarcut(const std::vector<std::vector<OpenMesh::VertexHandle>>& boundaries,
            const std::vector<std::vector<Eigen::Vector3f>>& polylines)
        {
            for (const auto& boundary : boundaries)
            {
                if (boundary.size() < 3) continue;

                // 1. 해당 구멍(Boundary)과 짝이 맞는 절단선(Ring) 찾기
                int matched_ring_idx = MatchBoundaryToRing(boundary, polylines);
                if (matched_ring_idx == -1) continue;

                const auto& ring = polylines[matched_ring_idx];

                // 2. 시작점 찾기 (절단선과 가장 가까운 경계선 상의 정점)
                int start_idx = 0;
                float min_dist = std::numeric_limits<float>::max();

                int num_b_original = static_cast<int>(ring.size());
                int num_b = (ring.front() - ring.back()).norm() < 1e-6f ? num_b_original - 1 : num_b_original;

                for (size_t n = 0; n < boundary.size(); ++n)
                {
                    Eigen::Vector3f p_a(point(boundary[n]).data());
                    for (int m = 0; m < num_b; ++m)
                    {
                        float d = (p_a - ring[m]).squaredNorm();
                        if (d < min_dist)
                        {
                            min_dist = d;
                            start_idx = static_cast<int>(n);
                        }
                    }
                }

                // 3. Earcut(귀 자르기)를 위해 시작점을 기준으로 배열(Loop) 재정렬 (Rotate)
                std::vector<OpenMesh::VertexHandle> ordered_boundary = boundary;
                std::rotate(ordered_boundary.begin(), ordered_boundary.begin() + start_idx, ordered_boundary.end());

                // 4. 구멍을 덮을 다각형의 법선 벡터 계산 (Newell's Method)
                Eigen::Vector3f normal = Eigen::Vector3f::Zero();
                for (size_t i = 0; i < ordered_boundary.size(); ++i)
                {
                    Eigen::Vector3f p0(point(ordered_boundary[i]).data());
                    Eigen::Vector3f p1(point(ordered_boundary[(i + 1) % ordered_boundary.size()]).data());
                    normal.x() += (p0.y() - p1.y()) * (p0.z() + p1.z());
                    normal.y() += (p0.z() - p1.z()) * (p0.x() + p1.x());
                    normal.z() += (p0.x() - p1.x()) * (p0.y() + p1.y());
                }

                if (normal.squaredNorm() > 1e-8f)
                {
                    normal.normalize();
                }

                // 5. 정렬된 폴리곤 배열과 법선 벡터를 이용해 Earcut 실행
                TriangulatePolygon(ordered_boundary, normal);
            }

            // 구멍을 메운 후 내부 데이터 구조 갱신
            garbage_collection();
            BuildSpatialHashMap();
        }

        bool ValidateLoopData(const std::vector<OpenMesh::VertexHandle>& loop_a, const std::vector<OpenMesh::VertexHandle>& loop_b) const
        {
            //std::cout << "[Validation] Checking Loop Data..." << std::endl;
            if (loop_a.size() < 3 || loop_b.size() < 3)
            {
                //std::cout << "[Validation Error] Loop size is too small. Loop A: " << loop_a.size() << ", Loop B: " << loop_b.size() << std::endl;
                return false;
            }

            for (auto vh : loop_a)
            {
                if (!vh.is_valid())
                {
                    //std::cout << "[Validation Error] Invalid VertexHandle in Loop A." << std::endl;
                    return false;
                }
            }

            for (auto vh : loop_b)
            {
                if (!vh.is_valid())
                {
                    //std::cout << "[Validation Error] Invalid VertexHandle in Loop B." << std::endl;
                    return false;
                }
            }

            //std::cout << "[Validation Success] Loop Data is valid." << std::endl;
            return true;
        }

        OpenMesh::FaceHandle AddFaceWithLog(OpenMesh::VertexHandle v0, OpenMesh::VertexHandle v1, OpenMesh::VertexHandle v2)
        {
            OpenMesh::FaceHandle face_handle = add_face(v0, v1, v2);
            if (face_handle.is_valid())
            {
                //std::cout << "[StitchLog] Face SUCCESS: (" << v0.idx() << ", " << v1.idx() << ", " << v2.idx() << ")" << std::endl;
            }
            else
            {
                //std::cout << "[StitchLog] Face ERROR: Reversing winding order for (" << v0.idx() << ", " << v1.idx() << ", " << v2.idx() << ")" << std::endl;
                face_handle = add_face(v0, v2, v1);
                if (face_handle.is_valid())
                {
                    //std::cout << "[StitchLog] Face SUCCESS (Reversed): (" << v0.idx() << ", " << v2.idx() << ", " << v1.idx() << ")" << std::endl;
                }
                else
                {
                    //std::cout << "[StitchLog] Face FATAL: Failed both winding orders!" << std::endl;
                }
            }
            return face_handle;
        }

        int FindBestMatchingRing(const std::vector<OpenMesh::VertexHandle>& boundary, const std::vector<std::vector<Eigen::Vector3f>>& polylines) const
        {
            //std::cout << "\n[Step 1] Finding Best Matching Ring (Dynamic Scale)..." << std::endl;

            // 1. 모델의 전체 바운딩 박스를 계산하여 스케일을 파악합니다.
            Eigen::Vector3f b_min(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            Eigen::Vector3f b_max(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

            for (auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
            {
                Eigen::Vector3f p(point(*v_it).data());
                b_min = b_min.cwiseMin(p);
                b_max = b_max.cwiseMax(p);
            }

            // 모델의 대각선 길이를 기준으로 동적 임계값을 설정합니다 (예: 전체 크기의 5%)
            float model_diagonal = (b_max - b_min).norm();
            float dynamic_threshold = model_diagonal * 0.05f;

            //std::cout << "[StitchLog] Model Diagonal: " << model_diagonal << ", Dynamic Threshold: " << dynamic_threshold << std::endl;

            int best_ring_index = -1;
            float min_average_distance = std::numeric_limits<float>::max();

            // 2. 모든 절단선과의 평균 거리를 계산합니다.
            for (size_t r = 0; r < polylines.size(); ++r)
            {
                const auto& ring = polylines[r];
                if (ring.size() < 3)
                {
                    continue;
                }

                float total_distance = 0.0f;
                for (auto vh : boundary)
                {
                    Eigen::Vector3f pa(point(vh).data());
                    float min_d = std::numeric_limits<float>::max();
                    for (const auto& pb : ring)
                    {
                        min_d = std::min(min_d, (pa - pb).norm());
                    }
                    total_distance += min_d;
                }
                float average_distance = total_distance / static_cast<float>(boundary.size());

                //std::cout << "[StitchLog] Ring " << r << " Average Distance: " << average_distance << std::endl;

                if (average_distance < min_average_distance)
                {
                    min_average_distance = average_distance;
                    best_ring_index = static_cast<int>(r);
                }
            }

            // 3. 고정값이 아닌, 모델 스케일에 비례하는 동적 임계값으로 필터링합니다.
            if (min_average_distance > dynamic_threshold)
            {
                //std::cout << "[StitchLog] No valid ring found. Minimum average distance (" << min_average_distance << ") exceeds dynamic threshold (" << dynamic_threshold << ")." << std::endl;
                return -1;
            }

            //std::cout << "[StitchLog] Selected Ring Index: " << best_ring_index << std::endl;
            return best_ring_index;
        }

        void FindBestStartPoints(const std::vector<OpenMesh::VertexHandle>& loop_a, const std::vector<Eigen::Vector3f>& loop_b, int& out_best_a, int& out_best_b) const
        {
            //std::cout << "[Step 2] Finding Best Start Points..." << std::endl;
            out_best_a = 0;
            out_best_b = 0;
            float min_start_distance = std::numeric_limits<float>::max();

            int number_a = static_cast<int>(loop_a.size());
            int number_b = static_cast<int>(loop_b.size());

            for (int i = 0; i < number_a; ++i)
            {
                Eigen::Vector3f pa(point(loop_a[i]).data());
                for (int j = 0; j < number_b; ++j)
                {
                    float distance = (pa - loop_b[j]).squaredNorm();
                    if (distance < min_start_distance)
                    {
                        min_start_distance = distance;
                        out_best_a = i;
                        out_best_b = j;
                    }
                }
            }
            //std::cout << "[StitchLog] Start Points Found -> Loop A Index: " << out_best_a << ", Loop B Index: " << out_best_b << " (Squared Distance: " << min_start_distance << ")" << std::endl;
        }

        float CalculateDPCost(const std::vector<OpenMesh::VertexHandle>& loop_a, const std::vector<Eigen::Vector3f>& loop_b, std::vector<int>& out_path) const
        {
            int number_a = static_cast<int>(loop_a.size());
            int number_b = static_cast<int>(loop_b.size());

            std::vector<float> dp((number_a + 1) * (number_b + 1), 0.0f);
            std::vector<int> choice((number_a + 1) * (number_b + 1), 0);

            auto get_distance = [&](int i, int j) -> float
                {
                    Eigen::Vector3f pa(point(loop_a[i % number_a]).data());
                    Eigen::Vector3f pb = loop_b[j % number_b];
                    return (pa - pb).norm();
                };

            dp[0] = 0.0f;
            for (int i = 1; i <= number_a; ++i)
            {
                dp[i * (number_b + 1) + 0] = dp[(i - 1) * (number_b + 1) + 0] + get_distance(i, 0);
                choice[i * (number_b + 1) + 0] = 1;
            }

            for (int j = 1; j <= number_b; ++j)
            {
                dp[0 * (number_b + 1) + j] = dp[0 * (number_b + 1) + (j - 1)] + get_distance(0, j);
                choice[0 * (number_b + 1) + j] = 2;
            }

            for (int i = 1; i <= number_a; ++i)
            {
                for (int j = 1; j <= number_b; ++j)
                {
                    float cost_a = dp[(i - 1) * (number_b + 1) + j];
                    float cost_b = dp[i * (number_b + 1) + (j - 1)];

                    float current_distance = get_distance(i, j);

                    if (cost_a < cost_b)
                    {
                        dp[i * (number_b + 1) + j] = cost_a + current_distance;
                        choice[i * (number_b + 1) + j] = 1;
                    }
                    else
                    {
                        dp[i * (number_b + 1) + j] = cost_b + current_distance;
                        choice[i * (number_b + 1) + j] = 2;
                    }
                }
            }

            out_path.clear();
            int current_i = number_a;
            int current_j = number_b;

            while (current_i > 0 || current_j > 0)
            {
                int step = choice[current_i * (number_b + 1) + current_j];
                out_path.push_back(step);

                if (step == 1)
                {
                    current_i--;
                }
                else
                {
                    current_j--;
                }
            }
            std::reverse(out_path.begin(), out_path.end());

            return dp[number_a * (number_b + 1) + number_b];
        }

        std::vector<Eigen::Vector3f> DetermineOptimalRingDirection(const std::vector<OpenMesh::VertexHandle>& loop_a, const std::vector<Eigen::Vector3f>& original_ring, int best_b) const
        {
            int number_a = static_cast<int>(loop_a.size());
            int number_b = static_cast<int>(original_ring.size());

            std::vector<Eigen::Vector3f> ring_forward;
            std::vector<Eigen::Vector3f> ring_reverse;

            for (int i = 0; i < number_b; ++i)
            {
                ring_forward.push_back(original_ring[(best_b + i) % number_b]);
                ring_reverse.push_back(original_ring[(best_b - i + number_b) % number_b]);
            }

            if (number_a < 3 || number_b < 3)
            {
                return ring_forward;
            }

            int step_a = std::max(1, number_a / 10);
            int step_b = std::max(1, number_b / 10);

            Eigen::Vector3f start_point_a(point(loop_a[0]).data());
            Eigen::Vector3f target_point_a(point(loop_a[step_a]).data());
            Eigen::Vector3f direction_a = (target_point_a - start_point_a).normalized();

            Eigen::Vector3f start_point_b = ring_forward[0];

            Eigen::Vector3f target_point_b_forward = ring_forward[step_b];
            Eigen::Vector3f direction_b_forward = (target_point_b_forward - start_point_b).normalized();

            Eigen::Vector3f target_point_b_reverse = ring_reverse[step_b];
            Eigen::Vector3f direction_b_reverse = (target_point_b_reverse - start_point_b).normalized();

            if (direction_a.dot(direction_b_forward) >= direction_a.dot(direction_b_reverse))
            {
                return ring_forward;
            }
            else
            {
                return ring_reverse;
            }
        }

        void PerformParameterizationZippering(const std::vector<OpenMesh::VertexHandle>& loop_a, const std::vector<OpenMesh::VertexHandle>& loop_b)
        {
            int number_a = static_cast<int>(loop_a.size());
            int number_b = static_cast<int>(loop_b.size());

            std::vector<float> parameters_a(number_a, 0.0f);
            float current_length_a = 0.0f;
            for (int i = 0; i < number_a; ++i)
            {
                parameters_a[i] = current_length_a;
                current_length_a += (Eigen::Vector3f(point(loop_a[(i + 1) % number_a]).data()) - Eigen::Vector3f(point(loop_a[i]).data())).norm();
            }
            if (current_length_a > 1e-6f)
            {
                for (int i = 0; i < number_a; ++i)
                {
                    parameters_a[i] /= current_length_a;
                }
            }

            std::vector<float> parameters_b(number_b, 0.0f);
            float current_length_b = 0.0f;
            for (int i = 0; i < number_b; ++i)
            {
                parameters_b[i] = current_length_b;
                current_length_b += (Eigen::Vector3f(point(loop_b[(i + 1) % number_b]).data()) - Eigen::Vector3f(point(loop_b[i]).data())).norm();
            }
            if (current_length_b > 1e-6f)
            {
                for (int i = 0; i < number_b; ++i)
                {
                    parameters_b[i] /= current_length_b;
                }
            }

            int index_a = 0;
            int index_b = 0;

            while (index_a < number_a || index_b < number_b)
            {
                int next_a = (index_a + 1) % number_a;
                int next_b = (index_b + 1) % number_b;

                bool advance_a = false;

                if (index_a == number_a)
                {
                    advance_a = false;
                }
                else if (index_b == number_b)
                {
                    advance_a = true;
                }
                else
                {
                    float parameter_next_a = (next_a == 0) ? 1.0f : parameters_a[next_a];
                    float parameter_next_b = (next_b == 0) ? 1.0f : parameters_b[next_b];
                    advance_a = (parameter_next_a <= parameter_next_b);
                }

                OpenMesh::FaceHandle face_handle;
                if (advance_a)
                {
                    face_handle = add_face(loop_a[next_a], loop_a[index_a % number_a], loop_b[index_b % number_b]);
                    if (!face_handle.is_valid())
                    {
                        add_face(loop_a[index_a % number_a], loop_a[next_a], loop_b[index_b % number_b]);
                    }
                    index_a++;
                }
                else
                {
                    face_handle = add_face(loop_b[next_b], loop_b[index_b % number_b], loop_a[index_a % number_a]);
                    if (!face_handle.is_valid())
                    {
                        add_face(loop_b[index_b % number_b], loop_b[next_b], loop_a[index_a % number_a]);
                    }
                    index_b++;
                }
            }
        }

        OpenMesh::FaceHandle FindFaceAtPosition(const Eigen::Vector3f& pt) const
        {
            if (hash_map.empty()) return OpenMesh::FaceHandle(-1);

            Eigen::Vector3f local_pos = pt - grid_min;
            int cx = static_cast<int>(std::floor(local_pos.x() / grid_cell_size.x()));
            int cy = static_cast<int>(std::floor(local_pos.y() / grid_cell_size.y()));
            int cz = static_cast<int>(std::floor(local_pos.z() / grid_cell_size.z()));

            OpenMesh::FaceHandle best_face(-1);
            float min_dist = std::numeric_limits<float>::max();

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

        // ---------------------------------------------------------
        // (Boolean) 특정 점(pt)이 '이 메쉬(this)' 내부에 있는지 판별 (Even-Odd Rule)
        // ---------------------------------------------------------
        bool IsPointInside(const Eigen::Vector3f& pt) const
        {
            Eigen::Vector3f dir(0.8523f, 0.3432f, 0.3951f);
            dir.normalize();

            int hit_count = CountRayIntersections(pt, dir);

            return (hit_count % 2) != 0;
        }

        int CountRayIntersections(const Eigen::Vector3f& origin, const Eigen::Vector3f& dir) const
        {
            if (hash_map.empty()) return 0;

            // 기존 IntersectGridRay의 고속 DDA(Digital Differential Analyzer) 알고리즘 재활용
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

            // 광선이 그리드 바운딩 박스를 아예 벗어났다면 충돌 0회
            if (t_enter > t_exit || t_exit < 0.0f) return 0;

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

            float max_t_search = t_exit + 0.1f;
            float current_t = t_enter;

            std::vector<bool> tested_faces(n_faces(), false);
            std::vector<float> hit_t_values;

            while (current_t <= max_t_search)
            {
                Eigen::Vector3i cell_pos(cx, cy, cz);
                auto it = hash_map.find(cell_pos);
                if (it != hash_map.end())
                {
                    for (int f_idx : it->second)
                    {
                        if (tested_faces[f_idx]) continue;
                        tested_faces[f_idx] = true; // 중복 검사 방지

                        auto fh = face_handle(f_idx);
                        if (status(fh).deleted()) continue;

                        IntersectionResult res = IntersectRayFaceWithSnap(origin, dir, fh);
                        if (res.type != IntersectionType::None && res.t > 1e-5f) // origin과 너무 가까운 충돌은 무시
                        {
                            // Edge나 Vertex에 맞았을 경우 인접한 Face들에 의해 중복 카운트되는 것을 방지합니다.
                            bool is_duplicate = false;
                            for (float t : hit_t_values) {
                                if (std::abs(t - res.t) < 1e-4f) {
                                    is_duplicate = true;
                                    break;
                                }
                            }
                            if (!is_duplicate) {
                                hit_t_values.push_back(res.t);
                            }
                        }
                    }
                }

                // DDA Step
                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) { cx += stepX; current_t = tMaxX; tMaxX += tDeltaX; }
                    else { cz += stepZ; current_t = tMaxZ; tMaxZ += tDeltaZ; }
                }
                else {
                    if (tMaxY < tMaxZ) { cy += stepY; current_t = tMaxY; tMaxY += tDeltaY; }
                    else { cz += stepZ; current_t = tMaxZ; tMaxZ += tDeltaZ; }
                }
            }

            return static_cast<int>(hit_t_values.size());
        }

        Eigen::Vector3f GetSafeSamplePoint() const
        {
            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                if (!status(*f_it).deleted())
                {
                    Eigen::Vector3f v0, v1, v2;
                    GetFaceVertices(*f_it, v0, v1, v2);
                    // 삼각형의 무게중심(Centroid) 반환
                    return (v0 + v1 + v2) / 3.0f;
                }
            }
            return Eigen::Vector3f::Zero(); // 폴백
        }

        // ---------------------------------------------------------
        // 메쉬의 모든 면(Face)을 뒤집는 함수 (완전 재구축 방식 - 절대 안 터짐)
        // ---------------------------------------------------------
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
