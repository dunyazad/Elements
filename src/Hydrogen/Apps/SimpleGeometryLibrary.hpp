#pragma once

#include <vector>
#include <algorithm>
#include <limits>
#include <execution>
#include <numeric>
#include <functional>
#include <mutex>
#include <Eigen/Dense>
#include <robin_hood/robin_hood.h>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

inline float DistanceToSegmentSquared(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    Eigen::Vector3f ab = b - a;
    float l2 = ab.squaredNorm();

    // 선분의 길이가 0인 경우 (a와 b가 같은 점인 경우)
    if (l2 < 1e-8f)
    {
        return (p - a).squaredNorm();
    }

    // 벡터 투영(Projection)을 통한 매개변수 t 계산
    float t = (p - a).dot(ab) / l2;

    // 선분 바깥으로 벗어나지 않도록 클램핑(Clamping)
    t = std::max(0.0f, std::min(1.0f, t));

    // 선분 위의 가장 가까운 점(Projection Point) 계산
    Eigen::Vector3f projection = a + t * ab;

    // 원래 점 P와 투영된 점 사이의 거리 제곱 반환
    return (p - projection).squaredNorm();
}

// ---------------------------------------------------------
// 점(p)과 선분(a ~ b) 사이의 실제 거리를 구하는 함수
// ---------------------------------------------------------
inline float DistanceToSegment(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
    return std::sqrt(DistanceToSegmentSquared(p, a, b));
}

namespace SGL
{
    using OMMesh = OpenMesh::TriMesh_ArrayKernelT<>;

    // ---------------------------------------------------------
    // 3D 그리드 좌표(int3)를 해싱하기 위한 초고속 해시 함수
    // ---------------------------------------------------------
    struct Int3Hash {
        size_t operator()(const Eigen::Vector3i& v) const {
            // 공간 좌표 해싱에 최적화된 프라임 곱셈 기법
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

    class Mesh : public OMMesh
    {
    public:
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

            TS(BuildSpatialHashMap);
            BuildSpatialHashMap();
            TE(BuildSpatialHashMap);
        }

        // 메모리 낭비가 전혀 없는 해시 기반 공간 분할 구축
        //void BuildSpatialHashMap()
        //{
        //    size_t num_faces = n_faces();
        //    if (num_faces == 0) return;

        //    std::vector<Eigen::Vector3f> centers(num_faces);
        //    grid_min.setConstant(std::numeric_limits<float>::max());
        //    grid_max.setConstant(-std::numeric_limits<float>::max());

        //    std::vector<int> face_indices(num_faces);
        //    std::iota(face_indices.begin(), face_indices.end(), 0);

        //    // 1. 센터 및 바운딩 박스 계산 (병렬)
        //    std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), [&](int i) {
        //        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        //        auto f_handle = face_handle(i);
        //        int count = 0;
        //        for (auto fv_it = cfv_iter(f_handle); fv_it.is_valid(); ++fv_it)
        //        {
        //            auto p = point(*fv_it);
        //            center += Eigen::Vector3f(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]));
        //            count++;
        //        }
        //        if (count > 0) center /= static_cast<float>(count);
        //        centers[i] = center;
        //        });

        //    for (const auto& c : centers)
        //    {
        //        grid_min = grid_min.cwiseMin(c);
        //        grid_max = grid_max.cwiseMax(c);
        //    }

        //    Eigen::Vector3f extent = grid_max - grid_min;
        //    grid_min -= extent * 0.01f;
        //    grid_max += extent * 0.01f;
        //    extent = grid_max - grid_min;

        //    // 2. 고정 해상도 대신 "셀의 물리적 크기"를 고정하여 공간 무한대 확장에 대비
        //    // 폴리곤 개수와 전체 부피를 기반으로 이상적인 셀 크기(Cell Size) 산출
        //    float volume = extent.x() * extent.y() * extent.z();
        //    float ideal_cell_vol = volume / (num_faces * 0.5f);
        //    float c_size = std::max(0.1f, std::cbrt(ideal_cell_vol)); // 셀 크기가 0이 되는 것 방지
        //    grid_cell_size = Eigen::Vector3f(c_size, c_size, c_size);

        //    hash_map.clear();
        //    // 데이터가 몰려 해시 충돌이 일어나는 것을 방지하기 위해 넉넉하게 예약
        //    hash_map.reserve(num_faces);

        //    // 3. 폴리곤들을 해당하는 해시 버킷에 쑤셔 넣기 (메모리 낭비 제로)
        //    for (int i = 0; i < num_faces; ++i)
        //    {
        //        Eigen::Vector3f local = centers[i] - grid_min;

        //        // 공간 크기 제약 없이 무한대로 좌표를 뻗어나갈 수 있음
        //        int cx = static_cast<int>(std::floor(local.x() / grid_cell_size.x()));
        //        int cy = static_cast<int>(std::floor(local.y() / grid_cell_size.y()));
        //        int cz = static_cast<int>(std::floor(local.z() / grid_cell_size.z()));

        //        Eigen::Vector3i cell_pos(cx, cy, cz);

        //        // 해시 맵에 삽입
        //        hash_map[cell_pos].push_back(i);
        //    }
        //}

        // 메모리 낭비가 전혀 없는 해시 기반 공간 분할 구축
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

            // 1. 모든 폴리곤의 바운딩 박스 계산 (병렬)
            std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), [&](int i) {
                Eigen::Vector3f v0, v1, v2;
                GetFaceVertices(face_handle(i), v0, v1, v2);

                face_bounds[i].min = v0.cwiseMin(v1).cwiseMin(v2);
                face_bounds[i].max = v0.cwiseMax(v1).cwiseMax(v2);
                });

            // 전체 영역 박스 갱신
            for (const auto& bounds : face_bounds)
            {
                grid_min = grid_min.cwiseMin(bounds.min);
                grid_max = grid_max.cwiseMax(bounds.max);
            }

            Eigen::Vector3f extent = grid_max - grid_min;
            grid_min -= extent * 0.01f;
            grid_max += extent * 0.01f;
            extent = grid_max - grid_min;

            // 2. 이상적인 셀 크기 계산
            float volume = extent.x() * extent.y() * extent.z();
            float ideal_cell_vol = volume / (num_faces * 0.5f);
            float c_size = std::max(0.1f, std::cbrt(ideal_cell_vol));
            grid_cell_size = Eigen::Vector3f(c_size, c_size, c_size);

            hash_map.clear();
            hash_map.reserve(num_faces * 2); // 걸쳐있는 폴리곤을 위해 넉넉하게 예약

            // 3. 폴리곤이 걸쳐있는 "모든" 복셀에 인덱스 삽입 (관통 방지 핵심)
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

        // Raycasting: 공간 제약 없이 무한 맵을 누비는 3D DDA
        bool IntersectGridRay(const Eigen::Vector3f& origin, const Eigen::Vector3f& dir, float& out_t, int& out_face) const
        {
            out_t = std::numeric_limits<float>::max();
            out_face = -1;

            if (hash_map.empty()) return false;

            Eigen::Vector3f inv_dir(
                dir.x() == 0.0f ? 1e-8f : 1.0f / dir.x(),
                dir.y() == 0.0f ? 1e-8f : 1.0f / dir.y(),
                dir.z() == 0.0f ? 1e-8f : 1.0f / dir.z()
            );

            // BBox 교차 검사를 통해 레이가 맵 전체 영역에 진입하는 시작점 찾기
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

            // 시작 셀 좌표
            int cx = static_cast<int>(std::floor(local_pos.x() / grid_cell_size.x()));
            int cy = static_cast<int>(std::floor(local_pos.y() / grid_cell_size.y()));
            int cz = static_cast<int>(std::floor(local_pos.z() / grid_cell_size.z()));

            int stepX = (dir.x() > 0.0f) ? 1 : -1;
            int stepY = (dir.y() > 0.0f) ? 1 : -1;
            int stepZ = (dir.z() > 0.0f) ? 1 : -1;

            float tDeltaX = std::abs(grid_cell_size.x() * inv_dir.x());
            float tDeltaY = std::abs(grid_cell_size.y() * inv_dir.y());
            float tDeltaZ = std::abs(grid_cell_size.z() * inv_dir.z());

            float tMaxX = (stepX > 0) ? ((cx + 1) * grid_cell_size.x() - local_pos.x()) * std::abs(inv_dir.x()) : (local_pos.x() - cx * grid_cell_size.x()) * std::abs(inv_dir.x());
            float tMaxY = (stepY > 0) ? ((cy + 1) * grid_cell_size.y() - local_pos.y()) * std::abs(inv_dir.y()) : (local_pos.y() - cy * grid_cell_size.y()) * std::abs(inv_dir.y());
            float tMaxZ = (stepZ > 0) ? ((cz + 1) * grid_cell_size.z() - local_pos.z()) * std::abs(inv_dir.z()) : (local_pos.z() - cz * grid_cell_size.z()) * std::abs(inv_dir.z());

            bool hit = false;
            float max_t_search = t_exit + 0.1f;
            float current_t = t_enter;

            while (current_t <= max_t_search)
            {
                // [수정된 부분] 현재 셀의 경계를 벗어나는 t 값을 미리 계산
                float next_cell_t = std::min({ tMaxX, tMaxY, tMaxZ });

                // 만약 이전에 발견한 충돌점(out_t)이 이번에 진입할 셀보다 더 가깝다면, 
                // 더 이상 탐색할 필요 없이 가장 가까운 충돌을 찾은 것이므로 즉시 종료합니다.
                if (hit && out_t < current_t)
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
                        Eigen::Vector3f v0, v1, v2;
                        GetFaceVertices(face_handle(f_idx), v0, v1, v2);

                        float t = 0.0f;
                        if (IntersectRayTriangle(origin, dir, v0, v1, v2, t))
                        {
                            // 무조건 가장 작은 t값을 유지하도록 갱신
                            if (t < out_t)
                            {
                                out_t = t;
                                out_face = f_idx;
                                hit = true;
                            }
                        }
                    }
                }

                // 다음 복셀로 전진
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

        // 플립하려는 엣지 주변의 두 삼각형이 볼록(Convex) 사각형을 이루는지 검사
        bool IsConvexQuadrilateral(OpenMesh::EdgeHandle eh) const
        {
            if (is_boundary(eh)) return false;

            auto h0 = halfedge_handle(eh, 0); // V0 -> V1
            auto h1 = halfedge_handle(eh, 1); // V1 -> V0

            // [핵심 수정]: 가운데 대각선(V0-V1)을 제외한 '순수한 바깥 테두리 4개의 정점'을 반시계(CCW) 순서대로 정확히 추출
            auto v_top = to_vertex_handle(next_halfedge_handle(h0)); // 위쪽 꼭짓점
            auto v_left = to_vertex_handle(h1);                       // 왼쪽 꼭짓점 (V0)
            auto v_bottom = to_vertex_handle(next_halfedge_handle(h1)); // 아래쪽 꼭짓점
            auto v_right = to_vertex_handle(h0);                       // 오른쪽 꼭짓점 (V1)

            Eigen::Vector3f p0(point(v_top).data());
            Eigen::Vector3f p1(point(v_left).data());
            Eigen::Vector3f p2(point(v_bottom).data());
            Eigen::Vector3f p3(point(v_right).data());

            // 바깥 테두리를 구성하는 4개의 엣지 벡터
            Eigen::Vector3f e0 = p1 - p0;
            Eigen::Vector3f e1 = p2 - p1;
            Eigen::Vector3f e2 = p3 - p2;
            Eigen::Vector3f e3 = p0 - p3;

            // 면의 대략적인 평균 법선(Normal) 계산
            Eigen::Vector3f n = (e0.cross(e1) + e2.cross(e3)).normalized();

            // 4개의 꼭짓점 내각 부호 확인
            float c0 = e0.cross(e1).dot(n);
            float c1 = e1.cross(e2).dot(n);
            float c2 = e2.cross(e3).dot(n);
            float c3 = e3.cross(e0).dot(n);

            // 약간의 곡면(굴곡)이 있는 경우 부동소수점 오차로 인해 0 이하로 떨어지는 것을 방지하기 위해 마진(-1e-4f) 적용
            const float EPSILON = -1e-4f;
            if (c0 < EPSILON || c1 < EPSILON || c2 < EPSILON || c3 < EPSILON)
            {
                return false;
            }

            return true;
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
            if (u < 0.0f || u > 1.0f) return false;

            Eigen::Vector3f qvec = tvec.cross(edge1);
            float v = direction.dot(qvec) * inv_det;
            if (v < 0.0f || u + v > 1.0f) return false;

            t = edge2.dot(qvec) * inv_det;
            return t > 1e-6f;
        }

    protected:
        Eigen::Vector3f grid_min;
        Eigen::Vector3f grid_max;
        Eigen::Vector3f grid_cell_size;

        // "존재하는 복셀"만 저장하는 1차원 초고속 해시 맵 (메모리 낭비율 0%)
        robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> hash_map;
    };
}
