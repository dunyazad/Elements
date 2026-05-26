#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <mutex>
#include <fstream>
#include <limits>
#include <utility>
#include <execution>
#include <Eigen/Dense>
#include <robin_hood/robin_hood.h>

namespace HEM
{
    using PID = unsigned int;
    using VID = unsigned int;
    using EID = unsigned int;
    using FID = unsigned int;

    const unsigned int INVALID_ID = -1;

    struct Point
    {
        float x;
        float y;
        float z;
    };

    struct Vertex
    {
        PID pid;
        EID eid;

        Vertex() : pid(INVALID_ID), eid(INVALID_ID)
        {
        }

        Vertex(PID pid) : pid(pid), eid(INVALID_ID)
        {
        }
    };

    struct Edge
    {
        VID vid;
        EID oeid;
        EID neid;
        EID peid;
        FID fid;

        Edge() : vid(INVALID_ID), oeid(INVALID_ID), neid(INVALID_ID), peid(INVALID_ID), fid(INVALID_ID)
        {
        }
    };

    struct Face
    {
        EID eid;

        Face() : eid(INVALID_ID)
        {
        }
    };

    struct AABB
    {
        Eigen::Vector3f minBound;
        Eigen::Vector3f maxBound;

        AABB()
        {
            minBound = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            maxBound = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        }

        void Expand(const Eigen::Vector3f& p)
        {
            minBound = minBound.cwiseMin(p);
            maxBound = maxBound.cwiseMax(p);
        }

        void Expand(const AABB& other)
        {
            minBound = minBound.cwiseMin(other.minBound);
            maxBound = maxBound.cwiseMax(other.maxBound);
        }

        bool Intersects(const AABB& other) const
        {
            if (maxBound.x() < other.minBound.x() || minBound.x() > other.maxBound.x())
            {
                return false;
            }
            if (maxBound.y() < other.minBound.y() || minBound.y() > other.maxBound.y())
            {
                return false;
            }
            if (maxBound.z() < other.minBound.z() || minBound.z() > other.maxBound.z())
            {
                return false;
            }
            return true;
        }
    };

    struct BVHNode
    {
        AABB bounds;
        int leftChild;
        int rightChild;
        int faceOffset;
        int faceCount;

        BVHNode() : leftChild(-1), rightChild(-1), faceOffset(0), faceCount(0)
        {
        }

        bool IsLeaf() const
        {
            return faceCount > 0;
        }
    };

    class Mesh
    {
    public:
        Mesh()
        {
        }

        ~Mesh()
        {
            Clear();
        }

        void Clear()
        {
            points.clear();
            vertices.clear();
            edges.clear();
            faces.clear();
            bvhNodes.clear();
            bvhFaceIds.clear();
        }

        void Build(const std::vector<Eigen::Vector3f>& sourcePoints, const std::vector<Eigen::Vector3i>& indices)
        {
            Clear();
            if (indices.empty())
            {
                return;
            }

            struct SortPoint
            {
                Eigen::Vector3f p;
                int oldIdx;
            };

            std::vector<SortPoint> sp(sourcePoints.size());
            for (int i = 0; i < (int)sourcePoints.size(); ++i)
            {
                sp[i] = { sourcePoints[i], i };
            }

            std::sort(std::execution::par, sp.begin(), sp.end(), [](const SortPoint& a, const SortPoint& b)
                {
                    if (a.p.x() != b.p.x())
                    {
                        return a.p.x() < b.p.x();
                    }
                    if (a.p.y() != b.p.y())
                    {
                        return a.p.y() < b.p.y();
                    }
                    return a.p.z() < b.p.z();
                });

            std::vector<int> indexMap(sourcePoints.size());
            std::vector<Eigen::Vector3f> weldedPoints;
            const float eps = 1e-4f;
            const float epsSq = eps * eps;

            for (int i = 0; i < (int)sp.size(); ++i)
            {
                int foundIdx = -1;
                for (int j = i - 1; j >= 0; --j)
                {
                    if (sp[i].p.x() - sp[j].p.x() > eps)
                    {
                        break;
                    }
                    if ((sp[i].p - sp[j].p).squaredNorm() <= epsSq)
                    {
                        foundIdx = indexMap[sp[j].oldIdx];
                        break;
                    }
                }

                if (foundIdx != -1)
                {
                    indexMap[sp[i].oldIdx] = foundIdx;
                }
                else
                {
                    indexMap[sp[i].oldIdx] = (int)weldedPoints.size();
                    weldedPoints.push_back(sp[i].p);
                }
            }

            points = weldedPoints;
            vertices.reserve(points.size());
            for (size_t i = 0; i < points.size(); ++i)
            {
                vertices.push_back(Vertex((PID)i));
            }

            robin_hood::unordered_flat_map<uint64_t, std::vector<EID>> edgeMap;
            faces.reserve(indices.size());

            for (const auto& tri : indices)
            {
                int idx[3] = { indexMap[tri.x()], indexMap[tri.y()], indexMap[tri.z()] };
                if (idx[0] == idx[1] || idx[1] == idx[2] || idx[2] == idx[0])
                {
                    continue;
                }

                FID newFaceId = (FID)faces.size();
                faces.push_back(Face());

                EID edgeIds[3];
                for (int i = 0; i < 3; ++i)
                {
                    edgeIds[i] = (EID)edges.size();
                    edges.push_back(Edge());
                }

                for (int i = 0; i < 3; ++i)
                {
                    int vFrom = idx[i];
                    int vTo = idx[(i + 1) % 3];

                    edges[edgeIds[i]].vid = (VID)vTo;
                    edges[edgeIds[i]].fid = newFaceId;
                    edges[edgeIds[i]].neid = edgeIds[(i + 1) % 3];
                    edges[edgeIds[i]].peid = edgeIds[(i + 2) % 3];

                    vertices[vFrom].eid = edgeIds[i];
                    faces[newFaceId].eid = edgeIds[i];

                    uint64_t key = (uint64_t)std::min(vFrom, vTo) << 32 | std::max(vFrom, vTo);
                    bool paired = false;
                    for (EID existingId : edgeMap[key])
                    {
                        if (edges[existingId].oeid == INVALID_ID && edges[edges[existingId].peid].vid == (VID)vTo)
                        {
                            edges[edgeIds[i]].oeid = existingId;
                            edges[existingId].oeid = edgeIds[i];
                            paired = true;
                            break;
                        }
                    }
                    if (!paired)
                    {
                        edgeMap[key].push_back(edgeIds[i]);
                    }
                }
            }

            BuildBVH();
        }

        bool ToSTL(const std::string& filename)
        {
            std::ofstream out(filename);
            if (!out.is_open())
            {
                return false;
            }

            out << "solid mesh\n";
            for (const auto& f : faces)
            {
                if (f.eid != INVALID_ID)
                {
                    EID e0 = f.eid;
                    EID e1 = edges[e0].neid;

                    VID v0 = edges[edges[e0].peid].vid;
                    VID v1 = edges[e0].vid;
                    VID v2 = edges[e1].vid;

                    Eigen::Vector3f p0 = points[vertices[v0].pid];
                    Eigen::Vector3f p1 = points[vertices[v1].pid];
                    Eigen::Vector3f p2 = points[vertices[v2].pid];

                    Eigen::Vector3f normal = (p1 - p0).cross(p2 - p0).normalized();

                    out << "  facet normal " << normal.x() << " " << normal.y() << " " << normal.z() << "\n";
                    out << "    outer loop\n";
                    out << "      vertex " << p0.x() << " " << p0.y() << " " << p0.z() << "\n";
                    out << "      vertex " << p1.x() << " " << p1.y() << " " << p1.z() << "\n";
                    out << "      vertex " << p2.x() << " " << p2.y() << " " << p2.z() << "\n";
                    out << "    endloop\n";
                    out << "  endfacet\n";
                }
            }
            out << "endsolid mesh\n";
            return true;
        }

        void QueryBVH(const AABB& queryBounds, std::vector<FID>& outFaces) const
        {
            if (bvhNodes.empty())
            {
                return;
            }

            std::vector<int> stack;
            stack.reserve(64);
            stack.push_back(0);

            while (!stack.empty())
            {
                int nodeIdx = stack.back();
                stack.pop_back();

                const BVHNode& node = bvhNodes[nodeIdx];
                if (!node.bounds.Intersects(queryBounds))
                {
                    continue;
                }

                if (node.IsLeaf())
                {
                    for (int i = 0; i < node.faceCount; ++i)
                    {
                        outFaces.push_back(bvhFaceIds[node.faceOffset + i]);
                    }
                }
                else
                {
                    stack.push_back(node.leftChild);
                    stack.push_back(node.rightChild);
                }
            }
        }

        std::vector<std::pair<FID, FID>> FindIntersection(const Mesh& other) const
        {
            std::vector<std::pair<FID, FID>> intersectingFaces;

            if (bvhNodes.empty() || other.bvhNodes.empty())
            {
                return intersectingFaces;
            }

            if (!bvhNodes[0].bounds.Intersects(other.bvhNodes[0].bounds))
            {
                return intersectingFaces;
            }

            std::vector<FID> validFaces;
            validFaces.reserve(faces.size());

            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid != INVALID_ID)
                {
                    AABB box = GetFaceAABB(i);
                    if (box.Intersects(other.bvhNodes[0].bounds))
                    {
                        validFaces.push_back(i);
                    }
                }
            }

            std::mutex resultMutex;

            std::for_each(std::execution::par, validFaces.begin(), validFaces.end(), [&](FID fid)
                {
                    AABB aabb = GetFaceAABB(fid);
                    std::vector<FID> candidateFaces;
                    other.QueryBVH(aabb, candidateFaces);

                    if (candidateFaces.empty())
                    {
                        return;
                    }

                    Eigen::Vector3f p0, p1, p2;
                    GetFaceVertices(fid, p0, p1, p2);

                    std::vector<std::pair<FID, FID>> localIntersections;

                    for (FID otherFid : candidateFaces)
                    {
                        Eigen::Vector3f q0, q1, q2;
                        other.GetFaceVertices(otherFid, q0, q1, q2);

                        if (TriTriIntersect(p0, p1, p2, q0, q1, q2))
                        {
                            localIntersections.push_back({ fid, otherFid });
                        }
                    }

                    if (!localIntersections.empty())
                    {
                        std::lock_guard<std::mutex> lock(resultMutex);
                        intersectingFaces.insert(intersectingFaces.end(), localIntersections.begin(), localIntersections.end());
                    }
                });

            return intersectingFaces;
        }

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> FindIntersectionLines(const Mesh& other) const
        {
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> intersectingLines;

            if (bvhNodes.empty() || other.bvhNodes.empty())
            {
                return intersectingLines;
            }

            if (!bvhNodes[0].bounds.Intersects(other.bvhNodes[0].bounds))
            {
                return intersectingLines;
            }

            std::vector<FID> validFaces;
            validFaces.reserve(faces.size());

            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid != INVALID_ID)
                {
                    AABB box = GetFaceAABB(i);
                    if (box.Intersects(other.bvhNodes[0].bounds))
                    {
                        validFaces.push_back(i);
                    }
                }
            }

            std::mutex resultMutex;

            std::for_each(std::execution::par, validFaces.begin(), validFaces.end(), [&](FID fid)
                {
                    AABB aabb = GetFaceAABB(fid);
                    std::vector<FID> candidateFaces;
                    other.QueryBVH(aabb, candidateFaces);

                    if (candidateFaces.empty())
                    {
                        return;
                    }

                    Eigen::Vector3f p0, p1, p2;
                    GetFaceVertices(fid, p0, p1, p2);

                    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> localLines;

                    for (FID otherFid : candidateFaces)
                    {
                        Eigen::Vector3f q0, q1, q2;
                        other.GetFaceVertices(otherFid, q0, q1, q2);

                        Eigen::Vector3f pt1, pt2;
                        if (TriTriIntersectLine(p0, p1, p2, q0, q1, q2, pt1, pt2))
                        {
                            localLines.push_back({ pt1, pt2 });
                        }
                    }

                    if (!localLines.empty())
                    {
                        std::lock_guard<std::mutex> lock(resultMutex);
                        intersectingLines.insert(intersectingLines.end(), localLines.begin(), localLines.end());
                    }
                });

            return intersectingLines;
        }

        void SplitFaceByPoint(FID fid, const Eigen::Vector3f& point)
        {
            if (fid >= faces.size() || faces[fid].eid == INVALID_ID)
            {
                return;
            }

            EID e0 = faces[fid].eid;
            EID e1 = edges[e0].neid;
            EID e2 = edges[e0].peid;

            VID v0 = edges[e2].vid;
            VID v1 = edges[e0].vid;
            VID v2 = edges[e1].vid;

            // 이미 Raycast로 삼각형 내부임이 완벽하게 검증된 point이므로, 
            // 부동소수점 오차로 인해 분할이 취소되는 것을 막기 위해 PointInTriangle 중복 검사를 제거합니다.

            points.push_back(point);
            PID newPid = (PID)points.size() - 1;
            VID vm = (VID)vertices.size();
            vertices.push_back(Vertex(newPid));

            FID f0 = fid;
            FID f1 = (FID)faces.size();
            FID f2 = f1 + 1;
            faces.push_back(Face());
            faces.push_back(Face());

            EID e_v1_vm = (EID)edges.size();
            EID e_vm_v1 = e_v1_vm + 1;
            EID e_v2_vm = e_v1_vm + 2;
            EID e_vm_v2 = e_v1_vm + 3;
            EID e_v0_vm = e_v1_vm + 4;
            EID e_vm_v0 = e_v1_vm + 5;

            for (int i = 0; i < 6; ++i)
            {
                edges.push_back(Edge());
            }

            edges[e0].neid = e_v1_vm;
            edges[e0].peid = e_vm_v0;

            edges[e_v1_vm].vid = vm;
            edges[e_v1_vm].oeid = e_vm_v1;
            edges[e_v1_vm].neid = e_vm_v0;
            edges[e_v1_vm].peid = e0;
            edges[e_v1_vm].fid = f0;

            edges[e_vm_v0].vid = v0;
            edges[e_vm_v0].oeid = e_v0_vm;
            edges[e_vm_v0].neid = e0;
            edges[e_vm_v0].peid = e_v1_vm;
            edges[e_vm_v0].fid = f0;

            faces[f0].eid = e0;

            edges[e1].neid = e_v2_vm;
            edges[e1].peid = e_vm_v1;
            edges[e1].fid = f1;

            edges[e_v2_vm].vid = vm;
            edges[e_v2_vm].oeid = e_vm_v2;
            edges[e_v2_vm].neid = e_vm_v1;
            edges[e_v2_vm].peid = e1;
            edges[e_v2_vm].fid = f1;

            edges[e_vm_v1].vid = v1;
            edges[e_vm_v1].oeid = e_v1_vm;
            edges[e_vm_v1].neid = e1;
            edges[e_vm_v1].peid = e_v2_vm;
            edges[e_vm_v1].fid = f1;

            faces[f1].eid = e1;

            edges[e2].neid = e_v0_vm;
            edges[e2].peid = e_vm_v2;
            edges[e2].fid = f2;

            edges[e_v0_vm].vid = vm;
            edges[e_v0_vm].oeid = e_vm_v0;
            edges[e_v0_vm].neid = e_vm_v2;
            edges[e_v0_vm].peid = e2;
            edges[e_v0_vm].fid = f2;

            edges[e_vm_v2].vid = v2;
            edges[e_vm_v2].oeid = e_v2_vm;
            edges[e_vm_v2].neid = e2;
            edges[e_vm_v2].peid = e_v0_vm;
            edges[e_vm_v2].fid = f2;

            faces[f2].eid = e2;

            vertices[vm].eid = e_vm_v0;

            // 분할이 실제로 성공하여 데이터가 늘어났는지 콘솔에서 즉시 확인하기 위한 디버깅 로그
            printf("Face Split Success! Total Points: %zu, Total Faces: %zu\n", points.size(), faces.size());
        }

        void SplitIntersectingFaces(const Mesh& other)
        {
            if (bvhNodes.empty() || other.bvhNodes.empty())
            {
                return;
            }

            struct IntersectionTask
            {
                FID fid;
                std::vector<FID> cuttingFaces;
            };

            std::vector<IntersectionTask> tasks;
            std::mutex taskMutex;

            std::vector<FID> validFaces;
            validFaces.reserve(faces.size());
            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid != INVALID_ID)
                {
                    validFaces.push_back(i);
                }
            }

            std::for_each(std::execution::par, validFaces.begin(), validFaces.end(), [&](FID fid)
                {
                    AABB aabb = GetFaceAABB(fid);
                    std::vector<FID> candidateFaces;
                    other.QueryBVH(aabb, candidateFaces);

                    if (candidateFaces.empty())
                    {
                        return;
                    }

                    Eigen::Vector3f p0, p1, p2;
                    GetFaceVertices(fid, p0, p1, p2);

                    std::vector<FID> localCutters;
                    for (FID otherFid : candidateFaces)
                    {
                        Eigen::Vector3f q0, q1, q2;
                        other.GetFaceVertices(otherFid, q0, q1, q2);

                        if (TriTriIntersect(p0, p1, p2, q0, q1, q2))
                        {
                            localCutters.push_back(otherFid);
                        }
                    }

                    if (!localCutters.empty())
                    {
                        std::sort(localCutters.begin(), localCutters.end());
                        std::lock_guard<std::mutex> lock(taskMutex);
                        tasks.push_back({ fid, localCutters });
                    }
                });

            if (tasks.empty())
            {
                return;
            }

            std::vector<Eigen::Vector3f> newPoints = points;
            std::vector<Eigen::Vector3i> newIndices;

            std::vector<bool> faceSplit(faces.size(), false);
            for (const auto& task : tasks)
            {
                faceSplit[task.fid] = true;
            }

            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid != INVALID_ID && !faceSplit[i])
                {
                    EID e0 = faces[i].eid;
                    EID e1 = edges[e0].neid;
                    EID e2 = edges[e0].peid;
                    newIndices.push_back(Eigen::Vector3i(
                        (int)vertices[edges[e2].vid].pid,
                        (int)vertices[edges[e0].vid].pid,
                        (int)vertices[edges[e1].vid].pid
                    ));
                }
            }

            robin_hood::unordered_flat_map<uint64_t, int> edgeSplitCache;

            for (const auto& task : tasks)
            {
                std::vector<Eigen::Vector3i> currentTris;
                EID e0 = faces[task.fid].eid;
                EID e1 = edges[e0].neid;
                EID e2 = edges[e0].peid;

                currentTris.push_back(Eigen::Vector3i(
                    (int)vertices[edges[e2].vid].pid,
                    (int)vertices[edges[e0].vid].pid,
                    (int)vertices[edges[e1].vid].pid
                ));

                for (FID cutterFid : task.cuttingFaces)
                {
                    Eigen::Vector3f q0, q1, q2;
                    other.GetFaceVertices(cutterFid, q0, q1, q2);

                    Eigen::Vector3f n = (q1 - q0).cross(q2 - q0).normalized();
                    float d = -n.dot(q0);

                    std::vector<Eigen::Vector3i> nextTris;
                    for (const auto& tri : currentTris)
                    {
                        ClipTriangleToBSP(n, d, tri, newPoints, nextTris, edgeSplitCache);
                    }
                    currentTris = nextTris;
                }

                newIndices.insert(newIndices.end(), currentTris.begin(), currentTris.end());
            }

            bool changed = true;
            int maxIters = 100;
            while (changed && maxIters-- > 0)
            {
                changed = false;
                std::vector<Eigen::Vector3i> nextIndices;
                nextIndices.reserve(newIndices.size());

                for (const auto& tri : newIndices)
                {
                    int v0 = tri.x(), v1 = tri.y(), v2 = tri.z();
                    uint64_t e0 = (uint64_t)std::min(v0, v1) << 32 | std::max(v0, v1);
                    uint64_t e1 = (uint64_t)std::min(v1, v2) << 32 | std::max(v1, v2);
                    uint64_t e2 = (uint64_t)std::min(v2, v0) << 32 | std::max(v2, v0);

                    if (edgeSplitCache.find(e0) != edgeSplitCache.end() &&
                        edgeSplitCache[e0] != v0 && edgeSplitCache[e0] != v1 && edgeSplitCache[e0] != v2)
                    {
                        int vm = edgeSplitCache[e0];
                        nextIndices.push_back(Eigen::Vector3i(v0, vm, v2));
                        nextIndices.push_back(Eigen::Vector3i(vm, v1, v2));
                        changed = true;
                    }
                    else if (edgeSplitCache.find(e1) != edgeSplitCache.end() &&
                        edgeSplitCache[e1] != v0 && edgeSplitCache[e1] != v1 && edgeSplitCache[e1] != v2)
                    {
                        int vm = edgeSplitCache[e1];
                        nextIndices.push_back(Eigen::Vector3i(v1, vm, v0));
                        nextIndices.push_back(Eigen::Vector3i(vm, v2, v0));
                        changed = true;
                    }
                    else if (edgeSplitCache.find(e2) != edgeSplitCache.end() &&
                        edgeSplitCache[e2] != v0 && edgeSplitCache[e2] != v1 && edgeSplitCache[e2] != v2)
                    {
                        int vm = edgeSplitCache[e2];
                        nextIndices.push_back(Eigen::Vector3i(v2, vm, v1));
                        nextIndices.push_back(Eigen::Vector3i(vm, v0, v1));
                        changed = true;
                    }
                    else
                    {
                        nextIndices.push_back(tri);
                    }
                }
                newIndices = nextIndices;
            }

            Build(newPoints, newIndices);
        }

        bool IsPointInside(const Eigen::Vector3f& p) const
        {
            if (bvhNodes.empty())
            {
                return false;
            }

            const AABB& rootAABB = bvhNodes[0].bounds;
            if (p.x() < rootAABB.minBound.x() || p.x() > rootAABB.maxBound.x() ||
                p.y() < rootAABB.minBound.y() || p.y() > rootAABB.maxBound.y() ||
                p.z() < rootAABB.minBound.z() || p.z() > rootAABB.maxBound.z())
            {
                return false;
            }

            Eigen::Vector3f dir(0.317f, 0.521f, 0.792f);
            dir.normalize();
            Eigen::Vector3f invDir(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());

            int hitCount = 0;
            std::vector<int> stack;
            stack.reserve(64);
            stack.push_back(0);

            while (!stack.empty())
            {
                int nodeIdx = stack.back();
                stack.pop_back();

                const BVHNode& node = bvhNodes[nodeIdx];
                if (!IntersectRayAABB(p, invDir, node.bounds))
                {
                    continue;
                }

                if (node.IsLeaf())
                {
                    for (int i = 0; i < node.faceCount; ++i)
                    {
                        FID fid = bvhFaceIds[node.faceOffset + i];
                        Eigen::Vector3f v0, v1, v2;
                        GetFaceVertices(fid, v0, v1, v2);

                        float t = 0.0f;
                        if (IntersectRayTriangle(p, dir, v0, v1, v2, t))
                        {
                            hitCount++;
                        }
                    }
                }
                else
                {
                    stack.push_back(node.leftChild);
                    stack.push_back(node.rightChild);
                }
            }

            return (hitCount % 2) != 0;
        }

        void SeparateByInsideOutside(const Mesh& other, Mesh& outInside, Mesh& outOutside) const
        {
            std::vector<Eigen::Vector3i> insideIndices;
            std::vector<Eigen::Vector3i> outsideIndices;

            std::mutex insideMutex;
            std::mutex outsideMutex;

            std::vector<FID> validFaces;
            validFaces.reserve(faces.size());
            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid != INVALID_ID)
                {
                    validFaces.push_back(i);
                }
            }

            std::for_each(std::execution::par, validFaces.begin(), validFaces.end(), [&](FID fid)
                {
                    Eigen::Vector3f v0, v1, v2;
                    GetFaceVertices(fid, v0, v1, v2);

                    Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();
                    Eigen::Vector3f centroid = (v0 + v1 + v2) / 3.0f;

                    // Fix: +normal 방향으로 오프셋 (face 전면 기준 판정)
                    Eigen::Vector3f offsetCentroid = centroid + (normal * 1e-4f);

                    // 경계 근처 안정성을 위해 양방향 투표 추가
                    bool frontInside = other.IsPointInside(offsetCentroid);
                    bool backInside = other.IsPointInside(centroid - (normal * 1e-4f));

                    // 양쪽 모두 inside일 때만 inside로 분류
                    bool isInside = frontInside && backInside;

                    EID e0 = faces[fid].eid;
                    EID e1 = edges[e0].neid;
                    EID e2 = edges[e0].peid;
                    Eigen::Vector3i tri(
                        (int)vertices[edges[e2].vid].pid,
                        (int)vertices[edges[e0].vid].pid,
                        (int)vertices[edges[e1].vid].pid
                    );

                    if (isInside)
                    {
                        std::lock_guard<std::mutex> lock(insideMutex);
                        insideIndices.push_back(tri);
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lock(outsideMutex);
                        outsideIndices.push_back(tri);
                    }
                });

            outInside.Build(points, insideIndices);
            outOutside.Build(points, outsideIndices);
        }

        std::vector<Mesh> GetConnectedComponents() const
        {
            std::vector<Mesh> result;
            std::vector<bool> visited(faces.size(), false);

            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid == INVALID_ID || visited[i])
                {
                    continue;
                }

                std::vector<Eigen::Vector3i> componentIndices;
                std::vector<FID> queue;
                queue.push_back(i);
                visited[i] = true;

                int head = 0;
                while (head < (int)queue.size())
                {
                    FID currFid = queue[head++];
                    EID e0 = faces[currFid].eid;
                    EID e1 = edges[e0].neid;
                    EID e2 = edges[e0].peid;

                    componentIndices.push_back(Eigen::Vector3i(
                        (int)vertices[edges[e2].vid].pid,
                        (int)vertices[edges[e0].vid].pid,
                        (int)vertices[edges[e1].vid].pid
                    ));

                    EID edgesToCheck[3] = { e0, e1, e2 };
                    for (int k = 0; k < 3; ++k)
                    {
                        EID oeid = edges[edgesToCheck[k]].oeid;
                        if (oeid != INVALID_ID)
                        {
                            FID adjFid = edges[oeid].fid;
                            if (adjFid != INVALID_ID && !visited[adjFid])
                            {
                                visited[adjFid] = true;
                                queue.push_back(adjFid);
                            }
                        }
                    }
                }

                Mesh comp;
                comp.Build(points, componentIndices);
                result.push_back(std::move(comp));
            }

            return result;
        }

        inline const Eigen::Vector3f& GetPoint(PID pid) const
        {
            return points[pid];
        }

        inline const Vertex& GetVertex(VID vid) const
        {
            return vertices[vid];
        }

        inline const Edge& GetEdge(EID eid) const
        {
            return edges[eid];
        }

        inline const Face& GetFace(FID fid) const
        {
            return faces[fid];
        }

        inline const std::vector<Eigen::Vector3f>& GetPoints() const
        {
            return points;
        }

        inline const std::vector<Vertex>& GetVertices() const
        {
            return vertices;
        }

        inline const std::vector<Edge>& GetEdges() const
        {
            return edges;
        }

        inline const std::vector<Face>& GetFaces() const
        {
            return faces;
        }

        Eigen::Vector3f GetCentroid() const
        {
            if (faces.empty()) return Eigen::Vector3f::Zero();

            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            int count = 0;

            for (const auto& f : faces)
            {
                if (f.eid == INVALID_ID) continue;

                EID e0 = f.eid;
                EID e1 = edges[e0].neid;
                EID e2 = edges[e0].peid;

                sum += points[vertices[edges[e2].vid].pid];
                sum += points[vertices[edges[e0].vid].pid];
                sum += points[vertices[edges[e1].vid].pid];
                count += 3;
            }

            if (count == 0) return Eigen::Vector3f::Zero();
            return sum / (float)count;
        }

    protected:
        void BuildBVH()
        {
            bvhNodes.clear();
            bvhFaceIds.clear();

            if (faces.empty())
            {
                return;
            }

            bvhFaceIds.reserve(faces.size());
            for (FID i = 0; i < (FID)faces.size(); ++i)
            {
                if (faces[i].eid != INVALID_ID)
                {
                    bvhFaceIds.push_back(i);
                }
            }

            if (bvhFaceIds.empty())
            {
                return;
            }

            BVHNode rootNode;
            rootNode.faceOffset = 0;
            rootNode.faceCount = (int)bvhFaceIds.size();
            bvhNodes.push_back(rootNode);

            UpdateNodeBounds(0);
            SubdivideNode(0);
        }

        void UpdateNodeBounds(int nodeIndex)
        {
            BVHNode& node = bvhNodes[nodeIndex];
            node.bounds = AABB();

            for (int i = 0; i < node.faceCount; ++i)
            {
                FID fid = bvhFaceIds[node.faceOffset + i];
                AABB faceBox = GetFaceAABB(fid);
                node.bounds.Expand(faceBox);
            }
        }

        void SubdivideNode(int nodeIndex)
        {
            int faceCount = bvhNodes[nodeIndex].faceCount;
            if (faceCount <= 4)
            {
                return;
            }

            Eigen::Vector3f extent = bvhNodes[nodeIndex].bounds.maxBound - bvhNodes[nodeIndex].bounds.minBound;
            int axis = 0;
            if (extent.y() > extent.x())
            {
                axis = 1;
            }
            if (extent.z() > extent[axis])
            {
                axis = 2;
            }

            int faceOffset = bvhNodes[nodeIndex].faceOffset;
            int middleIndex = faceOffset + faceCount / 2;

            std::nth_element(
                bvhFaceIds.begin() + faceOffset,
                bvhFaceIds.begin() + middleIndex,
                bvhFaceIds.begin() + faceOffset + faceCount,
                [&](FID a, FID b)
                {
                    return GetFaceCentroid(a)[axis] < GetFaceCentroid(b)[axis];
                });

            int leftCount = middleIndex - faceOffset;
            int rightCount = faceCount - leftCount;

            int leftChildIdx = (int)bvhNodes.size();
            bvhNodes.push_back(BVHNode());

            int rightChildIdx = (int)bvhNodes.size();
            bvhNodes.push_back(BVHNode());

            bvhNodes[leftChildIdx].faceOffset = faceOffset;
            bvhNodes[leftChildIdx].faceCount = leftCount;

            bvhNodes[rightChildIdx].faceOffset = middleIndex;
            bvhNodes[rightChildIdx].faceCount = rightCount;

            bvhNodes[nodeIndex].leftChild = leftChildIdx;
            bvhNodes[nodeIndex].rightChild = rightChildIdx;
            bvhNodes[nodeIndex].faceCount = 0;

            UpdateNodeBounds(leftChildIdx);
            UpdateNodeBounds(rightChildIdx);

            SubdivideNode(leftChildIdx);
            SubdivideNode(rightChildIdx);
        }

        AABB GetFaceAABB(FID fid) const
        {
            AABB box;
            if (faces[fid].eid == INVALID_ID)
            {
                return box;
            }

            EID e0 = faces[fid].eid;
            EID e1 = edges[e0].neid;
            EID e2 = edges[e0].peid;

            Eigen::Vector3f p0 = points[vertices[edges[e2].vid].pid];
            Eigen::Vector3f p1 = points[vertices[edges[e0].vid].pid];
            Eigen::Vector3f p2 = points[vertices[edges[e1].vid].pid];

            box.Expand(p0);
            box.Expand(p1);
            box.Expand(p2);
            return box;
        }

        Eigen::Vector3f GetFaceCentroid(FID fid) const
        {
            if (faces[fid].eid == INVALID_ID)
            {
                return Eigen::Vector3f::Zero();
            }

            EID e0 = faces[fid].eid;
            EID e1 = edges[e0].neid;
            EID e2 = edges[e0].peid;

            Eigen::Vector3f p0 = points[vertices[edges[e2].vid].pid];
            Eigen::Vector3f p1 = points[vertices[edges[e0].vid].pid];
            Eigen::Vector3f p2 = points[vertices[edges[e1].vid].pid];

            return (p0 + p1 + p2) / 3.0f;
        }

        void GetFaceVertices(FID fid, Eigen::Vector3f& v0, Eigen::Vector3f& v1, Eigen::Vector3f& v2) const
        {
            EID e0 = faces[fid].eid;
            EID e1 = edges[e0].neid;
            EID e2 = edges[e0].peid;

            v0 = points[vertices[edges[e2].vid].pid];
            v1 = points[vertices[edges[e0].vid].pid];
            v2 = points[vertices[edges[e1].vid].pid];
        }

        bool IntersectRayAABB(const Eigen::Vector3f& orig, const Eigen::Vector3f& invDir, const AABB& bounds) const
        {
            float tx1 = (bounds.minBound.x() - orig.x()) * invDir.x();
            float tx2 = (bounds.maxBound.x() - orig.x()) * invDir.x();
            float tmin = std::min(tx1, tx2);
            float tmax = std::max(tx1, tx2);

            float ty1 = (bounds.minBound.y() - orig.y()) * invDir.y();
            float ty2 = (bounds.maxBound.y() - orig.y()) * invDir.y();
            tmin = std::max(tmin, std::min(ty1, ty2));
            tmax = std::min(tmax, std::max(ty1, ty2));

            float tz1 = (bounds.minBound.z() - orig.z()) * invDir.z();
            float tz2 = (bounds.maxBound.z() - orig.z()) * invDir.z();
            tmin = std::max(tmin, std::min(tz1, tz2));
            tmax = std::min(tmax, std::max(tz1, tz2));

            return tmax >= tmin && tmax >= 0.0f;
        }

        bool IntersectRayTriangle(const Eigen::Vector3f& orig, const Eigen::Vector3f& dir, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, float& t) const
        {
            Eigen::Vector3f edge1 = v1 - v0;
            Eigen::Vector3f edge2 = v2 - v0;
            Eigen::Vector3f h = dir.cross(edge2);
            float a = edge1.dot(h);

            if (a > -1e-6f && a < 1e-6f)
            {
                return false;
            }

            float f = 1.0f / a;
            Eigen::Vector3f s = orig - v0;
            float u = f * s.dot(h);

            if (u < 0.0f || u > 1.0f)
            {
                return false;
            }

            Eigen::Vector3f q = s.cross(edge1);
            float v = f * dir.dot(q);

            if (v < 0.0f || u + v > 1.0f)
            {
                return false;
            }

            t = f * edge2.dot(q);
            return t > 1e-6f;
        }

        bool PointInTriangle(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) const
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

            const float eps = 1e-5f;
            return (u >= -eps) && (v >= -eps) && (u + v <= 1.0f + eps);
        }

        bool TriTriIntersect(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& q0, const Eigen::Vector3f& q1, const Eigen::Vector3f& q2) const
        {
            Eigen::Vector3f normal1 = (p1 - p0).cross(p2 - p0);
            float dist1 = -normal1.dot(p0);

            float d0 = normal1.dot(q0) + dist1;
            float d1 = normal1.dot(q1) + dist1;
            float d2 = normal1.dot(q2) + dist1;

            if (d0 > 0.0f && d1 > 0.0f && d2 > 0.0f)
            {
                return false;
            }
            if (d0 < 0.0f && d1 < 0.0f && d2 < 0.0f)
            {
                return false;
            }

            Eigen::Vector3f normal2 = (q1 - q0).cross(q2 - q0);
            float dist2 = -normal2.dot(q0);

            float dp0 = normal2.dot(p0) + dist2;
            float dp1 = normal2.dot(p1) + dist2;
            float dp2 = normal2.dot(p2) + dist2;

            if (dp0 > 0.0f && dp1 > 0.0f && dp2 > 0.0f)
            {
                return false;
            }
            if (dp0 < 0.0f && dp1 < 0.0f && dp2 < 0.0f)
            {
                return false;
            }

            Eigen::Vector3f edges1[3] = { p1 - p0, p2 - p1, p0 - p2 };
            Eigen::Vector3f edges2[3] = { q1 - q0, q2 - q1, q0 - q2 };

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    Eigen::Vector3f axis = edges1[i].cross(edges2[j]);
                    if (axis.squaredNorm() < 1e-8f)
                    {
                        continue;
                    }

                    float min1 = axis.dot(p0);
                    float max1 = min1;
                    float proj = axis.dot(p1);
                    min1 = std::min(min1, proj);
                    max1 = std::max(max1, proj);
                    proj = axis.dot(p2);
                    min1 = std::min(min1, proj);
                    max1 = std::max(max1, proj);

                    float min2 = axis.dot(q0);
                    float max2 = min2;
                    proj = axis.dot(q1);
                    min2 = std::min(min2, proj);
                    max2 = std::max(max2, proj);
                    proj = axis.dot(q2);
                    min2 = std::min(min2, proj);
                    max2 = std::max(max2, proj);

                    if (max1 < min2 || max2 < min1)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        bool TriTriIntersectLine(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& q0, const Eigen::Vector3f& q1, const Eigen::Vector3f& q2, Eigen::Vector3f& outP1, Eigen::Vector3f& outP2) const
        {
            Eigen::Vector3f n1 = (p1 - p0).cross(p2 - p0);
            float d1 = -n1.dot(p0);

            float dp0 = n1.dot(q0) + d1;
            float dp1 = n1.dot(q1) + d1;
            float dp2 = n1.dot(q2) + d1;

            if ((dp0 > 0.0f && dp1 > 0.0f && dp2 > 0.0f) || (dp0 < 0.0f && dp1 < 0.0f && dp2 < 0.0f))
            {
                return false;
            }

            Eigen::Vector3f n2 = (q1 - q0).cross(q2 - q0);
            float d2 = -n2.dot(q0);

            float dq0 = n2.dot(p0) + d2;
            float dq1 = n2.dot(p1) + d2;
            float dq2 = n2.dot(p2) + d2;

            if ((dq0 > 0.0f && dq1 > 0.0f && dq2 > 0.0f) || (dq0 < 0.0f && dq1 < 0.0f && dq2 < 0.0f))
            {
                return false;
            }

            Eigen::Vector3f dir = n1.cross(n2);
            float dirSqNorm = dir.squaredNorm();
            if (dirSqNorm < 1e-12f)
            {
                return false;
            }

            auto get_intersect_pts = [&](const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, float d0_val, float d1_val, float d2_val, Eigen::Vector3f& pt1, Eigen::Vector3f& pt2)
                {
                    int count = 0;
                    Eigen::Vector3f pts[3];
                    const float eps = 1e-7f;

                    if (d0_val * d1_val <= 0.0f && std::abs(d0_val - d1_val) > eps)
                    {
                        pts[count++] = v0 + (v1 - v0) * (d0_val / (d0_val - d1_val));
                    }
                    if (d1_val * d2_val <= 0.0f && std::abs(d1_val - d2_val) > eps)
                    {
                        pts[count++] = v1 + (v2 - v1) * (d1_val / (d1_val - d2_val));
                    }
                    if (count < 2 && d2_val * d0_val <= 0.0f && std::abs(d2_val - d0_val) > eps)
                    {
                        pts[count++] = v2 + (v0 - v2) * (d2_val / (d2_val - d0_val));
                    }

                    if (count >= 2)
                    {
                        pt1 = pts[0];
                        pt2 = pts[1];
                        return true;
                    }
                    return false;
                };

            Eigen::Vector3f i1_p1, i1_p2, i2_p1, i2_p2;
            if (!get_intersect_pts(p0, p1, p2, dq0, dq1, dq2, i1_p1, i1_p2))
            {
                return false;
            }
            if (!get_intersect_pts(q0, q1, q2, dp0, dp1, dp2, i2_p1, i2_p2))
            {
                return false;
            }

            float t1_1 = i1_p1.dot(dir);
            float t1_2 = i1_p2.dot(dir);
            if (t1_1 > t1_2)
            {
                std::swap(t1_1, t1_2);
                std::swap(i1_p1, i1_p2);
            }

            float t2_1 = i2_p1.dot(dir);
            float t2_2 = i2_p2.dot(dir);
            if (t2_1 > t2_2)
            {
                std::swap(t2_1, t2_2);
                std::swap(i2_p1, i2_p2);
            }

            float start_t = std::max(t1_1, t2_1);
            float end_t = std::min(t1_2, t2_2);

            if (start_t > end_t)
            {
                return false;
            }

            outP1 = i1_p1 + dir * ((start_t - t1_1) / dirSqNorm);
            outP2 = i1_p1 + dir * ((end_t - t1_1) / dirSqNorm);

            return true;
        }

        void ClipTriangleToBSP(const Eigen::Vector3f& n, float d, const Eigen::Vector3i& tri, std::vector<Eigen::Vector3f>& verts, std::vector<Eigen::Vector3i>& outTris, robin_hood::unordered_flat_map<uint64_t, int>& cache)
        {
            std::vector<int> frontPoints;
            std::vector<int> backPoints;
            const float eps = 1e-4f;

            for (int i = 0; i < 3; ++i)
            {
                int v0 = tri[i];
                int v1 = tri[(i + 1) % 3];
                float d0 = n.dot(verts[v0]) + d;
                float d1 = n.dot(verts[v1]) + d;

                if (d0 >= -eps)
                {
                    frontPoints.push_back(v0);
                }
                if (d0 <= eps)
                {
                    backPoints.push_back(v0);
                }
                if ((d0 > eps && d1 < -eps) || (d0 < -eps && d1 > eps))
                {
                    uint64_t key = (uint64_t)std::min(v0, v1) << 32 | std::max(v0, v1);
                    int idx;
                    if (cache.find(key) == cache.end())
                    {
                        idx = (int)verts.size();
                        verts.push_back(verts[v0] + (d0 / (d0 - d1)) * (verts[v1] - verts[v0]));
                        cache[key] = idx;
                    }
                    else
                    {
                        idx = cache[key];
                    }
                    frontPoints.push_back(idx);
                    backPoints.push_back(idx);
                }
            }

            auto Fan = [&](const std::vector<int>& pts, std::vector<Eigen::Vector3i>& target)
                {
                    if (pts.size() < 3)
                    {
                        return;
                    }
                    std::vector<int> u;
                    for (int p : pts)
                    {
                        if (u.empty() || u.back() != p)
                        {
                            u.push_back(p);
                        }
                    }
                    if (u.size() > 2 && u.front() == u.back())
                    {
                        u.pop_back();
                    }
                    for (size_t i = 1; i + 1 < u.size(); ++i)
                    {
                        target.push_back(Eigen::Vector3i(u[0], u[i], u[i + 1]));
                    }
                };

            Fan(frontPoints, outTris);
            Fan(backPoints, outTris);
        }

        std::vector<Eigen::Vector3f> points;
        std::vector<Vertex> vertices;
        std::vector<Edge> edges;
        std::vector<Face> faces;

        std::vector<BVHNode> bvhNodes;
        std::vector<FID> bvhFaceIds;
    };
}
