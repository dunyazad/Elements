#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include "HalfEdgeMesh.hpp"

namespace Eigen
{
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;
    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

void AddMesh(const std::string& name, const HEM::Mesh& mesh, const Eigen::Vector4f& color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f), const Eigen::Vector3f& offset = Eigen::Vector3f::Zero())
{
    auto entity = Helium.CreateEntity(name);
    auto renderable = Helium.CreateComponent<Renderable>(entity);
    renderable->Initialize(Renderable::Triangles);
    renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
    renderable->SetFaceCullingMode(Renderable::BackFace);

    Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [renderable](Entity e, const KeyEvent& event)
        {
            if (event.action == 1 && KeyCode::D1 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::Solid);
            }
            else if (event.action == 1 && KeyCode::D2 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::WireFrame);
            }
            else if (event.action == 1 && KeyCode::D3 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::WireFrameOverSolid);
            }
            else if (event.action == 1 && KeyCode::D4 == event.keyCode)
            {
                renderable->SetDrawingMode(Renderable::Point);
            }
        });

    Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [renderable](Entity e, const MouseButtonEvent& event)
        {
            if (event.action == 1 && event.button == MouseButton::Left)
            {
                auto cameraEntity = Helium.GetEntityByName("MainCamera");
                auto camera = Helium.GetComponent<Camera>(cameraEntity);
                if (nullptr == camera)
                {
                    return;
                }

                Ray ray = camera->ScreenPointToRay(
                    (float)event.xpos,
                    (float)event.ypos,
                    Helium.GetWidth(),
                    Helium.GetHeight()
                );

                int pickedTriangleIndex = -1;
                float pickedDistance = std::numeric_limits<float>::max();

                for (size_t i = 0; i < renderable->GetNumberOfIndices() / 3; i++)
                {
                    auto i0 = renderable->GetIndex(i * 3);
                    auto i1 = renderable->GetIndex(i * 3 + 1);
                    auto i2 = renderable->GetIndex(i * 3 + 2);
                    Eigen::Vector3f v0 = renderable->GetVertex(i0);
                    Eigen::Vector3f v1 = renderable->GetVertex(i1);
                    Eigen::Vector3f v2 = renderable->GetVertex(i2);

                    auto normal = (v1 - v0).cross(v2 - v0).normalized();

                    float t = 0.0f;
                    if (ray.IntersectTriangle(v0, v1, v2, t))
                    {
                        if (t < pickedDistance)
                        {
                            pickedDistance = t;
                            pickedTriangleIndex = (int)i;
                        }
                    }
                }

                if (-1 != pickedTriangleIndex)
                {
                    VD::Clear("HitPoint");
                    VD::Clear("HitTriangle");

                    auto i0 = renderable->GetIndex(pickedTriangleIndex * 3);
                    auto i1 = renderable->GetIndex(pickedTriangleIndex * 3 + 1);
                    auto i2 = renderable->GetIndex(pickedTriangleIndex * 3 + 2);
                    Eigen::Vector3f v0 = renderable->GetVertex(i0);
                    Eigen::Vector3f v1 = renderable->GetVertex(i1);
                    Eigen::Vector3f v2 = renderable->GetVertex(i2);
                    auto normal = (v1 - v0).cross(v2 - v0).normalized();

                    Eigen::Vector3f hitPoint = ray.origin + ray.direction * pickedDistance;
                    VD::AddSphere("HitPoint", hitPoint, normal, 0.01f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));

                    VD::AddTriangle("HitTriangle", v0, v1, v2, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));

                    if (event.modifiers & static_cast<int>(KeyModifiers::Control))
                    {
                        camera->SetTarget(hitPoint);
                    }
                }
            }
        });

    const auto& points = mesh.GetPoints();
    const auto& vertices = mesh.GetVertices();
    const auto& edges = mesh.GetEdges();
    const auto& faces = mesh.GetFaces();

    std::vector<Eigen::Vector3f> positions(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
    {
        positions[i] = points[vertices[i].pid] + offset;
    }

    std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        Eigen::Vector3f normal = (positions[v1] - positions[v0]).cross(positions[v2] - positions[v0]).normalized();
        normals[v0] += normal;
        normals[v1] += normal;
        normals[v2] += normal;
    }

    for (size_t i = 0; i < normals.size(); i++)
    {
        normals[i].normalize();
    }

    std::vector<unsigned int> indices;
    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        indices.push_back((unsigned int)v0);
        indices.push_back((unsigned int)v1);
        indices.push_back((unsigned int)v2);
    }

    renderable->AddVertices(positions);
    renderable->AddNormals(normals);
    renderable->AddColors4(std::vector<Eigen::Vector4f>(positions.size(), color));
    renderable->AddIndices(indices);
}

void RefreshMesh(Entity entity, const HEM::Mesh& mesh, const Eigen::Vector4f& color, const Eigen::Vector3f& offset)
{
    auto renderable = Helium.GetComponent<Renderable>(entity);
    if (nullptr == renderable)
    {
        return;
    }

    const auto& points = mesh.GetPoints();
    const auto& vertices = mesh.GetVertices();
    const auto& edges = mesh.GetEdges();
    const auto& faces = mesh.GetFaces();

    std::vector<Eigen::Vector3f> positions(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
    {
        positions[i] = points[vertices[i].pid] + offset;
    }

    std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());

    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        Eigen::Vector3f normal = (positions[v1] - positions[v0]).cross(positions[v2] - positions[v0]).normalized();
        normals[v0] += normal;
        normals[v1] += normal;
        normals[v2] += normal;
    }

    for (size_t i = 0; i < normals.size(); i++)
    {
        normals[i].normalize();
    }

    std::vector<unsigned int> indices;
    for (const auto& face : faces)
    {
        if (face.eid == HEM::INVALID_ID)
        {
            continue;
        }

        HEM::EID e0 = face.eid;
        HEM::EID e1 = edges[e0].neid;
        HEM::EID e2 = edges[e0].peid;

        HEM::VID v0 = edges[e2].vid;
        HEM::VID v1 = edges[e0].vid;
        HEM::VID v2 = edges[e1].vid;

        indices.push_back((unsigned int)v0);
        indices.push_back((unsigned int)v1);
        indices.push_back((unsigned int)v2);
    }

    renderable->Clear();
    renderable->AddVertices(positions);
    renderable->AddNormals(normals);
    renderable->AddColors4(std::vector<Eigen::Vector4f>(positions.size(), color));
    renderable->AddIndices(indices);
}

class AppHalfEdgeMesh : public App
{
public:
    void Execute_Boolean()
    {
        TS(TotalBoolean);

        TS(LoadMeshes);
        HEM::Mesh mesh_A, mesh_B;
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\Cube.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            mesh_A.Build(stl.GetPoints(), indices);
        }
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\Cylinder.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            mesh_B.Build(stl.GetPoints(), indices);
        }
        TE(LoadMeshes);

        TS(CopyOriginalMeshes);
        HEM::Mesh mesh_A_orig = mesh_A;
        HEM::Mesh mesh_B_orig = mesh_B;
        TE(CopyOriginalMeshes);

        TS(SplitFaces_A);
        mesh_A.SplitIntersectingFaces(mesh_B_orig);
        TE(SplitFaces_A);

        TS(SplitFaces_B);
        mesh_B.SplitIntersectingFaces(mesh_A_orig);
        TE(SplitFaces_B);

        TS(SeparateInsideOutside_A);
        HEM::Mesh A_Inside, A_Outside;
        mesh_A.SeparateByInsideOutside(mesh_B_orig, A_Inside, A_Outside);
        TE(SeparateInsideOutside_A);

        TS(SeparateInsideOutside_B);
        HEM::Mesh B_Inside, B_Outside;
        mesh_B.SeparateByInsideOutside(mesh_A_orig, B_Inside, B_Outside);
        TE(SeparateInsideOutside_B);

        // 절단면(Hole)을 찾아 Ear Clipping으로 뚜껑을 덮는(Capping) 로직이 포함된 Merge
        auto Merge = [](const HEM::Mesh& ma, const HEM::Mesh& mb, bool flipB) -> HEM::Mesh
            {
                std::vector<Eigen::Vector3f> pts;
                std::vector<Eigen::Vector3i> tris;

                // 1. Mesh A (Outside) 복사
                const auto& pa = ma.GetPoints();
                const auto& va = ma.GetVertices();
                const auto& ea = ma.GetEdges();
                const auto& fa = ma.GetFaces();

                for (const auto& p : pa) pts.push_back(p);

                for (const auto& f : fa)
                {
                    if (f.eid == HEM::INVALID_ID) continue;
                    HEM::EID e0 = f.eid;
                    HEM::EID e1 = ea[e0].neid;
                    HEM::EID e2 = ea[e0].peid;
                    int v0 = (int)va[ea[e2].vid].pid;
                    int v1 = (int)va[ea[e0].vid].pid;
                    int v2 = (int)va[ea[e1].vid].pid;
                    tris.push_back(Eigen::Vector3i(v0, v1, v2));
                }

                // 2. Mesh B (Inside) 복사
                const auto& pb = mb.GetPoints();
                const auto& vb = mb.GetVertices();
                const auto& eb = mb.GetEdges();
                const auto& fb = mb.GetFaces();

                int baseB = (int)pts.size();
                for (const auto& p : pb) pts.push_back(p);

                for (const auto& f : fb)
                {
                    if (f.eid == HEM::INVALID_ID) continue;
                    HEM::EID e0 = f.eid;
                    HEM::EID e1 = eb[e0].neid;
                    HEM::EID e2 = eb[e0].peid;
                    int v0 = baseB + (int)vb[eb[e2].vid].pid;
                    int v1 = baseB + (int)vb[eb[e0].vid].pid;
                    int v2 = baseB + (int)vb[eb[e1].vid].pid;

                    if (flipB)
                        tris.push_back(Eigen::Vector3i(v0, v2, v1));
                    else
                        tris.push_back(Eigen::Vector3i(v0, v1, v2));
                }

                // 3. O(N log N) 고속 스냅 (A와 B의 절단면 꼭짓점 일치)
                struct SnapPoint {
                    Eigen::Vector3f p;
                    int idx;
                };

                std::vector<SnapPoint> sortedA(baseB);
                for (int i = 0; i < baseB; ++i) {
                    sortedA[i] = { pts[i], i };
                }

                std::sort(std::execution::par, sortedA.begin(), sortedA.end(), [](const SnapPoint& a, const SnapPoint& b) {
                    return a.p.x() < b.p.x();
                    });

                const float snapDist = 1e-4f;
                const float snapDistSq = snapDist * snapDist;

                std::vector<int> bToA_Map(pts.size() - baseB, -1);

                std::for_each(std::execution::par, pts.begin() + baseB, pts.end(), [&](Eigen::Vector3f& ptB) {
                    int bIdx = &ptB - &pts[baseB];
                    auto it = std::lower_bound(sortedA.begin(), sortedA.end(), ptB.x() - snapDist,
                        [](const SnapPoint& sp, float val) { return sp.p.x() < val; });

                    while (it != sortedA.end() && (it->p.x() - ptB.x()) <= snapDist) {
                        if ((it->p - ptB).squaredNorm() < snapDistSq) {
                            ptB = it->p;
                            bToA_Map[bIdx] = it->idx;
                            break;
                        }
                        ++it;
                    }
                    });

                // 4. Boundary Loop (절단면 테두리) 추출 및 Ear Clipping (Triangulation)
                // Half-Edge 구조를 활용하여 짝이 없는(oeid == INVALID_ID) Edge들을 찾습니다.
                // 편의상 인덱스 매핑을 통해 병합된 엣지들을 추적합니다.

                robin_hood::unordered_flat_map<uint64_t, int> edgeCount;
                for (const auto& tri : tris)
                {
                    int v[3] = { tri.x(), tri.y(), tri.z() };
                    for (int i = 0; i < 3; ++i)
                    {
                        // 스냅된 인덱스로 변환하여 검사 (A에 스냅된 B 꼭짓점은 A의 인덱스를 사용)
                        int idFrom = v[i];
                        if (idFrom >= baseB && bToA_Map[idFrom - baseB] != -1) idFrom = bToA_Map[idFrom - baseB];

                        int idTo = v[(i + 1) % 3];
                        if (idTo >= baseB && bToA_Map[idTo - baseB] != -1) idTo = bToA_Map[idTo - baseB];

                        // 유효하지 않은 엣지(면적 0)는 카운트하지 않음
                        if (idFrom != idTo)
                        {
                            uint64_t key = (uint64_t)std::min(idFrom, idTo) << 32 | std::max(idFrom, idTo);
                            edgeCount[key]++;
                        }
                    }
                }

                // 외곽선(Boundary) 엣지 수집 (한 번만 사용된 엣지)
                std::vector<std::pair<int, int>> boundaryEdges;
                for (const auto& tri : tris)
                {
                    int v[3] = { tri.x(), tri.y(), tri.z() };
                    for (int i = 0; i < 3; ++i)
                    {
                        int idFrom = v[i];
                        if (idFrom >= baseB && bToA_Map[idFrom - baseB] != -1) idFrom = bToA_Map[idFrom - baseB];

                        int idTo = v[(i + 1) % 3];
                        if (idTo >= baseB && bToA_Map[idTo - baseB] != -1) idTo = bToA_Map[idTo - baseB];

                        if (idFrom != idTo)
                        {
                            uint64_t key = (uint64_t)std::min(idFrom, idTo) << 32 | std::max(idFrom, idTo);
                            // 방향성이 중요하므로 순서대로 저장
                            if (edgeCount[key] == 1)
                            {
                                boundaryEdges.push_back({ idFrom, idTo });
                            }
                        }
                    }
                }

                // 연결된 다각형(Loop) 구성
                std::vector<std::vector<int>> loops;
                std::vector<bool> edgeUsed(boundaryEdges.size(), false);

                for (size_t i = 0; i < boundaryEdges.size(); ++i)
                {
                    if (edgeUsed[i]) continue;

                    std::vector<int> currentLoop;
                    currentLoop.push_back(boundaryEdges[i].first);
                    int currentTarget = boundaryEdges[i].second;
                    edgeUsed[i] = true;

                    bool loopClosed = false;
                    while (!loopClosed)
                    {
                        bool foundNext = false;
                        for (size_t j = 0; j < boundaryEdges.size(); ++j)
                        {
                            if (!edgeUsed[j] && boundaryEdges[j].first == currentTarget)
                            {
                                currentLoop.push_back(currentTarget);
                                currentTarget = boundaryEdges[j].second;
                                edgeUsed[j] = true;
                                foundNext = true;

                                if (currentTarget == currentLoop[0])
                                {
                                    loopClosed = true;
                                }
                                break;
                            }
                        }
                        if (!foundNext) break; // 열린 루프(Open Loop)인 경우 비정상 상황이므로 중단
                    }

                    if (loopClosed && currentLoop.size() >= 3)
                    {
                        loops.push_back(currentLoop);
                    }
                }

                // Ear Clipping 구현 (기본적인 O(N^2) 버전 - 복잡한 구멍이 없는 단순 다각형 가정)
                for (const auto& loop : loops)
                {
                    std::vector<int> poly = loop;

                    // 폴리곤의 대략적인 법선 벡터(Normal) 계산 (Ear 판별용)
                    Eigen::Vector3f polyNormal = Eigen::Vector3f::Zero();
                    for (size_t i = 0; i < poly.size(); ++i) {
                        Eigen::Vector3f curr = pts[poly[i]];
                        Eigen::Vector3f next = pts[poly[(i + 1) % poly.size()]];
                        polyNormal.x() += (curr.y() - next.y()) * (curr.z() + next.z());
                        polyNormal.y() += (curr.z() - next.z()) * (curr.x() + next.x());
                        polyNormal.z() += (curr.x() - next.x()) * (curr.y() + next.y());
                    }
                    polyNormal.normalize();

                    // 귀(Ear)인지 판별하는 람다 함수
                    auto isEar = [&](int prev, int curr, int next, const std::vector<int>& pList) {
                        Eigen::Vector3f a = pts[prev];
                        Eigen::Vector3f b = pts[curr];
                        Eigen::Vector3f c = pts[next];

                        // 볼록한 꼭짓점(Convex) 확인
                        Eigen::Vector3f cross = (b - a).cross(c - b);
                        if (cross.dot(polyNormal) <= 0.0f) return false; // 오목(Concave)함

                        // 내부에 다른 꼭짓점이 있는지 확인
                        for (int pIdx : pList) {
                            if (pIdx == prev || pIdx == curr || pIdx == next) continue;
                            Eigen::Vector3f p = pts[pIdx];

                            // Barycentric coordinates로 내부에 있는지 확인
                            Eigen::Vector3f v0 = c - a, v1 = b - a, v2 = p - a;
                            float d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1);
                            float d20 = v2.dot(v0), d21 = v2.dot(v1);
                            float denom = d00 * d11 - d01 * d01;
                            if (denom == 0.0f) continue;

                            float v = (d11 * d20 - d01 * d21) / denom;
                            float w = (d00 * d21 - d01 * d20) / denom;
                            float u = 1.0f - v - w;

                            if (u >= 0.0f && v >= 0.0f && w >= 0.0f) return false; // 다른 꼭짓점이 내부에 있음
                        }
                        return true;
                        };

                    int maxIters = poly.size() * 2;
                    while (poly.size() >= 3 && maxIters-- > 0)
                    {
                        bool earFound = false;
                        for (size_t i = 0; i < poly.size(); ++i)
                        {
                            int prev = poly[(i + poly.size() - 1) % poly.size()];
                            int curr = poly[i];
                            int next = poly[(i + 1) % poly.size()];

                            if (isEar(prev, curr, next, poly))
                            {
                                tris.push_back(Eigen::Vector3i(prev, curr, next));
                                poly.erase(poly.begin() + i);
                                earFound = true;
                                break;
                            }
                        }
                        // 만약 완벽한 Ear를 찾지 못했다면(자기 교차 등), 무리해서 닫지 않고 루프 탈출
                        if (!earFound) break;
                    }
                }

                HEM::Mesh result;
                result.Build(pts, tris);
                return result;
            };

        TS(MergeMeshes);
        HEM::Mesh unionMesh = Merge(A_Outside, B_Outside, false);
        HEM::Mesh subtractionMesh = Merge(A_Outside, B_Inside, true);
        HEM::Mesh intersectionMesh = Merge(A_Inside, B_Inside, false);
        TE(MergeMeshes);

        TS(SaveSTL);
        unionMesh.ToSTL("D:\\temp\\boolean_union.stl");
        subtractionMesh.ToSTL("D:\\temp\\boolean_subtraction.stl");
        intersectionMesh.ToSTL("D:\\temp\\boolean_intersection.stl");
        TE(SaveSTL);

        TS(AddMeshToScene);
        Eigen::Vector3f offset(20.0f, 0.0f, 0.0f);
        AddMesh("Union", unionMesh, Eigen::Vector4f(0.3f, 0.7f, 1.0f, 1.0f), -offset);
        AddMesh("Subtraction", subtractionMesh, Eigen::Vector4f(1.0f, 0.5f, 0.3f, 1.0f), Eigen::Vector3f::Zero());
        AddMesh("Intersection", intersectionMesh, Eigen::Vector4f(0.3f, 1.0f, 0.5f, 1.0f), offset);
        TE(AddMeshToScene);

        TE(TotalBoolean);
    }

    void Execute_Custom()
    {
        TS(TotalBoolean);

        TS(LoadMeshes);
        HEM::Mesh mesh_A, mesh_B;
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            mesh_A.Build(stl.GetPoints(), indices);
        }
        {
            STLFormat stl;
            stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");
            std::vector<Eigen::Vector3i> indices;
            for (size_t i = 0; i < stl.GetPoints().size() / 3; i++)
                indices.push_back(Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
            mesh_B.Build(stl.GetPoints(), indices);
        }
        TE(LoadMeshes);

        auto originalMesh_A = mesh_A;

        // 교차선 추출 및 시각화 (보라색 선)
        auto intersectionLines = mesh_A.FindIntersectionLines(mesh_B);
        for (auto& line : intersectionLines)
        {
            auto p0 = line.first;
            auto p1 = line.second;
            VD::AddLine("IntersectionLines", p0, p1, Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f));
        }

        // 1. 교차선을 기준으로 Mesh_A의 삼각형들을 정밀 분할
        TS(SplitByIntersectionLines);
        //mesh_A.SplitByIntersectionLines(intersectionLines);
        TE(SplitByIntersectionLines);

        TS(MarkSeamEdges);
        mesh_A.MarkSeamEdges(intersectionLines);
        TE(MarkSeamEdges);

        TS(ExtractConnectedComponents);
        auto chunks = mesh_A.ExtractConnectedComponents();
        TE(ExtractConnectedComponents);

        auto colors = Color::GetContrastingColorsWithoutBWRGB(chunks.size() + 1);

        for (auto& mesh : chunks)
        {
            auto& color = colors[&mesh - &chunks[0]];
			AddMesh("Chunk_" + std::to_string(&mesh - &chunks[0]), mesh, color);
        }

#if 0
        // 2. 분할된 Mesh_A의 삼각형들을 mesh_B 영역을 기준으로 내부/외부 덩어리로 완전히 분리
        HEM::Mesh A_Inside, A_Outside;
        mesh_A.SeparateByInsideOutside(mesh_B, A_Inside, A_Outside);

        {
            auto meshes_inside = A_Inside.ExtractConnectedComponents();
            auto colors = Color::GetContrastingColorsWithoutBWRGB(meshes_inside.size() + 1);
            //std::vector<Eigen::Vector4f> colors = { Color::red(), Color::green(), Color::blue(), Color::yellow(), Color::cyan(), Color::magenta(), Color::orange(), Color::purple() };
            for (size_t i = 0; i < meshes_inside.size(); i++)
            {
                auto& mesh = meshes_inside[i];
                auto& color = colors[i % colors.size()];
                AddMesh("mesh_inside_A_Component_" + std::to_string(i), mesh, color);
            }
        }
        {
            auto meshes_outside = A_Outside.ExtractConnectedComponents();
            auto colors = Color::GetContrastingColorsWithoutBWRGB(meshes_outside.size() + 1);
            //std::vector<Eigen::Vector4f> colors = { Color::red(), Color::green(), Color::blue(), Color::yellow(), Color::cyan(), Color::magenta(), Color::orange(), Color::purple() };
            std::reverse(colors.begin(), colors.end()); // 내부와 외부가 색상이 겹치지 않도록 반전
            for (size_t i = 0; i < meshes_outside.size(); i++)
            {
                auto& mesh = meshes_outside[i];
                auto& color = colors[i % colors.size()];
                AddMesh("mesh_outside_A_Component_" + std::to_string(i), mesh, color);
            }
        }
#endif // 0

        // 자르는 기준이 된 mesh_B는 내부 덩어리를 잘 볼 수 있도록 반투명(Alpha 0.2) 혹은 연한 색상으로 렌더링
        //AddMesh("mesh_B", mesh_B, Eigen::Vector4f(0.8f, 0.8f, 0.8f, 0.2f));

        TE(TotalBoolean);
    }

    virtual void Execute() override
    {
        //Execute_Boolean();

        Execute_Custom();
    }
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
