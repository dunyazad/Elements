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


namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

#include <vector>
#include <map>
#include <utility>
#include <Eigen/Dense>

namespace HEM
{
	struct Face;
	struct Edge;

	struct Vertex
	{
		Eigen::Vector3f position;
		Edge* edge;

		Vertex() : position(0.0f, 0.0f, 0.0f), edge(nullptr)
		{
		}

		Vertex(const Eigen::Vector3f& pos) : position(pos), edge(nullptr)
		{
		}
	};

	struct Edge
	{
		Vertex* vertex;
		Edge* pair;
		Edge* next;
		Edge* prev;
		Face* face;

		Edge() : vertex(nullptr), pair(nullptr), next(nullptr), prev(nullptr), face(nullptr)
		{
		}
	};

	struct Face
	{
		Edge* edge;

		Face() : edge(nullptr)
		{
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
			for (auto v : vertices) delete v;
			for (auto e : edges) delete e;
			for (auto f : faces) delete f;

			vertices.clear();
			edges.clear();
			faces.clear();
		}

		void Build(const std::vector<Eigen::Vector3f>& inPoints, const std::vector<Eigen::Vector3i>& inIndices)
		{
			Clear();

			vertices.reserve(inPoints.size());
			for (const auto& p : inPoints)
			{
				vertices.push_back(new Vertex(p));
			}

			std::map<std::pair<int, int>, Edge*> edgeMap;

			faces.reserve(inIndices.size());
			for (const auto& tri : inIndices)
			{
				Face* newFace = new Face();
				faces.push_back(newFace);

				Edge* e[3];
				for (int i = 0; i < 3; ++i)
				{
					e[i] = new Edge();
					edges.push_back(e[i]);
				}

				for (int i = 0; i < 3; ++i)
				{
					int idx0 = tri[i];
					int idx1 = tri[(i + 1) % 3];

					e[i]->vertex = vertices[idx1];
					e[i]->face = newFace;
					e[i]->next = e[(i + 1) % 3];
					e[i]->prev = e[(i + 2) % 3];

					vertices[idx0]->edge = e[i];
					newFace->edge = e[i];

					std::pair<int, int> currentKey = { idx0, idx1 };
					std::pair<int, int> flipKey = { idx1, idx0 };

					if (edgeMap.find(flipKey) != edgeMap.end())
					{
						Edge* twin = edgeMap[flipKey];
						e[i]->pair = twin;
						twin->pair = e[i];
					}

					edgeMap[currentKey] = e[i];
				}
			}
		}

		const std::vector<Vertex*>& GetVertices() const
		{
			return vertices;
		}

		const std::vector<Edge*>& GetEdges() const
		{
			return edges;
		}

		const std::vector<Face*>& GetFaces() const
		{
			return faces;
		}

	protected:
		std::vector<Vertex*> vertices;
		std::vector<Edge*> edges;
		std::vector<Face*> faces;
	};
}

class AppHalfEdgeMesh : public App
{
public:
	virtual void Execute() override
	{
		auto entity = Helium.CreateEntity("Mesh");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		renderable->AddShader(Helium.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
		renderable->SetFaceCullingMode(Renderable::NoCulling);

		STLFormat stl;
		stl.Deserialize("D:\\Temp\\Upper(Lilivis).stl");

		std::vector<Eigen::Vector3f> normals;
		for (size_t i = 0; i < stl.GetPoints().size(); i += 3)
		{
			Eigen::Vector3f v0 = stl.GetPoints()[i];
			Eigen::Vector3f v1 = stl.GetPoints()[i + 1];
			Eigen::Vector3f v2 = stl.GetPoints()[i + 2];
			Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();
			normals.push_back(normal);
			normals.push_back(normal);
			normals.push_back(normal);
		}

		std::vector<unsigned int> indices(stl.GetPoints().size());
		std::iota(indices.begin(), indices.end(), 0);

		renderable->AddVertices(stl.GetPoints());
		renderable->AddNormals(normals);
		renderable->AddColors4(std::vector<Eigen::Vector4f>(stl.GetPoints().size(), Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f)));
		renderable->AddIndices(indices);

		Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [renderable](Entity e, const KeyEvent& event) {
			if (event.action == 1 && KeyCode::D1 == event.keyCode)
			{
				renderable->SetDrawingMode(Renderable::Solid);
			}
			else if(event.action == 1 && KeyCode::D2 == event.keyCode)
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

		//renderable->SetVisible(false);

		HEM::Mesh mesh;
		std::vector<Eigen::Vector3i> meshIndices;
		for (size_t i = 0; i < indices.size() / 3; i++)
		{
			meshIndices.push_back(Eigen::Vector3i(indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]));
		}

		mesh.Build(stl.GetPoints(), meshIndices);

		for (auto& face : mesh.GetFaces())
		{
			auto v0 = face->edge->vertex->position;
			auto v1 = face->edge->next->vertex->position;
			auto v2 = face->edge->prev->vertex->position;

			//VD::AddTriangle("Mesh", v0, v1, v2, Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
		}

		for (auto& edge : mesh.GetEdges())
		{
			//if (edge->pair == nullptr)
			{
				auto v0 = edge->vertex->position;
				auto v1 = edge->prev->vertex->position;
				VD::AddLine("Mesh", v0, v1, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
			}
		}
	}
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
