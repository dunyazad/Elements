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
	}
};

REGISTER_APP(AppHalfEdgeMesh, "AppHalfEdgeMesh");
