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

#include <manifold/manifold.h>

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

class AppManifold : public App
{
public:
	virtual void Initialize() override
	{
	}

	virtual void Execute() override
	{
		TS(LoadingMeshA);
		manifold::MeshGL meshA;
		{
			STLFormat stl;
			stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit.stl");

			meshA.numProp = 3;
			meshA.vertProperties.resize(stl.GetPoints().size() * 3);
			for (size_t i = 0; i < stl.GetPoints().size(); i++)
			{
				auto p = stl.GetPoints()[i];
				meshA.vertProperties[i * 3] = p.x();
				meshA.vertProperties[i * 3 + 1] = p.y();
				meshA.vertProperties[i * 3 + 2] = p.z();
			}
			meshA.triVerts.resize(stl.GetPoints().size());
			std::iota(meshA.triVerts.begin(), meshA.triVerts.end(), 0);

			meshA.Merge();
		}
		TE(LoadingMeshA);

		TS(LoadingMeshB);
		manifold::MeshGL meshB;
		{
			STLFormat stl;
			stl.Deserialize("D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");

			meshB.numProp = 3;
			meshB.vertProperties.resize(stl.GetPoints().size() * 3);
			for (size_t i = 0; i < stl.GetPoints().size(); i++)
			{
				auto p = stl.GetPoints()[i];
				meshB.vertProperties[i * 3] = p.x();
				meshB.vertProperties[i * 3 + 1] = p.y();
				meshB.vertProperties[i * 3 + 2] = p.z();
			}
			meshB.triVerts.resize(stl.GetPoints().size());
			std::iota(meshB.triVerts.begin(), meshB.triVerts.end(), 0);

			meshB.Merge();
		}
		TE(LoadingMeshB);

		TS(Subtract);
		manifold::Manifold manifoldA(meshA);
		manifold::Manifold manifoldB(meshB);

		manifold::Manifold result = manifoldA - manifoldB;
		TE(Subtract);

		{
			manifold::MeshGL outMesh = result.GetMeshGL();

			STLFormat stl;
			for (size_t i = 0; i < outMesh.NumTri(); i++)
			{
				auto i0 = outMesh.triVerts[i * 3];
				auto i1 = outMesh.triVerts[i * 3 + 1];
				auto i2 = outMesh.triVerts[i * 3 + 2];

				auto v0 = Eigen::Vector3f(outMesh.vertProperties[i0 * 3], outMesh.vertProperties[i0 * 3 + 1], outMesh.vertProperties[i0 * 3 + 2]);
				auto v1 = Eigen::Vector3f(outMesh.vertProperties[i1 * 3], outMesh.vertProperties[i1 * 3 + 1], outMesh.vertProperties[i1 * 3 + 2]);
				auto v2 = Eigen::Vector3f(outMesh.vertProperties[i2 * 3], outMesh.vertProperties[i2 * 3 + 1], outMesh.vertProperties[i2 * 3 + 2]);

				stl.AddTriangle(v0, v1, v2);
			}
			stl.Serialize("D:\\Temp\\result.stl");
		}

		manifold::MeshGL outMesh = result.GetMeshGL();

		auto entity = Helium.CreateEntity("Mesh");
		auto renderable = Helium.CreateComponent<Renderable>(entity);
		renderable->Initialize(Renderable::Triangles);
		renderable->AddShader(Helium.CreateShader("TwoSide", File("../../res/Shaders/TwoSide.vs"), File("../../res/Shaders/TwoSide.fs")));
		renderable->SetFaceCullingMode(Renderable::NoCulling);

		std::vector<Eigen::Vector3f> positions(outMesh.vertProperties.size() / 3);
		std::vector<Eigen::Vector3f> normals(outMesh.vertProperties.size() / 3);
		std::vector<Eigen::Vector4f> colors(outMesh.vertProperties.size() / 3);
		std::vector<unsigned int> indices(outMesh.triVerts.size());

		for (size_t i = 0; i < outMesh.NumTri(); i++)
		{
			auto i0 = outMesh.triVerts[i * 3];
			auto i1 = outMesh.triVerts[i * 3 + 1];
			auto i2 = outMesh.triVerts[i * 3 + 2];

			auto v0 = Eigen::Vector3f(outMesh.vertProperties[i0 * 3], outMesh.vertProperties[i0 * 3 + 1], outMesh.vertProperties[i0 * 3 + 2]);
			auto v1 = Eigen::Vector3f(outMesh.vertProperties[i1 * 3], outMesh.vertProperties[i1 * 3 + 1], outMesh.vertProperties[i1 * 3 + 2]);
			auto v2 = Eigen::Vector3f(outMesh.vertProperties[i2 * 3], outMesh.vertProperties[i2 * 3 + 1], outMesh.vertProperties[i2 * 3 + 2]);

			positions[i0] = v0;
			positions[i1] = v1;
			positions[i2] = v2;

			auto normal = (v1 - v0).cross(v2 - v0).normalized();
			normals[i0] = normal;
			normals[i1] = normal;
			normals[i2] = normal;

			colors[i0] = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);
			colors[i1] = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);
			colors[i2] = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

			indices[i * 3] = i0;
			indices[i * 3 + 1] = i1;
			indices[i * 3 + 2] = i2;

			//VD::AddTriangle("MeshA", v0, v1, v2, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
		}

		renderable->AddVertices(positions);
		renderable->AddNormals(normals);
		renderable->AddColors4(colors);
		renderable->AddIndices(indices);

		Helium.CreateEventCallback<KeyEvent>(entity, "3D", [renderable](Entity e, const KeyEvent& event) {
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
	}
};

REGISTER_APP(AppManifold, "AppManifold");
