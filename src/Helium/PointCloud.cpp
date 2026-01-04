#include "pch.h"

#include <Helium/PointCloud.h>
#include <Helium/HeliumCore.h>
#include <Helium/GeometryBuilder.h>
#include <Helium/Serialization.hpp>

PointCloud::PointCloud()
{
}

PointCloud::~PointCloud()
{
}

bool PointCloud::LoadFromPLY(const std::string& filename)
{
	if(InvalidEntity == entity)
	{
		entity = Helium.CreateEntity("PointCloud");
		entityName = "PointCloud";
		renderable = Helium.CreateComponent<Renderable>(entity);
	}

    auto entity = Helium.CreateEntity("PointCloud");

    Helium.CreateEventCallback<KeyEvent>(entity, [](Entity entity, const KeyEvent& event) {
        auto renderable = Helium.GetComponent<Renderable>(entity);
        if (nullptr == renderable) return;

        if (0 == event.action)
        {
            if (GLFW_KEY_GRAVE_ACCENT == event.keyCode)
            {
                renderable->NextDrawingMode();
            }
            else if (GLFW_KEY_1 == event.keyCode)
            {
                renderable->SetActiveShaderIndex(0);
            }
            else if (GLFW_KEY_2 == event.keyCode)
            {
                renderable->SetActiveShaderIndex(1);
            }
        }
        });

    auto renderable = Helium.CreateComponent<Renderable>(entity);

    PLYFormat ply;
    ply.Deserialize(filename);
    ply.SwapAxisYZ();

    renderable->Initialize(Renderable::GeometryMode::Triangles);

    renderable->AddShader(Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
    renderable->AddShader(Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
    renderable->SetActiveShaderIndex(1);

    GeometryBuilder::BuildSphere(renderable, { 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6, { 1.0f, 1.0f, 1.0f, 1.0f });

    size_t pointCount = ply.GetPoints().size();

    for (size_t i = 0; i < pointCount; i++)
    {
        const Eigen::Vector3f& p = ply.GetPoints()[i];

        Eigen::Vector3f n = (i < ply.GetNormals().size()) ? ply.GetNormals()[i] : Eigen::Vector3f::Zero();

        renderable->AddInstanceNormal(n);

        Eigen::Vector4f c = (i < ply.GetColors().size()) ? ply.GetColors()[i] : Eigen::Vector4f::Ones();

        renderable->AddInstanceColor(c);
        // Deep Learning Classes
        // if (!ply.GetDeepLearningClasses().empty())
        // {
        // 	pointCloud.SetPointDeepLearningClassID(i, ply.GetDeepLearningClasses()[i]);
        // }

        Eigen::Affine3f tm = Eigen::Affine3f::Identity();
        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();

        if (n.norm() > 0.0001f)
        {
            Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
            Eigen::Vector3f normalDir = n.normalized();

            Eigen::Vector3f axis = up.cross(normalDir);
            float dot = up.dot(normalDir);

            if (dot > 1.0f) dot = 1.0f;
            else if (dot < -1.0f) dot = -1.0f;

            float angle = std::acos(dot);

            if (axis.norm() > 0.0001f)
            {
                axis.normalize();
                rot = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
            }
            else if (dot < -0.9f)
            {
                rot = Eigen::AngleAxisf(3.1415926f, Eigen::Vector3f::UnitX()).toRotationMatrix();
            }
        }

        tm.translate(p);
        tm.rotate(rot);
        tm.scale(0.1f);

        renderable->AddInstanceTransform(tm.matrix());
        renderable->IncreaseNumberOfInstances();
    }

    renderable->EnableInstancing();

	return true;
}
