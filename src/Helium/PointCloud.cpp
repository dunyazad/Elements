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
    if (m_isLoading) return false; // Already loading

    m_isLoading = true;

    // Launch async task for IO and CPU heavy math
    m_loadingFuture = std::async(std::launch::async, [filename]() -> ProcessedInstanceData {
        ProcessedInstanceData data;
        PLYFormat ply;

        // IO Operation
        ply.Deserialize(filename);
        ply.SwapAxisYZ();

        data.pointCount = ply.GetPoints().size();

        // Pre-allocate memory to avoid reallocation overhead
        data.normals.reserve(data.pointCount);
        data.colors.reserve(data.pointCount);
        data.transforms.reserve(data.pointCount);

        // CPU Heavy Math Loop
        for (size_t i = 0; i < data.pointCount; i++)
        {
            const Eigen::Vector3f& p = ply.GetPoints()[i];
            Eigen::Vector3f n = (i < ply.GetNormals().size()) ? ply.GetNormals()[i] : Eigen::Vector3f::Zero();
            Eigen::Vector4f c = (i < ply.GetColors().size()) ? ply.GetColors()[i] : Eigen::Vector4f::Ones();

            data.normals.push_back(n);
            data.colors.push_back(c);

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

            data.transforms.push_back(tm.matrix());
        }

        return data;
        });

    return true;
}

void PointCloud::UpdateLoading()
{
    if (m_isLoading && m_loadingFuture.valid())
    {
        // Check if the async task is ready (non-blocking check)
        if (m_loadingFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            // Get the data from the thread
            ProcessedInstanceData data = m_loadingFuture.get();

            // --- Main Thread Operations (GPU / Entity Creation) ---

            if (InvalidEntity == entity)
            {
                entity = Helium.CreateEntity("PointCloud");
                entityName = "PointCloud";
            }

            // Register Events
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

            renderable = Helium.CreateComponent<Renderable>(entity);
            renderable->Initialize(Renderable::GeometryMode::Triangles);

            // Shader Setup
            renderable->AddShader(Helium.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
            renderable->AddShader(Helium.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
            renderable->SetActiveShaderIndex(1);

            // Geometry Setup
            GeometryBuilder::BuildSphere(renderable, { 0.0f, 0.0f, 0.0f }, 0.5f, 6, 6, { 1.0f, 1.0f, 1.0f, 1.0f });

            // Apply processed data to Renderable
            for (size_t i = 0; i < data.pointCount; i++)
            {
                renderable->AddInstanceNormal(data.normals[i]);
                renderable->AddInstanceColor(data.colors[i]);
                renderable->AddInstanceTransform(data.transforms[i]);
                renderable->IncreaseNumberOfInstances();
            }

            renderable->EnableInstancing();

            // Cleanup
            m_isLoading = false;
        }
    }
}

bool PointCloud::SetVisible(bool isVisible)
{
    if (renderable)
    {
        renderable->SetVisible(isVisible);
        return true;
    }
    return false;
}
