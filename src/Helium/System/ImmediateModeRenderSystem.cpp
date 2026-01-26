#include "pch.h"
#include <Helium/Systems/ImmediateModeRenderSystem.h>
#include <Helium/HeliumCore.h>
#include <Helium/Components/Components.h>
#include <Helium/Components/Camera.h>
#include <glad/glad.h>
#include <Eigen/Dense>

ImmediateModeRenderSystem::ImmediateModeRenderSystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

ImmediateModeRenderSystem::~ImmediateModeRenderSystem()
{
}

void ImmediateModeRenderSystem::Initialize()
{
}

void ImmediateModeRenderSystem::Update(float dt)
{
    if (!enabled || !core) return;

    glUseProgram(0);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, Helium.GetWidth(), Helium.GetHeight());

    glPointSize(10.0f);
    glLineWidth(2.0f);

    if (gridGizmoEnabled)
    {
        glPushMatrix();

        Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), gridNormal.normalized());
        Eigen::Matrix4f gridTransform = Eigen::Matrix4f::Identity();
        gridTransform.block<3, 3>(0, 0) = rotation.toRotationMatrix();

        glMultMatrixf(gridTransform.data());

        glColor4f(gridGizmoColor.x(), gridGizmoColor.y(), gridGizmoColor.z(), gridGizmoColor.w());
        glBegin(GL_LINES);
        for (float i = -100.0f; i <= 100.0f; i += gridSpacing)
        {
            // Lines parallel to X-axis
            glVertex3f(-100.0f, i, 0.0f);
            glVertex3f(100.0f, i, 0.0f);
            // Lines parallel to Y-axis
            glVertex3f(i, -100.0f, 0.0f);
            glVertex3f(i, 100.0f, 0.0f);
        }
        glEnd();
        glPopMatrix();
    }

    if (axisGizmoEnabled)
    {
        // Draw X-axis (Red)
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(-100.f, 0.0f, 0.0f);
        glVertex3f(100.0f, 0.0f, 0.0f);
        glEnd();

        // Draw Y-axis (Green)
        glBegin(GL_LINES);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, -100.0f, 0.0f);
        glVertex3f(0.0f, 100.0f, 0.0f);
        glEnd();

        // Draw Z-axis (Blue)
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, -100.0f);
        glVertex3f(0.0f, 0.0f, 100.0f);
        glEnd();
    }

    if (centerGizmoEnabled)
    {
        auto& registry = Helium.GetRegistry();
        auto entites = registry.view<Camera>();
        for (auto& entity : entites)
        {
            auto& camera = entites.get<Camera>(entity);

            const auto& projection = camera.GetProjectionMatrix();
            glMatrixMode(GL_PROJECTION);
            glLoadMatrixf(projection.data());

            const auto& view = camera.GetViewMatrix();
            auto target = camera.GetTarget();

            glMatrixMode(GL_MODELVIEW);
            glLoadMatrixf(view.data());

            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_POINTS);
            glVertex3f(target.x(), target.y(), target.z());
            glEnd();

            // Draw X-axis (Red) at Target
            glBegin(GL_LINES);
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(target.x() - 0.5f, target.y(), target.z());
            glVertex3f(target.x() + 1.0f, target.y(), target.z());
            glEnd();

            // Draw Y-axis (Green) at Target
            glBegin(GL_LINES);
            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3f(target.x(), target.y() - 0.5f, target.z());
            glVertex3f(target.x(), target.y() + 1.0f, target.z());
            glEnd();

            // Draw Z-axis (Blue) at Target
            glBegin(GL_LINES);
            glColor3f(0.0f, 0.0f, 1.0f);
            glVertex3f(target.x(), target.y(), target.z() - 0.5f);
            glVertex3f(target.x(), target.y(), target.z() + 1.0f);
            glEnd();
        }
    }

    glEnable(GL_DEPTH_TEST);
}
