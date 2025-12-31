#include "pch.h"
#include <Helium/Components/Camera.h>

Camera::Camera()
{
    UpdateViewMatrix();
    UpdateProjectionMatrix();
}

Camera::~Camera()
{
}

void Camera::SetProjectionMode(ProjectionMode mode)
{
    if (this->mode != mode)
    {
        this->mode = mode;
        dirty = true;
    }
}

void Camera::Update(unsigned int frameNo, float timeDelta)
{
    if (perspectiveSettings.IsDirty())
    {
        dirty = true;
        perspectiveSettings.SetDirty(false);
    }
    if (orthogonalSettings.IsDirty())
    {
        dirty = true;
        orthogonalSettings.SetDirty(false);
    }

    if (dirty)
    {
        UpdateViewMatrix();
        UpdateProjectionMatrix();
        dirty = false;
    }
}

void Camera::UpdateViewMatrix()
{
    Eigen::Vector3f f = (target - eye).normalized();
    Eigen::Vector3f u = up.normalized();
    Eigen::Vector3f s = f.cross(u).normalized();
    u = s.cross(f);

    viewMatrix = Eigen::Matrix4f::Identity();

    viewMatrix(0, 0) = s.x(); viewMatrix(0, 1) = s.y(); viewMatrix(0, 2) = s.z();
    viewMatrix(1, 0) = u.x(); viewMatrix(1, 1) = u.y(); viewMatrix(1, 2) = u.z();
    viewMatrix(2, 0) = -f.x(); viewMatrix(2, 1) = -f.y(); viewMatrix(2, 2) = -f.z();

    viewMatrix(0, 3) = -s.dot(eye);
    viewMatrix(1, 3) = -u.dot(eye);
    viewMatrix(2, 3) = f.dot(eye);
}

void Camera::UpdateProjectionMatrix()
{
    if (mode == Perspective)
    {
        float tanHalfFovy = tan(perspectiveSettings.fovy / 2.0f);
        float aspect = perspectiveSettings.aspectRatio;
        float zNear = perspectiveSettings.zNear;
        float zFar = perspectiveSettings.zFar;

        projectionMatrix = Eigen::Matrix4f::Zero();
        projectionMatrix(0, 0) = 1.0f / (aspect * tanHalfFovy);
        projectionMatrix(1, 1) = 1.0f / (tanHalfFovy);
        projectionMatrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
        projectionMatrix(3, 2) = -1.0f;
        projectionMatrix(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
    }
    else
    {
        float left = orthogonalSettings.left;
        float right = orthogonalSettings.right;
        float bottom = orthogonalSettings.bottom;
        float top = orthogonalSettings.top;
        float zNear = orthogonalSettings.zNear;
        float zFar = orthogonalSettings.zFar;

        projectionMatrix = Eigen::Matrix4f::Identity();
        projectionMatrix(0, 0) = 2.0f / (right - left);
        projectionMatrix(1, 1) = 2.0f / (top - bottom);
        projectionMatrix(2, 2) = -2.0f / (zFar - zNear);

        projectionMatrix(0, 3) = -(right + left) / (right - left);
        projectionMatrix(1, 3) = -(top + bottom) / (top - bottom);
        projectionMatrix(2, 3) = -(zFar + zNear) / (zFar - zNear);
    }
}

Ray Camera::ScreenPointToRay(float mouseX, float mouseY, int screenWidth, int screenHeight)
{
    float ndcX = (2.0f * mouseX) / (float)screenWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * mouseY) / (float)screenHeight;

    Eigen::Vector4f clipCoords(ndcX, ndcY, -1.0f, 1.0f);

    Eigen::Vector4f eyeCoords = projectionMatrix.inverse() * clipCoords;
    eyeCoords.z() = -1.0f;
    eyeCoords.w() = 0.0f;

    Eigen::Vector4f worldCoords = viewMatrix.inverse() * eyeCoords;
    Eigen::Vector3f rayDir = Eigen::Vector3f(worldCoords.x(), worldCoords.y(), worldCoords.z()).normalized();

    Eigen::Vector3f rayOrigin;

    if (mode == Perspective)
    {
        rayOrigin = eye;
    }
    else
    {
        Eigen::Vector4f clipPos(ndcX, ndcY, -1.0f, 1.0f);
        Eigen::Vector4f worldPos = (projectionMatrix * viewMatrix).inverse() * clipPos;
        worldPos /= worldPos.w();

        rayOrigin = Eigen::Vector3f(worldPos.x(), worldPos.y(), worldPos.z());
    }

    return Ray{ rayOrigin, rayDir };
}