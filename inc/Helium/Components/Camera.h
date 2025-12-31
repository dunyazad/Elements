#pragma once

#include <Eigen/Dense>
#include <cmath>

#ifndef DEG2RAD
#define DEG2RAD (static_cast<float>(EIGEN_PI) / 180.0f)
#endif
#ifndef RAD2DEG
#define RAD2DEG (180.0f / static_cast<float>(EIGEN_PI))
#endif

#include <Helium/TypeDefinitions.h>

class Camera
{
public:
    enum ProjectionMode
    {
        Perspective,
        Orthogonal
    };

    class PerspectiveSettings
    {
    public:
        inline bool IsDirty() const { return dirty; }
        inline void SetDirty(bool isDirty) { dirty = isDirty; }

        inline float GetFovy() const { return fovy * RAD2DEG; }
        inline void SetFovy(float fovyDeg) { this->fovy = fovyDeg * DEG2RAD; dirty = true; }

        inline float GetAspectRatio() const { return aspectRatio; }
        inline void SetAspectRatio(float aspectRatio) { this->aspectRatio = aspectRatio; dirty = true; }

        inline float GetZNear() const { return zNear; }
        inline void SetZNear(float zNear) { this->zNear = zNear; dirty = true; }

        inline float GetZFar() const { return zFar; }
        inline void SetZFar(float zFar) { this->zFar = zFar; dirty = true; }

        friend class Camera;

    private:
        bool dirty = true;
        float fovy = 45.0f * DEG2RAD;
        float aspectRatio = 1.0f;
        float zNear = 0.1f;
        float zFar = 1000.0f;
    };

    class OrthogonalSettings
    {
    public:
        inline bool IsDirty() const { return dirty; }
        inline void SetDirty(bool isDirty) { dirty = isDirty; }

        inline float GetLeft() const { return left; }
        inline void SetLeft(float left) { this->left = left; dirty = true; }

        inline float GetRight() const { return right; }
        inline void SetRight(float right) { this->right = right; dirty = true; }

        inline float GetBottom() const { return bottom; }
        inline void SetBottom(float bottom) { this->bottom = bottom; dirty = true; }

        inline float GetTop() const { return top; }
        inline void SetTop(float top) { this->top = top; dirty = true; }

        inline float GetZNear() const { return zNear; }
        inline void SetZNear(float zNear) { this->zNear = zNear; dirty = true; }

        inline float GetZFar() const { return zFar; }
        inline void SetZFar(float zFar) { this->zFar = zFar; dirty = true; }

        friend class Camera;

    private:
        bool dirty = true;
        float left = -10.0f;
        float right = 10.0f;
        float bottom = -10.0f;
        float top = 10.0f;
        float zNear = -100.0f;
        float zFar = 100.0f;
    };

public:
    Camera();
    ~Camera();

    void Update(unsigned int frameNo, float timeDelta);

    Ray ScreenPointToRay(float mouseX, float mouseY, int screenWidth, int screenHeight);

    void SetProjectionMode(ProjectionMode mode);
    inline ProjectionMode GetProjectionMode() const { return mode; }

    inline bool IsDirty() const { return dirty; }
    inline void SetDirty(bool isDirty) { dirty = isDirty; }

    inline Eigen::Vector3f& GetEye() { return eye; }
    inline Eigen::Vector3f& GetTarget() { return target; }
    inline Eigen::Vector3f& GetUp() { return up; }

    inline void SetEye(const Eigen::Vector3f& eye) { this->eye = eye; dirty = true; }
    inline void SetTarget(const Eigen::Vector3f& target) { this->target = target; dirty = true; }
    inline void SetUp(const Eigen::Vector3f& up) { this->up = up; dirty = true; }

    inline const Eigen::Matrix4f& GetProjectionMatrix() const { return projectionMatrix; }
    inline const Eigen::Matrix4f& GetViewMatrix() const { return viewMatrix; }

    inline PerspectiveSettings& GetPerspectiveSettings() { dirty = true; return perspectiveSettings; }
    inline OrthogonalSettings& GetOrthogonalSettings() { dirty = true; return orthogonalSettings; }

private:
    void UpdateViewMatrix();
    void UpdateProjectionMatrix();

private:
    bool dirty = true;
    ProjectionMode mode = Perspective;

    PerspectiveSettings perspectiveSettings;
    OrthogonalSettings orthogonalSettings;

    Eigen::Vector3f eye = { 0.0f, 0.0f, 50.0f };
    Eigen::Vector3f target = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f up = { 0.0f, 1.0f, 0.0f };

    Eigen::Matrix4f projectionMatrix = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f viewMatrix = Eigen::Matrix4f::Identity();
};
