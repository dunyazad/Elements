#pragma once

#include <Eigen/Dense>

class Renderable;
class DebuggingRenderable;

class GeometryBuilder
{
public:
    static void BuildPlane(
        Renderable* renderable,
        float width,
        float height,
        unsigned int hSegments,
        unsigned int vSegments,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& normal,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildPlane(
        DebuggingRenderable* debuggingRenderable,
        float width,
        float height,
        unsigned int hSegments,
        unsigned int vSegments,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& normal,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildBox(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& dimension,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildBox(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& dimension,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildSphere(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        float radius,
        unsigned int latitudeSegments,
        unsigned int longitudeSegments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildSphere(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        float radius,
        unsigned int latitudeSegments,
        unsigned int longitudeSegments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildDisk(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& normal,
        float radius,
        unsigned int segments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildDisk(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& normal,
        float radius,
        unsigned int segments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildCylinder(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        float radius,
        float height,
        unsigned int segments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildCylinder(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        float radius,
        float height,
        unsigned int segments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildCone(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        float radius,
        float height,
        unsigned int segments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildCone(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        float radius,
        float height,
        unsigned int segments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildCapsule(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        float radius,
        float height,
        unsigned int segments,
        unsigned int rings,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildCapsule(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        float radius,
        float height,
        unsigned int segments,
        unsigned int rings,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildTorus(
        Renderable* renderable,
        const Eigen::Vector3f& center,
        float majorRadius,
        float minorRadius,
        unsigned int majorSegments,
        unsigned int minorSegments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildTorus(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& center,
        float majorRadius,
        float minorRadius,
        unsigned int majorSegments,
        unsigned int minorSegments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildTube(
        Renderable* renderable,
        const std::vector<Eigen::Vector3f>& controlPoints,
        float radius,
        unsigned int curveSegments,
        unsigned int radialSegments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildTube(
        DebuggingRenderable* debuggingRenderable,
        const std::vector<Eigen::Vector3f>& controlPoints,
        float radius,
        unsigned int curveSegments,
        unsigned int radialSegments,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildArrow(
        Renderable* renderable,
        const Eigen::Vector3f& start,
        const Eigen::Vector3f& end,
        float stemRadius,
        float headRadius,
        float headLength,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildArrow(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Vector3f& start,
        const Eigen::Vector3f& end,
        float stemRadius,
        float headRadius,
        float headLength,
        const Eigen::Vector4f& color,
        bool wireframe = false);

    static void BuildFrustum(
        Renderable* renderable,
        const Eigen::Matrix4f& invViewProj,
        const Eigen::Vector4f& color,
        bool wireframe = true);

    static void BuildFrustum(
        DebuggingRenderable* debuggingRenderable,
        const Eigen::Matrix4f& invViewProj,
        const Eigen::Vector4f& color,
        bool wireframe = true);

    static void BuildGrid(
        Renderable* renderable,
        float size,
        unsigned int divisions,
        const Eigen::Vector4f& color);

    static void BuildGrid(
        DebuggingRenderable* debuggingRenderable,
        float size,
        unsigned int divisions,
        const Eigen::Vector4f& color);
};
