#include "pch.h"

#include <Helium/GeometryBuilder.h>
#include <Helium/Components/Renderable.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>

template <typename T>
static void BuildPlaneInternal(
    T* target,
    float width,
    float height,
    unsigned int hSegments,
    unsigned int vSegments,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& normal,
    const Eigen::Vector4f& color)
{
    if (hSegments < 1 || vSegments < 1)
    {
        return;
    }

    std::vector<unsigned int> indices;
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector4f> colors;
    std::vector<Eigen::Vector2f> uvs;

    unsigned int numVertices = (hSegments + 1) * (vSegments + 1);
    unsigned int numIndices = hSegments * vSegments * 6;
    vertices.reserve(numVertices);
    normals.reserve(numVertices);
    colors.reserve(numVertices);
    uvs.reserve(numVertices);
    indices.reserve(numIndices);

    float halfWidth = width * 0.5f;
    float halfHeight = height * 0.5f;

    Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), normal);

    for (unsigned int j = 0; j <= vSegments; ++j)
    {
        for (unsigned int i = 0; i <= hSegments; ++i)
        {
            float x = (static_cast<float>(i) / hSegments) * width - halfWidth;
            float y = (static_cast<float>(j) / vSegments) * height - halfHeight;

            Eigen::Vector3f localPos(x, y, 0.0f);

            vertices.push_back(center + (rotation * localPos));
            normals.push_back(normal);
            colors.push_back(color);

            uvs.push_back(Eigen::Vector2f(
                static_cast<float>(i) / hSegments,
                static_cast<float>(j) / vSegments
            ));
        }
    }

    for (unsigned int j = 0; j < vSegments; ++j)
    {
        for (unsigned int i = 0; i < hSegments; ++i)
        {
            unsigned int row1 = j * (hSegments + 1);
            unsigned int row2 = (j + 1) * (hSegments + 1);

            indices.push_back(row1 + i);
            indices.push_back(row1 + i + 1);
            indices.push_back(row2 + i);

            indices.push_back(row2 + i);
            indices.push_back(row1 + i + 1);
            indices.push_back(row2 + i + 1);
        }
    }

    target->SetVertices(vertices);
    target->SetNormals(normals);
    target->SetColors4(colors);
    target->SetUVs(uvs);
    target->SetIndices(indices);
}

void GeometryBuilder::BuildPlane(
    Renderable* renderable,
    float width,
    float height,
    unsigned int hSegments,
    unsigned int vSegments,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& normal,
    const Eigen::Vector4f& color)
{
    if (renderable == nullptr)
        return;

    BuildPlaneInternal(renderable, width, height, hSegments, vSegments, center, normal, color);
}

void GeometryBuilder::BuildPlane(
    DebuggingRenderable* debuggingRenderable,
    float width,
    float height,
    unsigned int hSegments,
    unsigned int vSegments,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& normal,
    const Eigen::Vector4f& color)
{
    if (debuggingRenderable == nullptr)
        return;

    BuildPlaneInternal(debuggingRenderable, width, height, hSegments, vSegments, center, normal, color);
}
