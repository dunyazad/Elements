#pragma once

#include <Helium/HeliumCommon.h>

struct TransformComponent
{
    Eigen::Vector3f Position = { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f Rotation = { 0.0f, 0.0f, 0.0f }; // Euler Angles
    Eigen::Vector3f Scale = { 1.0f, 1.0f, 1.0f };

    // 행렬 계산 헬퍼
    Eigen::Matrix4f GetTransform() const
    {
        Eigen::Affine3f t = Eigen::Affine3f::Identity();
        t.translate(Position);
        t.rotate(Eigen::AngleAxisf(Rotation.x(), Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf(Rotation.y(), Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(Rotation.z(), Eigen::Vector3f::UnitZ()));
        t.scale(Scale);
        return t.matrix();
    }
};
