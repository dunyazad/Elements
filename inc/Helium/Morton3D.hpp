#pragma once
#include <cstdint>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

namespace Eigen {
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;

    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

class Morton3D
{
public:
    // Encode separate x,y,z into morton
    static uint64_t Encode(uint32_t x, uint32_t y, uint32_t z)
    {
        uint64_t xx = Part1By2(x);
        uint64_t yy = Part1By2(y) << 1;
        uint64_t zz = Part1By2(z) << 2;
        return xx | yy | zz;
    }

    static uint64_t Encode(const Eigen::Vector3i& index)
    {
        uint64_t xx = Part1By2(index.x());
        uint64_t yy = Part1By2(index.y()) << 1;
        uint64_t zz = Part1By2(index.z()) << 2;
        return xx | yy | zz;
    }

    static uint64_t Encode(const Eigen::Vector3ui& index)
    {
        uint64_t xx = Part1By2(index.x());
        uint64_t yy = Part1By2(index.y()) << 1;
        uint64_t zz = Part1By2(index.z()) << 2;
        return xx | yy | zz;
    }

    // Decode morton to x,y,z
    static void Decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z)
    {
        x = Compact1By2(code);
        y = Compact1By2(code >> 1);
        z = Compact1By2(code >> 2);
    }

    static Eigen::Vector3i KeyToIndex(uint64_t code)
    {
        uint32_t x = Compact1By2(code);
        uint32_t y = Compact1By2(code >> 1);
        uint32_t z = Compact1By2(code >> 2);
        return Eigen::Vector3i(x, y, z);
    }

    static uint64_t IndexToKey(const Eigen::Vector3i& index)
    {
        return Encode(index);
    }

    static uint64_t IndexToKey(const Eigen::Vector3ui& index)
    {
        return Encode(index);
    }

    static Eigen::Vector3ui PositionToIndex(
        const Eigen::Vector3f& p,
        const Eigen::Vector3f& origin,
        float voxelSize)
    {
        float inv = 1.0f / voxelSize;

        int ix = (int)std::floor((p.x() - origin.x()) * inv);
        int iy = (int)std::floor((p.y() - origin.y()) * inv);
        int iz = (int)std::floor((p.z() - origin.z()) * inv);

        return Eigen::Vector3ui(ix, iy, iz);
    }

    static uint64_t EncodeFromPosition(
        const Eigen::Vector3f& p,
        const Eigen::Vector3f& origin,
        float voxelSize)
    {
        return Encode(PositionToIndex(p, origin, voxelSize));
    }

    static Eigen::Vector3f IndexToPosition(
        const Eigen::Vector3ui& index,
        const Eigen::Vector3f& origin,
        float voxelSize)
    {
        return origin +
            Eigen::Vector3f(
                (float)index.x() + 0.5f,
                (float)index.y() + 0.5f,
                (float)index.z() + 0.5f
            ) * voxelSize;
    }

    static Eigen::Vector3f DecodeToPosition(
        uint64_t code,
        const Eigen::Vector3f& origin,
        float voxelSize)
    {
        uint32_t ix, iy, iz;
        Decode(code, ix, iy, iz);

        return IndexToPosition({ ix, iy, iz }, origin, voxelSize);
    }

private:
    static uint64_t Part1By2(uint32_t x)
    {
        uint64_t r = x;

        r &= 0x1fffff;
        r = (r | (r << 32)) & 0x1f00000000ffff;
        r = (r | (r << 16)) & 0x1f0000ff0000ff;
        r = (r | (r << 8)) & 0x100f00f00f00f00f;
        r = (r | (r << 4)) & 0x10c30c30c30c30c3;
        r = (r | (r << 2)) & 0x1249249249249249;

        return r;
    }

    static uint32_t Compact1By2(uint64_t x)
    {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
        x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
        x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
        x = (x ^ (x >> 16)) & 0x1f00000000ffff;
        x = (x ^ (x >> 32)) & 0x1fffff;
        return (uint32_t)x;
    }
};
