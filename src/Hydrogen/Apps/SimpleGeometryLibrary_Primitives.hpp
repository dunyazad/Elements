#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <mutex>
#include <fstream>
#include <limits>
#include <utility>
#include <execution>
#include <Eigen/Dense>
#include <robin_hood/robin_hood.h>

// Simple Geometry Library (SGL)
namespace SGL
{
    struct Ray
    {
        Eigen::Vector3f origin;
        Eigen::Vector3f direction;
	};

    struct Line
    {
        Eigen::Vector3f start;
        Eigen::Vector3f end;
    };

    struct Plane
    {
        Eigen::Vector3f normal;
        Eigen::Vector3f planePoint;
	};

    struct Triangle
    {
        Eigen::Vector3f v0;
        Eigen::Vector3f v1;
		Eigen::Vector3f v2;
    };

    using PID = unsigned int;
    using VID = unsigned int;
    using EID = unsigned int;
    using FID = unsigned int;

    const unsigned int INVALID_ID = -1;

    struct Vertex
    {
        PID pid;
        EID eid;

        Vertex() : pid(INVALID_ID), eid(INVALID_ID)
        {
        }

        Vertex(PID pid) : pid(pid), eid(INVALID_ID)
        {
        }
    };

    struct HalfEdge
    {
        VID vid;
        EID oeid;
        EID neid;
        EID peid;
        FID fid;

        HalfEdge() : vid(INVALID_ID), oeid(INVALID_ID), neid(INVALID_ID), peid(INVALID_ID), fid(INVALID_ID)
        {
        }
    };

    struct Face
    {
        EID eid;

        Face() : eid(INVALID_ID)
        {
        }
    };

    struct AABB
    {
        Eigen::Vector3f minBound;
        Eigen::Vector3f maxBound;

        AABB()
        {
            minBound = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            maxBound = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        }

        void Expand(const Eigen::Vector3f& p)
        {
            minBound = minBound.cwiseMin(p);
            maxBound = maxBound.cwiseMax(p);
        }

        void Expand(const AABB& other)
        {
            minBound = minBound.cwiseMin(other.minBound);
            maxBound = maxBound.cwiseMax(other.maxBound);
        }

        bool Intersects(const AABB& other) const
        {
            if (maxBound.x() < other.minBound.x() || minBound.x() > other.maxBound.x())
            {
                return false;
            }
            if (maxBound.y() < other.minBound.y() || minBound.y() > other.maxBound.y())
            {
                return false;
            }
            if (maxBound.z() < other.minBound.z() || minBound.z() > other.maxBound.z())
            {
                return false;
            }
            return true;
        }
    };
}
