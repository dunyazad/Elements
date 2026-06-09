#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <memory>

#include <Core/Core.h>
#include <Core/DataStructures/VoxelDataBase.h>

#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <Helium/IVisualDebugging.h>
using VD = IVisualDebugging;

using namespace Huvitz;


#include "Apps.h"
class AppDataFrames : public App
{
public:
    virtual void Initialize() override
    {
    }

    virtual void Execute() override
    {
        nvDriverSetting.forceGPUPerformance();

        {
            thrust::device_vector<int> dummy(1 << 20);
            thrust::sequence(dummy.begin(), dummy.end());
            thrust::sort(dummy.begin(), dummy.end(), thrust::greater<int>());
            cudaDeviceSynchronize();
        }

        DataFrameReader reader("D:\\Debug\\DevicePoints.dfm");

        while (auto frame = reader.next())
        {
            size_t bytesPerPoint = sizeof(float3) * 2 + sizeof(uchar3) + sizeof(unsigned int) * 2;
            unsigned int pointCount = static_cast<unsigned int>(frame->data.size() / bytesPerPoint);

            if (110 > frame->frameIndex) continue;
            if (120 < frame->frameIndex) break;

            printf("Frame Index: %u, Tag: %u, Point count: %u\n", frame->frameIndex, frame->tag, pointCount);

            if (pointCount == 0)
            {
                continue;
            }

            uint8_t* basePtr = frame->data.data();
            size_t offset = 0;

            const float3* positions = reinterpret_cast<const float3*>(basePtr + offset);
            offset += sizeof(float3) * pointCount;

            const float3* normals = reinterpret_cast<const float3*>(basePtr + offset);
            offset += sizeof(float3) * pointCount;

            const uchar3* colors = reinterpret_cast<const uchar3*>(basePtr + offset);
            offset += sizeof(uchar3) * pointCount;

            const unsigned int* labels = reinterpret_cast<const unsigned int*>(basePtr + offset);
            offset += sizeof(unsigned int) * pointCount;

            const unsigned int* tags = reinterpret_cast<const unsigned int*>(basePtr + offset);
            offset += sizeof(unsigned int) * pointCount;

            for (size_t i = 0; i < pointCount; i++)
            {
                const float3& pos = positions[i];
                const float3& norm = normals[i];
                const uchar3& col = colors[i];

                VD::AddSphere("DevicePoints",
                    { pos.x, pos.y, pos.z },
                    { norm.x, norm.y, norm.z },
                    0.05f,
                    { (float)col.x / 255.0f, (float)col.y / 255.0f, (float)col.z / 255.0f, 1.0f });
            }
        }
    }
};

REGISTER_APP(AppDataFrames, "AppDataFrames");
