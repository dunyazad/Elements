#pragma once
#include <cuda_runtime.h>

#define MAX_CONTROL_POINTS 8

struct CuControlPoint {
    float t;
    float3 color;
};

struct CuTransferFunction
{
    float minVal = 0.0f;
    float maxVal = 1.0f;
    bool invert = false;

    bool useAlphaClip = false;
    float alphaClipThreshold = 0.0f;
    unsigned char baseAlpha = 255;

    CuControlPoint points[MAX_CONTROL_POINTS];
    int pointCount = 0;

    __host__ __device__ CuTransferFunction() {}
    __host__ __device__ CuTransferFunction(float min, float max) : minVal(min), maxVal(max) {}

#ifdef __CUDACC__
    __device__ __forceinline__ uchar4 Map(float value) const
    {
        float t = (value - minVal) / (maxVal - minVal + 1e-6f);
        t = __saturatef(t);

        if (useAlphaClip && t < alphaClipThreshold) return make_uchar4(0, 0, 0, 0);

        if (invert) t = 1.0f - t;

        float3 rgb;
        if (pointCount > 0) {
            rgb = SampleLinear(t);
        }
        else {
            rgb = SampleTurboAnalytic(t);
        }

        return make_uchar4(
            (unsigned char)(rgb.x * 255.0f),
            (unsigned char)(rgb.y * 255.0f),
            (unsigned char)(rgb.z * 255.0f),
            baseAlpha
        );
    }
#else
    uchar4 Map(float value) const = delete;
#endif

    __host__ void AddPoint(float t, float r, float g, float b) {
        if (pointCount >= MAX_CONTROL_POINTS) return;
        t = fmaxf(0.0f, fminf(1.0f, t));

        int i = pointCount - 1;
        while (i >= 0 && points[i].t > t) {
            points[i + 1] = points[i];
            i--;
        }
        points[i + 1] = { t, make_float3(r, g, b) };
        pointCount++;
    }

    __host__ void Clear() { pointCount = 0; }

    __host__ void SetJet() {
        Clear();
        AddPoint(0.00f, 0.0f, 0.0f, 0.5f); // Dark Blue
        AddPoint(0.12f, 0.0f, 0.0f, 1.0f); // Blue
        AddPoint(0.34f, 0.0f, 1.0f, 1.0f); // Cyan
        AddPoint(0.65f, 1.0f, 1.0f, 0.0f); // Yellow
        AddPoint(0.90f, 1.0f, 0.0f, 0.0f); // Red
        AddPoint(1.00f, 0.5f, 0.0f, 0.0f); // Dark Red
    }

    __host__ void SetGray() {
        Clear();
        AddPoint(0.0f, 0.0f, 0.0f, 0.0f);
        AddPoint(1.0f, 1.0f, 1.0f, 1.0f);
    }

    __host__ void SetDeepSea() {
        Clear();
        AddPoint(0.0f, 0.0f, 0.0f, 0.0f); // Black (밀도 0)
        AddPoint(0.4f, 0.0f, 0.0f, 1.0f); // Blue
        AddPoint(0.8f, 0.0f, 1.0f, 1.0f); // Cyan
        AddPoint(1.0f, 1.0f, 1.0f, 1.0f); // White (밀도 높음)
    }

    __host__ void SetTurboApprox() {
        Clear();
        AddPoint(0.00f, 0.19f, 0.07f, 0.23f);
        AddPoint(0.15f, 0.28f, 0.32f, 0.96f);
        AddPoint(0.30f, 0.12f, 0.63f, 0.99f);
        AddPoint(0.45f, 0.17f, 0.84f, 0.63f);
        AddPoint(0.60f, 0.57f, 0.89f, 0.24f);
        AddPoint(0.75f, 0.92f, 0.73f, 0.13f);
        AddPoint(0.90f, 0.94f, 0.38f, 0.04f);
        AddPoint(1.00f, 0.48f, 0.02f, 0.01f);
    }

private:
    __device__ __forceinline__ float3 SampleLinear(float t) const
    {
        if (t <= points[0].t) return points[0].color;
        if (t >= points[pointCount - 1].t) return points[pointCount - 1].color;

        for (int i = 0; i < pointCount - 1; ++i) {
            if (t >= points[i].t && t <= points[i + 1].t) {
                float range = points[i + 1].t - points[i].t;
                if (range < 1e-6f) return points[i].color;

                float alpha = (t - points[i].t) / range;

                // Lerp
                float3 c1 = points[i].color;
                float3 c2 = points[i + 1].color;
                return make_float3(
                    c1.x + (c2.x - c1.x) * alpha,
                    c1.y + (c2.y - c1.y) * alpha,
                    c1.z + (c2.z - c1.z) * alpha
                );
            }
        }
        return points[pointCount - 1].color;
    }

    __device__ __forceinline__ float3 SampleTurboAnalytic(float x) const
    {
        const float4 kRedVec4 = make_float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
        const float4 kGreenVec4 = make_float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
        const float4 kBlueVec4 = make_float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
        const float2 kRedVec2 = make_float2(-152.94239396f, 59.28637943f);
        const float2 kGreenVec2 = make_float2(4.27729857f, 2.82956604f);
        const float2 kBlueVec2 = make_float2(-89.90310912f, 27.34824973f);

        float4 v4 = make_float4(1.0f, x, x * x, x * x * x);
        float2 v2 = make_float2(v4.w * x, v4.w * x * x);

        return make_float3(
            kRedVec4.x * v4.x + kRedVec4.y * v4.y + kRedVec4.z * v4.z + kRedVec4.w * v4.w + kRedVec2.x * v2.x + kRedVec2.y * v2.y,
            kGreenVec4.x * v4.x + kGreenVec4.y * v4.y + kGreenVec4.z * v4.z + kGreenVec4.w * v4.w + kGreenVec2.x * v2.x + kGreenVec2.y * v2.y,
            kBlueVec4.x * v4.x + kBlueVec4.y * v4.y + kBlueVec4.z * v4.z + kBlueVec4.w * v4.w + kBlueVec2.x * v2.x + kBlueVec2.y * v2.y
        );
    }
};
