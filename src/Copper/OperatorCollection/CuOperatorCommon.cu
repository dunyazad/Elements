#include <Copper/OperatorCollection/CuOperatorCommon.h>
#include <Copper/OperatorCollection/CuOperatorCommonDevice.h>

__global__ void computeDensityKernel(
    const float3* __restrict__ positions, const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
    float* __restrict__ outValues, int numParticles, float radius, int mode, float cellSize, int3 gridSize, float3 worldOrigin)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    float3 myPos = FETCH(positions, index);
    float rSq = radius * radius;
    float invHSq = 1.0f / ((radius * 0.5f) * (radius * 0.5f));

    float invCellSize = 1.0f / cellSize;
    int gridX = max(0, min((int)((myPos.x - worldOrigin.x) * invCellSize), gridSize.x - 1));
    int gridY = max(0, min((int)((myPos.y - worldOrigin.y) * invCellSize), gridSize.y - 1));
    int gridZ = max(0, min((int)((myPos.z - worldOrigin.z) * invCellSize), gridSize.z - 1));

    int searchRange = (int)ceilf(radius * invCellSize);
    float density = 0.0f;

    for (int z = -searchRange; z <= searchRange; ++z)
    {
        for (int y = -searchRange; y <= searchRange; ++y)
        {
            for (int x = -searchRange; x <= searchRange; ++x)
            {
                int nx = gridX + x; int ny = gridY + y; int nz = gridZ + z;
                if (nx >= 0 && nx < gridSize.x && ny >= 0 && ny < gridSize.y && nz >= 0 && nz < gridSize.z)
                {
                    int hash = (nz * gridSize.y + ny) * gridSize.x + nx;
                    int start = FETCH(cellStart, hash);
                    if (start != -1)
                    {
                        int end = FETCH(cellEnd, hash);
                        for (int j = start; j < end; ++j)
                        {
                            float d2 = getDistSq(myPos, FETCH(positions, j));
                            if (d2 <= rSq)
                            {
                                if (mode == 0) density += 1.0f; // LDE
                                else density += expf(-d2 * invHSq); // KDE
                            }
                        }
                    }
                }
            }
        }
    }
    outValues[index] = density;
}

__global__ void applyTransferFunctionKernel(uchar3* colors, const float* values, int numParticles, CuTransferFunction tf)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    uchar4 c4 = tf.Map(values[index]);
    colors[index] = make_uchar3(c4.x, c4.y, c4.z);
}
