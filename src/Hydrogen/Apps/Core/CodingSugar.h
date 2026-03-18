#pragma once

#define XY(v) (v).x, (v).y
#define XY_(v) (v).x(), (v).y()
#define XYZ(v) (v).x, (v).y, (v).z
#define XYZ_(v) (v).x(), (v).y(), (v).z()
#define XYZW(v) (v).x, (v).y, (v).z, (v).w
#define XYZW_(v) (v).x(), (v).y(), (v).z(), (v).w()

//#define FLT_VALID(x) ((x) < FLT_MAX / 2)
#define FLT_VALID(x) ((x) < 3.402823466e+36F)
#define VECTOR3F_VALID(v) (FLT_VALID((v).x) && FLT_VALID((v).y) && FLT_VALID((v).z))
#define VECTOR3F_VALID_(v) (FLT_VALID((v).x()) && FLT_VALID((v).y()) && FLT_VALID((v).z()))
#define SHORT_VALID(x) ((x) != SHRT_MAX)
#define USHORT_VALID(x) ((x) != USHRT_MAX)
#define INT_VALID(x) ((x) != INT_MAX)
#define UINT_VALID(x) ((x) != UINT_MAX)
#define VECTOR3U_VALID(v) (UINT_VALID((v).x) && UINT_VALID((v).y) && UINT_VALID((v).z))
#define VECTOR3U_VALID_(v) (UINT_VALID((v).x()) && UINT_VALID((v).y()) && UINT_VALID((v).z()))

#ifndef CUDA_TS
#ifdef __CUDACC__
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#else
#define CUDA_TS(name)
#endif
#endif

#ifndef CUDA_TE
#ifdef __CUDACC__
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[%s] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#else
#define CUDA_TE(name)
#endif
#endif

#ifndef CUDA_MALLOC
#define CUDA_MALLOC(ptr, size) cudaMalloc(ptr, size);
#endif

#ifndef CUDA_FREE
#define CUDA_FREE(ptr) cudaFree(ptr);
#endif

#ifndef CUDA_SAFE_FREE
#define CUDA_SAFE_FREE(ptr) { if(ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; } }
//#define CUDA_SAFE_FREE(ptr) \
//    do { \
//        if (ptr) { \
//            cudaFree(ptr); \
//            ptr = nullptr; \
//        } \
//    } while(0)
#endif

#ifndef CUDA_MEMSET
#define CUDA_MEMSET(ptr, value, size) cudaMemset(ptr, value, size);
#endif

#ifndef CUDA_COPY_D2D
#define CUDA_COPY_D2D(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyDeviceToDevice);
#endif

#ifndef CUDA_COPY_D2H
#define CUDA_COPY_D2H(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
#endif

#ifndef CUDA_COPY_H2D
#define CUDA_COPY_H2D(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
#endif

#ifndef CUDA_COPY_H2H
#define CUDA_COPY_H2H(to, from, size) cudaMemcpy(to, from, size, cudaMemcpyHostToHost);
#endif

#ifndef CUDA_SYNC
#define CUDA_SYNC() cudaDeviceSynchronize();
#endif

#ifndef RAW_PTR
#define RAW_PTR(x) (thrust::raw_pointer_cast((x).data()))
#endif