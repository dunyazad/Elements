#ifndef COPPER_CUDA_MATH_H
#define COPPER_CUDA_MATH_H

#include <cuda_runtime.h>
#include <math.h>

// --- [Constants] ---

#define CUDA_MATH_EPSILON 1e-8f

// --- [Utility Functions] ---

/**
 * @brief Normalizes a 3D vector to unit length.
 * @param v Input vector
 * @return Normalized vector. Returns zero vector if input length is near zero.
 */__device__ __forceinline__ float3 Normalize(float3 v)
{
	float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
	if (lenSq < CUDA_MATH_EPSILON) return make_float3(0.0f, 0.0f, 0.0f);
	float invLen = 1.0f / sqrtf(lenSq);
	return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

/**
* @brief Computes the dot product of two 3D vectors.
* @param a First vector
* @param b Second vector
* @return Dot product a ¡¤ b
*/
__device__ __forceinline__ float Dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
* @brief Computes the cross product of two 3D vectors.
* @param a First vector
* @param b Second vector
* @return Cross product a ¡¿ b
*/
__device__ __forceinline__ float3 Cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/**
 * @brief Computes the squared length of a 3D vector.
 * @param v Input vector
 * @return Squared length |v|^2
 */
__device__ __forceinline__ float LengthSq(float3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

/**
 * @brief Computes the squared distance between two 3D points.
 * @param a First point
 * @param b Second point
 * @return Squared distance |a - b|^2
 */
__device__ __forceinline__ float DistanceSq(float3 a, float3 b)
{
	float3 d = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	return d.x * d.x + d.y * d.y + d.z * d.z;
}

// --- [Mat3 Structure & Functions] ---

struct Mat3
{
	float3 r0;
	float3 r1;
	float3 r2;
};

/**
 * @brief Creates a 3x3 zero matrix.
 * @return Zero matrix
 */
__device__ __forceinline__ Mat3 SetMat3Zero()
{
	Mat3 result;
	result.r0 = make_float3(0.0f, 0.0f, 0.0f);
	result.r1 = make_float3(0.0f, 0.0f, 0.0f);
	result.r2 = make_float3(0.0f, 0.0f, 0.0f);
	return result;
}

/**
 * @brief Creates a 3x3 identity matrix.
 * @return Identity matrix
 */
__device__ __forceinline__ Mat3 SetMat3Identity()
{
	Mat3 result;
	result.r0 = make_float3(1.0f, 0.0f, 0.0f);
	result.r1 = make_float3(0.0f, 1.0f, 0.0f);
	result.r2 = make_float3(0.0f, 0.0f, 1.0f);
	return result;
}

/**
 * @brief Creates a 3x3 matrix filled with a specific value.
 * @param value Value to fill all elements
 * @return Filled matrix
 */
__device__ __forceinline__ Mat3 SetMat3Fill(float value)
{
	Mat3 result;
	result.r0 = make_float3(value, value, value);
	result.r1 = make_float3(value, value, value);
	result.r2 = make_float3(value, value, value);
	return result;
}

/**
 * @brief Creates a 3x3 diagonal matrix.
 * @param v Diagonal values
 * @return Diagonal matrix
 */
__device__ __forceinline__ Mat3 SetMat3Diagonal(float3 v)
{
	Mat3 result = SetMat3Zero();
	result.r0.x = v.x;
	result.r1.y = v.y;
	result.r2.z = v.z;
	return result;
}

/**
 * @brief Creates a 3x3 matrix from a float array (row-major order).
 * @param data Array of 9 floats
 * @return Matrix constructed from array
 */
__device__ __forceinline__ Mat3 Mat3FromArray(const float* data)
{
	Mat3 result;
	result.r0 = make_float3(data[0], data[1], data[2]);
	result.r1 = make_float3(data[3], data[4], data[5]);
	result.r2 = make_float3(data[6], data[7], data[8]);
	return result;
}

/**
 * @brief Converts a 3x3 matrix to a float array (row-major order).
 * @param matrix Input matrix
 * @param outData Output array (must be at least 9 floats)
 */
__device__ __forceinline__ void Mat3ToArray(const Mat3 matrix, float* outData)
{
	outData[0] = matrix.r0.x; outData[1] = matrix.r0.y; outData[2] = matrix.r0.z;
	outData[3] = matrix.r1.x; outData[4] = matrix.r1.y; outData[5] = matrix.r1.z;
	outData[6] = matrix.r2.x; outData[7] = matrix.r2.y; outData[8] = matrix.r2.z;
}

/**
 * @brief Adds two 3x3 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Sum of matrices
 */
__device__ __forceinline__ Mat3 AddMat3(const Mat3 left, const Mat3 right)
{
	Mat3 result;
	result.r0 = make_float3(left.r0.x + right.r0.x, left.r0.y + right.r0.y, left.r0.z + right.r0.z);
	result.r1 = make_float3(left.r1.x + right.r1.x, left.r1.y + right.r1.y, left.r1.z + right.r1.z);
	result.r2 = make_float3(left.r2.x + right.r2.x, left.r2.y + right.r2.y, left.r2.z + right.r2.z);
	return result;
}

/**
 * @brief Subtracts two 3x3 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Difference of matrices
 */
__device__ __forceinline__ Mat3 SubtractMat3(const Mat3 left, const Mat3 right)
{
	Mat3 result;
	result.r0 = make_float3(left.r0.x - right.r0.x, left.r0.y - right.r0.y, left.r0.z - right.r0.z);
	result.r1 = make_float3(left.r1.x - right.r1.x, left.r1.y - right.r1.y, left.r1.z - right.r1.z);
	result.r2 = make_float3(left.r2.x - right.r2.x, left.r2.y - right.r2.y, left.r2.z - right.r2.z);
	return result;
}

/**
 * @brief Multiplies two 3x3 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Product of matrices (left ¡¿ right)
 */
__device__ __forceinline__ Mat3 MultiplyMat3(const Mat3 left, const Mat3 right)
{
	Mat3 result;
	result.r0.x = left.r0.x * right.r0.x + left.r0.y * right.r1.x + left.r0.z * right.r2.x;
	result.r0.y = left.r0.x * right.r0.y + left.r0.y * right.r1.y + left.r0.z * right.r2.y;
	result.r0.z = left.r0.x * right.r0.z + left.r0.y * right.r1.z + left.r0.z * right.r2.z;
	result.r1.x = left.r1.x * right.r0.x + left.r1.y * right.r1.x + left.r1.z * right.r2.x;
	result.r1.y = left.r1.x * right.r0.y + left.r1.y * right.r1.y + left.r1.z * right.r2.y;
	result.r1.z = left.r1.x * right.r0.z + left.r1.y * right.r1.z + left.r1.z * right.r2.z;
	result.r2.x = left.r2.x * right.r0.x + left.r2.y * right.r1.x + left.r2.z * right.r2.x;
	result.r2.y = left.r2.x * right.r0.y + left.r2.y * right.r1.y + left.r2.z * right.r2.y;
	result.r2.z = left.r2.x * right.r0.z + left.r2.y * right.r1.z + left.r2.z * right.r2.z;
	return result;
}

/**
 * @brief Multiplies a 3x3 matrix by a scalar.
 * @param matrix Input matrix
 * @param scalar Scalar value
 * @return Scaled matrix
 */
__device__ __forceinline__ Mat3 MultiplyMat3Scalar(const Mat3 matrix, float scalar)
{
	Mat3 result;
	result.r0 = make_float3(matrix.r0.x * scalar, matrix.r0.y * scalar, matrix.r0.z * scalar);
	result.r1 = make_float3(matrix.r1.x * scalar, matrix.r1.y * scalar, matrix.r1.z * scalar);
	result.r2 = make_float3(matrix.r2.x * scalar, matrix.r2.y * scalar, matrix.r2.z * scalar);
	return result;
}

/**
 * @brief Divides a 3x3 matrix by a scalar.
 * @param matrix Input matrix
 * @param scalar Scalar value (must not be zero)
 * @return Scaled matrix
 * @warning No check for division by zero
 */
__device__ __forceinline__ Mat3 DivideMat3Scalar(const Mat3 matrix, float scalar)
{
	float invScalar = 1.0f / scalar;
	Mat3 result;
	result.r0 = make_float3(matrix.r0.x * invScalar, matrix.r0.y * invScalar, matrix.r0.z * invScalar);
	result.r1 = make_float3(matrix.r1.x * invScalar, matrix.r1.y * invScalar, matrix.r1.z * invScalar);
	result.r2 = make_float3(matrix.r2.x * invScalar, matrix.r2.y * invScalar, matrix.r2.z * invScalar);
	return result;
}

/**
 * @brief Performs element-wise multiplication of two 3x3 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Element-wise product
 */
__device__ __forceinline__ Mat3 MultiplyMat3ElementWise(const Mat3 left, const Mat3 right)
{
	Mat3 result;
	result.r0 = make_float3(left.r0.x * right.r0.x, left.r0.y * right.r0.y, left.r0.z * right.r0.z);
	result.r1 = make_float3(left.r1.x * right.r1.x, left.r1.y * right.r1.y, left.r1.z * right.r1.z);
	result.r2 = make_float3(left.r2.x * right.r2.x, left.r2.y * right.r2.y, left.r2.z * right.r2.z);
	return result;
}

/**
 * @brief Performs element-wise division of two 3x3 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Element-wise quotient
 * @warning No check for division by zero
 */
__device__ __forceinline__ Mat3 DivideMat3ElementWise(const Mat3 left, const Mat3 right)
{
	Mat3 result;
	result.r0 = make_float3(left.r0.x / right.r0.x, left.r0.y / right.r0.y, left.r0.z / right.r0.z);
	result.r1 = make_float3(left.r1.x / right.r1.x, left.r1.y / right.r1.y, left.r1.z / right.r1.z);
	result.r2 = make_float3(left.r2.x / right.r2.x, left.r2.y / right.r2.y, left.r2.z / right.r2.z);
	return result;
}

/**
 * @brief Calculates the determinant of a 3x3 matrix.
 * @param m Input matrix
 * @return Determinant value
 */
__device__ __forceinline__ float GetMat3Determinant(const Mat3 m)
{
	return m.r0.x * (m.r1.y * m.r2.z - m.r1.z * m.r2.y) -
		m.r0.y * (m.r1.x * m.r2.z - m.r1.z * m.r2.x) +
		m.r0.z * (m.r1.x * m.r2.y - m.r1.y * m.r2.x);
}

/**
 * @brief Computes the inverse of a 3x3 matrix.
 * @param m Input matrix
 * @return Inverse matrix. Returns zero matrix if determinant is near zero.
 * @warning Singular matrices return zero matrix
 */
__device__ __forceinline__ Mat3 InverseMat3(const Mat3 m)
{
	float det = GetMat3Determinant(m);
	if (fabsf(det) < CUDA_MATH_EPSILON) return SetMat3Zero();
	float invDet = 1.0f / det;
	Mat3 result;
	result.r0.x = (m.r1.y * m.r2.z - m.r1.z * m.r2.y) * invDet;
	result.r0.y = (m.r0.z * m.r2.y - m.r0.y * m.r2.z) * invDet;
	result.r0.z = (m.r0.y * m.r1.z - m.r0.z * m.r1.y) * invDet;
	result.r1.x = (m.r1.z * m.r2.x - m.r1.x * m.r2.z) * invDet;
	result.r1.y = (m.r0.x * m.r2.z - m.r0.z * m.r2.x) * invDet;
	result.r1.z = (m.r1.x * m.r0.z - m.r0.x * m.r1.z) * invDet;
	result.r2.x = (m.r1.x * m.r2.y - m.r1.y * m.r2.x) * invDet;
	result.r2.y = (m.r2.x * m.r0.y - m.r0.x * m.r2.y) * invDet;
	result.r2.z = (m.r0.x * m.r1.y - m.r0.y * m.r1.x) * invDet;
	return result;
}

/**
 * @brief Transposes a 3x3 matrix.
 * @param matrix Input matrix
 * @return Transposed matrix
 */
__device__ __forceinline__ Mat3 TransposeMat3(const Mat3 matrix)
{
	Mat3 result;
	result.r0 = make_float3(matrix.r0.x, matrix.r1.x, matrix.r2.x);
	result.r1 = make_float3(matrix.r0.y, matrix.r1.y, matrix.r2.y);
	result.r2 = make_float3(matrix.r0.z, matrix.r1.z, matrix.r2.z);
	return result;
}

/**
 * @brief Multiplies a 3x3 matrix by a 3D vector.
 * @param matrix Input matrix
 * @param vector Input vector
 * @return Transformed vector
 */
__device__ __forceinline__ float3 MultiplyMat3Vector(const Mat3 matrix, float3 vector)
{
	float3 result;
	result.x = matrix.r0.x * vector.x + matrix.r0.y * vector.y + matrix.r0.z * vector.z;
	result.y = matrix.r1.x * vector.x + matrix.r1.y * vector.y + matrix.r1.z * vector.z;
	result.z = matrix.r2.x * vector.x + matrix.r2.y * vector.y + matrix.r2.z * vector.z;
	return result;
}

/**
 * @brief Calculates the trace of a 3x3 matrix.
 * @param matrix Input matrix
 * @return Trace (sum of diagonal elements)
 */
__device__ __forceinline__ float GetMat3Trace(const Mat3 matrix)
{
	return matrix.r0.x + matrix.r1.y + matrix.r2.z;
}

/**
 * @brief Gets a row from a 3x3 matrix.
 * @param matrix Input matrix
 * @param rowIndex Row index (0-2)
 * @return Row vector. Returns zero vector if index is out of bounds.
 */
__device__ __forceinline__ float3 GetMat3Row(const Mat3 matrix, int rowIndex)
{
	if (rowIndex < 0 || rowIndex > 2) return make_float3(0.0f, 0.0f, 0.0f);
	const float3* rows = (const float3*)&matrix;
	return rows[rowIndex];
}

/**
 * @brief Sets a row in a 3x3 matrix.
 * @param matrix Pointer to matrix
 * @param rowIndex Row index (0-2)
 * @param rowValues Values to set
 */
__device__ __forceinline__ void SetMat3Row(Mat3* matrix, int rowIndex, float3 rowValues)
{
	if (rowIndex < 0 || rowIndex > 2) return;
	float3* rows = (float3*)matrix;
	rows[rowIndex] = rowValues;
}

/**
 * @brief Gets a column from a 3x3 matrix.
 * @param matrix Input matrix
 * @param colIndex Column index (0-2)
 * @return Column vector. Returns zero vector if index is out of bounds.
 */
__device__ __forceinline__ float3 GetMat3Column(const Mat3 matrix, int colIndex)
{
	if (colIndex < 0 || colIndex > 2) return make_float3(0.0f, 0.0f, 0.0f);
	if (colIndex == 0) return make_float3(matrix.r0.x, matrix.r1.x, matrix.r2.x);
	if (colIndex == 1) return make_float3(matrix.r0.y, matrix.r1.y, matrix.r2.y);
	return make_float3(matrix.r0.z, matrix.r1.z, matrix.r2.z);
}

/**
 * @brief Sets a column in a 3x3 matrix.
 * @param matrix Pointer to matrix
 * @param colIndex Column index (0-2)
 * @param colValues Values to set
 */
__device__ __forceinline__ void SetMat3Column(Mat3* matrix, int colIndex, float3 colValues)
{
	if (colIndex < 0 || colIndex > 2) return;
	if (colIndex == 0) { matrix->r0.x = colValues.x; matrix->r1.x = colValues.y; matrix->r2.x = colValues.z; }
	else if (colIndex == 1) { matrix->r0.y = colValues.x; matrix->r1.y = colValues.y; matrix->r2.y = colValues.z; }
	else if (colIndex == 2) { matrix->r0.z = colValues.x; matrix->r1.z = colValues.y; matrix->r2.z = colValues.z; }
}

/**
 * @brief Checks if two 3x3 matrices are approximately equal.
 * @param left First matrix
 * @param right Second matrix
 * @param epsilon Tolerance for comparison
 * @return true if matrices are equal within epsilon, false otherwise
 */
__device__ __forceinline__ bool IsEqualMat3(const Mat3 left, const Mat3 right, float epsilon)
{
	return fabsf(left.r0.x - right.r0.x) < epsilon && fabsf(left.r0.y - right.r0.y) < epsilon && fabsf(left.r0.z - right.r0.z) < epsilon &&
		fabsf(left.r1.x - right.r1.x) < epsilon && fabsf(left.r1.y - right.r1.y) < epsilon && fabsf(left.r1.z - right.r1.z) < epsilon &&
		fabsf(left.r2.x - right.r2.x) < epsilon && fabsf(left.r2.y - right.r2.y) < epsilon && fabsf(left.r2.z - right.r2.z) < epsilon;
}

// --- [Mat4 Structure & Functions] ---

struct Mat4
{
	float4 r0;
	float4 r1;
	float4 r2;
	float4 r3;
};

/**
 * @brief Gets an element from a 4x4 matrix.
 * @param matrix Input matrix
 * @param row Row index (0-3)
 * @param col Column index (0-3)
 * @return Element value. Returns 0.0f if indices are out of bounds.
 */
__device__ __forceinline__ float GetElement(const Mat4 matrix, int row, int col)
{
	if (row < 0 || row > 3 || col < 0 || col > 3) return 0.0f;
	if (row == 0) { if (col == 0) return matrix.r0.x; if (col == 1) return matrix.r0.y; if (col == 2) return matrix.r0.z; return matrix.r0.w; }
	if (row == 1) { if (col == 0) return matrix.r1.x; if (col == 1) return matrix.r1.y; if (col == 2) return matrix.r1.z; return matrix.r1.w; }
	if (row == 2) { if (col == 0) return matrix.r2.x; if (col == 1) return matrix.r2.y; if (col == 2) return matrix.r2.z; return matrix.r2.w; }
	if (col == 0) return matrix.r3.x; if (col == 1) return matrix.r3.y; if (col == 2) return matrix.r3.z; return matrix.r3.w;
}

/**
 * @brief Sets an element in a 4x4 matrix.
 * @param matrix Pointer to matrix
 * @param row Row index (0-3)
 * @param col Column index (0-3)
 * @param value Value to set
 */
__device__ __forceinline__ void SetElement(Mat4* matrix, int row, int col, float value)
{
	if (row < 0 || row > 3 || col < 0 || col > 3) return;
	float* ptr = (float*)matrix;
	ptr[row * 4 + col] = value;
}

/**
 * @brief Creates a 4x4 zero matrix.
 * @return Zero matrix
 */
__device__ __forceinline__ Mat4 SetZero()
{
	Mat4 result;
	result.r0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	result.r1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	result.r2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	result.r3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	return result;
}

/**
 * @brief Creates a 4x4 identity matrix.
 * @return Identity matrix
 */
__device__ __forceinline__ Mat4 SetIdentity()
{
	Mat4 result;
	result.r0 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	result.r1 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
	result.r2 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
	result.r3 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	return result;
}

/**
 * @brief Creates a 4x4 matrix filled with a specific value.
 * @param value Value to fill all elements
 * @return Filled matrix
 */
__device__ __forceinline__ Mat4 Fill(float value)
{
	Mat4 result;
	result.r0 = make_float4(value, value, value, value);
	result.r1 = make_float4(value, value, value, value);
	result.r2 = make_float4(value, value, value, value);
	result.r3 = make_float4(value, value, value, value);
	return result;
}

/**
 * @brief Creates a 4x4 diagonal matrix.
 * @param v Diagonal values
 * @return Diagonal matrix
 */
__device__ __forceinline__ Mat4 SetDiagonal(float4 v)
{
	Mat4 result = SetZero();
	result.r0.x = v.x;
	result.r1.y = v.y;
	result.r2.z = v.z;
	result.r3.w = v.w;
	return result;
}

/**
 * @brief Creates a 4x4 matrix from a float array (row-major order).
 * @param data Array of 16 floats
 * @return Matrix constructed from array
 */
__device__ __forceinline__ Mat4 FromArray(const float* data)
{
	Mat4 result;
	result.r0 = make_float4(data[0], data[1], data[2], data[3]);
	result.r1 = make_float4(data[4], data[5], data[6], data[7]);
	result.r2 = make_float4(data[8], data[9], data[10], data[11]);
	result.r3 = make_float4(data[12], data[13], data[14], data[15]);
	return result;
}

/**
 * @brief Converts a 4x4 matrix to a float array (row-major order).
 * @param matrix Input matrix
 * @param outData Output array (must be at least 16 floats)
 */
__device__ __forceinline__ void ToArray(const Mat4 matrix, float* outData)
{
	outData[0] = matrix.r0.x; outData[1] = matrix.r0.y; outData[2] = matrix.r0.z; outData[3] = matrix.r0.w;
	outData[4] = matrix.r1.x; outData[5] = matrix.r1.y; outData[6] = matrix.r1.z; outData[7] = matrix.r1.w;
	outData[8] = matrix.r2.x; outData[9] = matrix.r2.y; outData[10] = matrix.r2.z; outData[11] = matrix.r2.w;
	outData[12] = matrix.r3.x; outData[13] = matrix.r3.y; outData[14] = matrix.r3.z; outData[15] = matrix.r3.w;
}

/**
 * @brief Gets a row from a 4x4 matrix.
 * @param matrix Input matrix
 * @param rowIndex Row index (0-3)
 * @return Row vector. Returns zero vector if index is out of bounds.
 */
__device__ __forceinline__ float4 GetRow(const Mat4 matrix, int rowIndex)
{
	if (rowIndex < 0 || rowIndex > 3) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	const float4* rows = (const float4*)&matrix;
	return rows[rowIndex];
}

/**
 * @brief Sets a row in a 4x4 matrix.
 * @param matrix Pointer to matrix
 * @param rowIndex Row index (0-3)
 * @param values Values to set
 */
__device__ __forceinline__ void SetRow(Mat4* matrix, int rowIndex, float4 values)
{
	if (rowIndex < 0 || rowIndex > 3) return;
	float4* rows = (float4*)matrix;
	rows[rowIndex] = values;
}

/**
 * @brief Gets a column from a 4x4 matrix.
 * @param matrix Input matrix
 * @param colIndex Column index (0-3)
 * @return Column vector. Returns zero vector if index is out of bounds.
 */
__device__ __forceinline__ float4 GetColumn(const Mat4 matrix, int colIndex)
{
	if (colIndex < 0 || colIndex > 3) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (colIndex == 0) return make_float4(matrix.r0.x, matrix.r1.x, matrix.r2.x, matrix.r3.x);
	if (colIndex == 1) return make_float4(matrix.r0.y, matrix.r1.y, matrix.r2.y, matrix.r3.y);
	if (colIndex == 2) return make_float4(matrix.r0.z, matrix.r1.z, matrix.r2.z, matrix.r3.z);
	return make_float4(matrix.r0.w, matrix.r1.w, matrix.r2.w, matrix.r3.w);
}

/**
 * @brief Sets a column in a 4x4 matrix.
 * @param matrix Pointer to matrix
 * @param colIndex Column index (0-3)
 * @param values Values to set
 */
__device__ __forceinline__ void SetColumn(Mat4* matrix, int colIndex, float4 values)
{
	if (colIndex < 0 || colIndex > 3) return;
	if (colIndex == 0) { matrix->r0.x = values.x; matrix->r1.x = values.y; matrix->r2.x = values.z; matrix->r3.x = values.w; }
	else if (colIndex == 1) { matrix->r0.y = values.x; matrix->r1.y = values.y; matrix->r2.y = values.z; matrix->r3.y = values.w; }
	else if (colIndex == 2) { matrix->r0.z = values.x; matrix->r1.z = values.y; matrix->r2.z = values.z; matrix->r3.z = values.w; }
	else { matrix->r0.w = values.x; matrix->r1.w = values.y; matrix->r2.w = values.z; matrix->r3.w = values.w; }
}

/**
 * @brief Adds two 4x4 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Sum of matrices
 */
__device__ __forceinline__ Mat4 Add(const Mat4 left, const Mat4 right)
{
	Mat4 result;
	result.r0 = make_float4(left.r0.x + right.r0.x, left.r0.y + right.r0.y, left.r0.z + right.r0.z, left.r0.w + right.r0.w);
	result.r1 = make_float4(left.r1.x + right.r1.x, left.r1.y + right.r1.y, left.r1.z + right.r1.z, left.r1.w + right.r1.w);
	result.r2 = make_float4(left.r2.x + right.r2.x, left.r2.y + right.r2.y, left.r2.z + right.r2.z, left.r2.w + right.r2.w);
	result.r3 = make_float4(left.r3.x + right.r3.x, left.r3.y + right.r3.y, left.r3.z + right.r3.z, left.r3.w + right.r3.w);
	return result;
}

/**
 * @brief Subtracts two 4x4 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Difference of matrices
 */
__device__ __forceinline__ Mat4 Subtract(const Mat4 left, const Mat4 right)
{
	Mat4 result;
	result.r0 = make_float4(left.r0.x - right.r0.x, left.r0.y - right.r0.y, left.r0.z - right.r0.z, left.r0.w - right.r0.w);
	result.r1 = make_float4(left.r1.x - right.r1.x, left.r1.y - right.r1.y, left.r1.z - right.r1.z, left.r1.w - right.r1.w);
	result.r2 = make_float4(left.r2.x - right.r2.x, left.r2.y - right.r2.y, left.r2.z - right.r2.z, left.r2.w - right.r2.w);
	result.r3 = make_float4(left.r3.x - right.r3.x, left.r3.y - right.r3.y, left.r3.z - right.r3.z, left.r3.w - right.r3.w);
	return result;
}

/**
 * @brief Multiplies two 4x4 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Product of matrices (left ¡¿ right)
 * @note Matrix multiplication is not commutative
 */
__device__ __forceinline__ Mat4 Multiply(const Mat4 left, const Mat4 right)
{
	Mat4 result;
	result.r0.x = left.r0.x * right.r0.x + left.r0.y * right.r1.x + left.r0.z * right.r2.x + left.r0.w * right.r3.x;
	result.r0.y = left.r0.x * right.r0.y + left.r0.y * right.r1.y + left.r0.z * right.r2.y + left.r0.w * right.r3.y;
	result.r0.z = left.r0.x * right.r0.z + left.r0.y * right.r1.z + left.r0.z * right.r2.z + left.r0.w * right.r3.z;
	result.r0.w = left.r0.x * right.r0.w + left.r0.y * right.r1.w + left.r0.z * right.r2.w + left.r0.w * right.r3.w;
	result.r1.x = left.r1.x * right.r0.x + left.r1.y * right.r1.x + left.r1.z * right.r2.x + left.r1.w * right.r3.x;
	result.r1.y = left.r1.x * right.r0.y + left.r1.y * right.r1.y + left.r1.z * right.r2.y + left.r1.w * right.r3.y;
	result.r1.z = left.r1.x * right.r0.z + left.r1.y * right.r1.z + left.r1.z * right.r2.z + left.r1.w * right.r3.z;
	result.r1.w = left.r1.x * right.r0.w + left.r1.y * right.r1.w + left.r1.z * right.r2.w + left.r1.w * right.r3.w;
	result.r2.x = left.r2.x * right.r0.x + left.r2.y * right.r1.x + left.r2.z * right.r2.x + left.r2.w * right.r3.x;
	result.r2.y = left.r2.x * right.r0.y + left.r2.y * right.r1.y + left.r2.z * right.r2.y + left.r2.w * right.r3.y;
	result.r2.z = left.r2.x * right.r0.z + left.r2.y * right.r1.z + left.r2.z * right.r2.z + left.r2.w * right.r3.z;
	result.r2.w = left.r2.x * right.r0.w + left.r2.y * right.r1.w + left.r2.z * right.r2.w + left.r2.w * right.r3.w;
	result.r3.x = left.r3.x * right.r0.x + left.r3.y * right.r1.x + left.r3.z * right.r2.x + left.r3.w * right.r3.x;
	result.r3.y = left.r3.x * right.r0.y + left.r3.y * right.r1.y + left.r3.z * right.r2.y + left.r3.w * right.r3.y;
	result.r3.z = left.r3.x * right.r0.z + left.r3.y * right.r1.z + left.r3.z * right.r2.z + left.r3.w * right.r3.z;
	result.r3.w = left.r3.x * right.r0.w + left.r3.y * right.r1.w + left.r3.z * right.r2.w + left.r3.w * right.r3.w;
	return result;
}

/**
 * @brief Multiplies a 4x4 matrix by a 4D vector.
 * @param matrix Input matrix
 * @param vector Input vector
 * @return Transformed vector
 */
__device__ __forceinline__ float4 MultiplyVector(const Mat4 matrix, const float4 vector)
{
	float4 result;
	result.x = matrix.r0.x * vector.x + matrix.r0.y * vector.y + matrix.r0.z * vector.z + matrix.r0.w * vector.w;
	result.y = matrix.r1.x * vector.x + matrix.r1.y * vector.y + matrix.r1.z * vector.z + matrix.r1.w * vector.w;
	result.z = matrix.r2.x * vector.x + matrix.r2.y * vector.y + matrix.r2.z * vector.z + matrix.r2.w * vector.w;
	result.w = matrix.r3.x * vector.x + matrix.r3.y * vector.y + matrix.r3.z * vector.z + matrix.r3.w * vector.w;
	return result;
}

/**
 * @brief Multiplies a 4x4 matrix by a scalar.
 * @param matrix Input matrix
 * @param scalar Scalar value
 * @return Scaled matrix
 */
__device__ __forceinline__ Mat4 MultiplyScalar(const Mat4 matrix, float scalar)
{
	Mat4 result;
	result.r0 = make_float4(matrix.r0.x * scalar, matrix.r0.y * scalar, matrix.r0.z * scalar, matrix.r0.w * scalar);
	result.r1 = make_float4(matrix.r1.x * scalar, matrix.r1.y * scalar, matrix.r1.z * scalar, matrix.r1.w * scalar);
	result.r2 = make_float4(matrix.r2.x * scalar, matrix.r2.y * scalar, matrix.r2.z * scalar, matrix.r2.w * scalar);
	result.r3 = make_float4(matrix.r3.x * scalar, matrix.r3.y * scalar, matrix.r3.z * scalar, matrix.r3.w * scalar);
	return result;
}

/**
 * @brief Divides a 4x4 matrix by a scalar.
 * @param matrix Input matrix
 * @param scalar Scalar value (must not be zero)
 * @return Scaled matrix
 * @warning No check for division by zero
 */
__device__ __forceinline__ Mat4 DivideScalar(const Mat4 matrix, float scalar)
{
	float invScalar = 1.0f / scalar;
	Mat4 result;
	result.r0 = make_float4(matrix.r0.x * invScalar, matrix.r0.y * invScalar, matrix.r0.z * invScalar, matrix.r0.w * invScalar);
	result.r1 = make_float4(matrix.r1.x * invScalar, matrix.r1.y * invScalar, matrix.r1.z * invScalar, matrix.r1.w * invScalar);
	result.r2 = make_float4(matrix.r2.x * invScalar, matrix.r2.y * invScalar, matrix.r2.z * invScalar, matrix.r2.w * invScalar);
	result.r3 = make_float4(matrix.r3.x * invScalar, matrix.r3.y * invScalar, matrix.r3.z * invScalar, matrix.r3.w * invScalar);
	return result;
}

/**
 * @brief Performs element-wise multiplication of two 4x4 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Element-wise product
 */
__device__ __forceinline__ Mat4 MultiplyElementWise(const Mat4 left, const Mat4 right)
{
	Mat4 result;
	result.r0 = make_float4(left.r0.x * right.r0.x, left.r0.y * right.r0.y, left.r0.z * right.r0.z, left.r0.w * right.r0.w);
	result.r1 = make_float4(left.r1.x * right.r1.x, left.r1.y * right.r1.y, left.r1.z * right.r1.z, left.r1.w * right.r1.w);
	result.r2 = make_float4(left.r2.x * right.r2.x, left.r2.y * right.r2.y, left.r2.z * right.r2.z, left.r2.w * right.r2.w);
	result.r3 = make_float4(left.r3.x * right.r3.x, left.r3.y * right.r3.y, left.r3.z * right.r3.z, left.r3.w * right.r3.w);
	return result;
}

/**
 * @brief Performs element-wise division of two 4x4 matrices.
 * @param left First matrix
 * @param right Second matrix
 * @return Element-wise quotient
 * @warning No check for division by zero
 */
__device__ __forceinline__ Mat4 DivideElementWise(const Mat4 left, const Mat4 right)
{
	Mat4 result;
	result.r0 = make_float4(left.r0.x / right.r0.x, left.r0.y / right.r0.y, left.r0.z / right.r0.z, left.r0.w / right.r0.w);
	result.r1 = make_float4(left.r1.x / right.r1.x, left.r1.y / right.r1.y, left.r1.z / right.r1.z, left.r1.w / right.r1.w);
	result.r2 = make_float4(left.r2.x / right.r2.x, left.r2.y / right.r2.y, left.r2.z / right.r2.z, left.r2.w / right.r2.w);
	result.r3 = make_float4(left.r3.x / right.r3.x, left.r3.y / right.r3.y, left.r3.z / right.r3.z, left.r3.w / right.r3.w);
	return result;
}

/**
 * @brief Calculates the determinant of a 4x4 matrix.
 * @param m Input matrix
 * @return Determinant value
 */
__device__ __forceinline__ float GetDeterminant(const Mat4 m)
{
	float f0 = m.r0.x * (m.r1.y * (m.r2.z * m.r3.w - m.r2.w * m.r3.z) - m.r1.z * (m.r2.y * m.r3.w - m.r2.w * m.r3.y) + m.r1.w * (m.r2.y * m.r3.z - m.r2.z * m.r3.y));
	float f1 = m.r0.y * (m.r1.x * (m.r2.z * m.r3.w - m.r2.w * m.r3.z) - m.r1.z * (m.r2.x * m.r3.w - m.r2.w * m.r3.x) + m.r1.w * (m.r2.x * m.r3.z - m.r2.z * m.r3.x));
	float f2 = m.r0.z * (m.r1.x * (m.r2.y * m.r3.w - m.r2.w * m.r3.y) - m.r1.y * (m.r2.x * m.r3.w - m.r2.w * m.r3.x) + m.r1.w * (m.r2.x * m.r3.y - m.r2.y * m.r3.x));
	float f3 = m.r0.w * (m.r1.x * (m.r2.y * m.r3.z - m.r2.z * m.r3.y) - m.r1.y * (m.r2.x * m.r3.z - m.r2.z * m.r3.x) + m.r1.z * (m.r2.x * m.r3.y - m.r2.y * m.r3.x));
	return f0 - f1 + f2 - f3;
}

/**
 * @brief Computes the inverse of a 4x4 matrix.
 * @param m Input matrix
 * @return Inverse matrix. Returns zero matrix if determinant is near zero.
 * @warning Singular matrices return zero matrix
 */
__device__ __forceinline__ Mat4 Inverse(const Mat4 m)
{
	float n11 = m.r0.x, n12 = m.r0.y, n13 = m.r0.z, n14 = m.r0.w;
	float n21 = m.r1.x, n22 = m.r1.y, n23 = m.r1.z, n24 = m.r1.w;
	float n31 = m.r2.x, n32 = m.r2.y, n33 = m.r2.z, n34 = m.r2.w;
	float n41 = m.r3.x, n42 = m.r3.y, n43 = m.r3.z, n44 = m.r3.w;
	float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
	float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
	float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
	float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
	float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
	if (fabsf(det) < CUDA_MATH_EPSILON) return SetZero();
	float invDet = 1.0f / det;
	Mat4 res;
	res.r0 = make_float4(t11 * invDet, t12 * invDet, t13 * invDet, t14 * invDet);
	res.r1.x = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * invDet;
	res.r1.y = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * invDet;
	res.r1.z = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * invDet;
	res.r1.w = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * invDet;
	res.r2.x = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * invDet;
	res.r2.y = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * invDet;
	res.r2.z = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * invDet;
	res.r2.w = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * invDet;
	res.r3.x = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * invDet;
	res.r3.y = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * invDet;
	res.r3.z = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * invDet;
	res.r3.w = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * invDet;
	return res;
}

/**
 * @brief Transposes a 4x4 matrix.
 * @param matrix Input matrix
 * @return Transposed matrix
 */
__device__ __forceinline__ Mat4 Transpose(const Mat4 matrix)
{
	Mat4 result;
	result.r0 = make_float4(matrix.r0.x, matrix.r1.x, matrix.r2.x, matrix.r3.x);
	result.r1 = make_float4(matrix.r0.y, matrix.r1.y, matrix.r2.y, matrix.r3.y);
	result.r2 = make_float4(matrix.r0.z, matrix.r1.z, matrix.r2.z, matrix.r3.z);
	result.r3 = make_float4(matrix.r0.w, matrix.r1.w, matrix.r2.w, matrix.r3.w);
	return result;
}

/**
 * @brief Extracts the upper-left 3x3 submatrix from a 4x4 matrix.
 * @param matrix Input 4x4 matrix
 * @return 3x3 matrix
 */
__device__ __forceinline__ Mat3 GetMat3(const Mat4 matrix)
{
	Mat3 result;
	result.r0 = make_float3(matrix.r0.x, matrix.r0.y, matrix.r0.z);
	result.r1 = make_float3(matrix.r1.x, matrix.r1.y, matrix.r1.z);
	result.r2 = make_float3(matrix.r2.x, matrix.r2.y, matrix.r2.z);
	return result;
}

/**
 * @brief Extracts the translation component from a 4x4 transformation matrix.
 * @param matrix Input transformation matrix
 * @return Translation vector
 */
__device__ __forceinline__ float3 GetTranslation(const Mat4 matrix)
{
	return make_float3(matrix.r0.w, matrix.r1.w, matrix.r2.w);
}

/**
 * @brief Extracts the scale component from a 4x4 transformation matrix.
 * @param matrix Input transformation matrix
 * @return Scale vector
 * @note Handles negative scale if determinant is negative
 */
__device__ __forceinline__ float3 GetScale(const Mat4 matrix)
{
	float3 s;
	s.x = sqrtf(matrix.r0.x * matrix.r0.x + matrix.r1.x * matrix.r1.x + matrix.r2.x * matrix.r2.x);
	s.y = sqrtf(matrix.r0.y * matrix.r0.y + matrix.r1.y * matrix.r1.y + matrix.r2.y * matrix.r2.y);
	s.z = sqrtf(matrix.r0.z * matrix.r0.z + matrix.r1.z * matrix.r1.z + matrix.r2.z * matrix.r2.z);
	// Handle negative scale if determinant is negative
	if (GetDeterminant(matrix) < 0.0f) s.x = -s.x;
	return s;
}

/**
 * @brief Transforms a 3D point by a 4x4 matrix (applies translation).
 * @param matrix Transformation matrix
 * @param p Point to transform
 * @return Transformed point
 */
__device__ __forceinline__ float3 TransformPoint(const Mat4 matrix, float3 p)
{
	float3 res;
	res.x = matrix.r0.x * p.x + matrix.r0.y * p.y + matrix.r0.z * p.z + matrix.r0.w;
	res.y = matrix.r1.x * p.x + matrix.r1.y * p.y + matrix.r1.z * p.z + matrix.r1.w;
	res.z = matrix.r2.x * p.x + matrix.r2.y * p.y + matrix.r2.z * p.z + matrix.r2.w;
	return res;
}

/**
 * @brief Transforms a 3D direction by a 4x4 matrix (ignores translation).
 * @param matrix Transformation matrix
 * @param d Direction to transform
 * @return Transformed direction
 */
__device__ __forceinline__ float3 TransformDirection(const Mat4 matrix, float3 d)
{
	float3 res;
	res.x = matrix.r0.x * d.x + matrix.r0.y * d.y + matrix.r0.z * d.z;
	res.y = matrix.r1.x * d.x + matrix.r1.y * d.y + matrix.r1.z * d.z;
	res.z = matrix.r2.x * d.x + matrix.r2.y * d.y + matrix.r2.z * d.z;
	return res;
}

/**
 * @brief Transforms a normal vector by a 4x4 matrix (uses inverse-transpose).
 * @param matrix Transformation matrix
 * @param n Normal to transform
 * @return Transformed and normalized normal
 */
__device__ __forceinline__ float3 MultiplyNormalMat3(const Mat4 matrix, float3 n)
{
	Mat3 m3 = GetMat3(matrix);
	Mat3 invTrp = TransposeMat3(InverseMat3(m3));
	return Normalize(MultiplyMat3Vector(invTrp, n));
}

/**
 * @brief Creates a 4x4 translation matrix.
 * @param x Translation along X axis
 * @param y Translation along Y axis
 * @param z Translation along Z axis
 * @return Translation matrix
 */
__device__ __forceinline__ Mat4 SetTranslation(float x, float y, float z)
{
	Mat4 res = SetIdentity();
	res.r0.w = x; res.r1.w = y; res.r2.w = z;
	return res;
}

/**
 * @brief Creates a 4x4 scale matrix.
 * @param x Scale along X axis
 * @param y Scale along Y axis
 * @param z Scale along Z axis
 * @return Scale matrix
 */
__device__ __forceinline__ Mat4 SetScale(float x, float y, float z)
{
	Mat4 res = SetZero();
	res.r0.x = x; res.r1.y = y; res.r2.z = z; res.r3.w = 1.0f;
	return res;
}

/**
 * @brief Creates a 4x4 shear matrix.
 * @param xy Shear X by Y
 * @param xz Shear X by Z
 * @param yx Shear Y by X
 * @param yz Shear Y by Z
 * @param zx Shear Z by X
 * @param zy Shear Z by Y
 * @return Shear matrix
 */
__device__ __forceinline__ Mat4 SetShear(float xy, float xz, float yx, float yz, float zx, float zy)
{
	Mat4 res = SetIdentity();
	res.r0.y = xy; res.r0.z = xz; res.r1.x = yx; res.r1.z = yz; res.r2.x = zx; res.r2.y = zy;
	return res;
}

/**
 * @brief Creates a 4x4 rotation matrix around X axis.
 * @param a Angle in radians
 * @return Rotation matrix
 */
__device__ __forceinline__ Mat4 SetRotationX(float a)
{
	Mat4 res = SetIdentity();
	float s = sinf(a), c = cosf(a);
	res.r1.y = c; res.r1.z = -s; res.r2.y = s; res.r2.z = c;
	return res;
}

/**
 * @brief Creates a 4x4 rotation matrix around Y axis.
 * @param a Angle in radians
 * @return Rotation matrix
 */
__device__ __forceinline__ Mat4 SetRotationY(float a)
{
	Mat4 res = SetIdentity();
	float s = sinf(a), c = cosf(a);
	res.r0.x = c; res.r0.z = s; res.r2.x = -s; res.r2.z = c;
	return res;
}

/**
 * @brief Creates a 4x4 rotation matrix around Z axis.
 * @param a Angle in radians
 * @return Rotation matrix
 */
__device__ __forceinline__ Mat4 SetRotationZ(float a)
{
	Mat4 res = SetIdentity();
	float s = sinf(a), c = cosf(a);
	res.r0.x = c; res.r0.y = -s; res.r1.x = s; res.r1.y = c;
	return res;
}

/**
 * @brief Creates a 4x4 rotation matrix around an arbitrary axis.
 * @param axis Rotation axis (will be normalized)
 * @param angle Rotation angle in radians
 * @return Rotation matrix
 */
__device__ __forceinline__ Mat4 SetRotationAxis(float3 axis, float angle)
{
	Mat4 res = SetIdentity();
	float3 a = Normalize(axis);
	float s = sinf(angle), c = cosf(angle), oc = 1.0f - c;
	res.r0.x = oc * a.x * a.x + c; res.r0.y = oc * a.x * a.y - a.z * s; res.r0.z = oc * a.z * a.x + a.y * s;
	res.r1.x = oc * a.x * a.y + a.z * s; res.r1.y = oc * a.y * a.y + c; res.r1.z = oc * a.y * a.z - a.x * s;
	res.r2.x = oc * a.z * a.x - a.y * s; res.r2.y = oc * a.y * a.z + a.x * s; res.r2.z = oc * a.z * a.z + c;
	return res;
}

/**
 * @brief Creates a 4x4 transformation matrix from translation, rotation (Euler angles), and scale.
 * @param t Translation vector
 * @param r_euler Rotation as Euler angles (in radians)
 * @param s Scale vector
 * @return Transformation matrix (T * R * S)
 */
__device__ __forceinline__ Mat4 SetTransform(float3 t, float3 r_euler, float3 s)
{
	Mat4 mt = SetTranslation(t.x, t.y, t.z);
	Mat4 rx = SetRotationX(r_euler.x);
	Mat4 ry = SetRotationY(r_euler.y);
	Mat4 rz = SetRotationZ(r_euler.z);
	Mat4 ms = SetScale(s.x, s.y, s.z);
	Mat4 r = Multiply(rz, Multiply(ry, rx));
	return Multiply(mt, Multiply(r, ms));
}

/**
 * @brief Creates a view matrix (look-at matrix).
 * @param eye Camera position
 * @param center Target position
 * @param up Up vector
 * @return View matrix
 */
__device__ __forceinline__ Mat4 LookAt(float3 eye, float3 center, float3 up)
{
	float3 f = Normalize(make_float3(center.x - eye.x, center.y - eye.y, center.z - eye.z));
	float3 s = Normalize(Cross(f, up));
	float3 u = Cross(s, f);
	Mat4 res = SetIdentity();
	res.r0 = make_float4(s.x, s.y, s.z, -(s.x * eye.x + s.y * eye.y + s.z * eye.z));
	res.r1 = make_float4(u.x, u.y, u.z, -(u.x * eye.x + u.y * eye.y + u.z * eye.z));
	res.r2 = make_float4(-f.x, -f.y, -f.z, (f.x * eye.x + f.y * eye.y + f.z * eye.z));
	return res;
}

/**
 * @brief Creates a perspective projection matrix.
 * @param fov Field of view in radians
 * @param aspect Aspect ratio (width/height)
 * @param nearZ Near clipping plane
 * @param farZ Far clipping plane
 * @return Perspective projection matrix
 */
__device__ __forceinline__ Mat4 Perspective(float fov, float aspect, float nearZ, float farZ)
{
	float h = 1.0f / tanf(fov * 0.5f);
	Mat4 res = SetZero();
	res.r0.x = h / aspect; res.r1.y = h;
	res.r2.z = -(farZ + nearZ) / (farZ - nearZ); res.r2.w = -(2.0f * farZ * nearZ) / (farZ - nearZ);
	res.r3.z = -1.0f;
	return res;
}

/**
 * @brief Creates an orthographic projection matrix.
 * @param l Left clipping plane
 * @param r Right clipping plane
 * @param b Bottom clipping plane
 * @param t Top clipping plane
 * @param n Near clipping plane
 * @param f Far clipping plane
 * @return Orthographic projection matrix
 */
__device__ __forceinline__ Mat4 Ortho(float l, float r, float b, float t, float n, float f)
{
	Mat4 res = SetIdentity();
	res.r0.x = 2.0f / (r - l); res.r0.w = -(r + l) / (r - l);
	res.r1.y = 2.0f / (t - b); res.r1.w = -(t + b) / (t - b);
	res.r2.z = -2.0f / (f - n); res.r2.w = -(f + n) / (f - n);
	return res;
}

/**
 * @brief Transforms a 3D point with projection and perspective divide.
 * @param matrix Projection matrix
 * @param point Point to transform
 * @return Transformed point (with perspective divide applied)
 */
__device__ __forceinline__ float3 TransformPointWithProjection(const Mat4 matrix, float3 point)
{
	float4 v = make_float4(point.x, point.y, point.z, 1.0f);
	float4 tr = MultiplyVector(matrix, v);
	if (tr.w != 0.0f && tr.w != 1.0f) { float inv = 1.0f / tr.w; return make_float3(tr.x * inv, tr.y * inv, tr.z * inv); }
	return make_float3(tr.x, tr.y, tr.z);
}

/**
 * @brief Transforms a 3D direction with projection (no perspective divide).
 * @param matrix Projection matrix
 * @param direction Direction to transform
 * @return Transformed direction
 */
__device__ __forceinline__ float3 TransformDirectionWithProjection(const Mat4 matrix, float3 direction)
{
	float4 v = make_float4(direction.x, direction.y, direction.z, 0.0f);
	float4 tr = MultiplyVector(matrix, v);
	return make_float3(tr.x, tr.y, tr.z);
}

/**
 * @brief Sets the upper-left 3x3 submatrix of a 4x4 matrix.
 * @param m Pointer to 4x4 matrix
 * @param m3 3x3 matrix to copy from
 */
__device__ __forceinline__ void SetUpper3x3(Mat4* m, const Mat3 m3)
{
	m->r0.x = m3.r0.x; m->r0.y = m3.r0.y; m->r0.z = m3.r0.z;
	m->r1.x = m3.r1.x; m->r1.y = m3.r1.y; m->r1.z = m3.r1.z;
	m->r2.x = m3.r2.x; m->r2.y = m3.r2.y; m->r2.z = m3.r2.z;
}

/**
 * @brief Extracts the upper-left 3x3 submatrix as a 4x4 matrix with identity in bottom-right.
 * @param m Input 4x4 matrix
 * @return 4x4 matrix with upper-left 3x3 copied and identity in bottom-right
 */
__device__ __forceinline__ Mat4 GetUpper3x3AsMat4(const Mat4 m)
{
	Mat4 res = SetZero();
	res.r0 = make_float4(m.r0.x, m.r0.y, m.r0.z, 0.0f);
	res.r1 = make_float4(m.r1.x, m.r1.y, m.r1.z, 0.0f);
	res.r2 = make_float4(m.r2.x, m.r2.y, m.r2.z, 0.0f);
	res.r3 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	return res;
}

/**
 * @brief Calculates the trace of a 4x4 matrix.
 * @param m Input matrix
 * @return Trace (sum of diagonal elements)
 */
__device__ __forceinline__ float GetTrace(const Mat4 m)
{
	return m.r0.x + m.r1.y + m.r2.z + m.r3.w;
}

/**
 * @brief Checks if two 4x4 matrices are approximately equal.
 * @param a First matrix
 * @param b Second matrix
 * @param e Tolerance for comparison
 * @return true if matrices are equal within epsilon, false otherwise
 */
__device__ __forceinline__ bool IsEqual(const Mat4 a, const Mat4 b, float e)
{
	return fabsf(a.r0.x - b.r0.x) < e && fabsf(a.r0.y - b.r0.y) < e && fabsf(a.r0.z - b.r0.z) < e && fabsf(a.r0.w - b.r0.w) < e &&
		fabsf(a.r1.x - b.r1.x) < e && fabsf(a.r1.y - b.r1.y) < e && fabsf(a.r1.z - b.r1.z) < e && fabsf(a.r1.w - b.r1.w) < e &&
		fabsf(a.r2.x - b.r2.x) < e && fabsf(a.r2.y - b.r2.y) < e && fabsf(a.r2.z - b.r2.z) < e && fabsf(a.r2.w - b.r2.w) < e &&
		fabsf(a.r3.x - b.r3.x) < e && fabsf(a.r3.y - b.r3.y) < e && fabsf(a.r3.z - b.r3.z) < e && fabsf(a.r3.w - b.r3.w) < e;
}

/**
 * @brief Linearly interpolates between two 4x4 matrices.
 * @param a First matrix
 * @param b Second matrix
 * @param t Interpolation factor (0.0 to 1.0)
 * @return Interpolated matrix
 */
__device__ __forceinline__ Mat4 Lerp(const Mat4 a, const Mat4 b, float t)
{
	Mat4 res;
	res.r0 = make_float4(a.r0.x + (b.r0.x - a.r0.x) * t, a.r0.y + (b.r0.y - a.r0.y) * t, a.r0.z + (b.r0.z - a.r0.z) * t, a.r0.w + (b.r0.w - a.r0.w) * t);
	res.r1 = make_float4(a.r1.x + (b.r1.x - a.r1.x) * t, a.r1.y + (b.r1.y - a.r1.y) * t, a.r1.z + (b.r1.z - a.r1.z) * t, a.r1.w + (b.r1.w - a.r1.w) * t);
	res.r2 = make_float4(a.r2.x + (b.r2.x - a.r2.x) * t, a.r2.y + (b.r2.y - a.r2.y) * t, a.r2.z + (b.r2.z - a.r2.z) * t, a.r2.w + (b.r2.w - a.r2.w) * t);
	res.r3 = make_float4(a.r3.x + (b.r3.x - a.r3.x) * t, a.r3.y + (b.r3.y - a.r3.y) * t, a.r3.z + (b.r3.z - a.r3.z) * t, a.r3.w + (b.r3.w - a.r3.w) * t);
	return res;
}
#endif COPPER_CUDA_MATH_H
