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
		const Eigen::Vector4f& color);

	static void BuildPlane(
		DebuggingRenderable* debuggingRenderable,
		float width,
		float height,
		unsigned int hSegments,
		unsigned int vSegments,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		const Eigen::Vector4f& color);
};