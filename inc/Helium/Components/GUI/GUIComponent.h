#pragma once

#include <string>

#include <Eigen/Dense>

struct GUIComponent
{
	int zIndex;
};

struct GUIRectangle : GUIComponent
{
	float x;
	float y;
	float width;
	float height;
	Eigen::Vector4f color;
};

struct GUIText : GUIComponent
{
	float x;
	float y;
	float fontSize;
	Eigen::Vector4f color;
	std::string text;
};
