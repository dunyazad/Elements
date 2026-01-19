#pragma once

#include <Helium/HeliumCommon.h>

#include <string>
#include <type_traits>

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

	GUIRectangle(float x, float y, float width, float height, const Eigen::Vector4f& color)
		: x(x), y(y), width(width), height(height), color(color) {}
};

struct GUIText : GUIComponent
{
	float x;
	float y;
	float fontSize;
	Eigen::Vector4f color;
	std::string text;

	GUIText(float x, float y, float fontSize, const Eigen::Vector4f& color, const std::string& text)
		: x(x), y(y), fontSize(fontSize), color(color), text(text) {}
};

template<typename T>
struct is_gui_component : std::false_type {};

template<> struct is_gui_component<GUIComponent> : std::true_type {};
template<> struct is_gui_component<GUIRectangle> : std::true_type {};
template<> struct is_gui_component<GUIText> : std::true_type {};
