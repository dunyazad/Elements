#pragma once

#include <Helium/HeliumCommon.h>

#include <string>
#include <type_traits>

#include <Eigen/Dense>

enum class TextHAlign
{
	Left,
	Center,
	Right
};

enum class TextVAlign
{
	Top,
	Middle,
	Bottom,
	Baseline
};

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
		: x(x), y(y), width(width), height(height), color(color) {
	}
};

struct GUICircle : GUIComponent
{
	float x;
	float y;
	float radius;
	Eigen::Vector4f color;
	GUICircle(float x, float y, float radius, const Eigen::Vector4f& color)
		: x(x), y(y), radius(radius), color(color) {
	}
};

struct GUIText : GUIComponent
{
	float x;
	float y;
	float fontSize;
	Eigen::Vector4f color;
	std::string text;

	TextHAlign hAlign;
	TextVAlign vAlign;

	float cachedWidth = 0.0f;
	std::string _lastText = "";
	float _lastFontSize = 0.0f;

	GUIText(float x, float y, float fontSize, const Eigen::Vector4f& color, const std::string& text,
		TextHAlign hAlign = TextHAlign::Left, TextVAlign vAlign = TextVAlign::Baseline)
		: x(x), y(y), fontSize(fontSize), color(color), text(text), hAlign(hAlign), vAlign(vAlign) {
	}
};

template<typename T>
struct is_gui_component : std::false_type {};

template<> struct is_gui_component<GUIComponent> : std::true_type {};
template<> struct is_gui_component<GUIRectangle> : std::true_type {};
template<> struct is_gui_component<GUICircle> : std::true_type {};
template<> struct is_gui_component<GUIText> : std::true_type {};