#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

#undef min
#undef max

namespace Color
{
	// glm::fract 대응을 위한 헬퍼 함수
	inline float fract(float x)
	{
		return x - std::floor(x);
	}

	inline Eigen::Vector4f black(float alpha = 1.0f) { return { 0.0f, 0.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f navy(float alpha = 1.0f) { return { 0.0f, 0.0f, 0.502f, alpha }; }
	inline Eigen::Vector4f midnightblue(float alpha = 1.0f) { return { 0.098f, 0.098f, 0.439f, alpha }; }
	inline Eigen::Vector4f darkblue(float alpha = 1.0f) { return { 0.0f, 0.0f, 0.545f, alpha }; }
	inline Eigen::Vector4f indigo(float alpha = 1.0f) { return { 0.294f, 0.0f, 0.51f, alpha }; }
	inline Eigen::Vector4f maroon(float alpha = 1.0f) { return { 0.502f, 0.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f purple(float alpha = 1.0f) { return { 0.502f, 0.0f, 0.502f, alpha }; }
	inline Eigen::Vector4f darkred(float alpha = 1.0f) { return { 0.545f, 0.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f darkmagenta(float alpha = 1.0f) { return { 0.545f, 0.0f, 0.545f, alpha }; }
	inline Eigen::Vector4f darkviolet(float alpha = 1.0f) { return { 0.58f, 0.0f, 0.827f, alpha }; }
	inline Eigen::Vector4f red(float alpha = 1.0f) { return { 1.0f, 0.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f mediumblue(float alpha = 1.0f) { return { 0.0f, 0.0f, 0.804f, alpha }; }
	inline Eigen::Vector4f blue(float alpha = 1.0f) { return { 0.0f, 0.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f darkslategray(float alpha = 1.0f) { return { 0.184f, 0.31f, 0.31f, alpha }; }
	inline Eigen::Vector4f darkslategrey(float alpha = 1.0f) { return { 0.184f, 0.31f, 0.31f, alpha }; }
	inline Eigen::Vector4f rebeccapurple(float alpha = 1.0f) { return { 0.4f, 0.2f, 0.6f, alpha }; }
	inline Eigen::Vector4f darkslateblue(float alpha = 1.0f) { return { 0.282f, 0.239f, 0.545f, alpha }; }
	inline Eigen::Vector4f brown(float alpha = 1.0f) { return { 0.647f, 0.165f, 0.165f, alpha }; }
	inline Eigen::Vector4f firebrick(float alpha = 1.0f) { return { 0.698f, 0.133f, 0.133f, alpha }; }
	inline Eigen::Vector4f blueviolet(float alpha = 1.0f) { return { 0.541f, 0.169f, 0.886f, alpha }; }
	inline Eigen::Vector4f darkgreen(float alpha = 1.0f) { return { 0.0f, 0.392f, 0.0f, alpha }; }
	inline Eigen::Vector4f green(float alpha = 1.0f) { return { 0.0f, 0.502f, 0.0f, alpha }; }
	inline Eigen::Vector4f teal(float alpha = 1.0f) { return { 0.0f, 0.502f, 0.502f, alpha }; }
	inline Eigen::Vector4f darkcyan(float alpha = 1.0f) { return { 0.0f, 0.545f, 0.545f, alpha }; }
	inline Eigen::Vector4f saddlebrown(float alpha = 1.0f) { return { 0.545f, 0.271f, 0.075f, alpha }; }
	inline Eigen::Vector4f darkolivegreen(float alpha = 1.0f) { return { 0.333f, 0.42f, 0.184f, alpha }; }
	inline Eigen::Vector4f sienna(float alpha = 1.0f) { return { 0.627f, 0.322f, 0.176f, alpha }; }
	inline Eigen::Vector4f forestgreen(float alpha = 1.0f) { return { 0.133f, 0.545f, 0.133f, alpha }; }
	inline Eigen::Vector4f dimgray(float alpha = 1.0f) { return { 0.412f, 0.412f, 0.412f, alpha }; }
	inline Eigen::Vector4f dimgrey(float alpha = 1.0f) { return { 0.412f, 0.412f, 0.412f, alpha }; }
	inline Eigen::Vector4f slategray(float alpha = 1.0f) { return { 0.439f, 0.502f, 0.565f, alpha }; }
	inline Eigen::Vector4f slategrey(float alpha = 1.0f) { return { 0.439f, 0.502f, 0.565f, alpha }; }
	inline Eigen::Vector4f royalblue(float alpha = 1.0f) { return { 0.255f, 0.412f, 0.882f, alpha }; }
	inline Eigen::Vector4f slateblue(float alpha = 1.0f) { return { 0.416f, 0.353f, 0.804f, alpha }; }
	inline Eigen::Vector4f crimson(float alpha = 1.0f) { return { 0.863f, 0.078f, 0.235f, alpha }; }
	inline Eigen::Vector4f darkorchid(float alpha = 1.0f) { return { 0.6f, 0.196f, 0.8f, alpha }; }
	inline Eigen::Vector4f mediumvioletred(float alpha = 1.0f) { return { 0.78f, 0.082f, 0.522f, alpha }; }
	inline Eigen::Vector4f olive(float alpha = 1.0f) { return { 0.502f, 0.502f, 0.0f, alpha }; }
	inline Eigen::Vector4f gray(float alpha = 1.0f) { return { 0.502f, 0.502f, 0.502f, alpha }; }
	inline Eigen::Vector4f grey(float alpha = 1.0f) { return { 0.502f, 0.502f, 0.502f, alpha }; }
	inline Eigen::Vector4f lightslategray(float alpha = 1.0f) { return { 0.467f, 0.533f, 0.6f, alpha }; }
	inline Eigen::Vector4f lightslategrey(float alpha = 1.0f) { return { 0.467f, 0.533f, 0.6f, alpha }; }
	inline Eigen::Vector4f mediumslateblue(float alpha = 1.0f) { return { 0.482f, 0.408f, 0.933f, alpha }; }
	inline Eigen::Vector4f steelblue(float alpha = 1.0f) { return { 0.275f, 0.510f, 0.706f, alpha }; }
	inline Eigen::Vector4f seagreen(float alpha = 1.0f) { return { 0.18f, 0.545f, 0.341f, alpha }; }
	inline Eigen::Vector4f olivedrab(float alpha = 1.0f) { return { 0.42f, 0.557f, 0.137f, alpha }; }
	inline Eigen::Vector4f cadetblue(float alpha = 1.0f) { return { 0.373f, 0.62f, 0.627f, alpha }; }
	inline Eigen::Vector4f cornflowerblue(float alpha = 1.0f) { return { 0.392f, 0.584f, 0.929f, alpha }; }
	inline Eigen::Vector4f mediumseagreen(float alpha = 1.0f) { return { 0.235f, 0.702f, 0.443f, alpha }; }
	inline Eigen::Vector4f dodgerblue(float alpha = 1.0f) { return { 0.118f, 0.565f, 1.0f, alpha }; }
	inline Eigen::Vector4f chocolate(float alpha = 1.0f) { return { 0.824f, 0.412f, 0.118f, alpha }; }
	inline Eigen::Vector4f darkgoldenrod(float alpha = 1.0f) { return { 0.722f, 0.525f, 0.043f, alpha }; }
	inline Eigen::Vector4f mediumpurple(float alpha = 1.0f) { return { 0.576f, 0.439f, 0.859f, alpha }; }
	inline Eigen::Vector4f peru(float alpha = 1.0f) { return { 0.804f, 0.522f, 0.247f, alpha }; }
	inline Eigen::Vector4f rosybrown(float alpha = 1.0f) { return { 0.737f, 0.561f, 0.561f, alpha }; }
	inline Eigen::Vector4f indianred(float alpha = 1.0f) { return { 0.804f, 0.361f, 0.361f, alpha }; }
	inline Eigen::Vector4f deeppink(float alpha = 1.0f) { return { 1.0f, 0.078f, 0.576f, alpha }; }
	inline Eigen::Vector4f orangered(float alpha = 1.0f) { return { 1.0f, 0.271f, 0.0f, alpha }; }
	inline Eigen::Vector4f mediumorchid(float alpha = 1.0f) { return { 0.729f, 0.333f, 0.827f, alpha }; }
	inline Eigen::Vector4f darkgray(float alpha = 1.0f) { return { 0.663f, 0.663f, 0.663f, alpha }; }
	inline Eigen::Vector4f darkgrey(float alpha = 1.0f) { return { 0.663f, 0.663f, 0.663f, alpha }; }
	inline Eigen::Vector4f darkseagreen(float alpha = 1.0f) { return { 0.561f, 0.737f, 0.561f, alpha }; }
	inline Eigen::Vector4f lightseagreen(float alpha = 1.0f) { return { 0.125f, 0.698f, 0.667f, alpha }; }
	inline Eigen::Vector4f goldenrod(float alpha = 1.0f) { return { 0.855f, 0.647f, 0.125f, alpha }; }
	inline Eigen::Vector4f palevioletred(float alpha = 1.0f) { return { 0.859f, 0.439f, 0.576f, alpha }; }
	inline Eigen::Vector4f mediumaquamarine(float alpha = 1.0f) { return { 0.4f, 0.804f, 0.667f, alpha }; }
	inline Eigen::Vector4f darksalmon(float alpha = 1.0f) { return { 0.914f, 0.588f, 0.478f, alpha }; }
	inline Eigen::Vector4f darkorange(float alpha = 1.0f) { return { 1.0f, 0.549f, 0.0f, alpha }; }
	inline Eigen::Vector4f limegreen(float alpha = 1.0f) { return { 0.196f, 0.804f, 0.196f, alpha }; }
	inline Eigen::Vector4f coral(float alpha = 1.0f) { return { 1.0f, 0.498f, 0.314f, alpha }; }
	inline Eigen::Vector4f silver(float alpha = 1.0f) { return { 0.753f, 0.753f, 0.753f, alpha }; }
	inline Eigen::Vector4f darkkhaki(float alpha = 1.0f) { return { 0.741f, 0.718f, 0.42f, alpha }; }
	inline Eigen::Vector4f mediumturquoise(float alpha = 1.0f) { return { 0.282f, 0.82f, 0.8f, alpha }; }
	inline Eigen::Vector4f tan(float alpha = 1.0f) { return { 0.824f, 0.706f, 0.549f, alpha }; }
	inline Eigen::Vector4f skyblue(float alpha = 1.0f) { return { 0.529f, 0.808f, 0.922f, alpha }; }
	inline Eigen::Vector4f lightskyblue(float alpha = 1.0f) { return { 0.529f, 0.808f, 0.98f, alpha }; }
	inline Eigen::Vector4f darkturquoise(float alpha = 1.0f) { return { 0.0f, 0.808f, 0.82f, alpha }; }
	inline Eigen::Vector4f burlywood(float alpha = 1.0f) { return { 0.871f, 0.722f, 0.529f, alpha }; }
	inline Eigen::Vector4f salmon(float alpha = 1.0f) { return { 0.98f, 0.502f, 0.447f, alpha }; }
	inline Eigen::Vector4f lightsteelblue(float alpha = 1.0f) { return { 0.690f, 0.769f, 0.871f, alpha }; }
	inline Eigen::Vector4f orange(float alpha = 1.0f) { return { 1.0f, 0.647f, 0.0f, alpha }; }
	inline Eigen::Vector4f lightcoral(float alpha = 1.0f) { return { 0.941f, 0.502f, 0.502f, alpha }; }
	inline Eigen::Vector4f orchid(float alpha = 1.0f) { return { 0.855f, 0.439f, 0.839f, alpha }; }
	inline Eigen::Vector4f hotpink(float alpha = 1.0f) { return { 1.0f, 0.412f, 0.706f, alpha }; }
	inline Eigen::Vector4f tomato(float alpha = 1.0f) { return { 1.0f, 0.388f, 0.278f, alpha }; }
	inline Eigen::Vector4f yellowgreen(float alpha = 1.0f) { return { 0.604f, 0.804f, 0.196f, alpha }; }
	inline Eigen::Vector4f lightgray(float alpha = 1.0f) { return { 0.827f, 0.827f, 0.827f, alpha }; }
	inline Eigen::Vector4f lightgrey(float alpha = 1.0f) { return { 0.827f, 0.827f, 0.827f, alpha }; }
	inline Eigen::Vector4f lightblue(float alpha = 1.0f) { return { 0.678f, 0.847f, 0.902f, alpha }; }
	inline Eigen::Vector4f gold(float alpha = 1.0f) { return { 1.0f, 0.843f, 0.0f, alpha }; }
	inline Eigen::Vector4f gainsboro(float alpha = 1.0f) { return { 0.863f, 0.863f, 0.863f, alpha }; }
	inline Eigen::Vector4f thistle(float alpha = 1.0f) { return { 0.847f, 0.749f, 0.847f, alpha }; }
	inline Eigen::Vector4f powderblue(float alpha = 1.0f) { return { 0.69f, 0.878f, 0.902f, alpha }; }
	inline Eigen::Vector4f lightsalmon(float alpha = 1.0f) { return { 1.0f, 0.627f, 0.478f, alpha }; }
	inline Eigen::Vector4f plum(float alpha = 1.0f) { return { 0.867f, 0.627f, 0.867f, alpha }; }
	inline Eigen::Vector4f sandybrown(float alpha = 1.0f) { return { 0.957f, 0.643f, 0.376f, alpha }; }
	inline Eigen::Vector4f deepskyblue(float alpha = 1.0f) { return { 0.0f, 0.749f, 1.0f, alpha }; }
	inline Eigen::Vector4f paleturquoise(float alpha = 1.0f) { return { 0.686f, 0.933f, 0.933f, alpha }; }
	inline Eigen::Vector4f aquamarine(float alpha = 1.0f) { return { 0.498f, 1.0f, 0.831f, alpha }; }
	inline Eigen::Vector4f lightgreen(float alpha = 1.0f) { return { 0.565f, 0.933f, 0.565f, alpha }; }
	inline Eigen::Vector4f turquoise(float alpha = 1.0f) { return { 0.251f, 0.878f, 0.816f, alpha }; }
	inline Eigen::Vector4f pink(float alpha = 1.0f) { return { 1.0f, 0.753f, 0.796f, alpha }; }
	inline Eigen::Vector4f khaki(float alpha = 1.0f) { return { 0.941f, 0.902f, 0.549f, alpha }; }
	inline Eigen::Vector4f violet(float alpha = 1.0f) { return { 0.933f, 0.510f, 0.933f, alpha }; }
	inline Eigen::Vector4f springgreen(float alpha = 1.0f) { return { 0.0f, 1.0f, 0.498f, alpha }; }
	inline Eigen::Vector4f palegreen(float alpha = 1.0f) { return { 0.596f, 0.984f, 0.596f, alpha }; }
	inline Eigen::Vector4f mediumspringgreen(float alpha = 1.0f) { return { 0.0f, 0.98f, 0.604f, alpha }; }
	inline Eigen::Vector4f lightpink(float alpha = 1.0f) { return { 1.0f, 0.714f, 0.757f, alpha }; }
	inline Eigen::Vector4f navajowhite(float alpha = 1.0f) { return { 1.0f, 0.871f, 0.678f, alpha }; }
	inline Eigen::Vector4f peachpuff(float alpha = 1.0f) { return { 1.0f, 0.855f, 0.725f, alpha }; }
	inline Eigen::Vector4f wheat(float alpha = 1.0f) { return { 0.961f, 0.871f, 0.702f, alpha }; }
	inline Eigen::Vector4f moccasin(float alpha = 1.0f) { return { 1.0f, 0.894f, 0.71f, alpha }; }
	inline Eigen::Vector4f palegoldenrod(float alpha = 1.0f) { return { 0.933f, 0.91f, 0.667f, alpha }; }
	inline Eigen::Vector4f beige(float alpha = 1.0f) { return { 0.961f, 0.961f, 0.863f, alpha }; }
	inline Eigen::Vector4f whitesmoke(float alpha = 1.0f) { return { 0.961f, 0.961f, 0.961f, alpha }; }
	inline Eigen::Vector4f lavender(float alpha = 1.0f) { return { 0.902f, 0.902f, 0.98f, alpha }; }
	inline Eigen::Vector4f antiquewhite(float alpha = 1.0f) { return { 0.98f, 0.922f, 0.843f, alpha }; }
	inline Eigen::Vector4f mistyrose(float alpha = 1.0f) { return { 1.0f, 0.894f, 0.882f, alpha }; }
	inline Eigen::Vector4f bisque(float alpha = 1.0f) { return { 1.0f, 0.894f, 0.769f, alpha }; }
	inline Eigen::Vector4f blanchedalmond(float alpha = 1.0f) { return { 1.0f, 0.922f, 0.804f, alpha }; }
	inline Eigen::Vector4f linen(float alpha = 1.0f) { return { 0.980f, 0.941f, 0.902f, alpha }; }
	inline Eigen::Vector4f papayawhip(float alpha = 1.0f) { return { 1.0f, 0.937f, 0.835f, alpha }; }
	inline Eigen::Vector4f oldlace(float alpha = 1.0f) { return { 0.992f, 0.961f, 0.902f, alpha }; }
	inline Eigen::Vector4f ghostwhite(float alpha = 1.0f) { return { 0.973f, 0.973f, 1.0f, alpha }; }
	inline Eigen::Vector4f aliceblue(float alpha = 1.0f) { return { 0.941f, 0.973f, 1.0f, alpha }; }
	inline Eigen::Vector4f seashell(float alpha = 1.0f) { return { 1.0f, 0.961f, 0.933f, alpha }; }
	inline Eigen::Vector4f cornsilk(float alpha = 1.0f) { return { 1.0f, 0.973f, 0.863f, alpha }; }
	inline Eigen::Vector4f lavenderblush(float alpha = 1.0f) { return { 1.0f, 0.941f, 0.961f, alpha }; }
	inline Eigen::Vector4f lightgoldenrodyellow(float alpha = 1.0f) { return { 0.980f, 0.980f, 0.824f, alpha }; }
	inline Eigen::Vector4f snow(float alpha = 1.0f) { return { 1.0f, 0.98f, 0.98f, alpha }; }
	inline Eigen::Vector4f floralwhite(float alpha = 1.0f) { return { 1.0f, 0.98f, 0.941f, alpha }; }
	inline Eigen::Vector4f mintcream(float alpha = 1.0f) { return { 0.961f, 1.0f, 0.98f, alpha }; }
	inline Eigen::Vector4f honeydew(float alpha = 1.0f) { return { 0.941f, 1.0f, 0.941f, alpha }; }
	inline Eigen::Vector4f lemonchiffon(float alpha = 1.0f) { return { 1.0f, 0.98f, 0.804f, alpha }; }
	inline Eigen::Vector4f lawngreen(float alpha = 1.0f) { return { 0.486f, 0.988f, 0.0f, alpha }; }
	inline Eigen::Vector4f greenyellow(float alpha = 1.0f) { return { 0.678f, 1.0f, 0.184f, alpha }; }
	inline Eigen::Vector4f chartreuse(float alpha = 1.0f) { return { 0.498f, 1.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f lime(float alpha = 1.0f) { return { 0.0f, 1.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f lightcyan(float alpha = 1.0f) { return { 0.878f, 1.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f azure(float alpha = 1.0f) { return { 0.941f, 1.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f cyan(float alpha = 1.0f) { return { 0.0f, 1.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f aqua(float alpha = 1.0f) { return { 0.0f, 1.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f lightyellow(float alpha = 1.0f) { return { 1.0f, 1.0f, 0.878f, alpha }; }
	inline Eigen::Vector4f ivory(float alpha = 1.0f) { return { 1.0f, 1.0f, 0.941f, alpha }; }
	inline Eigen::Vector4f fuchsia(float alpha = 1.0f) { return { 1.0f, 0.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f magenta(float alpha = 1.0f) { return { 1.0f, 0.0f, 1.0f, alpha }; }
	inline Eigen::Vector4f yellow(float alpha = 1.0f) { return { 1.0f, 1.0f, 0.0f, alpha }; }
	inline Eigen::Vector4f white(float alpha = 1.0f) { return { 1.0f, 1.0f, 1.0f, alpha }; }

	inline Eigen::Vector4f FromRGB(float r, float g, float b, float a = 1.0f)
	{
		return Eigen::Vector4f(r, g, b, a);
	}

	inline Eigen::Vector4f Lerp(const Eigen::Vector4f& a, const Eigen::Vector4f& b, float t)
	{
		t = std::clamp(t, 0.0f, 1.0f);
		return a * (1.0f - t) + b * t;
	}

	/**
	 * \brief HSV(Hue, Saturation, Value) 값을 RGBA Eigen::Vector4f로 변환
	 * \param h Hue(색상) 값 (0.0f ~ 1.0f). 1.0f를 초과하면 순환
	 * \param s Saturation(채도) 값 (0.0f ~ 1.0f).
	 * \param v Value(명도) 값 (0.0f ~ 1.0f).
	 * \param a Alpha(투명도) 값 (0.0f ~ 1.0f). 기본값은 1.0f
	 * \return RGBA 형식의 Eigen::Vector4f
	 */
	inline Eigen::Vector4f FromHSV(float h, float s, float v, float a = 1.0f)
	{
		float r = 0.0f, g = 0.0f, b = 0.0f;

		// 채도(s)가 0이면 무채색(회색)
		if (s <= 0.0f)
		{
			r = v;
			g = v;
			b = v;
		}
		else
		{
			// H 값을 [0, 1) 범위로 순환
			h = fract(h);

			float h_i = std::floor(h * 6.0f);
			float f = (h * 6.0f) - h_i;
			float p = v * (1.0f - s);
			float q = v * (1.0f - f * s);
			float t = v * (1.0f - (1.0f - f) * s);

			int sector = static_cast<int>(h_i) % 6;

			switch (sector)
			{
			case 0: r = v; g = t; b = p; break; // Red -> Yellow
			case 1: r = q; g = v; b = p; break; // Yellow -> Green
			case 2: r = p; g = v; b = t; break; // Green -> Cyan
			case 3: r = p; g = q; b = v; break; // Cyan -> Blue
			case 4: r = t; g = p; b = v; break; // Blue -> Magenta
			case 5: r = v; g = p; b = q; break; // Magenta -> Red
			}
		}

		return Eigen::Vector4f(r, g, b, a);
	}

	inline std::vector<Eigen::Vector4f> GetContrastingColors(size_t count)
	{
		// 함수가 여러 번 호출되더라도 색상 목록은 한 번만 생성되도록 static으로 선언
		static const std::vector<Eigen::Vector4f> allColors = {
			aliceblue(), antiquewhite(), aqua(), aquamarine(), azure(), beige(), bisque(), //black(),
			blanchedalmond(), blue(), blueviolet(), brown(), burlywood(), cadetblue(), chartreuse(),
			chocolate(), coral(), cornflowerblue(), cornsilk(), crimson(), cyan(), darkblue(),
			darkcyan(), darkgoldenrod(), darkgray(), darkgreen(), darkkhaki(), darkmagenta(),
			darkolivegreen(), darkorange(), darkorchid(), darkred(), darksalmon(), darkseagreen(),
			darkslateblue(), darkslategray(), darkturquoise(), darkviolet(), deeppink(), deepskyblue(),
			dimgray(), dodgerblue(), firebrick(), floralwhite(), forestgreen(), fuchsia(),
			gainsboro(), ghostwhite(), gold(), goldenrod(), gray(), green(), greenyellow(),
			honeydew(), hotpink(), indianred(), indigo(), ivory(), khaki(), lavender(),
			lavenderblush(), lawngreen(), lemonchiffon(), lightblue(), lightcoral(), lightcyan(),
			lightgoldenrodyellow(), lightgray(), lightgreen(), lightpink(), lightsalmon(),
			lightseagreen(), lightskyblue(), lightslategray(), lightsteelblue(), lightyellow(),
			lime(), limegreen(), linen(), magenta(), maroon(), mediumaquamarine(), mediumblue(),
			mediumorchid(), mediumpurple(), mediumseagreen(), mediumslateblue(), mediumspringgreen(),
			mediumturquoise(), mediumvioletred(), midnightblue(), mintcream(), mistyrose(),
			moccasin(), navajowhite(), navy(), oldlace(), olive(), olivedrab(), orange(),
			orangered(), orchid(), palegoldenrod(), palegreen(), paleturquoise(), palevioletred(),
			papayawhip(), peachpuff(), peru(), pink(), plum(), powderblue(), purple(),
			rebeccapurple(), red(), rosybrown(), royalblue(), saddlebrown(), salmon(), sandybrown(),
			seagreen(), seashell(), sienna(), silver(), skyblue(), slateblue(), slategray(), snow(),
			springgreen(), steelblue(), tan(), teal(), thistle(), tomato(), turquoise(), violet(),
			wheat(), white(), whitesmoke(), yellow(), yellowgreen()
		};

		if (count == 0)
		{
			return {};
		}
		if (count >= allColors.size())
		{
			return allColors; // 요청된 개수가 전체 색상 수보다 많으면 모든 색상을 반환
		}

		std::vector<Eigen::Vector4f> result;
		result.reserve(count);

		// 어떤 색상이 이미 선택되었는지 추적하기 위한 벡터
		std::vector<bool> usedIndices(allColors.size(), false);

		// 시작점으로 검은색(black)을 찾아서 선택, 대비를 위한 좋은 기준점
		size_t startIndex = 0;
		//for (size_t i = 0; i < allColors.size(); ++i)
		//{
		//	if (allColors[i] == black())
		//	{
		//		startIndex = i;
		//		break;
		//	}
		//}

		result.push_back(allColors[startIndex]);
		usedIndices[startIndex] = true;

		for (size_t i = 1; i < count; ++i)
		{
			float maxMinDist = -1.0f;
			size_t bestIndex = 0;

			// 아직 선택되지 않은 모든 색상을 순회
			for (size_t j = 0; j < allColors.size(); ++j)
			{
				if (usedIndices[j])
				{
					continue;
				}

				// 현재 색상(allColors[j])과 이미 선택된 색상들(result) 사이의 최소 거리
				float min_dist_to_result = std::numeric_limits<float>::max();
				for (const auto& selectedColor : result)
				{
					float dist = (allColors[j] - selectedColor).norm();
					if (dist < min_dist_to_result)
					{
						min_dist_to_result = dist;
					}
				}

				// 이 최소 거리가 이전에 찾은 최대-최소 거리보다 크다면, 이 색상을 후보로 선택
				if (min_dist_to_result > maxMinDist)
				{
					maxMinDist = min_dist_to_result;
					bestIndex = j;
				}
			}

			// 가장 멀리 떨어진 색상을 결과에 추가하고 사용됨으로 표시
			result.push_back(allColors[bestIndex]);
			usedIndices[bestIndex] = true;
		}

		return result;
	}

	inline std::vector<Eigen::Vector4f> GetContrastingColorsWithoutBWRGB(size_t count)
	{
		// 함수가 여러 번 호출되더라도 색상 목록은 한 번만 생성되도록 static으로 선언
		static const std::vector<Eigen::Vector4f> allColors = {
			aliceblue(), antiquewhite(), aqua(), aquamarine(), azure(), beige(), bisque(), /*black(),*/
			blanchedalmond(), /*blue(),*/ blueviolet(), brown(), burlywood(), cadetblue(), chartreuse(),
			chocolate(), coral(), cornflowerblue(), cornsilk(), crimson(), cyan(), darkblue(),
			darkcyan(), darkgoldenrod(), darkgray(), darkgreen(), darkkhaki(), darkmagenta(),
			darkolivegreen(), darkorange(), darkorchid(), darkred(), darksalmon(), darkseagreen(),
			darkslateblue(), darkslategray(), darkturquoise(), darkviolet(), deeppink(), deepskyblue(),
			dimgray(), dodgerblue(), firebrick(), floralwhite(), forestgreen(), fuchsia(),
			gainsboro(), ghostwhite(), gold(), goldenrod(), gray(), /*green(),*/ greenyellow(),
			honeydew(), hotpink(), indianred(), indigo(), ivory(), khaki(), lavender(),
			lavenderblush(), lawngreen(), lemonchiffon(), lightblue(), lightcoral(), lightcyan(),
			lightgoldenrodyellow(), lightgray(), lightgreen(), lightpink(), lightsalmon(),
			lightseagreen(), lightskyblue(), lightslategray(), lightsteelblue(), lightyellow(),
			lime(), limegreen(), linen(), magenta(), maroon(), mediumaquamarine(), mediumblue(),
			mediumorchid(), mediumpurple(), mediumseagreen(), mediumslateblue(), mediumspringgreen(),
			mediumturquoise(), mediumvioletred(), midnightblue(), mintcream(), mistyrose(),
			moccasin(), navajowhite(), navy(), oldlace(), olive(), olivedrab(), orange(),
			orangered(), orchid(), palegoldenrod(), palegreen(), paleturquoise(), palevioletred(),
			papayawhip(), peachpuff(), peru(), pink(), plum(), powderblue(), purple(),
			rebeccapurple(), /*red(),*/ rosybrown(), royalblue(), saddlebrown(), salmon(), sandybrown(),
			seagreen(), seashell(), sienna(), silver(), skyblue(), slateblue(), slategray(), snow(),
			springgreen(), steelblue(), tan(), teal(), thistle(), tomato(), turquoise(), violet(),
			wheat(), /*white(),*/ whitesmoke(), yellow(), yellowgreen()
		};

		if (count == 0)
		{
			return {};
		}
		if (count >= allColors.size())
		{
			return allColors;
		}

		std::vector<Eigen::Vector4f> result;
		result.reserve(count);

		// 어떤 색상이 이미 선택되었는지 추적하기 위한 벡터
		std::vector<bool> usedIndices(allColors.size(), false);

		// 시작점으로 검은색(black)을 찾아서 선택, 대비를 위한 좋은 기준점
		size_t startIndex = 0;
		//for (size_t i = 0; i < allColors.size(); ++i)
		//{
		//	if (allColors[i] == black())
		//	{
		//		startIndex = i;
		//		break;
		//	}
		//}

		result.push_back(allColors[startIndex]);
		usedIndices[startIndex] = true;

		// 요청된 개수만큼 색상을 선택할 때까지 반복
		for (size_t i = 1; i < count; ++i)
		{
			float maxMinDist = -1.0f;
			size_t bestIndex = 0;

			// 아직 선택되지 않은 모든 색상을 순회
			for (size_t j = 0; j < allColors.size(); ++j)
			{
				if (usedIndices[j])
				{
					continue;
				}

				// 현재 색상(allColors[j])과 이미 선택된 색상들(result) 사이의 최소 거리
				float min_dist_to_result = std::numeric_limits<float>::max();
				for (const auto& selectedColor : result)
				{
					float dist = (allColors[j] - selectedColor).norm();
					if (dist < min_dist_to_result)
					{
						min_dist_to_result = dist;
					}
				}

				// 이 최소 거리가 이전에 찾은 최대-최소 거리보다 크다면, 이 색상을 후보로 선택
				if (min_dist_to_result > maxMinDist)
				{
					maxMinDist = min_dist_to_result;
					bestIndex = j;
				}
			}

			// 가장 멀리 떨어진 색상을 결과에 추가하고 사용됨으로 표시
			result.push_back(allColors[bestIndex]);
			usedIndices[bestIndex] = true;
		}

		return result;
	}

	inline std::vector<Eigen::Vector4f> InterpolateColors(const std::vector<Eigen::Vector4f>& colors, unsigned int count)
	{
		std::vector<Eigen::Vector4f> result;

		if (colors.empty() || count == 0)
			return result;

		// 기준 색상이 하나뿐이면 그 색만 반복 반환
		if (colors.size() == 1)
		{
			result.resize(count, colors[0]);
			return result;
		}

		result.reserve(count);

		// 구간 수
		const size_t numSegments = colors.size() - 1;

		// 전체 구간을 count개로 나누기
		for (unsigned int i = 0; i < count; ++i)
		{
			// 전체 진행 비율 [0, 1]
			float globalT = (count == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(count - 1);
			globalT = std::clamp(globalT, 0.0f, 1.0f);

			// 어떤 구간에 속하는지 계산
			float segmentF = globalT * numSegments;
			size_t segmentIndex = static_cast<size_t>(segmentF);
			float localT = segmentF - static_cast<float>(segmentIndex);

			// 마지막 구간 경계 보정
			if (segmentIndex >= numSegments)
			{
				result.push_back(colors.back());
				continue;
			}

			// 보간
			Eigen::Vector4f c = colors[segmentIndex] * (1.0f - localT) + colors[segmentIndex + 1] * localT;
			result.push_back(c);
		}

		return result;
	}

	inline std::vector<Eigen::Vector4f> GetPalette(int numberOfColors)
	{
		if (numberOfColors <= 0)
		{
			return {};
		}

		// 1. 후보군이 될 모든 색상 정의 (static으로 선언하여 최초 1회만 초기화)
		static const std::vector<Eigen::Vector4f> allColors = {
			aliceblue(), antiquewhite(), aqua(), aquamarine(), azure(), beige(), bisque(), black(),
			blanchedalmond(), blue(), blueviolet(), brown(), burlywood(), cadetblue(), chartreuse(),
			chocolate(), coral(), cornflowerblue(), cornsilk(), crimson(), cyan(), darkblue(),
			darkcyan(), darkgoldenrod(), darkgray(), darkgreen(), darkkhaki(), darkmagenta(),
			darkolivegreen(), darkorange(), darkorchid(), darkred(), darksalmon(), darkseagreen(),
			darkslateblue(), darkslategray(), darkturquoise(), darkviolet(), deeppink(), deepskyblue(),
			dimgray(), dodgerblue(), firebrick(), floralwhite(), forestgreen(), fuchsia(),
			gainsboro(), ghostwhite(), gold(), goldenrod(), gray(), green(), greenyellow(),
			honeydew(), hotpink(), indianred(), indigo(), ivory(), khaki(), lavender(),
			lavenderblush(), lawngreen(), lemonchiffon(), lightblue(), lightcoral(), lightcyan(),
			lightgoldenrodyellow(), lightgray(), lightgreen(), lightpink(), lightsalmon(),
			lightseagreen(), lightskyblue(), lightslategray(), lightsteelblue(), lightyellow(),
			lime(), limegreen(), linen(), magenta(), maroon(), mediumaquamarine(), mediumblue(),
			mediumorchid(), mediumpurple(), mediumseagreen(), mediumslateblue(), mediumspringgreen(),
			mediumturquoise(), mediumvioletred(), midnightblue(), mintcream(), mistyrose(),
			moccasin(), navajowhite(), navy(), oldlace(), olive(), olivedrab(), orange(),
			orangered(), orchid(), palegoldenrod(), palegreen(), paleturquoise(), palevioletred(),
			papayawhip(), peachpuff(), peru(), pink(), plum(), powderblue(), purple(),
			rebeccapurple(), red(), rosybrown(), royalblue(), saddlebrown(), salmon(), sandybrown(),
			seagreen(), seashell(), sienna(), silver(), skyblue(), slateblue(), slategray(), snow(),
			springgreen(), steelblue(), tan(), teal(), thistle(), tomato(), turquoise(), violet(),
			wheat(), white(), whitesmoke(), yellow(), yellowgreen()
		};

		// 2. 흰색/검은색 계열을 제외한 유효 후보군 생성 (최초 1회만 수행)
		static std::vector<Eigen::Vector4f> validCandidates;
		if (validCandidates.empty())
		{
			validCandidates.reserve(allColors.size());
			for (const auto& c : allColors)
			{
				// 밝기(Luminance) 혹은 RGB 값을 기준으로 너무 어둡거나 너무 밝은 색 제외
				// (0.05 미만은 거의 검은색, 0.95 초과는 거의 흰색으로 간주)
				// Eigen::Vector4f는 .r, .g, .b 접근자가 없으므로 .x(), .y(), .z() 사용
				bool isTooDark = (c.x() < 0.05f && c.y() < 0.05f && c.z() < 0.05f);
				bool isTooBright = (c.x() > 0.95f && c.y() > 0.95f && c.z() > 0.95f);

				if (!isTooDark && !isTooBright)
				{
					validCandidates.push_back(c);
				}
			}
		}

		// 요청된 개수가 유효 후보군보다 많으면 전체 반환
		if (static_cast<size_t>(numberOfColors) >= validCandidates.size())
		{
			return validCandidates;
		}

		std::vector<Eigen::Vector4f> result;
		result.reserve(numberOfColors);

		std::vector<bool> usedIndices(validCandidates.size(), false);

		// 3. 첫 번째 색상 선택
		// 검은색/흰색이 없으므로, 시각적으로 강렬한 'Red' 혹은 'Blue'를 시작점으로 잡는 것이 좋습니다.
		// 여기서는 빨간색과 가장 가까운 색을 시작점으로 잡거나, 목록의 첫 번째를 사용합니다.
		// (알고리즘의 안정성을 위해 0번 인덱스보다는 명시적인 유색 컬러를 시작점으로 추천)
		size_t startIndex = 0;
		float minDistToRed = std::numeric_limits<float>::max();

		// 비교 대상 Red
		Eigen::Vector4f targetRed(1.0f, 0.0f, 0.0f, 1.0f);

		for (size_t i = 0; i < validCandidates.size(); ++i)
		{
			float d = (validCandidates[i] - targetRed).norm();
			if (d < minDistToRed)
			{
				minDistToRed = d;
				startIndex = i;
			}
		}

		result.push_back(validCandidates[startIndex]);
		usedIndices[startIndex] = true;

		// 4. Maximin Distance 알고리즘으로 나머지 색상 선택
		for (int i = 1; i < numberOfColors; ++i)
		{
			float maxMinDist = -1.0f;
			size_t bestIndex = 0;

			for (size_t j = 0; j < validCandidates.size(); ++j)
			{
				if (usedIndices[j]) continue;

				// 현재 후보 색상(candidate)이 기존 결과(result)들에 대해 가지는 최소 거리 계산
				float minDistToExisting = std::numeric_limits<float>::max();
				for (const auto& existing : result)
				{
					float d = (validCandidates[j] - existing).norm();
					if (d < minDistToExisting)
					{
						minDistToExisting = d;
					}
				}

				// 그 최소 거리가 가장 큰(멀리 떨어진) 색상을 선택
				if (minDistToExisting > maxMinDist)
				{
					maxMinDist = minDistToExisting;
					bestIndex = j;
				}
			}

			result.push_back(validCandidates[bestIndex]);
			usedIndices[bestIndex] = true;
		}

		return result;
	}

	// 히트맵 (Blue -> Green -> Red)
	inline Eigen::Vector4f GetHeatMapColor(float value, float minVal, float maxVal, float alpha = 1.0f)
	{
		float t = std::clamp((value - minVal) / (maxVal - minVal), 0.0f, 1.0f);

		// Blue (0.0) -> Green (0.5) -> Red (1.0)
		if (t < 0.5f)
		{
			// Blue to Green
			float localT = t * 2.0f;
			return Eigen::Vector4f(0.0f, localT, 1.0f - localT, alpha);
		}
		else
		{
			// Green to Red
			float localT = (t - 0.5f) * 2.0f;
			return Eigen::Vector4f(localT, 1.0f - localT, 0.0f, alpha);
		}
	}

	//inline Eigen::Vector4f Random(float alpha = 1.0f)
	//{
	//	// thread-safe, 초기화 1회
	//	static thread_local std::mt19937 rng{ std::random_device{}() };
	//	static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	//	return Eigen::Vector4f(
	//		dist(rng),
	//		dist(rng),
	//		dist(rng),
	//		alpha
	//	);
	//}

	//inline Eigen::Vector4f RandomHSV(
	//	float sMin = 0.6f, float sMax = 1.0f,
	//	float vMin = 0.6f, float vMax = 1.0f,
	//	float alpha = 1.0f)
	//{
	//	static thread_local std::mt19937 rng{ std::random_device{}() };
	//	std::uniform_real_distribution<float> hueDist(0.0f, 1.0f);
	//	std::uniform_real_distribution<float> satDist(sMin, sMax);
	//	std::uniform_real_distribution<float> valDist(vMin, vMax);

	//	return FromHSV(
	//		hueDist(rng),
	//		satDist(rng),
	//		valDist(rng),
	//		alpha
	//	);
	//}

	inline Eigen::Vector4f RandomFromIndex(size_t index, float alpha = 1.0f)
	{
		// Golden ratio 기반 Hue 분산
		const float goldenRatio = 0.61803398875f;
		float h = fract(index * goldenRatio);

		return FromHSV(h, 0.75f, 0.95f, alpha);
	}
}
