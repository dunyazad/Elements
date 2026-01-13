#pragma once

#include <any>

#include <Helium/Systems/HeliumSystem.h>
#include <stb/stb_truetype.h>
#include <Eigen/Dense>

class HeliumCore;
class Shader;

enum class GUICommandType
{
    Rectangle,
    Text
};

struct GUIRenderCommand
{
    int zIndex;
	GUICommandType type;
    std::any component;
};

class GUISystem : public HeliumSystem
{
public:
    GUISystem(HeliumCore* core);
    ~GUISystem();

    void Initialize();
    void Update(float dt);
    void Render();
    void Shutdown();

    void Resize(int width, int height);

	void RenderRectangle(float x, float y, float width, float height, const Eigen::Vector4f& color);
    void RenderText(const std::string& text, float x, float y, float scale, const Eigen::Vector4f& color);

private:
    unsigned int textureID = 0;
    std::vector<stbtt_packedchar> charData;

    unsigned int VAO = 0;
    unsigned int VBO = 0;
    Shader* textShader = nullptr;
    Shader* geometryShader = nullptr;

    std::vector<GUIRenderCommand> renderCommandQueue;
    template<typename T>
    void Submit(int zIndex, GUICommandType type, const T& data)
    {
        renderCommandQueue.push_back({ zIndex, type, data });
    }
};
