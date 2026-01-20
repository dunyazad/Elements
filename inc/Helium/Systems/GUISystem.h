#pragma once

#include <vector>
#include <string>
#include <any> 
#include <Eigen/Dense>
#include <stb/stb_truetype.h>

#include <Helium/Systems/HeliumSystem.h>
#include <Helium/Components/GUI/GUIComponent.h>

class Shader;

enum class GUICommandType
{
    Rectangle,
    Circle,
    Text
};

struct GUIRenderCommand
{
    int zIndex;
    GUICommandType type;
    std::any component;
};

class HELIUM_API GUISystem : public HeliumSystem
{
public:
    GUISystem(HeliumCore* core);
    virtual ~GUISystem();

    virtual void Initialize() override;
    virtual void Update(float dt) override;
    virtual void Render() override;
    virtual void Shutdown() override;

    virtual void Resize(int width, int height);

    template <typename T>
    void Submit(int zIndex, GUICommandType type, const T& component)
    {
        GUIRenderCommand cmd;
        cmd.zIndex = zIndex;
        cmd.type = type;
        cmd.component = component;
        renderCommandQueue.push_back(cmd);
    }

    void RenderText(const std::string& text, float x, float y, float textWidth, float fontSize, const Eigen::Vector4f& color,
        TextHAlign hAlign = TextHAlign::Left, TextVAlign vAlign = TextVAlign::Baseline);

    void RenderRectangle(float x, float y, float width, float height, const Eigen::Vector4f& color);
    void RenderCircle(float x, float y, float radius, const Eigen::Vector4f& color);

    static int GetNextZIndex() { return zIndexCounter++; }

private:
    std::vector<GUIRenderCommand> renderCommandQueue;
    static int zIndexCounter;

    unsigned int VAO = 0, VBO = 0;
    unsigned int textureID = 0;

    Shader* textShader = nullptr;
    Shader* geometryShader = nullptr;

    std::vector<stbtt_packedchar> charData;

    float fontAscent = 0.0f;
    float fontDescent = 0.0f;
    float fontLineGap = 0.0f;
};