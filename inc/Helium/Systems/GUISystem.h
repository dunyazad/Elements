#pragma once

#include <Helium/Systems/HeliumSystem.h>
#include <stb/stb_truetype.h>
#include <Eigen/Dense>

class HeliumCore;
class Shader;

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

    void RenderText(const std::string& text, float x, float y, float scale, const Eigen::Vector4f& color);

private:
    unsigned int textureID = 0;
    std::vector<stbtt_packedchar> charData;

    unsigned int VAO = 0;
    unsigned int VBO = 0;
    Shader* textShader = nullptr;
};
