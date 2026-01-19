#pragma once

#include <Helium/HeliumCommon.h>

#include <Helium/File.h>

#include <glad/glad.h>

class HELIUM_API Texture
{
public:
	Texture();
	~Texture();

	void Bind();
	void Unbind();

	void LoadFile(const File& file);

	void AllocTextureData(unsigned int width = 1024, unsigned int height = 1024);
	void SetTextureData(unsigned int width = 1024, unsigned int height = 1024, unsigned char* data = nullptr);

	inline unsigned int GetWidth() const { return width; }
	inline unsigned int GetHeight() const { return height; }
	inline GLuint GetTextureID() const { return textureID; }

private:
	GLuint textureID = UINT32_MAX;
	unsigned int width = 1024;
	unsigned int height = 1024;
	unsigned char* data = nullptr;
};
