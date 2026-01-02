#pragma once

class Texture
{
public:
	Texture();
	~Texture();

	void Bind();
	void Unbind();

	void LoadFile(const File& file);

	void AllocTextureData(ui32 width = 1024, ui32 height = 1024);
	void SetTextureData(ui32 width = 1024, ui32 height = 1024, ui8* data = nullptr);

	inline ui32 GetWidth() const { return width; }
	inline ui32 GetHeight() const { return height; }
	inline GLuint GetTextureID() const { return textureID; }

private:
	GLuint textureID = ui32_max;
	ui32 width = 1024;
	ui32 height = 1024;
	ui8* data = nullptr;
};
