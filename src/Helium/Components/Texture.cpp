#include "pch.h"

#include <Helium/Components/Texture.h>

Texture::Texture()
    : width(0), height(0), data(nullptr)
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // 텍스처 파라미터 설정 (반드시 바인드 이후)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture()
{
    glDeleteTextures(1, &textureID);

    if (nullptr != data)
    {
        delete[] data;
		data = nullptr;
    }
}

void Texture::Bind()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
}

void Texture::Unbind()
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::LoadFile(const File& file)
{
    //stbi_set_flip_vertically_on_load(true);

    //int width, height, channels;
    //unsigned char* imageData = stbi_load(file.GetFileName().c_str(), &width, &height, &channels, 4);
    //if (imageData)
    //{
    //    SetTextureData(width, height, imageData);
    //    stbi_image_free(imageData);
    //}
    //else
    //{
    //    std::cerr << "Failed to load texture file: " << file.GetFileName() << std::endl;
    //}
}

void Texture::AllocTextureData(unsigned int width, unsigned int height)
{
    this->width = width;
    this->height = height;

    if (this->data)
        delete[] this->data;
    this->data = new unsigned char[width * height * 4];
    memset(this->data, 0, sizeof(unsigned char) * width * height * 4);

    Bind();

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    Unbind();
}

void Texture::SetTextureData(unsigned int width, unsigned int height, unsigned char* data)
{
    if (this->data == nullptr || this->width * this->height < width * height)
    {
        if (this->data)
            delete[] this->data;
        this->data = new unsigned char[width * height * 4];
    }
    this->width = width;
    this->height = height;
    memcpy(this->data, data, sizeof(unsigned char) * width * height * 4);

    Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    Unbind();
}
