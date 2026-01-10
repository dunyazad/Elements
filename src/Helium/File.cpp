#include "pch.h"

#include <Helium/File.h>

#include <filesystem>

bool File::Exists(const std::string& filename)
{
	return std::filesystem::exists(filename);
}

File::File()
	: fileStream(nullptr)
{
	fileStream = new std::fstream();
}

File::File(const std::string& fileName, bool isBinary)
	: fileName(fileName)
{
	if (Exists(fileName))
	{
		fileStream = new std::fstream();

		Open(fileName, isBinary);
	}
}

File::~File()
{
	if (nullptr != fileStream)
	{
		delete fileStream;
		fileStream = nullptr;
	}
}

void File::Create(const std::string& fileName, bool isBinary)
{
	if (isBinary)
	{
		(*fileStream).open(fileName, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
	}
	else
	{
		(*fileStream).open(fileName, std::ios::in | std::ios::out | std::ios::trunc);
	}
}

bool File::Open(const std::string& fileName, bool isBinary)
{
	this->fileName = fileName;
	if (false == Exists(fileName)) return false;

	if (isBinary)
	{
		(*fileStream).open(fileName, std::ios::binary | std::ios::in);
	}
	else
	{
		(*fileStream).open(fileName, std::ios::in);
	}

	if ((*fileStream).is_open())
	{
		(*fileStream).seekg(0, std::ios::end);
		fileLength = int((*fileStream).tellg());
		(*fileStream).seekg(0, std::ios::beg);

		return true;
	}

	return false;
}

void File::Close()
{
	(*fileStream).close();
}

bool File::isOpen()
{
	if (nullptr == fileStream) return false;

	return (*fileStream).is_open();
}

bool File::GetWord(std::string& word)
{
	if (nullptr == fileStream) return false;

	return !((*fileStream) >> word).eof();
}

bool File::GetLine(std::string& line)
{
	if (nullptr == fileStream) return false;

	return !(getline((*fileStream), line).eof());
}

void File::Read(char* buffer, int length) const
{
	if (nullptr == fileStream) return;

	(*fileStream).read(buffer, length);
}

#include "pch.h"
#include <Helium/File.h>
#include <filesystem>

// ... (이전 함수들은 동일하므로 생략하거나 기존 유지) ...

std::string File::ReadAllString() const
{
	if (nullptr == fileStream || !(*fileStream).is_open())
	{
		return "";
	}

	(*fileStream).clear();
	(*fileStream).seekg(0, std::ios::end);

	std::streamsize length = (*fileStream).tellg();

	(*fileStream).seekg(0, std::ios::beg);

	if (length <= 0)
	{
		return "";
	}

	std::string content;
	content.resize(static_cast<size_t>(length));

	(*fileStream).read(&content[0], length);

	return content;
}

std::vector<unsigned char> File::ReadAllBytes() const
{
	if (nullptr == fileStream || !(*fileStream).is_open())
	{
		return {};
	}

	(*fileStream).clear();
	(*fileStream).seekg(0, std::ios::end);

	std::streamsize length = (*fileStream).tellg();

	(*fileStream).seekg(0, std::ios::beg);

	if (length <= 0)
	{
		return {};
	}

	std::vector<unsigned char> content(static_cast<size_t>(length));

	// vector<unsigned char> 데이터를 char*로 캐스팅하여 읽기
	(*fileStream).read(reinterpret_cast<char*>(content.data()), length);

	return content;
}

void File::Write(char* buffer, int length)
{
	if (nullptr == fileStream) return;

	(*fileStream).write(buffer, length);
}

std::fstream& File::operator << (bool data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (short data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (unsigned short data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (int data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (unsigned int data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (long data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (unsigned long data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (float data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (double data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (std::string& data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (const std::string& data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (char* data)
{
	return (std::fstream&)((*fileStream) << data);
}

std::fstream& File::operator << (const char* data)
{
	return (std::fstream&)((*fileStream) << data);
}

//

std::fstream& File::operator >> (bool& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (short& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (unsigned short& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (int& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (unsigned int& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (long& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (unsigned long& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (float& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (double& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

std::fstream& File::operator >> (std::string& data)
{
	return (std::fstream&)((*fileStream) >> data);
}

//std::fstream& File::operator >> (char* data)
//{
//	return (std::fstream&)((*m_pFileStream) >> data);
//}
