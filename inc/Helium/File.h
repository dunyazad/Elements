#pragma once

#include <string>
#include <fstream>
#include <vector>

class File
{
public:
	File();
	File(const std::string& fileName, bool isBinary = false);
	~File();

	static bool Exists(const std::string& filename);

	void Create(const std::string& fileName, bool isBinary);
	bool Open(const std::string& fileName, bool isBinary);
	void Close();
	bool isOpen();

	bool GetWord(std::string& word);
	bool GetLine(std::string& line);

	void Read(char* buffer, int length) const;

	std::string ReadAllString() const;
	std::vector<unsigned char> ReadAllBytes() const;

	void Write(char* buffer, int length);

	std::fstream& operator << (bool data);
	std::fstream& operator << (short data);
	std::fstream& operator << (unsigned short data);
	std::fstream& operator << (int data);
	std::fstream& operator << (unsigned int data);
	std::fstream& operator << (long data);
	std::fstream& operator << (unsigned long data);
	std::fstream& operator << (float data);
	std::fstream& operator << (double data);
	std::fstream& operator << (std::string& data);
	std::fstream& operator << (const std::string& data);
	std::fstream& operator << (char* data);
	std::fstream& operator << (const char* data);

	std::fstream& operator >> (bool& data);
	std::fstream& operator >> (short& data);
	std::fstream& operator >> (unsigned short& data);
	std::fstream& operator >> (int& data);
	std::fstream& operator >> (unsigned int& data);
	std::fstream& operator >> (long& data);
	std::fstream& operator >> (unsigned long& data);
	std::fstream& operator >> (float& data);
	std::fstream& operator >> (double& data);
	std::fstream& operator >> (std::string& data);
	//std::fstream& operator >> (char* data);

	inline const std::string& GetFileName() const { return fileName; }
	inline int GetFileLength() const { return fileLength; }

private:
	std::fstream* fileStream = nullptr;
	std::string fileName = "";
	int fileLength = 0;
};
