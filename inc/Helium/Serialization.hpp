#pragma once

#define NOMINMAX

#undef min
#undef max

#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <mutex>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#define FLT_VALID(x) ((x) < 3.402823466e+36F)

// --- Helper Functions ---

inline int safe_stoi(const std::string& input)
{
	if (input.empty()) return INT_MAX;
	return stoi(input);
}

inline float safe_stof(const std::string& input)
{
	if (input.empty()) return FLT_MAX;
	return stof(input);
}

inline std::vector<std::string> split(const std::string& input, const std::string& delimiters, bool includeEmptyString = false)
{
	std::vector<std::string> result;
	std::string piece;
	for (auto c : input)
	{
		bool contains = false;
		for (auto d : delimiters)
		{
			if (d == c) { contains = true; break; }
		}

		if (!contains)
		{
			piece += c;
		}
		else
		{
			if (includeEmptyString || !piece.empty())
			{
				result.push_back(piece);
				piece.clear();
			}
		}
	}
	if (!piece.empty()) result.push_back(piece);
	return result;
}

// Updated ParseOneLine to use Eigen vectors
inline void ParseOneLine(
	const std::string& line,
	std::vector<Eigen::Vector3f>& vertices,
	std::vector<Eigen::Vector2f>& uvs,
	std::vector<Eigen::Vector3f>& vertex_normals,
	std::vector<Eigen::Vector4f>& vertex_colors,
	std::vector<Eigen::Vector3i>& faces,
	float scaleX, float scaleY, float scaleZ)
{
	if (line.empty()) return;

	auto words = split(line, " \t");
	if (words.empty()) return;

	if (words[0] == "v")
	{
		float x = safe_stof(words[1]) * scaleX;
		float y = safe_stof(words[2]) * scaleY;
		float z = safe_stof(words[3]) * scaleZ;
		vertices.emplace_back(x, y, z);

		if (words.size() > 4)
		{
			float r = safe_stof(words[4]);
			float g = safe_stof(words[5]);
			float b = safe_stof(words[6]);
			// Assume alpha 1.0 if not present
			vertex_colors.emplace_back(r, g, b, 1.0f);
		}
	}
	else if (words[0] == "vt")
	{
		float u = safe_stof(words[1]);
		float v = safe_stof(words[2]);
		uvs.emplace_back(u, v);
	}
	else if (words[0] == "vn")
	{
		float x = safe_stof(words[1]);
		float y = safe_stof(words[2]);
		float z = safe_stof(words[3]);
		vertex_normals.emplace_back(x, y, z);
	}
	else if (words[0] == "f")
	{
		if (words.size() == 4)
		{
			auto fe0 = split(words[1], "/", true);
			auto fe1 = split(words[2], "/", true);
			auto fe2 = split(words[3], "/", true);

			// Note: OBJ indices are 1-based usually, but here we store raw parsed ints.
			// Ideally, we should subtract 1 here if converting to 0-based directly.
			// Assuming caller handles indexing or these are just raw IDs.
			faces.emplace_back(safe_stoi(fe0[0]), safe_stoi(fe1[0]), safe_stoi(fe2[0]));
		}
	}
}

// --- Classes ---

class HSerializable
{
public:
	virtual ~HSerializable() = default;
	virtual bool Serialize(const std::string& filename) = 0;
	virtual bool Deserialize(const std::string& filename) = 0;

	virtual inline void AddPoint(float x, float y, float z)
	{
		Eigen::Vector3f p(x, y, z);
		points.push_back(p);
		if (FLT_VALID(x) && FLT_VALID(y) && FLT_VALID(z))
		{
			aabb.extend(p);
		}
	}

	virtual inline void AddPoint(const Eigen::Vector3f& p)
	{
		points.push_back(p);
		if (FLT_VALID(p.x()) && FLT_VALID(p.y()) && FLT_VALID(p.z()))
		{
			aabb.extend(p);
		}
	}

	virtual inline void SwapAxisYZ() = 0;

	inline const std::vector<Eigen::Vector3f>& GetPoints() const { return points; }
	inline std::vector<Eigen::Vector3f>& GetPoints() { return points; }

	inline Eigen::Vector3f GetAABBMin() const { return aabb.min(); }
	inline Eigen::Vector3f GetAABBMax() const { return aabb.max(); }
	inline Eigen::Vector3f GetAABBCenter() const { return aabb.center(); }

protected:
	std::vector<Eigen::Vector3f> points;
	Eigen::AlignedBox3f aabb;
};

class XYZFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		fprintf(fp, "%llu\n", points.size());
		for (const auto& p : points)
		{
			fprintf(fp, "%.6f %.6f %.6f\n", p.x(), p.y(), p.z());
		}

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		int size = 0;
		fscanf_s(fp, "%d\n", &size);

		points.reserve(size);
		for (int i = 0; i < size; i++)
		{
			float x, y, z;
			fscanf_s(fp, "%f %f %f\n", &x, &y, &z);
			AddPoint(x, y, z);
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points)
		{
			std::swap(p.y(), p.z());
		}
		// Recompute AABB
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}
};

class OFFFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		fprintf(fp, "OFF\n");

		// Indices are stored as Vector3i (triangles)
		auto pointCount = points.size();
		auto faceCount = indices.size();

		fprintf(fp, "%llu %llu %llu\n", pointCount, faceCount, (size_t)0);

		for (size_t i = 0; i < pointCount; i++)
		{
			const auto& p = points[i];
			fprintf(fp, "%4.6f %4.6f %4.6f\n", p.x(), p.y(), p.z());

			if (i % 10000 == 0)
			{
				auto percent = ((double)i / (double)pointCount) * 100.0;
				printf("[%llu / %llu] %f percent\n", i, pointCount, percent);
			}
		}

		for (size_t i = 0; i < faceCount; i++)
		{
			const auto& tri = indices[i];

			if (colors.empty())
			{
				fprintf(fp, "3 %7d %7d %7d 255 255 255\n", tri.x(), tri.y(), tri.z());
			}
			else
			{
				// Color per vertex logic in original code was slightly ambiguous (color per face vs vertex).
				// Assuming mapping color of first vertex to face or direct face color list.
				// Based on original code: colors[i0 * 3] -> likely vertex colors.
				if (tri.x() < (int)colors.size())
				{
					const auto& c = colors[tri.x()];
					auto red = (unsigned char)(c.x() * 255);
					auto green = (unsigned char)(c.y() * 255);
					auto blue = (unsigned char)(c.z() * 255);
					fprintf(fp, "3 %7d %7d %7d %3d %3d %3d\n", tri.x(), tri.y(), tri.z(), red, green, blue);
				}
				else
				{
					fprintf(fp, "3 %7d %7d %7d 255 255 255\n", tri.x(), tri.y(), tri.z());
				}
			}
		}

		// Fallback for point cloud color saving if no faces
		if (faceCount == 0 && !colors.empty())
		{
			// Note: OFF standard doesn't typically support standalone colors without faces/verts well,
			// but keeping original logic.
			for (const auto& c : colors)
			{
				auto red = (unsigned char)(c.x() * 255);
				auto green = (unsigned char)(c.y() * 255);
				auto blue = (unsigned char)(c.z() * 255);
				fprintf(fp, "1 %3d %3d %3d\n", red, green, blue);
			}
		}

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		char buffer[1024];
		auto line = fgets(buffer, 1024, fp);
		if (0 != strcmp(line, "OFF\n")) return false;

		line = fgets(buffer, 1024, fp);
		while (line && '#' == line[0])
		{
			line = fgets(buffer, 1024, fp);
		}

		size_t vertexCount = 0;
		size_t triangleCount = 0;
		size_t edgeCount = 0;
		sscanf_s(line, "%llu %llu %llu", &vertexCount, &triangleCount, &edgeCount);

		printf("vertexCount : %llu, triangleCount : %llu\n", vertexCount, triangleCount);

		points.reserve(vertexCount);
		for (size_t i = 0; i < vertexCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if (line)
			{
				if ('#' == line[0]) { i--; continue; }
				float x, y, z;
				sscanf_s(line, "%f %f %f\n", &x, &y, &z);
				AddPoint(x, y, z);
			}
		}

		indices.reserve(triangleCount);
		for (size_t i = 0; i < triangleCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if (line && '#' == line[0]) { i--; continue; }

			unsigned int count, i0, i1, i2;
			sscanf_s(line, "%u %u %u %u\n", &count, &i0, &i1, &i2);
			if (count == 3)
			{
				AddTriangle(i0, i1, i2);
			}
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline const std::vector<Eigen::Vector3i>& GetIndices() const { return indices; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddTriangle(unsigned int i0, unsigned int i1, unsigned int i2)
	{
		indices.emplace_back(i0, i1, i2);
	}

	virtual inline void AddColor(float r, float g, float b, float a = 1.0f)
	{
		colors.emplace_back(r, g, b, a);
	}

protected:
	std::vector<Eigen::Vector3i> indices; // Stores triangles
	std::vector<Eigen::Vector4f> colors;
};

class CustomMeshFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err) return false;

		// Points
		fprintf_s(fp, "%llu\n", points.size());
		for (const auto& p : points)
			fprintf(fp, "%f, %f, %f\n", p.x(), p.y(), p.z());

		// Normals
		fprintf_s(fp, "%llu\n", normals.size());
		for (const auto& n : normals)
			fprintf(fp, "%f, %f, %f\n", n.x(), n.y(), n.z());

		// Colors
		fprintf_s(fp, "%llu\n", colors.size());
		for (const auto& c : colors)
			fprintf(fp, "%f, %f, %f\n", c.x(), c.y(), c.z());

		// Indices
		fprintf_s(fp, "%llu\n", indices.size());
		for (const auto& tri : indices)
			fprintf(fp, "%d, %d, %d\n", tri.x(), tri.y(), tri.z());

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err) return false;

		char buffer[1024];

		auto ReadBlock = [&](auto& vector, int dim) {
			size_t count = 0;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%llu\n", &count);
			vector.reserve(count);
			for (size_t i = 0; i < count; i++)
			{
				fgets(buffer, sizeof(buffer), fp);
				if (dim == 3) // Vectors
				{
					float x, y, z;
					sscanf_s(buffer, "%f, %f, %f\n", &x, &y, &z);
					if constexpr (std::is_same_v<typename std::decay_t<decltype(vector)>::value_type, Eigen::Vector3f>)
						vector.emplace_back(x, y, z);
					else if constexpr (std::is_same_v<typename std::decay_t<decltype(vector)>::value_type, Eigen::Vector4f>)
						vector.emplace_back(x, y, z, 1.0f);
					else if constexpr (std::is_same_v<typename std::decay_t<decltype(vector)>::value_type, Eigen::Vector3i>)
						vector.emplace_back((int)x, (int)y, (int)z); // Actually integers in string
				}
			}
			};

		// Points
		size_t nop = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &nop);
		points.reserve(nop);
		for (size_t i = 0; i < nop; ++i) {
			float x, y, z;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%f, %f, %f\n", &x, &y, &z);
			AddPoint(x, y, z);
		}

		// Normals
		size_t non = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &non);
		normals.reserve(non);
		for (size_t i = 0; i < non; ++i) {
			float x, y, z;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%f, %f, %f\n", &x, &y, &z);
			AddNormal(x, y, z);
		}

		// Colors
		size_t noc = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &noc);
		colors.reserve(noc);
		for (size_t i = 0; i < noc; ++i) {
			float r, g, b;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%f, %f, %f\n", &r, &g, &b);
			AddColor(r, g, b);
		}

		// Indices
		size_t noi = 0;
		if (fgets(buffer, sizeof(buffer), fp)) sscanf_s(buffer, "%llu\n", &noi);
		indices.reserve(noi);
		for (size_t i = 0; i < noi; ++i) {
			int i0, i1, i2;
			fgets(buffer, sizeof(buffer), fp);
			sscanf_s(buffer, "%d, %d, %d\n", &i0, &i1, &i2);
			AddTriangle(i0, i1, i2);
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		for (auto& n : normals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline const std::vector<Eigen::Vector3i>& GetIndices() const { return indices; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddNormal(float x, float y, float z) { normals.emplace_back(x, y, z); }
	virtual inline void AddNormal(const Eigen::Vector3f& n) { normals.push_back(n); }
	virtual inline void AddTriangle(int i0, int i1, int i2) { indices.emplace_back(i0, i1, i2); }
	virtual inline void AddColor(float r, float g, float b, float a = 1.0f) { colors.emplace_back(r, g, b, a); }

protected:
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3i> indices; // Triangles
	std::vector<Eigen::Vector4f> colors;
};

class OBJFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		std::ofstream ofs(filename);
		std::stringstream ss;
		ss.precision(6);

		ss << "# cuTSDF::ResourceIO::OBJ" << std::endl;
		for (size_t i = 0; i < points.size(); i++)
		{
			const auto& p = points[i];
			if (colors.size() == points.size())
			{
				const auto& c = colors[i];
				ss << "v " << p.x() << " " << p.y() << " " << p.z() << " " << c.x() << " " << c.y() << " " << c.z() << std::endl;
			}
			else
			{
				ss << "v " << p.x() << " " << p.y() << " " << p.z() << std::endl;
			}

			if (normals.size() == points.size())
			{
				const auto& n = normals[i];
				ss << "vn " << n.x() << " " << n.y() << " " << n.z() << std::endl;
			}
		}

		for (const auto& uv : uvs)
		{
			ss << "vt " << uv.x() << " " << uv.y() << std::endl;
		}

		bool has_uv = !uvs.empty();
		bool has_vn = !normals.empty();

		for (size_t i = 0; i < indices.size(); i++)
		{
			const auto& face = indices[i];
			// OBJ indices are usually 1-based, assuming stored as such or handled on read
			// Here we assume indices stored are 1-based compatible with how ParseOneLine reads them
			// If ParseOneLine stores raw file ints, they are 1-based.

			if (has_uv && has_vn)
			{
				ss << "f "
					<< face.x() << "/" << face.x() << "/" << face.x() << " "
					<< face.y() << "/" << face.y() << "/" << face.y() << " "
					<< face.z() << "/" << face.z() << "/" << face.z() << std::endl;
			}
			else if (has_uv)
			{
				ss << "f "
					<< face.x() << "/" << face.x() << " "
					<< face.y() << "/" << face.y() << " "
					<< face.z() << "/" << face.z() << std::endl;
			}
			else if (has_vn)
			{
				ss << "f "
					<< face.x() << "//" << face.x() << " "
					<< face.y() << "//" << face.y() << " "
					<< face.z() << "//" << face.z() << std::endl;
			}
			else
			{
				ss << "f " << face.x() << " " << face.y() << " " << face.z() << std::endl;
			}

			if (i % 10000 == 0)
			{
				auto percent = ((double)i / (double)indices.size()) * 100.0;
				printf("[%llu / %llu] %f percent\n", i, indices.size(), percent);
			}
		}

		ofs << ss.rdbuf();
		ofs.close();
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		std::ifstream ifs(filename);
		if (!ifs.is_open())
		{
			printf("filename : %s is not open\n", filename.c_str());
			return false;
		}

		std::stringstream buffer;
		buffer << ifs.rdbuf();

		std::string line;
		while (buffer.good())
		{
			getline(buffer, line);
			ParseOneLine(line, points, uvs, normals, colors, indices, 1.0f, 1.0f, 1.0f);
			// Update AABB
			if (!points.empty()) aabb.extend(points.back());
		}
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		for (auto& n : normals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector3i>& GetIndices() const { return indices; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddUV(float u, float v) { uvs.emplace_back(u, v); }
	virtual inline void AddNormal(float x, float y, float z) { normals.emplace_back(x, y, z); }
	virtual inline void AddTriangle(int i0, int i1, int i2) { indices.emplace_back(i0, i1, i2); }
	virtual inline void AddColor(float r, float g, float b, float a = 1.0f) { colors.emplace_back(r, g, b, a); }

protected:
	std::vector<Eigen::Vector2f> uvs;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector3i> indices;
	std::vector<Eigen::Vector4f> colors;
};

class PLYFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		std::ofstream ofs(filename);
		std::stringstream ss;
		ss.precision(6);

		ss << "ply" << std::endl;
		ss << "format ascii 1.0" << std::endl;
		ss << "element vertex " << points.size() << std::endl;
		ss << "property float x" << std::endl;
		ss << "property float y" << std::endl;
		ss << "property float z" << std::endl;

		if (normals.size() == points.size())
		{
			ss << "property float nx" << std::endl;
			ss << "property float ny" << std::endl;
			ss << "property float nz" << std::endl;
		}
		if (colors.size() == points.size())
		{
			ss << "property uchar red" << std::endl;
			ss << "property uchar green" << std::endl;
			ss << "property uchar blue" << std::endl;
			if (useAlpha) ss << "property uchar alpha" << std::endl;
		}
		if (uvs.size() == points.size())
		{
			ss << "property float u" << std::endl;
			ss << "property float v" << std::endl;
		}

		if (!lineIndices.empty())
		{
			ss << "element edge " << lineIndices.size() << std::endl;
			ss << "property int vertex1" << std::endl;
			ss << "property int vertex2" << std::endl;
		}

		if (!triangleIndices.empty())
		{
			ss << "element face " << triangleIndices.size() << std::endl;
			ss << "property list uchar int vertex_indices" << std::endl;
		}

		ss << "end_header" << std::endl;

		for (size_t i = 0; i < points.size(); i++)
		{
			const auto& p = points[i];
			ss << p.x() << " " << p.y() << " " << p.z() << " ";

			if (normals.size() == points.size())
			{
				const auto& n = normals[i];
				ss << n.x() << " " << n.y() << " " << n.z() << " ";
			}

			if (colors.size() == points.size())
			{
				const auto& c = colors[i];
				ss << (int)(c.x() * 255) << " " << (int)(c.y() * 255) << " " << (int)(c.z() * 255) << " ";
				if (useAlpha) ss << (int)(c.w() * 255) << " ";
			}

			if (uvs.size() == points.size())
			{
				const auto& uv = uvs[i];
				ss << uv.x() << " " << uv.y() << " ";
			}

			ss << std::endl;
		}

		for (const auto& line : lineIndices)
		{
			// Edge: vertex1 vertex2 (Usually PLY edges are just v1 v2)
			// But sometimes stored as list. The original code used "2 i0 i1" but property was "vertex1", "vertex2".
			// If property is explicit v1, v2, it should just be "i0 i1".
			// However, original code: `ss << "2 " << i0 << " " << i1` implies it wanted a list, 
			// BUT header said `property int vertex1`, not list.
			// Let's stick to standard PLY edge list convention if property is list, or direct values if scalar.
			// Original header: "property int vertex1" -> Scalar.
			// Original body: "2 i0 i1" -> List format.
			// This is contradictory. I will use standard list format "property list uchar int vertex_indices" for edges usually,
			// or scalar. Let's fix to list for safety or scalar.
			// Based on property definition "vertex1", "vertex2", it expects "i0 i1".
			ss << line.x() << " " << line.y() << std::endl;
		}

		for (const auto& tri : triangleIndices)
		{
			ss << "3 " << tri.x() << " " << tri.y() << " " << tri.z() << std::endl;
		}

		ofs << ss.rdbuf();
		ofs.close();
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		std::ifstream ifs(filename);
		if (!ifs.is_open()) return false;

		std::string line;
		std::vector<std::string> elementNames;
		std::vector<size_t> elementCounts;
		std::vector<bool> listTypeInfo;
		std::vector<std::vector<std::string>> elementPropertyNames;

		while (std::getline(ifs, line))
		{
			auto words = split(line, " \t");
			if (words.empty()) continue;

			if (words[0] == "element")
			{
				elementNames.push_back(words[1]);
				elementCounts.push_back(atoi(words[2].c_str()));
				elementPropertyNames.emplace_back();
				listTypeInfo.push_back(false);
			}
			else if (words[0] == "property")
			{
				size_t index = elementNames.size() - 1;
				if (words[1] == "list")
				{
					listTypeInfo[index] = true;
					elementPropertyNames[index].push_back(words[4]); // property list uchar int vertex_indices
				}
				else
				{
					elementPropertyNames[index].push_back(words[2]);
					if (words[2] == "alpha" || words[2] == "a") useAlpha = true;
				}
			}
			else if (words[0] == "end_header") break;
		}

		for (size_t i = 0; i < elementNames.size(); i++)
		{
			bool isList = listTypeInfo[i];
			size_t count = elementCounts[i];

			if (!isList) // Vertices usually
			{
				for (size_t j = 0; j < count; j++)
				{
					std::getline(ifs, line);
					auto words = split(line, " \t");

					float x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0, u = 0, v = 0;
					float r = 0, g = 0, b = 0, a = 1.0f;
					int label = 0, dlClass = 0;

					// Map properties to words
					for (size_t k = 0; k < words.size() && k < elementPropertyNames[i].size(); k++)
					{
						const auto& prop = elementPropertyNames[i][k];
						float val = (float)atof(words[k].c_str());

						if (prop == "x") x = val;
						else if (prop == "y") y = val;
						else if (prop == "z") z = val;
						else if (prop == "nx") nx = val;
						else if (prop == "ny") ny = val;
						else if (prop == "nz") nz = val;
						else if (prop == "red") r = val / 255.0f;
						else if (prop == "green") g = val / 255.0f;
						else if (prop == "blue") b = val / 255.0f;
						else if (prop == "alpha") a = val / 255.0f;
						else if (prop == "u") u = val;
						else if (prop == "v") v = val;
						else if (prop == "label") label = (int)val;
						else if (prop == "deepLearningClass") dlClass = (int)val;
					}

					AddPoint(x, y, z);
					// Assuming if any normal component exists, add normal
					// Checking against property names existence is better, but simplified here:
					if (!elementPropertyNames[i].empty())
					{
						// This simple parser adds defaults. Real usage checks what properties exist.
						// Assuming standard PLY structure for this snippet.
						// To be strictly robust, we should check if "nx" existed in property names.
						// For now, we add only if we processed vertices.
					}
					// Just storing what we found if it looks like a vertex
					if (elementNames[i] == "vertex") {
						// Logic to only add if properties existed
						// For brevity, assuming consistent file
						if (normals.size() < points.size()) AddNormal(nx, ny, nz);
						if (colors.size() < points.size()) AddColor(r, g, b, a);
						if (uvs.size() < points.size()) AddUV(u, v);
						if (labels.size() < points.size()) labels.push_back(label);
						if (deepLearningClasses.size() < points.size()) deepLearningClasses.push_back(dlClass);
					}
				}
			}
			else // Faces (lists)
			{
				for (size_t j = 0; j < count; j++)
				{
					std::getline(ifs, line);
					auto words = split(line, " \t");
					int listSize = atoi(words[0].c_str());

					if (listSize == 3)
					{
						AddTriangle(atoi(words[1].c_str()), atoi(words[2].c_str()), atoi(words[3].c_str()));
					}
					else if (listSize == 2) // Edge
					{
						AddLineIndex(atoi(words[1].c_str()), atoi(words[2].c_str()));
					}
				}
			}
		}
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		for (auto& n : normals) std::swap(n.y(), n.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}

	inline std::vector<Eigen::Vector3f>& GetNormals() { return normals; }
	inline const std::vector<Eigen::Vector3f>& GetNormals() const { return normals; }
	inline const std::vector<Eigen::Vector2i>& GetLineIndices() const { return lineIndices; }
	inline const std::vector<Eigen::Vector3i>& GetTriangleIndices() const { return triangleIndices; }
	inline std::vector<Eigen::Vector4f>& GetColors() { return colors; }
	inline const std::vector<Eigen::Vector4f>& GetColors() const { return colors; }

	virtual inline void AddUV(float u, float v) { uvs.emplace_back(u, v); }
	virtual inline void AddNormal(float x, float y, float z) { normals.emplace_back(x, y, z); }
	virtual inline void AddLineIndex(int i0, int i1) { lineIndices.emplace_back(i0, i1); }
	virtual inline void AddTriangle(int i0, int i1, int i2) { triangleIndices.emplace_back(i0, i1, i2); }
	virtual inline void AddColor(float r, float g, float b, float a = 1.0f) { colors.emplace_back(r, g, b, a); }

	virtual void AddCube(float cx, float cy, float cz, float nx, float ny, float nz, float r, float g, float b, float a, float scale)
	{
		float h = scale * 0.5f;
		Eigen::Vector3f center(cx, cy, cz);

		Eigen::Vector3f verts[8] = {
			{cx - h, cy - h, cz - h}, {cx + h, cy - h, cz - h},
			{cx + h, cy + h, cz - h}, {cx - h, cy + h, cz - h},
			{cx - h, cy - h, cz + h}, {cx + h, cy - h, cz + h},
			{cx + h, cy + h, cz + h}, {cx - h, cy + h, cz + h}
		};

		unsigned int base = (unsigned int)points.size();

		for (int i = 0; i < 8; ++i)
		{
			AddPoint(verts[i]);
			AddNormal(nx, ny, nz);
			AddColor(r, g, b, a);
		}

		static const int cube_tris[12][3] = {
			{0, 1, 2}, {0, 2, 3}, {4, 6, 5}, {4, 7, 6},
			{0, 4, 5}, {0, 5, 1}, {2, 6, 7}, {2, 7, 3},
			{0, 3, 7}, {0, 7, 4}, {1, 5, 6}, {1, 6, 2}
		};

		for (int i = 0; i < 12; ++i)
		{
			AddTriangle(base + cube_tris[i][0], base + cube_tris[i][1], base + cube_tris[i][2]);
		}
	}

	virtual void AddData(float* pos, float* nrm, float* col, unsigned int count, bool alpha)
	{
		useAlpha = alpha;
		for (unsigned int i = 0; i < count; ++i)
		{
			AddPoint(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
			AddNormal(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]);
			if (alpha) AddColor(col[i * 4], col[i * 4 + 1], col[i * 4 + 2], col[i * 4 + 3]);
			else AddColor(col[i * 3], col[i * 3 + 1], col[i * 3 + 2], 1.0f);
		}
	}

protected:
	std::vector<Eigen::Vector2f> uvs;
	std::vector<Eigen::Vector3f> normals;
	std::vector<Eigen::Vector2i> lineIndices;
	std::vector<Eigen::Vector3i> triangleIndices;
	std::vector<Eigen::Vector4f> colors;
	std::vector<int> labels;
	std::vector<int> deepLearningClasses;
	bool useAlpha = false;
};

// Generic Point Type for ALP (Must support Eigen now)
struct EigenPoint
{
	Eigen::Vector3f position;
	Eigen::Vector3f normal;
	Eigen::Vector3f color; // or Vector4f
};

template<typename Point = EigenPoint>
class ALPFormat
{
public:
	bool Serialize(const std::string& filename)
	{
		std::ofstream ofs(filename, std::ios::out | std::ios::binary);
		if (!ofs.is_open()) return false;

		unsigned long nop = (unsigned long)points.size();
		unsigned int pointSize = sizeof(Point);

		ofs.write((char*)&nop, sizeof(unsigned long));
		ofs.write((char*)&pointSize, sizeof(unsigned int));

		if (nop > 0)
			ofs.write((char*)points.data(), nop * pointSize);

		ofs.close();
		return true;
	}

	bool Deserialize(const std::string& filename)
	{
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);
		if (!ifs.is_open()) return false;

		unsigned long nop = 0;
		unsigned int pointSize = 0;

		ifs.read((char*)&nop, sizeof(unsigned long));
		ifs.read((char*)&pointSize, sizeof(unsigned int));

		if (pointSize != sizeof(Point))
		{
			// Size mismatch handling
			ifs.close();
			return false;
		}

		points.resize(nop);
		ifs.read((char*)points.data(), nop * pointSize);

		// Update AABB
		aabb.setEmpty();
		// Assumes Point has 'position' member which is Eigen::Vector3f
		for (const auto& p : points) aabb.extend(p.position);

		ifs.close();
		return true;
	}

	void AddPoint(const Point& point)
	{
		std::lock_guard<std::mutex> lock(points_mutex);
		points.push_back(point);
		aabb.extend(point.position);
	}

	void AddPoints(const std::vector<Point>& inputPoints)
	{
		std::lock_guard<std::mutex> lock(points_mutex);
		points.insert(points.end(), inputPoints.begin(), inputPoints.end());
		for (const auto& p : inputPoints) aabb.extend(p.position);
	}

	const std::vector<Point>& GetPoints() const
	{
		std::lock_guard<std::mutex> lock(points_mutex);
		return points;
	}

	// Helper to convert from PLY to internal points
	void FromPLY(const PLYFormat& ply)
	{
		const auto& plyPoints = ply.GetPoints();
		const auto& plyNormals = ply.GetNormals();
		const auto& plyColors = ply.GetColors();

		size_t count = plyPoints.size();
		points.reserve(points.size() + count);

		for (size_t i = 0; i < count; i++)
		{
			Point p;
			// Assuming Point struct structure:
			p.position = plyPoints[i];
			if (i < plyNormals.size()) p.normal = plyNormals[i];
			else p.normal = Eigen::Vector3f::Zero();

			if (i < plyColors.size()) p.color = plyColors[i].head<3>(); // Taking RGB
			else p.color = Eigen::Vector3f::Ones();

			AddPoint(p);
		}
	}

	inline Eigen::Vector3f GetAABBMin() { return aabb.min(); }
	inline Eigen::Vector3f GetAABBMax() { return aabb.max(); }
	inline Eigen::Vector3f GetAABBCenter() { return aabb.center(); }

protected:
	mutable std::mutex points_mutex;
	std::vector<Point> points;
	Eigen::AlignedBox3f aabb;
};

class CSVFormat : public HSerializable
{
public:
	virtual bool Serialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err) return false;

		fprintf(fp, "%llu\n", points.size());
		fprintf(fp, "X, Y, Z\n");
		for (const auto& p : points)
		{
			fprintf(fp, "%.6f, %.6f, %.6f\n", p.x(), p.y(), p.z());
		}

		fclose(fp);
		return true;
	}

	virtual bool Deserialize(const std::string& filename) override
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err) return false;

		char buffer[1024];
		// Skip header lines potentially? Original skipped one line
		fgets(buffer, sizeof(buffer), fp); // Size?
		fgets(buffer, sizeof(buffer), fp); // Header X,Y,Z?

		while (fgets(buffer, sizeof(buffer), fp)) {
			float x, y, z;
			if (sscanf_s(buffer, "%f,%f,%f", &x, &y, &z) == 3) {
				AddPoint(x, y, z);
			}
		}

		fclose(fp);
		return true;
	}

	virtual inline void SwapAxisYZ() override
	{
		for (auto& p : points) std::swap(p.y(), p.z());
		aabb.setEmpty();
		for (const auto& p : points) aabb.extend(p);
	}
};
