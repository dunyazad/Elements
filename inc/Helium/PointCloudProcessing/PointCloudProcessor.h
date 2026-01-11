#pragma once

#include <any>
#include <map>
#include <string>
#include <vector>

enum class PointCloudVisualizationMode
{
	None,
	Gradient,
	Binary
};

class PointCloudProcessorParameters
{
public:
	PointCloudProcessorParameters() = default;
	virtual ~PointCloudProcessorParameters() = default;

	template<typename T>
	T GetParameter(const std::string& key, const T& defaultValue) const
	{
		auto it = parameters.find(key);
		if (it != parameters.end())
		{
			return std::any_cast<T>(it->second);
		}
		return defaultValue;
	}

	template<typename T>
	void SetParameter(const std::string& key, const T& value)
	{
		parameters[key] = value;
	}

protected:
	std::map<std::string, std::any> parameters;
};

class PointCloudProcessor
{
public:
	enum class PointCloudProcessorType
	{
		SOR,
		ROR,
		CurvatureAnalysis,
		NormalDeviation,
		PFOR
	};

	virtual ~PointCloudProcessor() = default;

	virtual std::vector<uint8_t> Process(const PointCloudProcessorParameters& parameters) = 0;

	inline PointCloudProcessorType GetType() const { return type; }

protected:
	PointCloudProcessor(PointCloudProcessorType t) : type(t) {}

private:
	PointCloudProcessorType type;
	PointCloudProcessorParameters parameters;
};
