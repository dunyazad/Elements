#pragma once

#include <any>
#include <map>
#include <string>
#include <vector>

namespace PointProcessing
{
	enum class PointVisualizationMode
	{
		None,
		Gradient,
		Binary,
		OutlierFiltered
	};

	class PointProcessorParameters
	{
	public:
		PointProcessorParameters() = default;
		virtual ~PointProcessorParameters() = default;

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

	class PointProcessor
	{
	public:
		enum class PointProcessorType
		{
			SOR,
			ROR,
			CurvatureAnalysis,
			NormalDeviation,
			PFOR
		};

		virtual ~PointProcessor() = default;

		virtual void Process(const PointProcessorParameters& parameters) = 0;

		inline PointProcessorType GetType() const { return type; }

		template<typename T>
		T GetProcessResult(const std::string& key, const T& defaultValue) const
		{
			auto it = processResults.find(key);
			if (it != processResults.end())
			{
				return std::any_cast<T>(it->second);
			}
			return defaultValue;
		}

	protected:
		PointProcessor(PointProcessorType t) : type(t) {}

		std::map<std::string, std::any> processResults;

	private:
		PointProcessorType type;
		PointProcessorParameters parameters;
	};
}
