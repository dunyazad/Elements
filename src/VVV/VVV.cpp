#include <VVV/VVV.h>
#include <unordered_map>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>

namespace VVV
{
	struct Morton64Hash
	{
		size_t operator()(const Morton64& key) const noexcept
		{
			return static_cast<size_t>(key.code);
		}
	};
}