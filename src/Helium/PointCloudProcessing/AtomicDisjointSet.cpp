#include "pch.h"

#include <Helium/PointCloudProcessing/AtomicDisjointSet.h>

#include <atomic>
#include <execution>
#include <vector>

void AtomicDisjointSet::Initialize(size_t n)
{
	size = n;
	parent = std::make_unique<std::atomic<int>[]>(n);

	std::vector<int> indices(n);
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
		parent[i].store(i, std::memory_order_relaxed);
		});
}

int AtomicDisjointSet::Find(int i)
{
	int p = parent[i].load(std::memory_order_relaxed);
	while (p != i)
	{
		int pp = parent[p].load(std::memory_order_relaxed);
		parent[i].store(pp, std::memory_order_relaxed);
		i = pp;
		p = parent[i].load(std::memory_order_relaxed);
	}
	return i;
}

void AtomicDisjointSet::Union(int i, int j)
{
	int rootA = Find(i);
	int rootB = Find(j);

	while (rootA != rootB)
	{
		if (rootA > rootB) std::swap(rootA, rootB);

		int expected = rootB;
		if (parent[rootB].compare_exchange_weak(expected, rootA))
		{
			return;
		}

		rootA = Find(rootA);
		rootB = Find(expected);
	}
}
