#pragma once

class AtomicDisjointSet
{
public:
	void Initialize(size_t n);

	int Find(int i);

	void Union(int i, int j);

private:
	std::unique_ptr<std::atomic<int>[]> parent;
	size_t size = 0;
};
