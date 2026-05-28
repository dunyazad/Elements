#pragma once

#include <Core/Common/DeviceCommon.h>

struct cached_allocator;

namespace Huvitz
{
    template <typename K>
    struct CudaHash
    {
        __device__ __forceinline__
            uint64_t operator()(const K& key) const
        {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&key);
            uint64_t hashValue = 14695981039346656037ULL;
            for (size_t i = 0; i < sizeof(K); ++i)
            {
                hashValue ^= static_cast<uint64_t>(bytes[i]);
                hashValue *= 1099511628211ULL;
            }
            return hashValue;
        }
    };

    template <>
    struct CudaHash<uint64_t>
    {
        __device__ __forceinline__ uint64_t operator()(uint64_t key) const
        {
            key ^= key >> 33;
            key *= 0xff51afd7ed558ccdULL;
            key ^= key >> 33;
            key *= 0xc4ceb9fe1a85ec53ULL;
            key ^= key >> 33;
            return key;
        }
    };

    template <typename Key>
    struct CudaEqual
    {
        __device__ __forceinline__ bool operator()(const Key& a, const Key& b) const
        {
            return a == b;
        }
    };

    template <typename Key>
    struct CudaSentinel
    {
        __host__ __device__ __forceinline__ static Key EmptyKey()
        {
            Key k;
            memset(&k, 0xFF, sizeof(Key));
            return k;
        }

        __host__ __device__ __forceinline__ static Key DeletedKey()
        {
            Key k;
            memset(&k, 0xFE, sizeof(Key));
            return k;
        }
    };

    template <>
    struct CudaSentinel<uint64_t>
    {
        __host__ __device__ __forceinline__ static uint64_t EmptyKey()
        {
            return 0xFFFFFFFFFFFFFFFFull;
        }

        __host__ __device__ __forceinline__ static uint64_t DeletedKey()
        {
            return 0xFFFFFFFFFFFFFFFEull;
        }
    };

    template <typename T, int Size = sizeof(T)>
    struct AtomicCAS;

    template <typename T>
    struct AtomicCAS<T, 4>
    {
        __device__ __forceinline__
            static T CAS(T* address, T expected, T desired)
        {
            unsigned int prev = atomicCAS(
                reinterpret_cast<unsigned int*>(address),
                *reinterpret_cast<unsigned int*>(&expected),
                *reinterpret_cast<unsigned int*>(&desired));
            return *reinterpret_cast<T*>(&prev);
        }
    };

    template <typename T>
    struct AtomicCAS<T, 8>
    {
        __device__ __forceinline__
            static T CAS(T* address, T expected, T desired)
        {
            unsigned long long int prev = atomicCAS(
                reinterpret_cast<unsigned long long int*>(address),
                *reinterpret_cast<unsigned long long int*>(&expected),
                *reinterpret_cast<unsigned long long int*>(&desired));
            return *reinterpret_cast<T*>(&prev);
        }
    };

    template <typename Key, typename Value>
    struct alignas(16) Slot
    {
        Key key;
        Value value;
    };

    template <typename Key, typename Value,
        typename Hash = CudaHash<Key>,
        typename Equal = CudaEqual<Key>,
        typename Sentinel = CudaSentinel<Key>>
        struct DeviceHashMap
    {
        Slot<Key, Value>* slotArray;
        uint64_t capacity;
        uint64_t mask;
        Hash hashFunction;
        Equal equalFunction;

        __device__ __forceinline__
            Value* InsertAndGet(const Key& key)
        {
            const Key emptyKey = Sentinel::EmptyKey();
            const Key deletedKey = Sentinel::DeletedKey();
            uint64_t index = hashFunction(key) & mask;
            uint64_t attempt = 0;

            while (attempt < capacity)
            {
                Slot<Key, Value>* slot = slotArray + index;
                Key* addr = &slot->key;
                Key cur = *reinterpret_cast<volatile Key*>(addr);

                if (cur == emptyKey || cur == deletedKey)
                {
                    Key prev = AtomicCAS<Key>::CAS(addr, cur, key);
                    if (prev == cur)
                    {
                        if (prev == deletedKey)
                        {
                            uint8_t* valuePtr = reinterpret_cast<uint8_t*>(&slot->value);
                            for (int i = 0; i < (int)sizeof(Value); ++i)
                            {
                                valuePtr[i] = 0;
                            }
                        }
                        return &slot->value;
                    }
                    cur = prev;
                }

                if (equalFunction(cur, key))
                {
                    return &slot->value;
                }
                index = (index + 1) & mask;
                ++attempt;
            }
            return nullptr;
        }

        __device__ __forceinline__
            Value* Find(const Key& key) const
        {
            const Key emptyKey = Sentinel::EmptyKey();
            const Key deletedKey = Sentinel::DeletedKey();
            uint64_t index = hashFunction(key) & mask;
            uint64_t attempt = 0;

            while (attempt < capacity)
            {
                Slot<Key, Value>* slot = slotArray + index;
                Key cur = *reinterpret_cast<volatile Key*>(&slot->key);

                if (cur == emptyKey)
                {
                    return nullptr;
                }
                if (cur == deletedKey)
                {
                    index = (index + 1) & mask;
                    ++attempt;
                    continue;
                }
                if (equalFunction(cur, key))
                {
                    return &slot->value;
                }

                index = (index + 1) & mask;
                ++attempt;
            }
            return nullptr;
        }

        __device__ __forceinline__
            bool Erase(const Key& key)
        {
            const Key emptyKey = Sentinel::EmptyKey();
            const Key deletedKey = Sentinel::DeletedKey();
            uint64_t index = hashFunction(key) & mask;
            uint64_t attempt = 0;

            while (attempt < capacity)
            {
                Slot<Key, Value>* slot = slotArray + index;
                Key* addr = &slot->key;
                Key cur = *reinterpret_cast<volatile Key*>(addr);

                if (cur == emptyKey)
                {
                    return false;
                }
                if (cur == deletedKey)
                {
                    index = (index + 1) & mask;
                    ++attempt;
                    continue;
                }

                if (equalFunction(cur, key))
                {
                    Key prev = AtomicCAS<Key>::CAS(addr, cur, deletedKey);
                    return prev == cur;
                }

                index = (index + 1) & mask;
                ++attempt;
            }
            return false;
        }
    };

    template <typename Key, typename Value>
    __global__ void Kernel_FillSlots(Slot<Key, Value>* slotArray, uint64_t count, Key emptyKey)
    {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count)
        {
            slotArray[idx].key = emptyKey;
            slotArray[idx].value = Value{};
        }
    }

    template <typename Key, typename Value, typename Hash, typename Equal, typename Sentinel>
    __global__ void Kernel_Rehash(Slot<Key, Value>* oldSlots, uint64_t oldCapacity, DeviceHashMap<Key, Value, Hash, Equal, Sentinel> newMap)
    {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < oldCapacity)
        {
            Key k = oldSlots[idx].key;
            if (k != Sentinel::EmptyKey() && k != Sentinel::DeletedKey())
            {
                Value* ptr = newMap.InsertAndGet(k);
                if (ptr != nullptr)
                {
                    *ptr = oldSlots[idx].value;
                }
            }
        }
    }

    template <typename Key, typename Value>
    __global__ void Kernel_CountValidSlots(
        const Slot<Key, Value>* slotArray,
        uint64_t count,
        Key emptyKey,
        Key deletedKey,
        unsigned long long* result)
    {
        __shared__ unsigned long long sharedCount[256];
        uint32_t tid = threadIdx.x;
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

        unsigned long long localCount = 0;
        if (idx < count)
        {
            Key k = slotArray[idx].key;
            if (k != emptyKey && k != deletedKey)
            {
                localCount = 1ULL;
            }
        }
        sharedCount[tid] = localCount;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                sharedCount[tid] += sharedCount[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            atomicAdd(result, sharedCount[0]);
        }
    }

    template <typename Key, typename Value,
        typename Hash = CudaHash<Key>,
        typename Equal = CudaEqual<Key>,
        typename Sentinel = CudaSentinel<Key>>
        class CudaHashMap
    {
    public:
        using DeviceView = DeviceHashMap<Key, Value, Hash, Equal, Sentinel>;

        explicit CudaHashMap(uint64_t capacityTarget = 1024, cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr)
            : capacity(NextPowerOfTwo(capacityTarget)), deviceSlotArray(nullptr), countResult(nullptr)
        {
            AllocateAndInit(allocator, stream);
        }

        ~CudaHashMap()
        {
            if (deviceSlotArray)
            {
                cudaFree(deviceSlotArray);
            }
            if (countResult)
            {
                cudaFree(countResult);
            }
        }

        CudaHashMap(const CudaHashMap&) = delete;
        CudaHashMap& operator=(const CudaHashMap&) = delete;

        CudaHashMap(CudaHashMap&& other) noexcept
            : capacity(other.capacity), deviceSlotArray(other.deviceSlotArray), countResult(other.countResult), deviceView(other.deviceView)
        {
            other.deviceSlotArray = nullptr;
            other.countResult = nullptr;
        }

        CudaHashMap& operator=(CudaHashMap&& other) noexcept
        {
            if (this != &other)
            {
                if (deviceSlotArray)
                {
                    cudaFree(deviceSlotArray);
                }
                if (countResult)
                {
                    cudaFree(countResult);
                }
                capacity = other.capacity;
                deviceSlotArray = other.deviceSlotArray;
                countResult = other.countResult;
                deviceView = other.deviceView;

                other.deviceSlotArray = nullptr;
                other.countResult = nullptr;
            }
            return *this;
        }

        DeviceView GetDeviceView() const
        {
            return deviceView;
        }

        uint64_t GetCapacity() const
        {
            return capacity;
        }

        Slot<Key, Value>* GetDeviceSlotArray()
        {
            return deviceSlotArray;
        }

        uint64_t CountHost(cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr) const
        {
            *countResult = 0;

            uint64_t blocks = (capacity + 255) / 256;
            Kernel_CountValidSlots<Key, Value> << <blocks, 256, 0, stream >> > (
                deviceSlotArray, capacity,
                Sentinel::EmptyKey(), Sentinel::DeletedKey(),
                countResult);

            if (stream)
            {
                cudaStreamSynchronize(stream);
            }
            else
            {
                cudaDeviceSynchronize();
            }

            return static_cast<uint64_t>(*countResult);
        }

        void Clear(cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr)
        {
            if (deviceSlotArray == nullptr)
            {
                return;
            }

            uint64_t blocks = (capacity + 255) / 256;
            Kernel_FillSlots<Key, Value> << <blocks, 256, 0, stream >> > (
                deviceSlotArray, capacity, Sentinel::EmptyKey());

            if (stream)
            {
                cudaStreamSynchronize(stream);
            }
            else
            {
                cudaDeviceSynchronize();
            }
        }

        void Resize(uint64_t targetCapacity, cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr)
        {
            uint64_t nextCapacity = NextPowerOfTwo(targetCapacity);
            if (nextCapacity <= capacity)
            {
                return;
            }

            Slot<Key, Value>* newdeviceSlotArray = nullptr;
            cudaMalloc(&newdeviceSlotArray, nextCapacity * sizeof(Slot<Key, Value>));

            uint64_t newBlocks = (nextCapacity + 255) / 256;
            Kernel_FillSlots<Key, Value> << <newBlocks, 256, 0, stream >> > (
                newdeviceSlotArray, nextCapacity, Sentinel::EmptyKey());

            if (stream)
            {
                cudaStreamSynchronize(stream);
            }
            else
            {
                cudaDeviceSynchronize();
            }

            DeviceView newView;
            newView.slotArray = newdeviceSlotArray;
            newView.capacity = nextCapacity;
            newView.mask = nextCapacity - 1;
            newView.hashFunction = Hash{};
            newView.equalFunction = Equal{};

            uint64_t oldBlocks = (capacity + 255) / 256;
            Kernel_Rehash<Key, Value, Hash, Equal, Sentinel> << <oldBlocks, 256, 0, stream >> > (
                deviceSlotArray, capacity, newView);

            if (stream)
            {
                cudaStreamSynchronize(stream);
            }
            else
            {
                cudaDeviceSynchronize();
            }

            cudaFree(deviceSlotArray);

            deviceSlotArray = newdeviceSlotArray;
            capacity = nextCapacity;
            deviceView = newView;
        }

    private:
        uint64_t capacity;
        Slot<Key, Value>* deviceSlotArray;
        unsigned long long* countResult;
        DeviceView deviceView;

        static uint64_t NextPowerOfTwo(uint64_t v)
        {
            if (v == 0)
            {
                return 1;
            }
            v--;
            v |= v >> 1; v |= v >> 2; v |= v >> 4;
            v |= v >> 8; v |= v >> 16; v |= v >> 32;
            return v + 1;
        }

        void AllocateAndInit(cached_allocator* allocator = nullptr, CUstream_st* stream = nullptr)
        {
            cudaMalloc(&deviceSlotArray, capacity * sizeof(Slot<Key, Value>));
            cudaMallocManaged(&countResult, sizeof(unsigned long long));

            Clear(allocator, stream);

            deviceView.slotArray = deviceSlotArray;
            deviceView.capacity = capacity;
            deviceView.mask = capacity - 1;
            deviceView.hashFunction = Hash{};
            deviceView.equalFunction = Equal{};
        }
    };
}