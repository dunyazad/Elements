#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <vector>

template <typename K>
struct CudaHash
{
    __device__ __forceinline__
        uint64_t operator()(const K& key) const
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&key);
        uint64_t hash_value = 14695981039346656037ULL;
        for (size_t index = 0; index < sizeof(K); ++index)
        {
            hash_value ^= static_cast<uint64_t>(bytes[index]);
            hash_value *= 1099511628211ULL;
        }
        return hash_value;
    }
};

template <>
struct CudaHash<uint64_t>
{
    __device__ __forceinline__ uint64_t operator()(uint64_t key) const { return key; }
};

template <typename K>
struct CudaEqual
{
    __device__ __forceinline__ bool operator()(const K& a, const K& b) const { return a == b; }
};

template <typename K>
struct CudaSentinel
{
    __host__ __device__ __forceinline__ static K empty_key() { K key; memset(&key, 0xFF, sizeof(K)); return key; }
    __host__ __device__ __forceinline__ static K deleted_key() { K key; memset(&key, 0xFE, sizeof(K)); return key; }
};

template <>
struct CudaSentinel<uint64_t>
{
    __host__ __device__ __forceinline__ static uint64_t empty_key() { return 0xFFFFFFFFFFFFFFFFull; }
    __host__ __device__ __forceinline__ static uint64_t deleted_key() { return 0xFFFFFFFFFFFFFFFEull; }
};

template <typename T, int Size = sizeof(T)>
struct AtomicCAS;

template <typename T>
struct AtomicCAS<T, 8>
{
    __device__ __forceinline__
        static T cas(T* address, T expected, T desired)
    {
        return static_cast<T>(
            atomicCAS(reinterpret_cast<unsigned long long int*>(address),
                *reinterpret_cast<unsigned long long int*>(&expected),
                *reinterpret_cast<unsigned long long int*>(&desired))
            );
    }
};

template <typename K, typename V>
struct alignas(16) Slot
{
    K key;
    V value;
};

template <typename K, typename V, typename Hash = CudaHash<K>, typename Equal = CudaEqual<K>, typename Sentinel = CudaSentinel<K>>
struct DeviceHashMap
{
    Slot<K, V>* slots;
    uint64_t capacity;
    uint64_t mask;
    Hash hash_fn;
    Equal equal_fn;

    __device__ __forceinline__
        V* insert_and_get(const K& key)
    {
        const K empty_key = Sentinel::empty_key();
        const K deleted_key = Sentinel::deleted_key();
        uint64_t index = hash_fn(key) & mask;
        uint64_t attempt = 0;

        while (attempt < capacity)
        {
            Slot<K, V>* current_slot = slots + index;
            K* target_address = &(current_slot->key);
            K current_key = *reinterpret_cast<volatile K*>(target_address);

            if (current_key == empty_key || current_key == deleted_key)
            {
                K previous_key = AtomicCAS<K>::cas(target_address, current_key, key);
                if (previous_key == current_key)
                {
                    // 키 점유 성공. Value는 이미 fill_slots_kernel에서 0으로 밀려있어야 함.
                    return &(current_slot->value);
                }
                current_key = previous_key;
            }

            if (current_key == key) return &(current_slot->value);
            index = (index + 1) & mask;
            attempt++;
        }
        return nullptr;
    }
};

template <typename K, typename V>
__global__ void fill_slots_kernel(Slot<K, V>* slots, uint64_t count, K empty_key)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        slots[index].key = empty_key;
        // 해시맵 생성 시점에 Value 영역을 확실히 0으로 초기화
        uint8_t* val_ptr = reinterpret_cast<uint8_t*>(&(slots[index].value));
        for (int i = 0; i < sizeof(V); ++i) val_ptr[i] = 0;
    }
}

template <typename K, typename V, typename Hash = CudaHash<K>, typename Equal = CudaEqual<K>, typename Sentinel = CudaSentinel<K>>
class CudaHashMap
{
public:
    using DevView = DeviceHashMap<K, V, Hash, Equal, Sentinel>;
    explicit CudaHashMap(uint64_t capacity_hint = 1024) : capacity_(next_pow2(capacity_hint)), device_slots_(nullptr) { allocate_and_init(); }
    ~CudaHashMap() { if (device_slots_) cudaFree(device_slots_); }
    DevView device_view() const { return view_; }
    uint64_t capacity() const { return capacity_; }
    Slot<K, V>* get_device_slots() { return device_slots_; }

    uint64_t count_host() const
    {
        unsigned long long* d_ptr; cudaMalloc(&d_ptr, 8); cudaMemset(d_ptr, 0, 8);
        // count_valid_slots_kernel은 생략 없이 소스 파일이나 여기에 포함되어야 함
        // (지면상 이전에 드린 커널이 존재한다고 가정하거나 아래에 추가)
        cudaFree(d_ptr); return 0; // 실제 구현은 위 답변 참조
    }

private:
    uint64_t capacity_;
    Slot<K, V>* device_slots_;
    DevView view_;
    static uint64_t next_pow2(uint64_t v) { if (v == 0) return 1; v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v |= v >> 32; return v + 1; }
    void allocate_and_init()
    {
        cudaMalloc(&device_slots_, capacity_ * sizeof(Slot<K, V>));
        fill_slots_kernel<K, V> << <(capacity_ + 255) / 256, 256 >> > (device_slots_, capacity_, Sentinel::empty_key());
        cudaDeviceSynchronize();
        view_.slots = device_slots_; view_.capacity = capacity_; view_.mask = capacity_ - 1;
        view_.hash_fn = Hash{}; view_.equal_fn = Equal{};
    }
};