#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <cstring>
#include <atomic>
#include <stdexcept>
#include <functional>

// ---------------------------------------------------------------------------
// Blob 및 레이아웃 정의
// ---------------------------------------------------------------------------
#ifndef BLOB_DEFINED
#define BLOB_DEFINED
using Blob = std::vector<uint8_t>;
#endif

#pragma pack(push, 1)
struct FrameHeader
{
    uint32_t magic;        // 0x44464D00 'DFM\0'
    uint32_t frameIndex;
    uint32_t tag;          // 데이터 뭉치 대표 태그 (성능을 위해 통합 저장 시 0 사용)
    uint64_t payloadBytes;
};
#pragma pack(pop)

static constexpr uint32_t DATAFRAME_MAGIC = 0x44464D00;

struct DataFrame
{
    uint32_t frameIndex;
    uint32_t tag;
    Blob data;
};

// ---------------------------------------------------------------------------
// AsyncFileWriter
// ---------------------------------------------------------------------------
class AsyncFileWriter
{
public:
    enum class OpenMode { Truncate, Append };

    AsyncFileWriter(
        const std::string& filepath,
        OpenMode mode = OpenMode::Truncate,
        std::function<void(std::string)> on_error = nullptr)
        : on_error(std::move(on_error))
    {
        auto ios_mode = std::ios::out | std::ios::binary;
        if (mode == OpenMode::Append)
        {
            ios_mode |= std::ios::app;
        }

        file.open(filepath, ios_mode);
        if (!file.is_open())
        {
            if (this->on_error)
            {
                this->on_error("Failed to open file: " + filepath);
            }
            return;
        }
        thread = std::thread(&AsyncFileWriter::worker_loop, this);
    }

    ~AsyncFileWriter()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop.store(true, std::memory_order_relaxed);
        }
        cv.notify_one();
        if (thread.joinable())
        {
            thread.join();
        }
        if (file.is_open())
        {
            file.flush();
        }
    }

    AsyncFileWriter(const AsyncFileWriter&) = delete;
    AsyncFileWriter& operator=(const AsyncFileWriter&) = delete;

    bool push(Blob blob)
    {
        if (stop.load(std::memory_order_relaxed))
        {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            io_queue.push(std::move(blob));
        }
        cv.notify_one();
        return true;
    }

    void flush()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv_empty.wait(lock, [this] { return io_queue.empty(); });
    }

    bool is_open() const
    {
        return file.is_open();
    }

private:
    void worker_loop()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this] {
                return !io_queue.empty() || stop.load(std::memory_order_relaxed);
                });
            drain(lock);
            if (stop.load(std::memory_order_relaxed))
            {
                break;
            }
        }
    }

    void drain(std::unique_lock<std::mutex>& lock)
    {
        while (!io_queue.empty())
        {
            Blob item = std::move(io_queue.front());
            io_queue.pop();
            if (io_queue.empty())
            {
                cv_empty.notify_all();
            }

            lock.unlock();
            if (file.is_open())
            {
                file.write(reinterpret_cast<const char*>(item.data()),
                    static_cast<std::streamsize>(item.size()));
            }
            lock.lock();
        }
    }

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cv_empty;
    std::queue<Blob> io_queue;
    std::ofstream file;
    std::function<void(std::string)> on_error;
    std::atomic<bool> stop{ false };
    std::thread thread;
};

// ---------------------------------------------------------------------------
// DataFrameRecorder
// ---------------------------------------------------------------------------
template<typename T>
class DataFrameRecorder
{
public:
    using Serializer = std::function<Blob(const T&)>;

    DataFrameRecorder(
        const std::string& filepath,
        Serializer serializer,
        std::function<void(std::string)> on_error = nullptr)
        : writer(filepath, AsyncFileWriter::OpenMode::Truncate, std::move(on_error)),
        serializer(std::move(serializer))
    {
    }

    void record_raw(const Blob& payload, uint32_t tag, uint32_t frame_idx)
    {
        FrameHeader hdr{};
        hdr.magic = DATAFRAME_MAGIC;
        hdr.frameIndex = frame_idx;
        hdr.tag = tag;
        hdr.payloadBytes = static_cast<uint64_t>(payload.size());

        Blob blob(sizeof(FrameHeader) + payload.size());
        std::memcpy(blob.data(), &hdr, sizeof(hdr));
        std::memcpy(blob.data() + sizeof(hdr), payload.data(), payload.size());

        writer.push(std::move(blob));
    }

    void flush()
    {
        writer.flush();
    }

private:
    AsyncFileWriter writer;
    Serializer serializer;
};

// ---------------------------------------------------------------------------
// DataFrameReader
// ---------------------------------------------------------------------------
class DataFrameReader
{
public:
    explicit DataFrameReader(const std::string& filepath)
    {
        file.open(filepath, std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("DataFrameReader: cannot open file: " + filepath);
        }
        index_file();
    }

    ~DataFrameReader()
    {
        if (file.is_open())
        {
            file.close();
        }
    }

    bool is_open() const
    {
        return file.is_open();
    }

    bool at_end() const
    {
        return cursor >= offsets.size();
    }

    std::optional<DataFrame> next()
    {
        if (cursor >= offsets.size())
        {
            return std::nullopt;
        }
        file.seekg(offsets[cursor]);
        auto frame = read_one();
        if (frame)
        {
            ++cursor;
        }
        return frame;
    }

    void rewind()
    {
        cursor = 0;
        file.clear();
        if (!offsets.empty())
        {
            file.seekg(offsets[0]);
        }
    }

private:
    void index_file()
    {
        file.seekg(0, std::ios::beg);
        offsets.clear();
        FrameHeader hdr{};
        while (file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)))
        {
            if (hdr.magic != DATAFRAME_MAGIC)
            {
                break;
            }
            offsets.push_back(static_cast<std::streamoff>(file.tellg()) - static_cast<std::streamoff>(sizeof(hdr)));
            file.seekg(static_cast<std::streamoff>(hdr.payloadBytes), std::ios::cur);
        }
        file.clear();
        rewind();
    }

    std::optional<DataFrame> read_one()
    {
        FrameHeader hdr{};
        if (!file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)))
        {
            return std::nullopt;
        }
        DataFrame frame;
        frame.frameIndex = hdr.frameIndex;
        frame.tag = hdr.tag;
        frame.data.resize(static_cast<size_t>(hdr.payloadBytes));
        file.read(reinterpret_cast<char*>(frame.data.data()), hdr.payloadBytes);
        return frame;
    }

    std::ifstream file;
    std::vector<std::streamoff> offsets;
    size_t cursor{ 0 };
};

// ---------------------------------------------------------------------------
// BackgroundDataWriter
// ---------------------------------------------------------------------------
struct DataFrameDeviceView
{
    unsigned int capacity;
    unsigned int* size_ptr;
    float3* positions;
    float3* normals;
    uchar3* colors;
    unsigned int* labels;
    unsigned int* tags;

#if defined(__CUDACC__)
    __device__ void AddPointWarpOptimized(bool keep, const float3& p, const float3& n, const uchar3& c, unsigned int label, unsigned int tag) const
    {
        unsigned int mask = __ballot_sync(0xFFFFFFFF, keep);
        int warp_count = __popc(mask);
        int lane_id = threadIdx.x % 32;
        int warp_offset = 0;

        if (lane_id == 0 && warp_count > 0)
        {
            warp_offset = atomicAdd(size_ptr, warp_count);
        }
        warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

        if (keep)
        {
            int prefix = __popc(mask & ((1 << lane_id) - 1));
            unsigned int idx = warp_offset + prefix;
            if (idx < capacity)
            {
                positions[idx] = p;
                normals[idx] = n;
                colors[idx] = c;
                labels[idx] = label;
                tags[idx] = tag;
            }
        }
    }
#endif
};

class BackgroundDataWriter
{
public:
    BackgroundDataWriter(unsigned int initial_capacity, const std::string& filepath)
        : capacity(0), stream(nullptr), sync_event(nullptr), device_size(nullptr), pinned_size(nullptr),
        device_positions(nullptr), pinned_positions(nullptr), device_normals(nullptr), pinned_normals(nullptr),
        device_colors(nullptr), pinned_colors(nullptr), device_labels(nullptr), pinned_labels(nullptr),
        device_tags(nullptr), pinned_tags(nullptr)
    {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming);
        AllocateMemory(initial_capacity);
        recorder = std::make_unique<DataFrameRecorder<Blob>>(filepath, [](const Blob& b) { return b; });
    }

    ~BackgroundDataWriter()
    {
        if (stream)
        {
            cudaStreamSynchronize(stream);
        }
        FreeMemory();
        if (stream)
        {
            cudaStreamDestroy(stream);
        }
        if (sync_event)
        {
            cudaEventDestroy(sync_event);
        }
    }

    void AsyncSaveFrame(int frame_index, uint32_t group_tag = 0)
    {
        bool needs_resize = false;
        unsigned int count = FetchToPinned(needs_resize);

        if (needs_resize)
        {
            Resize(static_cast<unsigned int>(count * 1.2f));
            return;
        }

        if (count > 0 && recorder)
        {
            size_t p_size = count * sizeof(float3);
            size_t n_size = count * sizeof(float3);
            size_t c_size = count * sizeof(uchar3);
            size_t l_size = count * sizeof(unsigned int);
            size_t t_size = count * sizeof(unsigned int);

            Blob payload(p_size + n_size + c_size + l_size + t_size);
            uint8_t* ptr = payload.data();

            std::memcpy(ptr, pinned_positions, p_size); ptr += p_size;
            std::memcpy(ptr, pinned_normals, n_size);   ptr += n_size;
            std::memcpy(ptr, pinned_colors, c_size);    ptr += c_size;
            std::memcpy(ptr, pinned_labels, l_size);    ptr += l_size;
            std::memcpy(ptr, pinned_tags, t_size);

            recorder->record_raw(payload, group_tag, frame_index);
        }
    }

    DataFrameDeviceView GetView() const
    {
        DataFrameDeviceView view;
        view.capacity = capacity;
        view.size_ptr = device_size;
        view.positions = device_positions;
        view.normals = device_normals;
        view.colors = device_colors;
        view.labels = device_labels;
        view.tags = device_tags;
        return view;
    }

    void PrepareNextFrame()
    {
        if (device_size)
        {
            cudaMemsetAsync(device_size, 0, sizeof(unsigned int), stream);
        }
    }

    void Resize(unsigned int new_capacity)
    {
        cudaStreamSynchronize(stream);
        FreeMemory();
        AllocateMemory(new_capacity);
    }

private:
    unsigned int FetchToPinned(bool& needs_resize)
    {
        cudaMemcpyAsync(pinned_size, device_size, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(sync_event, stream);
        while (cudaEventQuery(sync_event) == cudaErrorNotReady)
        {
            std::this_thread::yield();
        }

        unsigned int count = *pinned_size;
        if (count > capacity)
        {
            needs_resize = true;
            return count;
        }

        if (count > 0)
        {
            cudaMemcpyAsync(pinned_positions, device_positions, count * sizeof(float3), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(pinned_normals, device_normals, count * sizeof(float3), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(pinned_colors, device_colors, count * sizeof(uchar3), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(pinned_labels, device_labels, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(pinned_tags, device_tags, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(sync_event, stream);
            while (cudaEventQuery(sync_event) == cudaErrorNotReady)
            {
                std::this_thread::yield();
            }
        }
        needs_resize = false;
        return count;
    }

    void AllocateMemory(unsigned int new_capacity)
    {
        capacity = new_capacity;
        cudaMalloc(&device_size, sizeof(unsigned int));
        cudaMalloc(&device_positions, sizeof(float3) * capacity);
        cudaMalloc(&device_normals, sizeof(float3) * capacity);
        cudaMalloc(&device_colors, sizeof(uchar3) * capacity);
        cudaMalloc(&device_labels, sizeof(unsigned int) * capacity);
        cudaMalloc(&device_tags, sizeof(unsigned int) * capacity);

        cudaHostAlloc(&pinned_size, sizeof(unsigned int), cudaHostAllocDefault);
        cudaHostAlloc(&pinned_positions, sizeof(float3) * capacity, cudaHostAllocDefault);
        cudaHostAlloc(&pinned_normals, sizeof(float3) * capacity, cudaHostAllocDefault);
        cudaHostAlloc(&pinned_colors, sizeof(uchar3) * capacity, cudaHostAllocDefault);
        cudaHostAlloc(&pinned_labels, sizeof(unsigned int) * capacity, cudaHostAllocDefault);
        cudaHostAlloc(&pinned_tags, sizeof(unsigned int) * capacity, cudaHostAllocDefault);
    }

    void FreeMemory()
    {
        if (device_size)
        {
            cudaFree(device_size); cudaFree(device_positions); cudaFree(device_normals);
            cudaFree(device_colors); cudaFree(device_labels); cudaFree(device_tags);
            device_size = nullptr;
        }
        if (pinned_size)
        {
            cudaFreeHost(pinned_size); cudaFreeHost(pinned_positions); cudaFreeHost(pinned_normals);
            cudaFreeHost(pinned_colors); cudaFreeHost(pinned_labels); cudaFreeHost(pinned_tags);
            pinned_size = nullptr;
        }
    }

    unsigned int capacity;
    cudaStream_t stream;
    cudaEvent_t sync_event;
    unsigned int* device_size, * pinned_size;
    float3* device_positions, * pinned_positions, * device_normals, * pinned_normals;
    uchar3* device_colors, * pinned_colors;
    unsigned int* device_labels, * pinned_labels, * device_tags, * pinned_tags;
    std::unique_ptr<DataFrameRecorder<Blob>> recorder;
};
