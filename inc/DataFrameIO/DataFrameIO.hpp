#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Blob = std::vector<uint8_t>;

// ===========================================================================
// AsyncFileWriter
//
// Writes Blob data to a file on a background thread.
// push() is thread-safe and returns immediately.
// flush() blocks until the queue is fully drained to disk.
// ===========================================================================
class AsyncFileWriter
{
public:
    enum class OpenMode { Truncate, Append };

    AsyncFileWriter(
        const std::string& filepath,
        OpenMode                         mode = OpenMode::Truncate,
        std::function<void(std::string)> on_error = nullptr)
        : on_error(std::move(on_error))
    {
        auto ios_mode = std::ios::out | std::ios::binary;
        if (mode == OpenMode::Append)
            ios_mode |= std::ios::app;

        file.open(filepath, ios_mode);
        if (!file.is_open())
        {
            if (on_error) on_error("Failed to open file: " + filepath);
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
        if (thread.joinable()) thread.join();
        if (file.is_open())   file.flush();
    }

    AsyncFileWriter(const AsyncFileWriter&) = delete;
    AsyncFileWriter& operator=(const AsyncFileWriter&) = delete;
    AsyncFileWriter(AsyncFileWriter&&) = delete;
    AsyncFileWriter& operator=(AsyncFileWriter&&) = delete;

    bool push(Blob blob)
    {
        if (stop.load(std::memory_order_relaxed)) return false;
        { std::lock_guard<std::mutex> lock(mutex); io_queue.push(std::move(blob)); }
        cv.notify_one();
        return true;
    }

    void flush()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv_empty.wait(lock, [this] { return io_queue.empty(); });
    }

    bool   is_open() const { return file.is_open(); }
    size_t pending() const { std::lock_guard<std::mutex> l(mutex); return io_queue.size(); }

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
            if (stop.load(std::memory_order_relaxed)) break;
        }
    }

    void drain(std::unique_lock<std::mutex>& lock)
    {
        while (!io_queue.empty())
        {
            Blob item = std::move(io_queue.front());
            io_queue.pop();
            if (io_queue.empty()) cv_empty.notify_all();

            lock.unlock();
            if (file.is_open())
            {
                file.write(reinterpret_cast<const char*>(item.data()),
                    static_cast<std::streamsize>(item.size()));
                if (!file.good() && on_error) on_error("Write failed");
            }
            lock.lock();
        }
        if (file.is_open()) file.flush();
    }

    mutable std::mutex           mutex;
    std::condition_variable      cv;
    std::condition_variable      cv_empty;
    std::queue<Blob>             io_queue;
    std::ofstream                file;
    std::function<void(std::string)> on_error;
    std::atomic<bool>            stop{ false };
    std::thread                  thread;
};

// ===========================================================================
// On-disk layout per frame:
//
//   [FrameHeader]
//   [raw payload bytes]
// ===========================================================================
#pragma pack(push, 1)
struct FrameHeader
{
    uint32_t magic;        // 0x44464D00  'DFM\0'
    uint32_t frameIndex;
    uint64_t payloadBytes;
};
#pragma pack(pop)

static constexpr uint32_t DATAFRAME_MAGIC = 0x44464D00;

// ===========================================================================
// DataFrame  (host-side decoded frame)
// ===========================================================================
struct DataFrame
{
    uint32_t frameIndex;
    Blob     data;  // raw payload bytes; cast as needed
};

// ===========================================================================
// DataFrameRecorder<T>
//
// Pass one serializer lambda: (const T&) -> Blob
// The lambda owns all conversion logic (cudaMemcpy, casting, layout).
//
//   std::string ->
//       [](const std::string& s) -> Blob {
//           return Blob(s.begin(), s.end()); }
//
//   vector<int> ->
//       [](const std::vector<int>& v) -> Blob {
//           const auto* p = reinterpret_cast<const uint8_t*>(v.data());
//           return Blob(p, p + v.size() * sizeof(int)); }
//
//   CUDA params ->
//       [](const MyParams& p) -> Blob {
//           Blob buf(N * sizeof(float));
//           cudaMemcpy(buf.data(), p.d_depth, buf.size(), cudaMemcpyDeviceToHost);
//           return buf; }
// ===========================================================================
template<typename T>
class DataFrameRecorder
{
public:
    using Serializer = std::function<Blob(const T&)>;

    DataFrameRecorder(
        const std::string& filepath,
        Serializer serializer,
        std::function<void(std::string)> on_error = nullptr)
        : writer(filepath, AsyncFileWriter::OpenMode::Truncate, std::move(on_error))
        , serializer(std::move(serializer))
    {
    }

    ~DataFrameRecorder() = default;

    DataFrameRecorder(const DataFrameRecorder&) = delete;
    DataFrameRecorder& operator=(const DataFrameRecorder&) = delete;

    void record(const T& value)
    {
        record(serializer(value));
    }

    void record(const Blob& payload)
    {
        FrameHeader hdr{};
        hdr.magic = DATAFRAME_MAGIC;
        hdr.frameIndex = frameIndex;
        hdr.payloadBytes = static_cast<uint64_t>(payload.size());

        Blob blob(sizeof(FrameHeader) + payload.size());
        std::memcpy(blob.data(), &hdr, sizeof(hdr));
        std::memcpy(blob.data() + sizeof(hdr), payload.data(), payload.size());

        writer.push(std::move(blob));
        ++frameIndex;
    }

    void record_device(const T& value)
    {
        record(serializer(value));
	}

    void     flush() { writer.flush(); }
    bool     is_open()     const { return writer.is_open(); }
    uint32_t frame_count() const { return frameIndex; }

private:
    AsyncFileWriter writer;
    Serializer      serializer;
    uint32_t        frameIndex{ 0 };
};

// ===========================================================================
// DataFrameReader
//
// Not templated. Returns raw bytes per frame.
// Interpret frame.data with the same layout the serializer produced.
// ===========================================================================
class DataFrameReader
{
public:
    explicit DataFrameReader(const std::string& filepath)
    {
        file.open(filepath, std::ios::in | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error(
                "DataFrameReader: cannot open file: " + filepath);
        index_file();
    }

    ~DataFrameReader() = default;

    DataFrameReader(const DataFrameReader&) = delete;
    DataFrameReader& operator=(const DataFrameReader&) = delete;

    bool   is_open()     const { return file.is_open(); }
    bool   at_end()      const { return !file.good() || file.eof(); }
    size_t frame_count() const { return offsets.size(); }

    std::optional<DataFrame> next()
    {
        if (cursor >= offsets.size()) return std::nullopt;
        file.seekg(offsets[cursor]);
        if (!file.good()) return std::nullopt;
        auto frame = read_one();
        if (frame) ++cursor;
        return frame;
    }

    std::optional<DataFrame> read_at(size_t frameIndex)
    {
        if (frameIndex >= offsets.size())
        {
            throw std::out_of_range(
                "DataFrameReader::read_at - index " + std::to_string(frameIndex)
                + " out of range (total " + std::to_string(offsets.size()) + ")");
        }
        seek(frameIndex);
        return next();
    }

    void seek(size_t frameIndex)
    {
        if (frameIndex >= offsets.size())
            throw std::out_of_range(
                "DataFrameReader::seek - index " + std::to_string(frameIndex)
                + " out of range (total " + std::to_string(offsets.size()) + ")");
        cursor = frameIndex;
        file.clear();
        file.seekg(offsets[frameIndex]);
    }

    void rewind()
    {
        cursor = 0;
        file.clear();
        if (!offsets.empty()) file.seekg(offsets[0]);
    }

private:
    void index_file()
    {
        file.seekg(0, std::ios::beg);
        offsets.clear();

        FrameHeader hdr{};
        while (file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)))
        {
            if (hdr.magic != DATAFRAME_MAGIC) break;
            offsets.push_back(
                static_cast<std::streamoff>(file.tellg())
                - static_cast<std::streamoff>(sizeof(hdr)));
            file.seekg(static_cast<std::streamoff>(hdr.payloadBytes), std::ios::cur);
            if (!file.good()) break;
        }

        file.clear();
        file.seekg(offsets.empty() ? 0 : offsets[0]);
        cursor = 0;
    }

    std::optional<DataFrame> read_one()
    {
        FrameHeader hdr{};
        if (!file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) return std::nullopt;
        if (hdr.magic != DATAFRAME_MAGIC) return std::nullopt;

        DataFrame frame;
        frame.frameIndex = hdr.frameIndex;
        frame.data.resize(static_cast<size_t>(hdr.payloadBytes));

        if (hdr.payloadBytes > 0 &&
            !file.read(reinterpret_cast<char*>(frame.data.data()),
                static_cast<std::streamsize>(hdr.payloadBytes)))
            return std::nullopt;

        return frame;
    }

    std::ifstream               file;
    std::vector<std::streamoff> offsets;
    size_t                      cursor{ 0 };
};