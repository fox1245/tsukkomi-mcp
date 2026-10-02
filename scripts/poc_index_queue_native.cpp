// Isolated ctypes bridge for the shared-index queue PoC. The queue, workers,
// admission bound and promise settlement belong to NeoGraph RequestQueue.
#include <neograph/util/request_queue.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace {

using Callback = void (*)(std::uint64_t job_id, void* user_data);

void write_error(char* error, std::size_t error_size,
                 const char* message) noexcept {
    if (error == nullptr || error_size == 0) return;
    std::size_t length = 0;
    if (message != nullptr) {
        while (length + 1 < error_size && message[length] != '\0') ++length;
        if (length != 0) std::memcpy(error, message, length);
    }
    error[length] = '\0';
}

struct QueueBridge {
    QueueBridge(std::size_t workers, std::size_t capacity)
        : queue(workers, capacity) {}

    // Register IDs before callbacks can inspect their futures. Callbacks may
    // reenter the bridge; never hold this mutex while close joins workers.
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::future<void>> futures;
    bool closing = false;
    std::uint64_t closed_rejections = 0;

    // Destroy the native queue before the future registry and its mutex.
    neograph::util::RequestQueue queue;
};

} // namespace

extern "C" {

void* poc_queue_create(std::size_t workers, std::size_t capacity,
                       char* error, std::size_t error_size) noexcept {
    write_error(error, error_size, nullptr);
    try {
        return new QueueBridge(workers, capacity);
    } catch (const std::exception& exception) {
        write_error(error, error_size, exception.what());
    } catch (...) {
        write_error(error, error_size, "Unknown queue creation exception");
    }
    return nullptr;
}

int poc_queue_submit(void* queue, std::uint64_t job_id, Callback run,
                     void* user_data, char* error,
                     std::size_t error_size) noexcept {
    write_error(error, error_size, nullptr);
    if (queue == nullptr || run == nullptr) {
        write_error(error, error_size, "Invalid queue handle or callback");
        return -1;
    }
    try {
        auto& bridge = *static_cast<QueueBridge*>(queue);
        std::lock_guard<std::mutex> lock(bridge.mutex);
        if (bridge.closing) {
            // Close marks admission closed before releasing this mutex, then
            // joins outside it. Do not enqueue in the interval between those
            // operations; account for these rejections in the public stats.
            ++bridge.closed_rejections;
            write_error(error, error_size, "RequestQueue is closed");
            return -2;
        }

        // Allocate the registry entry before native admission. No allocation
        // is needed after acceptance, even if the callback has already begun.
        auto [entry, inserted] = bridge.futures.try_emplace(job_id);
        if (!inserted) {
            write_error(error, error_size, "Duplicate outstanding job ID");
            return -1;
        }

        std::pair<bool, std::future<void>> submitted;
        try {
            submitted = bridge.queue.submit([job_id, run, user_data] {
                run(job_id, user_data);
            });
        } catch (...) {
            bridge.futures.erase(entry);
            throw;
        }
        if (submitted.first) {
            entry->second = std::move(submitted.second);
            return 1;
        }

        bridge.futures.erase(entry);
        if (submitted.second.valid()) {
            // An enqueue failure is not capacity rejection. RequestQueue
            // supplies a ready, failed future even though admission failed.
            submitted.second.get();
            write_error(error, error_size, "RequestQueue enqueue failed");
            return -1;
        }
        if (bridge.queue.is_closed()) {
            write_error(error, error_size, "RequestQueue is closed");
            return -2;
        }
        write_error(error, error_size, "RequestQueue is full");
        return 0;
    } catch (const std::exception& exception) {
        write_error(error, error_size, exception.what());
    } catch (...) {
        write_error(error, error_size, "Unknown queue submission exception");
    }
    return -1;
}

int poc_queue_take(void* queue, std::uint64_t job_id,
                   char* error, std::size_t error_size) noexcept {
    write_error(error, error_size, nullptr);
    if (queue == nullptr) {
        write_error(error, error_size, "Invalid queue handle");
        return -1;
    }
    std::future<void> completed;
    try {
        auto& bridge = *static_cast<QueueBridge*>(queue);
        std::lock_guard<std::mutex> lock(bridge.mutex);
        auto entry = bridge.futures.find(job_id);
        if (entry == bridge.futures.end()) {
            write_error(error, error_size, "Unknown job ID");
            return -1;
        }
        if (entry->second.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready) {
            return 0;
        }
        // Only one collector may consume a ready future. Moving and erasing
        // under the mutex also allows reuse of the ID after collection.
        completed = std::move(entry->second);
        bridge.futures.erase(entry);
    } catch (const std::exception& exception) {
        write_error(error, error_size, exception.what());
        return -1;
    } catch (...) {
        write_error(error, error_size, "Unknown completion lookup exception");
        return -1;
    }

    try {
        completed.get();
        return 1;
    } catch (const std::exception& exception) {
        // In particular, queued close cancellation retains the native
        // "RequestQueue is closed" exception rather than invoking Python.
        write_error(error, error_size, exception.what());
    } catch (...) {
        write_error(error, error_size, "Unknown job exception");
    }
    return 2;
}

int poc_queue_stats(void* queue, std::uint64_t out[6]) noexcept {
    if (queue == nullptr || out == nullptr) return -1;
    try {
        auto& bridge = *static_cast<QueueBridge*>(queue);
        std::lock_guard<std::mutex> lock(bridge.mutex);
        const auto stats = bridge.queue.stats();
        out[0] = static_cast<std::uint64_t>(stats.pending);
        out[1] = static_cast<std::uint64_t>(stats.active);
        out[2] = static_cast<std::uint64_t>(stats.completed);
        out[3] = static_cast<std::uint64_t>(stats.rejected) +
                 bridge.closed_rejections;
        out[4] = static_cast<std::uint64_t>(stats.num_workers);
        out[5] = static_cast<std::uint64_t>(stats.max_queue_size);
        return 0;
    } catch (...) {
        return -1;
    }
}

int poc_queue_close(void* queue, char* error,
                    std::size_t error_size) noexcept {
    write_error(error, error_size, nullptr);
    if (queue == nullptr) {
        write_error(error, error_size, "Invalid queue handle");
        return -1;
    }
    try {
        auto& bridge = *static_cast<QueueBridge*>(queue);
        {
            std::lock_guard<std::mutex> lock(bridge.mutex);
            bridge.closing = true;
        }
        // Native close is idempotent, joins claimed work and settles every
        // unclaimed future with an exception. Always call it, including on a
        // repeated close, so an external caller waits for worker-led close.
        bridge.queue.close();
        return 0;
    } catch (const std::exception& exception) {
        write_error(error, error_size, exception.what());
    } catch (...) {
        write_error(error, error_size, "Unknown queue close exception");
    }
    return -1;
}

void poc_queue_destroy(void* queue) noexcept {
    if (queue == nullptr) return;
    // As with a normal owned C handle, destroy requires no concurrent API
    // calls and must not be called by its own callback. Keep ctypes callback
    // roots and user_data alive until this external close/destroy finishes.
    try {
        auto* bridge = static_cast<QueueBridge*>(queue);
        {
            std::lock_guard<std::mutex> lock(bridge->mutex);
            bridge->closing = true;
        }
        bridge->queue.close();
        delete bridge;
    } catch (...) {
        // A failed close cannot safely release storage still used by workers.
        // Callers needing diagnostics must call poc_queue_close first.
    }
}

const char* poc_queue_backend() noexcept {
    return "neograph::util::RequestQueue/moodycamel::ConcurrentQueue";
}

} // extern "C"
