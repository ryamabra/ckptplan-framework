#include "ckptplan/cuda_memory.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ckptplan::cuda {
namespace {

constexpr unsigned int threads_per_block = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* get() noexcept { return data_; }

private:
    void* data_{nullptr};
};

__global__ void split_activation_bytes(
    const std::uint64_t* activation_bytes,
    const std::uint8_t* checkpointed,
    std::uint64_t* retained,
    std::uint64_t* discarded,
    std::size_t count
) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const auto bytes = activation_bytes[index];
    const bool is_checkpointed = checkpointed[index] != 0;
    retained[index] = is_checkpointed ? 0 : bytes;
    discarded[index] = is_checkpointed ? bytes : 0;
}

__global__ void inclusive_scan_step(
    const std::uint64_t* input,
    std::uint64_t* output,
    std::size_t count,
    std::size_t offset
) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    std::uint64_t value = input[index];
    if (index >= offset) {
        value += input[index - offset];
    }
    output[index] = value;
}

__global__ void reduce_sum_and_max(
    const std::uint64_t* retained_prefix,
    const std::uint64_t* discarded,
    std::uint64_t* output,
    std::size_t count
) {
    __shared__ std::uint64_t retained_values[threads_per_block];
    __shared__ std::uint64_t discarded_values[threads_per_block];
    __shared__ std::uint64_t peak_values[threads_per_block];

    const auto lane = threadIdx.x;
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + lane;
    const auto retained_prefix_value = index < count ? retained_prefix[index] : 0;
    retained_values[lane] = index + 1 == count ? retained_prefix_value : 0;
    discarded_values[lane] = index < count ? discarded[index] : 0;
    peak_values[lane] = retained_prefix_value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            retained_values[lane] += retained_values[lane + stride];
            discarded_values[lane] += discarded_values[lane + stride];
            peak_values[lane] = max(peak_values[lane], peak_values[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&output[0]), retained_values[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&output[1]), discarded_values[0]);
        atomicMax(reinterpret_cast<unsigned long long*>(&output[2]), peak_values[0]);
    }
}

}  // namespace

MemorySummary summarize_memory(
    const std::uint64_t* activation_bytes,
    const std::uint8_t* checkpointed,
    std::size_t count,
    cudaStream_t stream
) {
    if (count == 0) {
        return {};
    }
    if (activation_bytes == nullptr || checkpointed == nullptr) {
        throw std::invalid_argument("activation_bytes and checkpointed must be non-null");
    }

    const auto vector_bytes = count * sizeof(std::uint64_t);
    DeviceBuffer retained_a(vector_bytes);
    DeviceBuffer retained_b(vector_bytes);
    DeviceBuffer discarded(vector_bytes);
    DeviceBuffer totals(3 * sizeof(std::uint64_t));
    check_cuda(cudaMemsetAsync(totals.get(), 0, 3 * sizeof(std::uint64_t), stream), "clear totals");

    const auto blocks = static_cast<unsigned int>((count + threads_per_block - 1) / threads_per_block);
    split_activation_bytes<<<blocks, threads_per_block, 0, stream>>>(
        activation_bytes,
        checkpointed,
        static_cast<std::uint64_t*>(retained_a.get()),
        static_cast<std::uint64_t*>(discarded.get()),
        count
    );
    check_cuda(cudaGetLastError(), "launch split_activation_bytes");

    auto* input = static_cast<std::uint64_t*>(retained_a.get());
    auto* output = static_cast<std::uint64_t*>(retained_b.get());
    for (std::size_t offset = 1; offset < count; offset <<= 1) {
        inclusive_scan_step<<<blocks, threads_per_block, 0, stream>>>(input, output, count, offset);
        check_cuda(cudaGetLastError(), "launch inclusive_scan_step");
        std::swap(input, output);
        if (offset > count / 2) {
            break;  // Prevent offset overflow after the final useful pass.
        }
    }

    reduce_sum_and_max<<<blocks, threads_per_block, 0, stream>>>(
        input,
        static_cast<std::uint64_t*>(discarded.get()),
        static_cast<std::uint64_t*>(totals.get()),
        count
    );
    check_cuda(cudaGetLastError(), "launch reduce_sum_and_max");

    std::uint64_t host_totals[3]{};
    check_cuda(
        cudaMemcpyAsync(host_totals, totals.get(), sizeof(host_totals), cudaMemcpyDeviceToHost, stream),
        "copy memory summary"
    );
    check_cuda(cudaStreamSynchronize(stream), "synchronize memory summary");
    return MemorySummary{host_totals[0], host_totals[1], host_totals[2]};
}

}  // namespace ckptplan::cuda
