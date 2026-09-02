#include "ckptplan/cuda_memory.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

int main() {
    const std::vector<std::uint64_t> bytes{64, 128, 96, 192, 80};
    const std::vector<std::uint8_t> checkpointed{0, 1, 0, 1, 0};

    std::uint64_t* device_bytes = nullptr;
    std::uint8_t* device_checkpointed = nullptr;
    require(cudaMalloc(&device_bytes, bytes.size() * sizeof(bytes.front())), "allocate activation bytes");
    require(cudaMalloc(&device_checkpointed, checkpointed.size()), "allocate decisions");
    require(
        cudaMemcpy(device_bytes, bytes.data(), bytes.size() * sizeof(bytes.front()), cudaMemcpyHostToDevice),
        "copy activation bytes"
    );
    require(
        cudaMemcpy(device_checkpointed, checkpointed.data(), checkpointed.size(), cudaMemcpyHostToDevice),
        "copy decisions"
    );

    const auto summary = ckptplan::cuda::summarize_memory(
        device_bytes,
        device_checkpointed,
        bytes.size()
    );
    std::cout << "retained bytes: " << summary.retained_bytes << '\n'
              << "checkpointed bytes: " << summary.checkpointed_bytes << '\n'
              << "peak retained prefix: " << summary.peak_prefix_bytes << '\n';

    cudaFree(device_checkpointed);
    cudaFree(device_bytes);
}
