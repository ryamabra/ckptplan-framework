#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ckptplan::cuda {

struct MemorySummary {
    std::uint64_t retained_bytes{};
    std::uint64_t checkpointed_bytes{};
    std::uint64_t peak_prefix_bytes{};
};

// Computes retained/checkpointed totals and the largest retained prefix on the
// GPU. `checkpointed[i] != 0` means block i will be recomputed in backward.
// The function synchronizes `stream` before returning the host-side summary.
MemorySummary summarize_memory(
    const std::uint64_t* activation_bytes,
    const std::uint8_t* checkpointed,
    std::size_t count,
    cudaStream_t stream = nullptr
);

}  // namespace ckptplan::cuda
