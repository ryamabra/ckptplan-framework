#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ckptplan {

enum class PlannerKind {
    greedy,
    dynamic_programming,
    uniform,
    checkpoint_all,
    no_checkpoint,
};

struct BlockProfile {
    std::string id;
    std::uint64_t activation_bytes{};
    double recompute_milliseconds{};
    bool eligible{true};
};

struct PlannerOptions {
    PlannerKind kind{PlannerKind::dynamic_programming};
    std::uint64_t target_retained_bytes{};
    std::uint64_t activation_bucket_bytes{1U << 20U};
    std::size_t scale_guard_cells{5'000'000};
    bool best_effort{false};
};

struct PlanResult {
    std::vector<bool> checkpointed;
    bool feasible{false};
    bool repair_applied{false};
    bool bucket_fallback_applied{false};
    std::uint64_t activation_bytes_before{};
    std::uint64_t activation_bytes_after{};
    double recompute_milliseconds{};
};

// Pure CPU cost-model implementation. It deliberately does not execute a model
// or call CUDA, which keeps planning deterministic and cheap to test.
[[nodiscard]] PlanResult plan(
    std::span<const BlockProfile> profiles,
    const PlannerOptions& options = {}
);

[[nodiscard]] std::string_view planner_name(PlannerKind kind) noexcept;

}  // namespace ckptplan
