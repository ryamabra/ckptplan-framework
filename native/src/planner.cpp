#include "ckptplan/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace ckptplan {
namespace {

using IndexList = std::vector<std::size_t>;

std::uint64_t checked_sum(std::span<const BlockProfile> profiles) {
    std::uint64_t total = 0;
    for (const auto& profile : profiles) {
        if (profile.activation_bytes > std::numeric_limits<std::uint64_t>::max() - total) {
            throw std::overflow_error("activation byte total exceeds uint64_t");
        }
        total += profile.activation_bytes;
    }
    return total;
}

void validate(std::span<const BlockProfile> profiles, const PlannerOptions& options) {
    if (profiles.empty()) {
        throw std::invalid_argument("plan requires at least one block profile");
    }
    if (options.kind == PlannerKind::dynamic_programming) {
        if (options.activation_bucket_bytes == 0) {
            throw std::invalid_argument("activation_bucket_bytes must be positive");
        }
        if (options.scale_guard_cells == 0) {
            throw std::invalid_argument("scale_guard_cells must be positive");
        }
    }

    std::vector<std::string_view> ids;
    ids.reserve(profiles.size());
    for (const auto& profile : profiles) {
        if (profile.id.empty()) {
            throw std::invalid_argument("block ids must be non-empty");
        }
        if (!std::isfinite(profile.recompute_milliseconds) || profile.recompute_milliseconds < 0.0) {
            throw std::invalid_argument("recompute time must be finite and non-negative");
        }
        ids.emplace_back(profile.id);
    }
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::invalid_argument("block ids must be unique");
    }
}

IndexList eligible_indices(std::span<const BlockProfile> profiles) {
    IndexList indices;
    indices.reserve(profiles.size());
    for (std::size_t index = 0; index < profiles.size(); ++index) {
        if (profiles[index].eligible) {
            indices.push_back(index);
        }
    }
    return indices;
}

IndexList density_order(std::span<const BlockProfile> profiles, const IndexList& eligible) {
    auto ordered = eligible;
    std::stable_sort(ordered.begin(), ordered.end(), [&](std::size_t left, std::size_t right) {
        const auto& a = profiles[left];
        const auto& b = profiles[right];
        const double a_density = a.recompute_milliseconds == 0.0
            ? std::numeric_limits<double>::infinity()
            : static_cast<double>(a.activation_bytes) / a.recompute_milliseconds;
        const double b_density = b.recompute_milliseconds == 0.0
            ? std::numeric_limits<double>::infinity()
            : static_cast<double>(b.activation_bytes) / b.recompute_milliseconds;
        return std::tuple{-a_density, -static_cast<long double>(a.activation_bytes),
                          a.recompute_milliseconds, left}
             < std::tuple{-b_density, -static_cast<long double>(b.activation_bytes),
                          b.recompute_milliseconds, right};
    });
    return ordered;
}

IndexList greedy(
    std::span<const BlockProfile> profiles,
    const IndexList& eligible,
    std::uint64_t saving_target
) {
    IndexList selected;
    std::uint64_t achieved = 0;
    for (const auto index : density_order(profiles, eligible)) {
        if (achieved >= saving_target) {
            break;
        }
        selected.push_back(index);
        achieved += profiles[index].activation_bytes;
    }
    return selected;
}

IndexList uniform(
    std::span<const BlockProfile> profiles,
    const IndexList& eligible,
    std::uint64_t saving_target
) {
    if (saving_target == 0 || eligible.empty()) {
        return {};
    }
    for (std::size_t count = 1; count <= eligible.size(); ++count) {
        IndexList selected;
        selected.reserve(count);
        std::uint64_t achieved = 0;
        for (std::size_t position = 0; position < count; ++position) {
            const auto eligible_position = position * eligible.size() / count;
            const auto index = eligible[eligible_position];
            selected.push_back(index);
            achieved += profiles[index].activation_bytes;
        }
        if (achieved >= saving_target) {
            return selected;
        }
    }
    return eligible;
}

struct DynamicResult {
    IndexList selected;
    bool repair_applied{false};
    bool bucket_fallback_applied{false};
};

DynamicResult dynamic_programming(
    std::span<const BlockProfile> profiles,
    const IndexList& eligible,
    std::uint64_t saving_target,
    const PlannerOptions& options
) {
    if (saving_target == 0) {
        return {};
    }
    const auto target_units = (saving_target + options.activation_bucket_bytes - 1)
        / options.activation_bucket_bytes;
    if (target_units > std::numeric_limits<std::size_t>::max() - 1) {
        throw std::length_error("dynamic-programming target is too large");
    }
    const auto columns = static_cast<std::size_t>(target_units + 1);
    if (!eligible.empty() && columns > options.scale_guard_cells / eligible.size()) {
        throw std::length_error("dynamic-programming table exceeds scale_guard_cells");
    }

    const auto infinity = std::numeric_limits<double>::infinity();
    std::vector<double> previous(columns, infinity);
    std::vector<double> current(columns, infinity);
    std::vector<std::vector<bool>> take(eligible.size(), std::vector<bool>(columns, false));
    previous[0] = 0.0;

    for (std::size_t row = 0; row < eligible.size(); ++row) {
        const auto& profile = profiles[eligible[row]];
        const auto units = static_cast<std::size_t>(profile.activation_bytes / options.activation_bucket_bytes);
        for (std::size_t achieved = 0; achieved < columns; ++achieved) {
            const auto prior_achieved = achieved > units ? achieved - units : 0;
            const auto with_cost = previous[prior_achieved] + profile.recompute_milliseconds;
            current[achieved] = previous[achieved];
            if (with_cost < current[achieved]) {
                current[achieved] = with_cost;
                take[row][achieved] = true;
            }
        }
        previous.swap(current);
        std::fill(current.begin(), current.end(), infinity);
    }

    if (!std::isfinite(previous.back())) {
        std::uint64_t exact_total = 0;
        for (const auto index : eligible) {
            exact_total += profiles[index].activation_bytes;
        }
        if (exact_total >= saving_target) {
            return {greedy(profiles, eligible, saving_target), false, true};
        }
        return {};
    }

    DynamicResult result;
    auto achieved_units = columns - 1;
    for (std::size_t row = eligible.size(); row > 0; --row) {
        if (!take[row - 1][achieved_units]) {
            continue;
        }
        const auto index = eligible[row - 1];
        result.selected.push_back(index);
        const auto units = static_cast<std::size_t>(profiles[index].activation_bytes / options.activation_bucket_bytes);
        achieved_units = achieved_units > units ? achieved_units - units : 0;
    }

    std::uint64_t real_achieved = 0;
    for (const auto index : result.selected) {
        real_achieved += profiles[index].activation_bytes;
    }
    if (real_achieved < saving_target) {
        result.repair_applied = true;
        const auto ordered = density_order(profiles, eligible);
        for (const auto index : ordered) {
            if (std::find(result.selected.begin(), result.selected.end(), index) != result.selected.end()) {
                continue;
            }
            result.selected.push_back(index);
            real_achieved += profiles[index].activation_bytes;
            if (real_achieved >= saving_target) {
                break;
            }
        }
    }
    return result;
}

}  // namespace

PlanResult plan(std::span<const BlockProfile> profiles, const PlannerOptions& options) {
    validate(profiles, options);
    const auto total = checked_sum(profiles);
    const auto saving_target = options.target_retained_bytes >= total
        ? std::uint64_t{0}
        : total - options.target_retained_bytes;
    const auto eligible = eligible_indices(profiles);

    IndexList selected;
    bool repair_applied = false;
    bool bucket_fallback_applied = false;
    switch (options.kind) {
        case PlannerKind::greedy:
            selected = greedy(profiles, eligible, saving_target);
            break;
        case PlannerKind::dynamic_programming: {
            auto result = dynamic_programming(profiles, eligible, saving_target, options);
            selected = std::move(result.selected);
            repair_applied = result.repair_applied;
            bucket_fallback_applied = result.bucket_fallback_applied;
            break;
        }
        case PlannerKind::uniform:
            selected = uniform(profiles, eligible, saving_target);
            break;
        case PlannerKind::checkpoint_all:
            selected = eligible;
            break;
        case PlannerKind::no_checkpoint:
            break;
    }

    std::uint64_t achieved = 0;
    for (const auto index : selected) {
        achieved += profiles[index].activation_bytes;
    }
    const bool feasible = achieved >= saving_target;
    if (!feasible && !options.best_effort) {
        throw std::runtime_error("requested activation target is infeasible");
    }
    if (!feasible && options.best_effort) {
        selected = eligible;
        achieved = 0;
        for (const auto index : selected) {
            achieved += profiles[index].activation_bytes;
        }
    }

    PlanResult result;
    result.checkpointed.assign(profiles.size(), false);
    for (const auto index : selected) {
        result.checkpointed[index] = true;
        result.recompute_milliseconds += profiles[index].recompute_milliseconds;
    }
    result.feasible = feasible;
    result.repair_applied = repair_applied;
    result.bucket_fallback_applied = bucket_fallback_applied;
    result.activation_bytes_before = total;
    result.activation_bytes_after = total - achieved;
    return result;
}

std::string_view planner_name(PlannerKind kind) noexcept {
    switch (kind) {
        case PlannerKind::greedy: return "greedy";
        case PlannerKind::dynamic_programming: return "dynamic_programming";
        case PlannerKind::uniform: return "uniform";
        case PlannerKind::checkpoint_all: return "checkpoint_all";
        case PlannerKind::no_checkpoint: return "no_checkpoint";
    }
    return "unknown";
}

}  // namespace ckptplan
