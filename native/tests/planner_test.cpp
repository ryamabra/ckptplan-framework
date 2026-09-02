#include "ckptplan/planner.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ckptplan::BlockProfile;
using ckptplan::PlannerKind;
using ckptplan::PlannerOptions;

void test_dynamic_programming_minimizes_cost() {
    const std::vector profiles{
        BlockProfile{"b0", 60, 8.0, true},
        BlockProfile{"b1", 40, 2.0, true},
        BlockProfile{"b2", 30, 1.0, true},
    };
    const auto result = ckptplan::plan(profiles, PlannerOptions{
        .kind = PlannerKind::dynamic_programming,
        .target_retained_bytes = 60,
        .activation_bucket_bytes = 10,
    });
    assert(result.feasible);
    assert(!result.checkpointed[0]);
    assert(result.checkpointed[1]);
    assert(result.checkpointed[2]);
    assert(result.activation_bytes_after == 60);
    assert(std::abs(result.recompute_milliseconds - 3.0) < 1e-12);
}
void test_ineligible_blocks_are_never_selected() {
    const std::vector profiles{
        BlockProfile{"eligible", 20, 2.0, true},
        BlockProfile{"stateful", 100, 0.1, false},
    };
    const auto result = ckptplan::plan(profiles, PlannerOptions{
        .kind = PlannerKind::checkpoint_all,
        .target_retained_bytes = 100,
    });
    assert(result.checkpointed[0]);
    assert(!result.checkpointed[1]);
}

void test_sub_bucket_values_use_deterministic_fallback() {
    const std::vector profiles{
        BlockProfile{"b0", 3, 5.0, true},
        BlockProfile{"b1", 4, 1.0, true},
        BlockProfile{"b2", 5, 2.0, true},
    };
    const auto result = ckptplan::plan(profiles, PlannerOptions{
        .kind = PlannerKind::dynamic_programming,
        .target_retained_bytes = 6,
        .activation_bucket_bytes = 10,
    });
    assert(result.feasible);
    assert(result.bucket_fallback_applied);
    assert(result.activation_bytes_after <= 6);
}

void test_validation() {
    bool rejected = false;
    try {
        const std::vector profiles{
            BlockProfile{"same", 10, 1.0, true},
            BlockProfile{"same", 20, 2.0, true},
        };
        (void)ckptplan::plan(profiles);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

}  // namespace

int main() {
    test_dynamic_programming_minimizes_cost();
    test_ineligible_blocks_are_never_selected();
    test_sub_bucket_values_use_deterministic_fallback();
    test_validation();
    std::cout << "ckptplan native planner tests passed\n";
}
