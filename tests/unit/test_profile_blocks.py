"""CPU tests for profile_blocks and the canonical chain helpers."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
import torch

from ckptplan import (
    BlockProfile,
    NoDifferentiableOutputError,
    UnsupportedBoundaryError,
    declare_blocks,
    profile_blocks,
)
from tests.conftest import build_tiny_batch, build_tiny_sequential_model


def _declare_modules(*modules: torch.nn.Module):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList(modules)
    return declare_blocks(
        model,
        [(f"block_{index}", module) for index, module in enumerate(modules)],
    )


def _profile(blocks, inputs, kwargs=None, **options):
    return profile_blocks(
        blocks,
        inputs,
        kwargs,
        device="cpu",
        dtype=options.pop("dtype", torch.float32),
        num_warmup=options.pop("num_warmup", 0),
        num_trials=options.pop("num_trials", 2),
        **options,
    )


def test_returns_ordered_complete_frozen_profiles_with_cpu_timings() -> None:
    model = build_tiny_sequential_model()
    blocks = declare_blocks(
        model,
        [(f"layer{index}", module) for index, module in enumerate(model.blocks)],
    )

    profiles = _profile(blocks, (build_tiny_batch(),), num_warmup=1, num_trials=2)

    assert isinstance(profiles, tuple)
    assert [profile.block_id for profile in profiles] == ["layer0", "layer1", "layer2", "layer3"]
    assert [profile.order for profile in profiles] == [0, 1, 2, 3]
    assert all(isinstance(profile, BlockProfile) for profile in profiles)
    assert {field.name for field in dataclasses.fields(BlockProfile)} == set(profiles[0].__dict__)
    assert profiles[0].param_count == 64 * 64 + 64
    assert profiles[0].trainable_param_count == profiles[0].param_count
    assert all(profile.device == "cpu" for profile in profiles)
    assert all(profile.dtype == "torch.float32" for profile in profiles)
    assert all(profile.timing_only for profile in profiles)
    assert all(profile.activation_bytes_estimate is None for profile in profiles)
    assert all(profile.activation_bytes_method is None for profile in profiles)
    assert all(profile.forward_time_ms_mean >= 0 for profile in profiles)
    assert all(profile.recompute_time_upper_bound_ms_mean >= 0 for profile in profiles)
    assert all(math.isfinite(profile.forward_time_ms_std) for profile in profiles)
    assert all(math.isfinite(profile.recompute_time_upper_bound_ms_std) for profile in profiles)
    assert all(
        profile.recompute_time_source == "measured_full_recompute_early_stop_disabled"
        for profile in profiles
    )
    assert all(profile.num_warmup == 1 and profile.num_trials == 2 for profile in profiles)
    assert all(profile.eligible_for_checkpoint for profile in profiles)
    assert all(profile.exclusion_reason is None for profile in profiles)
    with pytest.raises(dataclasses.FrozenInstanceError):
        profiles[0].block_id = "changed"  # type: ignore[misc]


class _KwargProducer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, *, scale: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.linear(x) * scale
        return {"right": value + 1, "left": value}


class _KwargConsumer(torch.nn.Module):
    def forward(self, *, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


def test_kwargs_and_dict_boundary_are_deterministic() -> None:
    blocks = _declare_modules(_KwargProducer(), _KwargConsumer())
    x = torch.randn(2, 4)
    scale = torch.tensor(2.0)

    first = _profile(blocks, (x,), {"scale": scale}, num_trials=1)
    second = _profile(blocks, (x,), {"scale": scale}, num_trials=1)

    assert first[0].input_shape_signature == second[0].input_shape_signature
    assert first[1].input_shape_signature == second[1].input_shape_signature
    assert "left=" in first[1].input_shape_signature
    assert first[1].input_shape_signature.index("left=") < first[1].input_shape_signature.index("right=")


class _ListOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x.sin(), x.cos()]


def test_unrestricted_final_list_output_is_profiled() -> None:
    blocks = _declare_modules(torch.nn.Linear(4, 4), _ListOutput())
    profiles = _profile(blocks, (torch.randn(2, 4),), num_trials=1)
    assert len(profiles) == 2
    assert "List[" in profiles[-1].output_shape_signature
    assert profiles[-1].eligible_for_checkpoint is True


def test_non_final_list_boundary_is_rejected() -> None:
    blocks = _declare_modules(_ListOutput(), torch.nn.Identity())
    with pytest.raises(UnsupportedBoundaryError, match="List|list"):
        _profile(blocks, (torch.randn(2, 4),), num_trials=1)


class _NestedOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        value = self.linear(x)
        return {"outer": [value, {"deep": value.square()}], "constant": "ok"}


def test_nested_differentiable_output_collection() -> None:
    profiles = _profile(_declare_modules(_NestedOutput()), (torch.randn(2, 4),))
    assert profiles[0].eligible_for_checkpoint is True
    assert profiles[0].exclusion_reason is None


def test_stochastic_block_is_flagged_warned_and_eligible() -> None:
    block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.5))
    profile = _profile(_declare_modules(block), (torch.randn(2, 4),))[0]
    assert profile.is_stochastic is True
    assert profile.stochastic_submodules == ("1",)
    assert profile.eligible_for_checkpoint is True
    assert profile.exclusion_reason is None
    assert any("RNG replay" in warning for warning in profile.warnings)


@pytest.mark.parametrize("training, expected_stateful, expected_eligible", [(True, True, False), (False, False, True)])
def test_stateful_classification_depends_on_training_mode(
    training: bool,
    expected_stateful: bool,
    expected_eligible: bool,
) -> None:
    block = torch.nn.BatchNorm1d(4, track_running_stats=True)
    block.train(training)
    profile = _profile(_declare_modules(block), (torch.randn(3, 4),))[0]
    assert profile.is_stateful is expected_stateful
    assert profile.eligible_for_checkpoint is expected_eligible
    assert profile.exclusion_reason == (
        None if expected_eligible else "stateful_mutation_in_train_mode"
    )
    assert profile.stateful_submodules == ("<root>",)


def test_mixed_training_flags_and_buffers_are_restored_on_success() -> None:
    block = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout())
    block.train()
    block[0].train()
    block[1].eval()
    original_flags = tuple(module.training for module in block.modules())
    original_buffers = {
        name: value.detach().clone() for name, value in block.named_buffers()
    }

    _profile(_declare_modules(block), (torch.randn(3, 4),))

    assert tuple(module.training for module in block.modules()) == original_flags
    for name, value in block.named_buffers():
        assert torch.equal(value, original_buffers[name])


class _RaiseOnSecondCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.register_buffer("counter", torch.tensor(7.0))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.counter.add_(1)
        if self.calls == 2:
            raise RuntimeError("intentional profiling failure")
        return self.linear(x)


def test_buffers_and_modes_are_restored_when_profiling_raises() -> None:
    block = _RaiseOnSecondCall()
    block.eval()
    original_buffer = block.counter.detach().clone()
    with pytest.raises(RuntimeError, match="intentional profiling failure"):
        _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)
    assert block.training is False
    assert torch.equal(block.counter, original_buffer)


def test_preexisting_gradients_are_preserved_exactly() -> None:
    block = torch.nn.Linear(4, 4)
    original_weight_grad = torch.full_like(block.weight, 3.5)
    block.weight.grad = original_weight_grad
    assert block.bias.grad is None

    _profile(_declare_modules(block), (torch.randn(2, 4),))

    assert block.weight.grad is original_weight_grad
    assert torch.equal(block.weight.grad, torch.full_like(block.weight, 3.5))
    assert block.bias.grad is None


class _RebindingBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.register_buffer("state", torch.tensor([5.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state = self.state + 1
        return self.linear(x)


def test_rebound_buffer_registration_identity_and_value_are_restored() -> None:
    block = _RebindingBuffer()
    original = block.state
    original_value = original.detach().clone()

    _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.state is original
    assert torch.equal(block.state, original_value)


class _UnchangedQuantizedBuffer(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.register_buffer(
            "quantized_state",
            torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), 0.1, 0, torch.qint8),
        )


def test_unchanged_quantized_buffer_does_not_break_state_restoration() -> None:
    block = _UnchangedQuantizedBuffer()
    original = block.quantized_state

    _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.quantized_state is original
    assert torch.equal(block.quantized_state.dequantize(), torch.tensor([1.0, 2.0]))


class _UnchangedSparseBuffer(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.register_buffer(
            "sparse_state",
            torch.sparse_coo_tensor(
                torch.tensor([[0, 1]]),
                torch.tensor([3.0, 4.0]),
                size=(2,),
            ).coalesce(),
        )


def test_unchanged_sparse_buffer_does_not_break_state_restoration() -> None:
    block = _UnchangedSparseBuffer()
    original = block.sparse_state
    original_value = original.clone()

    _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.sparse_state is original
    assert torch.equal(block.sparse_state.to_dense(), original_value.to_dense())


class _SparseResizeThenRaise(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.register_buffer(
            "sparse_state",
            torch.sparse_coo_tensor(
                torch.tensor([[0]]), torch.tensor([2.0]), size=(2,)
            ).coalesce(),
        )
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 2:
            self.sparse_state.sparse_resize_((3,), 1, 0)
            self.training = False
            self.weight.grad = torch.full_like(self.weight, 99)
            raise LookupError("original sparse profiling failure")
        return super().forward(x)


def test_sparse_resize_restores_state_and_preserves_original_exception() -> None:
    block = _SparseResizeThenRaise()
    original_sparse = block.sparse_state
    original_value = original_sparse.clone()
    original_grad = torch.full_like(block.weight, 7)
    block.weight.grad = original_grad

    with pytest.raises(LookupError, match="original sparse profiling failure"):
        _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.sparse_state is original_sparse
    assert block.sparse_state.size() == original_value.size()
    assert torch.equal(block.sparse_state.to_dense(), original_value.to_dense())
    assert block.training is True
    assert block.weight.grad is original_grad


class _CompressedSparseBuffer(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.register_buffer(
            "compressed_state",
            torch.sparse_csr_tensor(
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 1]),
                torch.tensor([2.0, 3.0]),
                size=(2, 2),
            ),
        )
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


def test_compressed_sparse_buffer_is_rejected_before_execution() -> None:
    block = _CompressedSparseBuffer()

    with pytest.raises(ValueError, match="unsupported layout.*sparse_csr"):
        _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.calls == 0


class _OverlappingBufferViews(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        base = torch.tensor([1.0, 2.0, 3.0])
        self.register_buffer("left", base[:2])
        self.register_buffer("right", base[1:])


def test_overlapping_buffer_storage_alias_is_preserved() -> None:
    block = _OverlappingBufferViews()
    original_left = block.left
    original_right = block.right
    original_storage = block.left.untyped_storage().data_ptr()
    original_right_offset = block.right.storage_offset()

    _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.left is original_left
    assert block.right is original_right
    assert block.left.untyped_storage().data_ptr() == original_storage
    assert block.right.untyped_storage().data_ptr() == original_storage
    assert block.right.storage_offset() == original_right_offset


class _ResizeThenRaise(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.tensor([2.0]))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 2:
            self.state.resize_(3)
            raise LookupError("original profiling failure")
        return x.sin()


def test_buffer_resize_restores_metadata_without_masking_original_exception() -> None:
    block = _ResizeThenRaise()
    original = block.state
    original_value = original.detach().clone()
    original_shape = original.shape

    with pytest.raises(LookupError, match="original profiling failure"):
        _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.state is original
    assert block.state.shape == original_shape
    assert torch.equal(block.state, original_value)


class _ForwardMutatesGrad(torch.nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.grad = torch.full_like(self.weight, 99)
        return super().forward(x)


def test_capture_pass_restores_explicit_forward_gradient_mutation() -> None:
    block = _ForwardMutatesGrad(4, 4)
    original_grad = torch.full_like(block.weight, 3)
    block.weight.grad = original_grad

    _profile(_declare_modules(block), (torch.randn(2, 4),), num_trials=1)

    assert block.weight.grad is original_grad
    assert torch.equal(block.weight.grad, torch.full_like(block.weight, 3))


class _CallCountingLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(x)


def test_dtype_mismatch_is_rejected_before_execution_without_mutation() -> None:
    block = _CallCountingLinear()
    original_weight = block.weight.detach().clone()
    with pytest.raises(ValueError, match="block_0.*parameter|dtype"):
        _profile(
            _declare_modules(block),
            (torch.randn(2, 4, dtype=torch.float64),),
            dtype=torch.float64,
            num_trials=1,
        )
    assert block.calls == 0
    assert torch.equal(block.weight, original_weight)


def test_device_mismatch_is_rejected_before_execution_without_mutation() -> None:
    block = _CallCountingLinear()
    with pytest.raises(ValueError, match="example_inputs.*device"):
        _profile(
            _declare_modules(block),
            (torch.empty(2, 4, device="meta"),),
            num_trials=1,
        )
    assert block.calls == 0


def test_dtype_none_rejects_ambiguous_mixed_floating_dtypes_before_execution() -> None:
    first = _CallCountingLinear()
    second = torch.nn.Linear(4, 4, dtype=torch.float64)
    blocks = _declare_modules(first, second)
    with pytest.raises(ValueError, match="mixed floating dtypes"):
        _profile(blocks, (torch.randn(2, 4),), dtype=None, num_trials=1)
    assert first.calls == 0


class _DoubleProducer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.double()


def test_intermediate_dtype_mismatch_is_rejected_before_consumer_executes() -> None:
    consumer = _CallCountingLinear()
    blocks = _declare_modules(_DoubleProducer(), consumer)

    with pytest.raises(ValueError, match="block_0.*output|dtype"):
        _profile(blocks, (torch.randn(2, 4),), dtype=None, num_trials=1)

    assert consumer.calls == 0


def test_non_cpu_request_is_explicitly_rejected_before_execution() -> None:
    block = _CallCountingLinear()
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        profile_blocks(
            _declare_modules(block),
            (torch.randn(2, 4),),
            device="cuda",
            num_warmup=0,
            num_trials=1,
        )
    assert block.calls == 0


class _IntegerOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int64)


class _IntegerToFloat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float())


def test_partial_no_differentiable_output_records_ineligible_measured_profile() -> None:
    blocks = _declare_modules(_IntegerOutput(), _IntegerToFloat())
    profiles = _profile(blocks, (torch.randn(2, 4),), num_trials=1)
    invalid, valid = profiles
    assert invalid.eligible_for_checkpoint is False
    assert invalid.exclusion_reason == "no_differentiable_output"
    assert invalid.is_stochastic is None
    assert invalid.is_stateful is None
    assert invalid.forward_time_ms_mean >= 0
    assert invalid.recompute_time_upper_bound_ms_mean >= 0
    assert any("autograd probe" in warning for warning in invalid.warnings)
    assert valid.eligible_for_checkpoint is True


def test_all_blocks_without_differentiable_output_raise_specific_error() -> None:
    blocks = _declare_modules(_IntegerOutput(), _IntegerOutput())
    with pytest.raises(NoDifferentiableOutputError) as excinfo:
        _profile(blocks, (torch.ones(2, 4, dtype=torch.int64),), dtype=None, num_trials=1)
    assert excinfo.value.block_ids == ("block_0", "block_1")


class _AliasSensitive(torch.nn.Module):
    def forward(self, a: torch.Tensor, *, b: torch.Tensor) -> torch.Tensor:
        if a is b:
            return (a + b).sin()
        return (a + b).detach().to(torch.int64)


def test_repeated_tensor_identity_is_preserved_across_args_and_kwargs() -> None:
    tensor = torch.randn(2, 4)
    profile = _profile(
        _declare_modules(_AliasSensitive()),
        (tensor,),
        {"b": tensor},
        num_trials=1,
    )[0]
    assert profile.eligible_for_checkpoint is True
    assert profile.exclusion_reason is None
