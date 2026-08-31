"""CPU and CUDA block profiling for the v0.1 implementation slice."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from ckptplan._execution import (
    _io_signature,
    _preserve_module_state,
    _run_block_chain,
)
from ckptplan.errors import NoDifferentiableOutputError, UnsupportedBoundaryError
from ckptplan.types import BlockProfile, CheckpointableBlock

PROFILER_VERSION = "0.1.0.dev0"
_RECOMPUTE_SOURCE = "measured_full_recompute_early_stop_disabled"

_STOCHASTIC_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)
_STATEFUL_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
)


def _named_tensor_leaves(value: Any, path: str) -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield path, value
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _named_tensor_leaves(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise UnsupportedBoundaryError(
                    f"non-string dict key {key!r} of type {type(key).__name__}"
                )
        for key in sorted(value):
            yield from _named_tensor_leaves(value[key], f"{path}[{key!r}]")


def _collect_differentiable_leaves(value: Any) -> list[torch.Tensor]:
    """Collect arbitrarily nested floating tensor leaves that participate in autograd."""
    return [
        tensor
        for _path, tensor in _named_tensor_leaves(value, "output")
        if tensor.requires_grad and tensor.is_floating_point()
    ]


def _fresh_leaf_copy(value: Any, memo: dict[int, torch.Tensor] | None = None) -> Any:
    """Recursively clone tensors, making floating tensors fresh autograd leaves."""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        if id(value) in memo:
            return memo[id(value)]
        copied = value.detach().clone(memory_format=torch.preserve_format)
        if copied.is_floating_point():
            copied.requires_grad_(True)
        memo[id(value)] = copied
        return copied
    if isinstance(value, tuple):
        return tuple(_fresh_leaf_copy(item, memo) for item in value)
    if isinstance(value, list):
        return [_fresh_leaf_copy(item, memo) for item in value]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise UnsupportedBoundaryError(
                    f"non-string dict key {key!r} of type {type(key).__name__}"
                )
        return {key: _fresh_leaf_copy(value[key], memo) for key in value}
    return value


def _normalize_cpu_device(device: torch.device | str) -> torch.device:
    try:
        normalized = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid profiling device {device!r}: {exc}") from exc
    if normalized.type not in {"cpu", "cuda"}:
        raise NotImplementedError(f"profile_blocks supports CPU and CUDA, got {normalized}")
    if normalized.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is not available")
    return normalized


def _validate_run_counts(num_warmup: int, num_trials: int) -> None:
    if isinstance(num_warmup, bool) or not isinstance(num_warmup, int) or num_warmup < 0:
        raise ValueError(f"num_warmup must be a non-negative int, got {num_warmup!r}")
    if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError(f"num_trials must be a positive int, got {num_trials!r}")


def _module_tensors(
    blocks: Sequence[CheckpointableBlock],
) -> Iterator[tuple[str, torch.Tensor]]:
    for block in blocks:
        for name, parameter in block.module.named_parameters(recurse=True):
            yield f"block {block.block_id!r} parameter {name!r}", parameter
        for name, buffer in block.module.named_buffers(recurse=True):
            yield f"block {block.block_id!r} buffer {name!r}", buffer


def _runtime_tensors(
    chain_records: Sequence[tuple[CheckpointableBlock, tuple[Any, ...], dict[str, Any], Any]],
) -> Iterator[tuple[str, torch.Tensor]]:
    for block, args, kwargs, output in chain_records:
        yield from _named_tensor_leaves(args, f"block {block.block_id!r} input args")
        yield from _named_tensor_leaves(kwargs, f"block {block.block_id!r} input kwargs")
        yield from _named_tensor_leaves(output, f"block {block.block_id!r} output")


def _validate_device_and_resolve_dtype(
    blocks: Sequence[CheckpointableBlock],
    chain_records: Sequence[tuple[CheckpointableBlock, tuple[Any, ...], dict[str, Any], Any]],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.dtype:
    if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
        raise ValueError(f"dtype must be a floating torch.dtype or None, got {dtype!r}")

    floating_dtypes: dict[torch.dtype, str] = {}
    for description, tensor in (*tuple(_module_tensors(blocks)), *tuple(_runtime_tensors(chain_records))):
        if tensor.device.type != device.type:
            raise ValueError(
                f"{description} is on device {tensor.device}, but profile_blocks "
                f"was requested for {device}; caller-owned values are never moved"
            )
        if tensor.is_floating_point():
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"{description} has dtype {tensor.dtype}, but profile_blocks "
                    f"was requested with dtype={dtype}; caller-owned values are never cast"
                )
            floating_dtypes.setdefault(tensor.dtype, description)

    if dtype is not None:
        return dtype
    if len(floating_dtypes) > 1:
        details = ", ".join(
            f"{found_dtype} at {description}"
            for found_dtype, description in floating_dtypes.items()
        )
        raise ValueError(
            "dtype=None cannot infer one truthful execution dtype from mixed "
            f"floating dtypes: {details}"
        )
    if not floating_dtypes:
        # A chain with no floating tensors cannot produce a differentiable
        # floating output. Let the public all-invalid policy raise its specific
        # NoDifferentiableOutputError after classification instead of masking it
        # with a generic dtype-inference error.
        return torch.float32
    return next(iter(floating_dtypes))


def _validate_before_execution(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> None:
    """Reject placement/dtype mismatches before any caller module can execute."""
    if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
        raise ValueError(f"dtype must be a floating torch.dtype or None, got {dtype!r}")
    supported_buffer_layouts = {torch.strided, torch.sparse_coo}
    for block in blocks:
        for name, buffer in block.module.named_buffers():
            if buffer.layout not in supported_buffer_layouts:
                raise ValueError(
                    f"block {block.block_id!r} buffer {name!r} has unsupported layout "
                    f"{buffer.layout}; v0.1 profiling supports strided and sparse COO "
                    "registered buffers only"
                )
    values = [
        *tuple(_module_tensors(blocks)),
        *tuple(_named_tensor_leaves(example_inputs, "example_inputs")),
        *tuple(_named_tensor_leaves(dict(example_kwargs or {}), "example_kwargs")),
    ]
    floating_dtypes: dict[torch.dtype, str] = {}
    for description, tensor in values:
        if tensor.device.type != device.type:
            raise ValueError(
                f"{description} is on device {tensor.device}, but profile_blocks "
                f"was requested for {device}; caller-owned values are never moved"
            )
        if tensor.is_floating_point():
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"{description} has dtype {tensor.dtype}, but profile_blocks "
                    f"was requested with dtype={dtype}; caller-owned values are never cast"
                )
            floating_dtypes.setdefault(tensor.dtype, description)
    if dtype is None and len(floating_dtypes) > 1:
        details = ", ".join(
            f"{found_dtype} at {description}"
            for found_dtype, description in floating_dtypes.items()
        )
        raise ValueError(
            "dtype=None cannot infer one truthful execution dtype from mixed "
            f"floating dtypes: {details}"
        )


def _clear_parameter_gradients(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


class _TimedCall:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.call_times_ms: list[float] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        output = self.module(*args, **kwargs)
        self.call_times_ms.append((time.perf_counter() - start) * 1000.0)
        return output


def _measure_cpu_trial(
    block: CheckpointableBlock,
    captured_args: tuple[Any, ...],
    captured_kwargs: Mapping[str, Any],
) -> tuple[float, float, bool]:
    alias_memo: dict[int, torch.Tensor] = {}
    fresh_args = _fresh_leaf_copy(captured_args, alias_memo)
    fresh_kwargs = _fresh_leaf_copy(dict(captured_kwargs), alias_memo)
    timed = _TimedCall(block.module)
    probe = torch.ones((), dtype=torch.float32, device="cpu", requires_grad=True)

    def call_with_probe(probe_value: torch.Tensor) -> tuple[Any, torch.Tensor]:
        output = timed(*fresh_args, **fresh_kwargs)
        # The zero-valued probe is not part of the block result. Its saved
        # square forces non-reentrant checkpoint to execute the complete timed
        # function again even when the real output has no differentiable leaves.
        return output, probe_value.square().sum() * 0

    with torch.enable_grad(), set_checkpoint_early_stop(False):
        output, recompute_probe = checkpoint(
            call_with_probe,
            probe,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        differentiable_leaves = _collect_differentiable_leaves(output)
        backward_outputs = [*differentiable_leaves, recompute_probe]
        torch.autograd.backward(
            backward_outputs,
            [torch.ones_like(tensor) for tensor in backward_outputs],
        )

    if len(timed.call_times_ms) != 2:
        raise RuntimeError(
            f"block {block.block_id!r}: expected one forward and one full "
            f"checkpoint recomputation, observed {len(timed.call_times_ms)} calls"
        )
    return timed.call_times_ms[0], timed.call_times_ms[1], bool(differentiable_leaves)


def _measure_cuda_trial(
    block: CheckpointableBlock,
    captured_args: tuple[Any, ...],
    captured_kwargs: Mapping[str, Any],
    device: torch.device,
) -> tuple[float, float, int, bool]:
    alias_memo: dict[int, torch.Tensor] = {}
    fresh_args = _fresh_leaf_copy(captured_args, alias_memo)
    fresh_kwargs = _fresh_leaf_copy(dict(captured_kwargs), alias_memo)
    input_bytes = sum(
        tensor.numel() * tensor.element_size()
        for _description, tensor in _named_tensor_leaves(fresh_args, "args")
        if tensor.is_floating_point()
    ) + sum(
        tensor.numel() * tensor.element_size()
        for _description, tensor in _named_tensor_leaves(fresh_kwargs, "kwargs")
        if tensor.is_floating_point()
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.enable_grad():
        output = block.module(*fresh_args, **fresh_kwargs)
    end.record()
    end.synchronize()
    mem_after = torch.cuda.memory_allocated(device)
    activation = max(0, mem_after - mem_before - input_bytes)
    leaves = _collect_differentiable_leaves(output)
    del output

    timed_calls: list[float] = []
    probe = torch.ones((), dtype=torch.float32, device=device, requires_grad=True)
    fresh_args = _fresh_leaf_copy(captured_args)
    fresh_kwargs = _fresh_leaf_copy(dict(captured_kwargs))

    def timed_call(probe_value: torch.Tensor) -> tuple[Any, torch.Tensor]:
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        result = block.module(*fresh_args, **fresh_kwargs)
        event_end.record()
        event_end.synchronize()
        timed_calls.append(event_start.elapsed_time(event_end))
        return result, probe_value.square().sum() * 0

    with set_checkpoint_early_stop(False), torch.enable_grad():
        output, recompute_probe = checkpoint(
            timed_call, probe, use_reentrant=False, preserve_rng_state=True
        )
        leaves = _collect_differentiable_leaves(output)
        torch.autograd.backward([*leaves, recompute_probe], [torch.ones_like(t) for t in (*leaves, recompute_probe)])
    if len(timed_calls) != 2:
        raise RuntimeError(f"block {block.block_id!r}: expected one forward and one recomputation")
    return timed_calls[0], timed_calls[1], int(round(activation)), bool(leaves)


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _classify_module(
    module: torch.nn.Module,
) -> tuple[bool | None, bool, tuple[str, ...], tuple[str, ...]]:
    stochastic_names: list[str] = []
    stateful_names: list[str] = []
    active_stateful = False
    for name, submodule in module.named_modules():
        display_name = name or "<root>"
        if isinstance(submodule, _STOCHASTIC_TYPES):
            stochastic_names.append(display_name)
        if isinstance(submodule, _STATEFUL_TYPES) and bool(
            getattr(submodule, "track_running_stats", False)
        ):
            stateful_names.append(display_name)
            active_stateful = active_stateful or submodule.training
    is_stochastic = True if stochastic_names else None
    return is_stochastic, active_stateful, tuple(stochastic_names), tuple(stateful_names)


def profile_blocks(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> tuple[BlockProfile, ...]:
    """Profile a declared block chain on CPU without moving caller-owned state."""
    normalized_device = _normalize_cpu_device(device)
    _validate_run_counts(num_warmup, num_trials)
    if not isinstance(example_inputs, tuple):
        raise ValueError(f"example_inputs must be a tuple, got {type(example_inputs).__name__}")
    if not blocks:
        raise ValueError("profile_blocks requires at least one declared block")

    _validate_before_execution(
        blocks,
        example_inputs,
        example_kwargs,
        device=normalized_device,
        dtype=dtype,
    )
    chain_records = []
    # The chain generator yields before converting/executing the next boundary.
    # Validate each newly observed output immediately so an invalid producer
    # cannot reach or execute its downstream consumer.
    for record in _run_block_chain(blocks, example_inputs, example_kwargs):
        chain_records.append(record)
        _validate_device_and_resolve_dtype(
            blocks,
            chain_records,
            device=normalized_device,
            dtype=dtype,
        )
    execution_dtype = _validate_device_and_resolve_dtype(
        blocks,
        chain_records,
        device=normalized_device,
        dtype=dtype,
    )

    profiles: list[BlockProfile] = []
    invalid_block_ids: list[str] = []
    for block, captured_args, captured_kwargs, captured_output in chain_records:
        forward_times: list[float] = []
        recompute_times: list[float] = []
        activation_estimates: list[int] = []
        differentiability_results: list[bool] = []

        with _preserve_module_state(block.module, isolate_gradients=True):
            for _ in range(num_warmup):
                if normalized_device.type == "cuda":
                    _forward, _recompute, _activation, has_differentiable_output = _measure_cuda_trial(block, captured_args, captured_kwargs, normalized_device)
                else:
                    _forward, _recompute, has_differentiable_output = _measure_cpu_trial(block, captured_args, captured_kwargs)
                differentiability_results.append(has_differentiable_output)
                _clear_parameter_gradients(block.module)
            for _ in range(num_trials):
                if normalized_device.type == "cuda":
                    forward_ms, recompute_ms, activation_bytes, has_differentiable_output = _measure_cuda_trial(block, captured_args, captured_kwargs, normalized_device)
                else:
                    forward_ms, recompute_ms, has_differentiable_output = _measure_cpu_trial(block, captured_args, captured_kwargs)
                    activation_bytes = None
                forward_times.append(forward_ms)
                recompute_times.append(recompute_ms)
                if normalized_device.type == "cuda":
                    activation_estimates.append(activation_bytes)
                differentiability_results.append(has_differentiable_output)
                _clear_parameter_gradients(block.module)

        if len(set(differentiability_results)) != 1:
            raise ValueError(
                f"block {block.block_id!r} changed whether its output was "
                "differentiable across profiling calls; data-dependent output "
                "structure is unsupported in v0.1"
            )
        has_differentiable_output = differentiability_results[0]
        warnings: list[str] = []

        if has_differentiable_output:
            is_stochastic, is_stateful, stochastic_names, stateful_names = _classify_module(
                block.module
            )
            eligible = not is_stateful
            exclusion_reason = None if eligible else "stateful_mutation_in_train_mode"
            if is_stochastic:
                warnings.append(
                    "checkpoint RNG replay only covers torch's CPU/CUDA RNG; other "
                    "randomness sources such as Python random and NumPy are not protected"
                )
            if is_stateful:
                warnings.append(
                    "block is ineligible because a running-statistics module is in training mode"
                )
        else:
            invalid_block_ids.append(block.block_id)
            is_stochastic = None
            is_stateful = None
            stochastic_names = ()
            stateful_names = ()
            eligible = False
            exclusion_reason = "no_differentiable_output"
            warnings.append(
                "block output has no differentiable floating tensor leaves; timing "
                "was measured with a zero-valued autograd probe and the block is ineligible"
            )

        forward_mean, forward_std = _mean_and_std(forward_times)
        recompute_mean, recompute_std = _mean_and_std(recompute_times)
        if normalized_device.type == "cuda":
            torch.cuda.empty_cache()
        profiles.append(
            BlockProfile(
                block_id=block.block_id,
                order=block.order,
                device=str(normalized_device),
                dtype=str(execution_dtype),
                input_shape_signature=_io_signature(captured_args, captured_kwargs),
                output_shape_signature=_io_signature((captured_output,), {}),
                param_count=sum(parameter.numel() for parameter in block.module.parameters()),
                trainable_param_count=sum(
                    parameter.numel()
                    for parameter in block.module.parameters()
                    if parameter.requires_grad
                ),
                timing_only=normalized_device.type != "cuda",
                activation_bytes_estimate=(
                    int(round(statistics.fmean(activation_estimates)))
                    if normalized_device.type == "cuda" else None
                ),
                activation_bytes_method="isolated_forward_delta" if normalized_device.type == "cuda" else None,
                forward_time_ms_mean=forward_mean,
                forward_time_ms_std=forward_std,
                recompute_time_upper_bound_ms_mean=recompute_mean,
                recompute_time_upper_bound_ms_std=recompute_std,
                recompute_time_source=_RECOMPUTE_SOURCE,
                num_warmup=num_warmup,
                num_trials=num_trials,
                is_stochastic=is_stochastic,
                is_stateful=is_stateful,
                stochastic_submodules=stochastic_names,
                stateful_submodules=stateful_names,
                eligible_for_checkpoint=eligible,
                exclusion_reason=exclusion_reason,
                warnings=tuple(warnings),
                pytorch_version=torch.__version__,
                profiler_version=PROFILER_VERSION,
            )
        )

    if len(invalid_block_ids) == len(blocks):
        raise NoDifferentiableOutputError(block_ids=tuple(invalid_block_ids))
    return tuple(profiles)
