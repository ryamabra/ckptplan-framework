"""Canonical private execution helpers shared across ckptplan stages.

Profiling owns measurement policy, but chain walking, boundary conversion, and
shape signatures are execution semantics. Future plan application must import
these helpers instead of defining a second interpretation of the block chain.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import sys
from typing import Any

import torch

from ckptplan.errors import UnsupportedBoundaryError
from ckptplan.types import CheckpointableBlock, ExecutionSignature


def _signature(value: Any) -> str:
    """Return the deterministic shape/structure signature from MVP_SPEC.md §6.4."""
    if torch.is_tensor(value):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device.type})"
        )
    if isinstance(value, (list, tuple)):
        kind = "List" if isinstance(value, list) else "Tuple"
        return f"{kind}[{', '.join(_signature(item) for item in value)}]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise UnsupportedBoundaryError(f"non-string dict key {key!r}")
        items = ", ".join(f"{key}={_signature(value[key])}" for key in sorted(value))
        return f"Dict{{{items}}}"
    return f"Const({type(value).__name__})"


def _io_signature(args: tuple[Any, ...], kwargs: Mapping[str, Any] | None) -> str:
    """Sign positional and keyword arguments together, with stable kwargs order."""
    normalized_kwargs = dict(sorted(dict(kwargs or {}).items()))
    return f"Args{_signature(tuple(args))}, Kwargs{_signature(normalized_kwargs)}"


def _boundary_convert(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Convert one block's output into the next block's complete call boundary."""
    if torch.is_tensor(value):
        return (value,), {}
    if isinstance(value, tuple):
        return value, {}
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise UnsupportedBoundaryError(
                    f"non-string dict key {key!r} at a block boundary"
                )
        return (), dict(value)
    raise UnsupportedBoundaryError(
        f"block output of type {type(value).__name__} is not a Tensor, tuple, or "
        "dict[str, Any] and cannot be used as the next block's input "
        "(MVP_SPEC.md §9.2)"
    )


@contextmanager
def _preserve_module_state(
    module: torch.nn.Module,
    *,
    isolate_gradients: bool,
) -> Iterator[None]:
    """Restore modes, buffers, and pre-existing gradients exactly on exit.

    Training flags are restored directly per submodule rather than via
    ``module.train(...)`` because the latter recursively flattens mixed-mode
    trees. When gradient isolation is requested, original ``Tensor`` objects
    are detached from parameters during profiling and reattached afterward,
    preserving both their identity and their values.
    """
    training_snapshot = tuple((submodule, submodule.training) for submodule in module.modules())
    # Exact restoration includes registration identity, aliases, None entries,
    # persistence metadata, rebinding, and buffers added/removed during forward.
    # PyTorch exposes no public API for snapshotting that complete registration
    # state, so this guard deliberately snapshots Module's canonical registries.
    buffer_registries = tuple(
        (
            submodule,
            dict(submodule._buffers),
            set(submodule._non_persistent_buffers_set),
        )
        for submodule in module.modules()
    )
    buffer_values: dict[
        int,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.UntypedStorage | None,
            int,
            torch.Size,
            tuple[int, ...],
        ],
    ] = {}
    for _submodule, registrations, _non_persistent in buffer_registries:
        for buffer in registrations.values():
            if buffer is not None and id(buffer) not in buffer_values:
                saved_value = (
                    buffer.detach().clone(memory_format=torch.preserve_format)
                    if buffer.layout == torch.strided
                    else buffer.detach().clone()
                )
                buffer_values[id(buffer)] = (
                    buffer,
                    saved_value,
                    buffer.untyped_storage() if buffer.layout == torch.strided else None,
                    buffer.storage_offset() if buffer.layout == torch.strided else 0,
                    buffer.size(),
                    buffer.stride() if buffer.layout == torch.strided else (),
                )
    gradient_snapshot = tuple((parameter, parameter.grad) for parameter in module.parameters())

    if isolate_gradients:
        for parameter, _gradient in gradient_snapshot:
            parameter.grad = None

    try:
        yield
    finally:
        active_exception = sys.exc_info()[1]
        restoration_error: BaseException | None = None
        try:
            with torch.no_grad():
                for (
                    original_buffer,
                    saved_value,
                    original_storage,
                    original_offset,
                    original_size,
                    original_stride,
                ) in buffer_values.values():
                    metadata_unchanged = (
                        original_buffer.layout == saved_value.layout
                        and original_buffer.size() == original_size
                        and (
                            original_buffer.layout != torch.strided
                            or (
                                original_buffer.stride() == original_stride
                                and original_buffer.storage_offset() == original_offset
                                and original_buffer.untyped_storage().data_ptr()
                                == original_storage.data_ptr()
                            )
                        )
                    )
                    if not metadata_unchanged:
                        if original_buffer.layout == torch.strided and original_storage is not None:
                            # Reattach the original backing storage and view metadata,
                            # preserving aliases between distinct registered buffers.
                            original_buffer.set_(
                                original_storage,
                                original_offset,
                                original_size,
                                original_stride,
                            )
                        elif original_buffer.layout == torch.sparse_coo:
                            original_buffer.sparse_resize_and_clear_(
                                original_size,
                                saved_value.sparse_dim(),
                                saved_value.dense_dim(),
                            )
                        else:
                            # Compressed sparse layouts expose resize_ rather
                            # than sparse_resize_and_clear_. If the forward
                            # successfully resized one, the inverse resize is
                            # the supported way to restore its metadata.
                            original_buffer.resize_(original_size)
                    # copy_ supports quantized and sparse tensors and, unlike set_
                    # to an independent clone, preserves existing storage aliases.
                    original_buffer.copy_(saved_value)
                for submodule, registrations, non_persistent in buffer_registries:
                    submodule._buffers.clear()
                    submodule._buffers.update(registrations)
                    submodule._non_persistent_buffers_set.clear()
                    submodule._non_persistent_buffers_set.update(non_persistent)
        except BaseException as error:  # cleanup below must still run
            restoration_error = error
        finally:
            for submodule, training in training_snapshot:
                submodule.training = training
            for parameter, original_gradient in gradient_snapshot:
                parameter.grad = original_gradient
        if restoration_error is not None:
            if active_exception is not None:
                active_exception.add_note(
                    f"An additional error occurred while restoring module state: "
                    f"{restoration_error!r}"
                )
            else:
                raise restoration_error


def _run_block_chain(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
) -> Iterator[tuple[CheckpointableBlock, tuple[Any, ...], dict[str, Any], Any]]:
    """Run the declared chain once under ``torch.no_grad()`` with state restored."""
    current_args = example_inputs
    current_kwargs = dict(example_kwargs or {})
    num_blocks = len(blocks)

    for index, block in enumerate(blocks):
        with _preserve_module_state(block.module, isolate_gradients=False):
            with torch.no_grad():
                output = block.module(*current_args, **current_kwargs)
        yield block, current_args, current_kwargs, output
        if index < num_blocks - 1:
            current_args, current_kwargs = _boundary_convert(output)


def compute_execution_signature(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
) -> ExecutionSignature:
    """Execute the declared chain once and record every boundary signature."""
    signatures: list[tuple[str, str, str]] = []
    entry_signature = _io_signature(example_inputs, example_kwargs)
    for block, args, kwargs, output in _run_block_chain(blocks, example_inputs, example_kwargs):
        # Match BlockProfile.output_shape_signature: a block return is represented
        # as one positional boundary value before any downstream conversion.
        signatures.append((block.block_id, _io_signature(args, kwargs), _io_signature((output,), {})))
    return ExecutionSignature(
        entry_signature=entry_signature,
        block_signatures=tuple(signatures),
        block_order=tuple(block.block_id for block in blocks),
    )
