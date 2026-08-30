"""Unit tests for ``declare_blocks`` and ``CheckpointableBlock``.

See MVP_SPEC.md Sec 3 (rules), Sec 5 (schema), Sec 10.3 (shared parameters and
buffers). Covers every invariant listed there, plus the "does not execute the
model" / "does not require example inputs" claims from Sec 3's last bullet.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

from ckptplan import BlockDeclarationError, CheckpointableBlock, declare_blocks


def _linear_model(n: int = 3, width: int = 4) -> torch.nn.Module:
    """A plain container model with n named Linear submodules: layer0..layer{n-1}."""
    model = torch.nn.Module()
    for i in range(n):
        model.add_module(f"layer{i}", torch.nn.Linear(width, width))
    return model


# -- happy path ---------------------------------------------------------------


def test_returns_tuple_of_checkpointable_block_in_order() -> None:
    model = _linear_model(3)
    blocks = declare_blocks(
        model=model,
        blocks=[("a", model.layer0), ("b", model.layer1), ("c", model.layer2)],
    )
    assert isinstance(blocks, tuple)
    assert all(isinstance(b, CheckpointableBlock) for b in blocks)
    assert [b.block_id for b in blocks] == ["a", "b", "c"]
    assert [b.order for b in blocks] == [0, 1, 2]
    assert blocks[0].module is model.layer0
    assert blocks[1].module is model.layer1
    assert blocks[2].module is model.layer2


def test_preserves_caller_order_not_model_definition_order() -> None:
    model = _linear_model(3)
    # Declared deliberately out of the model's own definition order.
    blocks = declare_blocks(
        model=model,
        blocks=[("c", model.layer2), ("a", model.layer0), ("b", model.layer1)],
    )
    assert [b.block_id for b in blocks] == ["c", "a", "b"]
    assert [b.order for b in blocks] == [0, 1, 2]
    assert blocks[0].module is model.layer2
    assert blocks[1].module is model.layer0
    assert blocks[2].module is model.layer1


def test_allows_declaring_the_root_model_itself() -> None:
    # The root model is reachable from its own named_modules() (name="").
    model = _linear_model(1)
    blocks = declare_blocks(model=model, blocks=[("root", model)])
    assert blocks[0].module is model


def test_reachability_works_for_nested_submodules() -> None:
    model = torch.nn.Module()
    encoder = torch.nn.Module()
    encoder.add_module("layer0", torch.nn.Linear(4, 4))
    model.add_module("encoder", encoder)
    blocks = declare_blocks(model=model, blocks=[("enc0", encoder.layer0)])
    assert blocks[0].module is encoder.layer0


def test_empty_block_list_returns_empty_tuple() -> None:
    model = _linear_model(1)
    assert declare_blocks(model=model, blocks=[]) == ()


# -- block_id validation -------------------------------------------------------


def test_rejects_empty_block_id() -> None:
    model = _linear_model(1)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("", model.layer0)])


def test_rejects_non_string_block_id() -> None:
    model = _linear_model(1)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[(0, model.layer0)])  # type: ignore[list-item]


def test_rejects_duplicate_block_id() -> None:
    model = _linear_model(2)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", model.layer0), ("a", model.layer1)])


# -- reachability ---------------------------------------------------------------


def test_rejects_module_not_reachable_from_model() -> None:
    model = _linear_model(1)
    outsider = torch.nn.Linear(4, 4)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", outsider)])


def test_reachability_is_identity_based_not_name_based() -> None:
    # model_b's layer0 sits at the same attribute name as model_a's layer0 but
    # is a different object, and must still be rejected against model_a.
    model_a = _linear_model(1)
    model_b = _linear_model(1)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model_a, blocks=[("a", model_b.layer0)])


# -- duplicate module instance ---------------------------------------------------


def test_rejects_repeated_module_instance_under_different_block_ids() -> None:
    model = _linear_model(1)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", model.layer0), ("b", model.layer0)])


# -- overlapping block subtrees (MVP_SPEC.md Sec 2, Sec 10.7, Revision 3.2) ---------


def test_rejects_parent_then_child() -> None:
    model = torch.nn.Module()
    encoder = torch.nn.Module()
    encoder.add_module("layer0", torch.nn.Linear(4, 4))
    model.add_module("encoder", encoder)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("encoder", encoder), ("enc0", encoder.layer0)])


def test_rejects_child_then_parent() -> None:
    # Same pair as above, declared in the opposite order: the check must be
    # symmetric/order-independent.
    model = torch.nn.Module()
    encoder = torch.nn.Module()
    encoder.add_module("layer0", torch.nn.Linear(4, 4))
    model.add_module("encoder", encoder)
    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("enc0", encoder.layer0), ("encoder", encoder)])


def test_siblings_are_allowed() -> None:
    # Two children of the same parent -- neither is an ancestor of the other.
    model = torch.nn.Module()
    parent = torch.nn.Module()
    parent.add_module("layer0", torch.nn.Linear(4, 4))
    parent.add_module("layer1", torch.nn.Linear(4, 4))
    model.add_module("parent", parent)
    blocks = declare_blocks(
        model=model, blocks=[("a", parent.layer0), ("b", parent.layer1)]
    )
    assert len(blocks) == 2


def test_rejects_deeply_nested_overlap() -> None:
    # Ancestor several levels above the declared descendant, not an immediate
    # parent -- the check must walk the full subtree, not just direct children.
    model = torch.nn.Module()
    level1 = torch.nn.Module()
    level2 = torch.nn.Module()
    level3 = torch.nn.Module()
    leaf = torch.nn.Linear(4, 4)
    level3.add_module("leaf", leaf)
    level2.add_module("level3", level3)
    level1.add_module("level2", level2)
    model.add_module("level1", level1)

    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("top", level1), ("deep", leaf)])


def test_overlap_check_does_not_falsely_flag_an_unrelated_third_block() -> None:
    # A is an ancestor of B; C is an unrelated sibling of A's parent, not
    # related to A or B by ancestry at all. Only the (A, B) pair must raise;
    # (A, C) and (B, C) must not.
    model = torch.nn.Module()
    a = torch.nn.Module()
    a.add_module("b", torch.nn.Linear(4, 4))
    model.add_module("a", a)
    model.add_module("c", torch.nn.Linear(4, 4))

    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", a), ("b", a.b), ("c", model.c)])

    # B and C alone (no overlap between them) must succeed.
    blocks = declare_blocks(model=model, blocks=[("b", a.b), ("c", model.c)])
    assert len(blocks) == 2


def test_overlap_error_names_both_block_ids() -> None:
    model = torch.nn.Module()
    encoder = torch.nn.Module()
    encoder.add_module("layer0", torch.nn.Linear(4, 4))
    model.add_module("encoder", encoder)
    with pytest.raises(BlockDeclarationError) as excinfo:
        declare_blocks(model=model, blocks=[("parent_block", encoder), ("child_block", encoder.layer0)])
    message = str(excinfo.value)
    assert "parent_block" in message
    assert "child_block" in message


def test_rejects_sibling_wrappers_that_share_a_descendant_module() -> None:
    model = torch.nn.Module()
    shared = torch.nn.Linear(4, 4)
    left = torch.nn.Module()
    right = torch.nn.Module()
    left.add_module("shared", shared)
    right.add_module("shared", shared)
    model.add_module("left", left)
    model.add_module("right", right)

    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("left", left), ("right", right)])


# -- shared buffers vs. shared parameters (MVP_SPEC.md Sec 10.3) ------------------


def test_rejects_shared_persistent_buffer_across_blocks() -> None:
    model = torch.nn.Module()
    shared_buffer = torch.zeros(4)
    block_a = torch.nn.Module()
    block_a.register_buffer("stat", shared_buffer, persistent=True)
    block_b = torch.nn.Module()
    block_b.register_buffer("stat", shared_buffer, persistent=True)
    model.add_module("a", block_a)
    model.add_module("b", block_b)

    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", block_a), ("b", block_b)])


def test_rejects_shared_non_persistent_buffer_across_blocks() -> None:
    # persistent=False controls serialization, not runtime mutation. A shared
    # registered buffer can still be mutated during checkpoint recomputation.
    model = torch.nn.Module()
    shared_buffer = torch.zeros(4)
    block_a = torch.nn.Module()
    block_a.register_buffer("stat", shared_buffer, persistent=False)
    block_b = torch.nn.Module()
    block_b.register_buffer("stat", shared_buffer, persistent=False)
    model.add_module("a", block_a)
    model.add_module("b", block_b)

    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", block_a), ("b", block_b)])
    assert "stat" not in block_a.state_dict()
    assert "stat" not in block_b.state_dict()


def test_rejects_mixed_persistence_buffer_alias_across_blocks() -> None:
    # The same tensor is a registered buffer in both blocks, so differing
    # serialization policy does not remove the recomputation-side-effect risk.
    model = torch.nn.Module()
    shared_buffer = torch.zeros(4)
    block_a = torch.nn.Module()
    block_a.register_buffer("stat", shared_buffer, persistent=True)
    block_b = torch.nn.Module()
    block_b.register_buffer("stat", shared_buffer, persistent=False)
    model.add_module("a", block_a)
    model.add_module("b", block_b)

    with pytest.raises(BlockDeclarationError):
        declare_blocks(model=model, blocks=[("a", block_a), ("b", block_b)])


def test_permits_shared_parameter_across_blocks() -> None:
    model = torch.nn.Module()
    shared_parameter = torch.nn.Parameter(torch.ones(4, 4))
    block_a = torch.nn.Module()
    block_a.register_parameter("weight", shared_parameter)
    block_b = torch.nn.Module()
    block_b.register_parameter("weight", shared_parameter)
    model.add_module("a", block_a)
    model.add_module("b", block_b)

    # Must not raise: shared parameters are explicitly permitted (Sec 10.3).
    blocks = declare_blocks(model=model, blocks=[("a", block_a), ("b", block_b)])
    assert len(blocks) == 2
    assert block_a.weight is block_b.weight


def test_own_buffer_declared_alone_is_not_flagged_as_shared() -> None:
    model = torch.nn.Module()
    only_block = torch.nn.Module()
    only_block.register_buffer("running_stat", torch.zeros(4))
    model.add_module("only", only_block)
    blocks = declare_blocks(model=model, blocks=[("only", only_block)])
    assert len(blocks) == 1


def test_two_blocks_with_distinct_own_buffers_does_not_raise() -> None:
    # Negative-space check: two *different* buffer instances across two
    # blocks must not be mistaken for one shared instance.
    model = torch.nn.Module()
    block_a = torch.nn.Module()
    block_a.register_buffer("stat", torch.zeros(4))
    block_b = torch.nn.Module()
    block_b.register_buffer("stat", torch.ones(4))
    model.add_module("a", block_a)
    model.add_module("b", block_b)

    blocks = declare_blocks(model=model, blocks=[("a", block_a), ("b", block_b)])
    assert len(blocks) == 2


def test_same_buffer_tensor_under_two_names_within_one_block_does_not_raise() -> None:
    # One tensor registered under two different buffer names, both within the
    # *same* declared block, must not be treated as a cross-block collision.
    # (Relies on nn.Module.named_buffers()'s remove_duplicate=True default,
    # which yields such a tensor only once -- see the comment in api.py.)
    model = torch.nn.Module()
    only_block = torch.nn.Module()
    shared_within_block = torch.zeros(4)
    only_block.register_buffer("stat_a", shared_within_block)
    only_block.register_buffer("stat_b", shared_within_block)
    model.add_module("only", only_block)

    blocks = declare_blocks(model=model, blocks=[("only", only_block)])
    assert len(blocks) == 1


# -- does not execute the model ---------------------------------------------------


class _SpyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - must never run
        self.forward_calls += 1
        return self.linear(x)


def test_never_calls_forward() -> None:
    model = torch.nn.Module()
    spy = _SpyModule()
    model.add_module("spy", spy)
    declare_blocks(model=model, blocks=[("spy", spy)])
    assert spy.forward_calls == 0


# -- does not require example inputs ------------------------------------------------


def test_signature_has_no_example_inputs_parameter() -> None:
    params = inspect.signature(declare_blocks).parameters
    assert set(params) == {"model", "blocks"}


# -- CheckpointableBlock immutability -------------------------------------------------


def test_checkpointable_block_is_frozen() -> None:
    model = _linear_model(1)
    blocks = declare_blocks(model=model, blocks=[("a", model.layer0)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        blocks[0].block_id = "changed"  # type: ignore[misc]
