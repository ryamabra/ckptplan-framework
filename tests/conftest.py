"""Shared pytest fixtures.

The tiny deterministic sequential model here matches MVP_SPEC.md's CPU
correctness fixture exactly (see MVP_SPEC.md, "CPU correctness and planner-logic
tests"): 4 blocks, each ``nn.Sequential(nn.Linear(64, 64), nn.ReLU())``,
``float32``, CPU, seed 0, batch shape ``(8, 64)``.

It is a plain ``torch.nn.Module`` fixture only. It is not wrapped in any
``ckptplan`` API (``declare_blocks``, ``CheckpointableBlock``, etc.) because that
API does not exist yet in this repository-foundation slice — see STATE.md.
"""

from __future__ import annotations

import pytest
import torch

NUM_BLOCKS = 4
BLOCK_WIDTH = 64
BATCH_SIZE = 8
SEED = 0


class TinySequentialModel(torch.nn.Module):
    """The 4-block fixture from MVP_SPEC.md, composed into a single module."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(BLOCK_WIDTH, BLOCK_WIDTH), torch.nn.ReLU())
            for _ in range(NUM_BLOCKS)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def build_tiny_sequential_model() -> TinySequentialModel:
    """Deterministic factory: always seeds before constructing, so two
    independent calls produce byte-identical parameters."""
    torch.manual_seed(SEED)
    return TinySequentialModel()


def build_tiny_batch() -> torch.Tensor:
    """Deterministic factory for the fixture's input batch, shape (8, 64)."""
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, BLOCK_WIDTH, dtype=torch.float32)


@pytest.fixture
def tiny_sequential_model() -> TinySequentialModel:
    return build_tiny_sequential_model()


@pytest.fixture
def tiny_batch() -> torch.Tensor:
    return build_tiny_batch()
