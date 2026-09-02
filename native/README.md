# ckptplan native core

This directory contains two intentionally narrow native components:

- `ckptplan::plan`, a dependency-free C++20 implementation of the deterministic
  cost model used by the Python planner;
- `ckptplan::cuda::summarize_memory`, a CUDA primitive that computes retained,
  checkpointed, and peak-prefix activation byte totals from device arrays.

The native planner does not execute models, inspect PyTorch modules, or replace
the Python public API. Python remains responsible for profiling real modules,
checking eligibility, applying a plan, preserving model state, and benchmarking
correctness. The native API accepts already-validated cost records so non-Python
runtimes can reuse the pure selection stage.

## CPU build

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

Consumers can link the `ckptplan::planner` CMake target and include
`ckptplan/planner.hpp`.

## CUDA build

```bash
cmake -S native -B native/build -DCKPTPLAN_ENABLE_CUDA=ON
cmake --build native/build --target ckptplan_cuda_demo
./native/build/ckptplan_cuda_demo
```

The CUDA function expects device pointers and synchronizes the supplied stream
before returning its host-side summary. It is separate from the PyTorch
allocator profiler; it provides a small integration primitive rather than
claiming to measure the process-wide CUDA peak.
