# Architecture

## Scope

`ckptplan` will provide a profiling, planning, application, and reporting pipeline for block-level gradient checkpointing in PyTorch.

The first implementation is intentionally restricted to an ordered collection of explicit checkpointable blocks. General tensor-level scheduling, arbitrary dynamic graphs, distributed training, and CPU/NVMe offloading are outside the initial scope.

## System Boundary

Inputs:

- a PyTorch model;
- representative example inputs;
- a declared set or ordered sequence of checkpointable blocks;
- an activation-memory budget or activation-memory-saving target (not a total
  GPU-capacity target; see `MVP_SPEC.md` §3, §7);
- profiling and planning options.

Outputs:

- an immutable, serializable checkpoint plan;
- a library-owned execution container composing exactly the declared blocks in
  order (not a transparent wrapper of arbitrary pre-existing model code; see
  `MVP_SPEC.md` §2, §9);
- profiler measurements and planner estimates;
- a report comparing estimated and measured outcomes.

## Proposed Package Layout

```text
ckptplan/
├── __init__.py
├── _execution.py
├── api.py
├── errors.py
├── types.py
├── profiling/
│   ├── profiler.py
│   ├── timing.py
│   └── memory.py
├── planning/
│   ├── base.py
│   ├── greedy.py
│   └── dynamic_programming.py
├── application/
│   ├── validate.py
│   └── wrapper.py
├── reporting/
│   ├── console.py
│   └── results.py
└── benchmarks/
    ├── models.py
    ├── baselines.py
    └── runner.py

tests/
├── unit/
├── integration/
└── correctness/
```

`_execution.py` is the accepted private home for canonical chain walking,
boundary conversion, shape signatures, and execution-state preservation.
Profiling and future plan application import these helpers so execution
semantics have one implementation. Subpackages not yet reached by the current
implementation slice remain uncreated rather than existing as empty stubs.

## Core Data Model

### BlockProfile

One record per checkpointable block and representative input shape. It should contain stable block identity, execution order, activation-size estimate, measured forward time, measured recomputation cost or an explicitly documented proxy, input/output metadata, device information, dtype, and measurement variance.

### CheckpointPlan

An immutable description of which blocks are checkpointed, the target constraint, predicted memory savings, predicted recomputation overhead, profiler/model fingerprints, planner name and version, and any assumptions required to apply the plan safely.

### BenchmarkResult

A structured record containing configuration, peak allocated and reserved memory, step latency distribution, throughput, warm-up count, trial count, numerical-correctness result, and environment metadata.

## Component Responsibilities

### Public API

Coordinates validation, profiling, planning, plan application, and reporting. It must expose explicit stages so advanced users can inspect or replace each component.

### Profiler

Runs representative forward and training steps, collects block-level measurements, synchronizes CUDA only where required for correct timing, resets memory statistics between controlled trials, and reports uncertainty rather than presenting a single noisy observation as exact.

Profiling must not silently claim that tensor size equals peak-memory savings. Activation lifetime, saved tensors, allocator behavior, and checkpoint boundaries can make those values differ.

Device and dtype options are validation-only: profiling never moves or casts
caller-owned modules or values. Profiling restores every submodule's individual
training flag, every supported registered-buffer value and registration state,
and every pre-existing parameter gradient on success and exception. Unsupported
buffer layouts are rejected during preflight before caller code executes.

### Planner

Consumes ordered block profiles and a target constraint. It produces a deterministic plan. The initial formulation must document exactly what is optimized and which dependencies or simplifying assumptions make the problem tractable.

At least one transparent baseline planner must exist alongside any more sophisticated planner.

### Plan Application

Applies checkpointing through supported PyTorch APIs. It validates that the target model matches the profile used to create the plan and refuses unsafe application when block identity, order, device, dtype, or relevant shape assumptions differ.

Random-number behavior and checkpoint API options must be explicit. Stateful or unsupported blocks should fail validation rather than produce silently incorrect training.

### Reporter

Shows the selected blocks, predicted trade-off, measured trade-off, profiler cost, and differences between prediction and observation. Initial reporting should be structured data plus concise terminal output; interactive visualization is not required for v0.1.

## Execution Flow

```text
model + representative inputs + eligible blocks
                    |
                    v
              validate inputs
                    |
                    v
             profile each block
                    |
                    v
          build ordered cost model
                    |
                    v
        plan under target constraint
                    |
                    v
       validate and apply checkpoint plan
                    |
                    v
       run correctness and performance trials
                    |
                    v
          emit plan and measured report
```

## Correctness Invariants

- A planned model produces outputs consistent with the unmodified model under the declared tolerance.
- Parameter gradients are consistent with the unmodified model under the declared tolerance.
- All trainable parameters expected to receive gradients still receive them.
- A serialized plan cannot be applied silently to an incompatible model or input profile.
- Unsupported stochastic or stateful behavior is either handled explicitly or rejected.
- Planner output is deterministic for identical profiles and options.

## Measurement Principles

- Separate profiling time from steady-state training time.
- Warm up CUDA before recording trials.
- Record multiple trials and distribution statistics.
- Measure both allocated and reserved CUDA memory when available.
- Capture software versions, GPU model, dtype, batch shape, and relevant environment settings.
- Use identical inputs and training-step semantics across baselines.
- Report failures and OOM boundaries rather than discarding them.

## Initial Baselines

- No checkpointing.
- Every eligible block checkpointed.
- Uniform selection of eligible blocks.
- Greedy selection based on a declared memory-saved/recompute-cost score.
- Dynamic-programming selection if the accepted formulation supports it.

## Compatibility Strategy

The MVP should pin a narrow tested compatibility matrix. CI can cover CPU correctness and planner behavior. CUDA correctness and performance should run as a separate reproducible benchmark workflow because hosted GPU availability and cost differ from normal unit testing.

## Explicit Non-Goals for v0.1

- Arbitrary operation- or tensor-level checkpoint placement.
- Automatic traversal and rewriting of every possible PyTorch autograd graph.
- Distributed, pipeline, tensor, or data-parallel scheduling.
- CPU or storage offloading.
- `torch.compile` and CUDA Graph guarantees unless separately validated.
- Mixture-of-experts and data-dependent control flow.
- A Rust or C++ optimization core without measured justification.
- Guaranteed global optimality for real peak GPU memory unless the implemented model proves that guarantee.

## Future Extension Points

- Repeated transformer-block adapters.
- Shape-bucketed profiles for variable sequence lengths.
- More accurate saved-tensor and activation-lifetime models.
- Tensor-level graph scheduling.
- Offload as a third keep/recompute/offload decision.
- Hugging Face Trainer and Lightning integrations.
- Additional solvers and Pareto-frontier generation.
- HTML reports and memory timelines.
