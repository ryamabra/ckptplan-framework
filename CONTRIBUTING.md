# Contributing to ckptplan

Use a focused pull request for a bug fix, documentation improvement, or feature.
For larger changes, open an issue first to agree on the scope. Include the
problem, the resulting behavior, and the checks you ran in the pull request.

## Python development

From a clone of this repository, use Python 3.10–3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -m mypy
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead of the `source` command.

The standard test suite runs on CPU. Activation-memory profiling requires
CUDA; CPU profiles contain timing measurements only. Do not treat a CPU test
run as evidence for GPU memory savings or gradient correctness on CUDA.
See the [main README](README.md#verified-evidence) for existing GPU evidence.

For bug fixes, add a regression test that reproduces the problem before the
fix. Keep GPU benchmarks separate from routine CPU checks and report the
hardware and workload when sharing benchmark results.

## Native planner

For changes under `native/`, use a C++20 compiler and CMake:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

CUDA support is optional; see [native/README.md](native/README.md#cuda-build).

## Dashboard

See [dashboard/README.md](dashboard/README.md) for Node.js requirements and
startup instructions. For dashboard changes, run these from `dashboard/`:

```bash
npm ci
npm test
npm run build
```

## Before opening a pull request

- Run the checks relevant to the files you changed and state their results.
- Document public behavior changes and keep examples consistent with the API.
- Keep generated build outputs, local environments, credentials, and unrelated
  edits out of the commit.
- Distinguish measured benchmark results from estimates; state unverified
  configurations and limitations.

Repository-maintenance sessions also record completed work and the next task
in [STATE.md](STATE.md), following [WORKFLOW.md](WORKFLOW.md).
