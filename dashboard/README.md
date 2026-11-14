# ckptplan benchmark explorer

A local React and TypeScript dashboard for inspecting saved ckptplan benchmark
artifacts. It recognizes all JSON layouts supported by `benchmarks/report.py`,
plus the progressive-scaling `steps` layout:

- `{"results": [{"planner": ...}]}`;
- `{"results": {"planner": {...}}}`;
- top-level planner objects;
- `{"steps": [{"batch_size": ..., "no_checkpoint": {...}}]}`.

The browser reads selected files directly. There is no server and no upload.
OOM results remain visible because a feasibility boundary is evidence, not a
row to silently discard.

## Run locally

From the repository root, with Node.js 20.19+ or 22.12+ installed:

```bash
cd dashboard
npm ci
npm run dev
```

Open the local URL printed by Vite. The initial view contains bundled sample
results; it does not run a benchmark or profile your GPU.

## Open your results

Click **Open benchmark JSON** and select a saved result file. For example, open
[`benchmarks/progressive_scaling_result.json`](../benchmarks/progressive_scaling_result.json)
from this repository to compare the measured batch-size boundary. The file is
read in your browser, without uploading it.

To collect new results, use the Python benchmarking workflow described in the
[main README](../README.md#the-pipeline). The dashboard only displays saved
measurements and does not launch training jobs.

Use `npm test` for the normalizer tests and `npm run build` to produce `dist/`.
