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

```bash
npm install
npm run dev
```

Use `npm test` for the normalizer tests and `npm run build` to produce `dist/`.
