import { useMemo, useRef, useState } from "react";
import { ComparisonTable, MemoryBars, MetricCard } from "./components";
import { normalizeBenchmark, pairComparisons } from "./normalize";
import { sampleDataset } from "./sample";
import type { BenchmarkDataset } from "./types";

export default function App() {
  const [dataset, setDataset] = useState<BenchmarkDataset>(sampleDataset);
  const [error, setError] = useState<string>();
  const inputRef = useRef<HTMLInputElement>(null);
  const comparisons = useMemo(() => pairComparisons(dataset), [dataset]);

  const successful = dataset.runs.filter((run) => !run.oom);
  const largestBatch = Math.max(...successful.map((run) => run.batchSize ?? 0), 0);
  const lowestMemory = successful
    .filter((run) => run.peakAllocatedBytes !== undefined)
    .sort((a, b) => (a.peakAllocatedBytes ?? Infinity) - (b.peakAllocatedBytes ?? Infinity))[0];
  const averageOverhead = comparisons
    .filter((item) => item.latencyOverheadPercent !== undefined && !item.baseline.oom)
    .reduce((sum, item, _, values) => sum + (item.latencyOverheadPercent ?? 0) / values.length, 0);

  const loadFile = async (file?: File) => {
    if (!file) return;
    try {
      const payload: unknown = JSON.parse(await file.text());
      setDataset(normalizeBenchmark(payload, file.name));
      setError(undefined);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read this benchmark file.");
    }
  };

  return (
    <main>
      <header className="hero">
        <nav>
          <a className="brand" href="https://github.com/ryamabra/ckptplan-framework">
            <span className="brand-mark">c</span>
            <span>ckptplan</span>
          </a>
          <button className="upload-button" onClick={() => inputRef.current?.click()}>
            Open benchmark JSON
          </button>
          <input
            ref={inputRef}
            hidden
            type="file"
            accept="application/json,.json"
            onChange={(event) => void loadFile(event.target.files?.[0])}
          />
        </nav>
        <div className="hero-copy">
          <p className="eyebrow">Activation checkpoint intelligence</p>
          <h1>Find the memory<br />you can afford to trade.</h1>
          <p className="lede">
            Explore measured GPU memory, recompute latency, and OOM boundaries from any
            ckptplan benchmark artifact—entirely in your browser.
          </p>
        </div>
        <div className="dataset-chip">
          <span className="live-dot" />
          <div><small>Viewing</small><b>{dataset.title}</b></div>
          <span>{dataset.runs.length} runs</span>
        </div>
      </header>

      {error && <div className="error-banner" role="alert">{error}</div>}

      <section className="metrics" aria-label="Benchmark summary">
        <MetricCard label="Largest feasible batch" value={String(largestBatch || "n/r")} detail="Across successful runs" />
        <MetricCard
          label="Lowest peak allocation"
          value={lowestMemory ? `${((lowestMemory.peakAllocatedBytes ?? 0) / 2 ** 30).toFixed(1)} GiB` : "n/r"}
          detail={lowestMemory?.planner.replaceAll("_", " ") ?? "No measured result"}
        />
        <MetricCard
          label="Mean recompute premium"
          value={comparisons.length ? `${averageOverhead >= 0 ? "+" : ""}${averageOverhead.toFixed(1)}%` : "n/r"}
          detail="Versus no checkpoint"
        />
        <MetricCard
          label="OOM observations"
          value={String(dataset.runs.filter((run) => run.oom).length)}
          detail="Preserved, never discarded"
        />
      </section>

      <div className="content-grid">
        <MemoryBars runs={dataset.runs} />
        <aside className="panel insight-panel">
          <p className="eyebrow">Read the boundary</p>
          <h2>Feasibility is the result.</h2>
          <p>
            A checkpoint plan is valuable when it makes a workload executable, even when a
            percentage comparison cannot exist because the baseline runs out of memory.
          </p>
          <div className="boundary-stat">
            <span>{dataset.runs.filter((run) => run.oom).length}</span>
            <p>baseline or candidate runs hit an honest OOM boundary.</p>
          </div>
        </aside>
      </div>
      <ComparisonTable comparisons={comparisons} />
      <footer>
        <span>Local-first · no uploads</span>
        <span>React + TypeScript</span>
      </footer>
    </main>
  );
}
