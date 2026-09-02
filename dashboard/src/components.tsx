import type { BenchmarkRun, Comparison } from "./types";

const gibibytes = (bytes?: number): string =>
  bytes === undefined ? "n/r" : `${(bytes / 2 ** 30).toFixed(2)} GiB`;

const milliseconds = (value?: number): string =>
  value === undefined ? "n/r" : `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;

const percent = (value?: number): string =>
  value === undefined ? "n/r" : `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;

export function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}
export function MemoryBars({ runs }: { runs: BenchmarkRun[] }) {
  const measured = runs.filter((run) => !run.oom && run.peakAllocatedBytes !== undefined);
  const maximum = Math.max(...measured.map((run) => run.peakAllocatedBytes ?? 0), 1);
  return (
    <section className="panel chart-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Peak allocation</p>
          <h2>Memory by workload</h2>
        </div>
        <span className="legend"><i /> measured on GPU</span>
      </div>
      <div className="bars" aria-label="Peak allocated memory comparison">
        {runs.map((run) => (
          <div className="bar-row" key={run.id}>
            <div className="bar-label">
              <b>{run.planner.replaceAll("_", " ")}</b>
              <span>batch {run.batchSize ?? "?"}</span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${run.oom ? "oom" : ""}`}
                style={{ width: run.oom ? "100%" : `${((run.peakAllocatedBytes ?? 0) / maximum) * 100}%` }}
              />
              <span>{run.oom ? "OOM" : gibibytes(run.peakAllocatedBytes)}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function ComparisonTable({ comparisons }: { comparisons: Comparison[] }) {
  return (
    <section className="panel table-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Cost curve</p>
          <h2>Checkpoint trade-offs</h2>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Workload</th>
              <th>Planner</th>
              <th>Memory saved</th>
              <th>Latency</th>
              <th>Latency Δ</th>
              <th>Blocks</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map(({ baseline, candidate, memoryReductionPercent, latencyOverheadPercent }) => (
              <tr key={`${baseline.id}:${candidate.id}`}>
                <td>b{candidate.batchSize ?? "?"} · s{candidate.sequenceLength ?? "?"}</td>
                <td><span className="planner-pill">{candidate.planner.replaceAll("_", " ")}</span></td>
                <td className="positive">{baseline.oom ? "baseline OOM" : percent(memoryReductionPercent)}</td>
                <td>{milliseconds(candidate.latencyMilliseconds)}</td>
                <td>{baseline.oom ? "—" : percent(latencyOverheadPercent)}</td>
                <td>{candidate.checkpointedBlocks.length || "n/r"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
