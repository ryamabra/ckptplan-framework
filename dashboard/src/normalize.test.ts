import { describe, expect, it } from "vitest";
import { normalizeBenchmark, pairComparisons } from "./normalize";

describe("normalizeBenchmark", () => {
  it("normalizes progressive scaling artifacts", () => {
    const dataset = normalizeBenchmark({
      steps: [{
        batch_size: 4,
        seq_len: 2048,
        no_checkpoint: { oom: true },
        dynamic_programming: {
          oom: false,
          peak_allocated_bytes: 100,
          selected_checkpoint_blocks: ["layer0"],
        },
      }],
    });
    expect(dataset.sourceShape).toBe("progressive");
    expect(dataset.runs).toHaveLength(2);
    expect(dataset.runs[1]).toMatchObject({
      planner: "dynamic_programming",
      batchSize: 4,
      sequenceLength: 2048,
      checkpointedBlocks: ["layer0"],
    });
  });

  it("normalizes list and map result schemas", () => {
    const list = normalizeBenchmark({ results: [{ planner: "greedy", oom: false }] });
    const map = normalizeBenchmark({ results: { uniform: { oom: false } } });
    expect(list.sourceShape).toBe("result-list");
    expect(list.runs[0].planner).toBe("greedy");
    expect(map.sourceShape).toBe("result-map");
    expect(map.runs[0].planner).toBe("uniform");
  });

  it("computes comparisons without inventing OOM percentages", () => {
    const dataset = normalizeBenchmark({
      steps: [{
        batch_size: 2,
        no_checkpoint: { oom: false, peak_allocated_bytes: 200, step_latency_ms_mean: 10 },
        greedy: { oom: false, peak_allocated_bytes: 100, step_latency_ms_mean: 12 },
      }],
    });
    expect(pairComparisons(dataset)[0]).toMatchObject({
      memoryReductionPercent: 50,
      latencyOverheadPercent: 20,
    });
  });

  it("rejects unrelated JSON", () => {
    expect(() => normalizeBenchmark({ hello: "world" })).toThrow("No benchmark runs found");
  });
});
