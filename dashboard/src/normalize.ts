import type { BenchmarkDataset, BenchmarkRun, Comparison } from "./types";

type JsonObject = Record<string, unknown>;

const isObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const numberValue = (object: JsonObject, ...keys: string[]): number | undefined => {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
};

const stringValue = (object: JsonObject, ...keys: string[]): string | undefined => {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "string") return value;
  }
  return undefined;
};

const booleanValue = (object: JsonObject, key: string): boolean | undefined => {
  const value = object[key];
  return typeof value === "boolean" ? value : undefined;
};

const stringList = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

const normalizeRun = (
  entry: JsonObject,
  planner: string,
  id: string,
  inherited: { batchSize?: number; sequenceLength?: number } = {},
): BenchmarkRun => ({
  id,
  planner,
  batchSize: numberValue(entry, "batch_size") ?? inherited.batchSize,
  sequenceLength: numberValue(entry, "seq_len", "sequence_length") ?? inherited.sequenceLength,
  peakAllocatedBytes: numberValue(entry, "peak_allocated_bytes"),
  peakReservedBytes: numberValue(entry, "peak_reserved_bytes"),
  latencyMilliseconds: numberValue(entry, "step_latency_ms_mean", "latency_ms_mean"),
  throughput: numberValue(entry, "throughput_samples_per_sec"),
  predictedActivationBytesAfter: numberValue(
    entry,
    "predicted_activation_bytes_after",
    "predicted_activation_after",
  ),
  checkpointedBlocks: stringList(
    entry.selected_checkpoint_blocks ?? entry.checkpointed_blocks,
  ),
  correctnessPassed:
    entry.correctness_passed === null
      ? null
      : booleanValue(entry, "correctness_passed"),
  oom: booleanValue(entry, "oom") ?? false,
  errorMessage: stringValue(entry, "error_message") ?? null,
});

const metadataFrom = (payload: JsonObject): BenchmarkDataset["metadata"] => {
  const metadata: BenchmarkDataset["metadata"] = {};
  for (const [key, value] of Object.entries(payload)) {
    if (["results", "steps"].includes(key)) continue;
    if (["string", "number", "boolean"].includes(typeof value)) {
      metadata[key] = value as string | number | boolean;
    }
  }
  return metadata;
};

export function normalizeBenchmark(payload: unknown, title = "Benchmark results"): BenchmarkDataset {
  if (!isObject(payload)) throw new Error("Expected a JSON object at the top level.");

  if (Array.isArray(payload.steps)) {
    const runs: BenchmarkRun[] = [];
    payload.steps.forEach((step, stepIndex) => {
      if (!isObject(step)) return;
      const inherited = {
        batchSize: numberValue(step, "batch_size"),
        sequenceLength: numberValue(step, "seq_len", "sequence_length"),
      };
      Object.entries(step).forEach(([planner, value]) => {
        if (!isObject(value)) return;
        runs.push(normalizeRun(value, planner, `${stepIndex}:${planner}`, inherited));
      });
    });
    if (!runs.length) throw new Error("The progressive result contains no planner runs.");
    return { title, sourceShape: "progressive", metadata: metadataFrom(payload), runs };
  }

  if (Array.isArray(payload.results)) {
    const runs = payload.results.flatMap((value, index) => {
      if (!isObject(value)) return [];
      const planner = stringValue(value, "planner", "config_name") ?? `config${index}`;
      return [normalizeRun(value, planner, `${index}:${planner}`)];
    });
    if (!runs.length) throw new Error("The results list contains no benchmark runs.");
    return { title, sourceShape: "result-list", metadata: metadataFrom(payload), runs };
  }

  if (isObject(payload.results)) {
    const runs = Object.entries(payload.results).flatMap(([planner, value], index) =>
      isObject(value) ? [normalizeRun(value, planner, `${index}:${planner}`)] : [],
    );
    if (!runs.length) throw new Error("The results map contains no benchmark runs.");
    return { title, sourceShape: "result-map", metadata: metadataFrom(payload), runs };
  }

  const runs = Object.entries(payload).flatMap(([planner, value], index) => {
    if (!isObject(value)) return [];
    const looksLikeRun = "oom" in value || "peak_allocated_bytes" in value;
    return looksLikeRun ? [normalizeRun(value, planner, `${index}:${planner}`)] : [];
  });
  if (!runs.length) {
    throw new Error(
      "No benchmark runs found. Expected steps, a results list/map, or top-level run objects.",
    );
  }
  return { title, sourceShape: "top-level", metadata: metadataFrom(payload), runs };
}
const percentChange = (before?: number, after?: number): number | undefined => {
  if (before === undefined || after === undefined || before === 0) return undefined;
  return ((after - before) / before) * 100;
};

export function pairComparisons(dataset: BenchmarkDataset): Comparison[] {
  const byWorkload = new Map<string, BenchmarkRun[]>();
  dataset.runs.forEach((run) => {
    const key = `${run.batchSize ?? "?"}:${run.sequenceLength ?? "?"}`;
    byWorkload.set(key, [...(byWorkload.get(key) ?? []), run]);
  });

  return [...byWorkload.values()].flatMap((runs) => {
    const baseline = runs.find((run) => run.planner === "no_checkpoint") ?? runs[0];
    return runs
      .filter((candidate) => candidate !== baseline)
      .map((candidate) => ({
        baseline,
        candidate,
        memoryReductionPercent:
          baseline.peakAllocatedBytes && candidate.peakAllocatedBytes !== undefined
            ? -percentChange(baseline.peakAllocatedBytes, candidate.peakAllocatedBytes)!
            : undefined,
        latencyOverheadPercent: percentChange(
          baseline.latencyMilliseconds,
          candidate.latencyMilliseconds,
        ),
        throughputChangePercent: percentChange(baseline.throughput, candidate.throughput),
      }));
  });
}
