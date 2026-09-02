export interface BenchmarkRun {
  id: string;
  planner: string;
  batchSize?: number;
  sequenceLength?: number;
  peakAllocatedBytes?: number;
  peakReservedBytes?: number;
  latencyMilliseconds?: number;
  throughput?: number;
  predictedActivationBytesAfter?: number;
  checkpointedBlocks: string[];
  correctnessPassed?: boolean | null;
  oom: boolean;
  errorMessage?: string | null;
}
export interface BenchmarkDataset {
  title: string;
  sourceShape: "progressive" | "result-list" | "result-map" | "top-level";
  metadata: Record<string, string | number | boolean>;
  runs: BenchmarkRun[];
}

export interface Comparison {
  baseline: BenchmarkRun;
  candidate: BenchmarkRun;
  memoryReductionPercent?: number;
  latencyOverheadPercent?: number;
  throughputChangePercent?: number;
}
