import { normalizeBenchmark } from "./normalize";

export const sampleDataset = normalizeBenchmark(
  {
    layers: 24,
    hidden: 2048,
    seq_len_start: 2048,
    dtype: "float32",
    steps: [
      {
        batch_size: 1,
        seq_len: 2048,
        no_checkpoint: {
          oom: false,
          peak_allocated_bytes: 9896961536,
          step_latency_ms_mean: 1629.93,
          throughput_samples_per_sec: 0.6135,
        },
        dynamic_programming: {
          oom: false,
          peak_allocated_bytes: 9887220736,
          step_latency_ms_mean: 1837.22,
          throughput_samples_per_sec: 0.5443,
          selected_checkpoint_blocks: ["layer0", "layer2", "layer4", "layer7"],
        },
      },
      {
        batch_size: 2,
        seq_len: 2048,
        no_checkpoint: {
          oom: false,
          peak_allocated_bytes: 15697372160,
          step_latency_ms_mean: 3273.23,
          throughput_samples_per_sec: 0.611,
        },
        dynamic_programming: {
          oom: false,
          peak_allocated_bytes: 10290021376,
          step_latency_ms_mean: 4192.36,
          throughput_samples_per_sec: 0.4771,
          selected_checkpoint_blocks: ["layer0", "layer1", "layer3", "layer5", "layer7"],
        },
      },
      {
        batch_size: 4,
        seq_len: 2048,
        no_checkpoint: { oom: true, error_message: "CUDA out of memory" },
        dynamic_programming: {
          oom: false,
          peak_allocated_bytes: 11767260160,
          step_latency_ms_mean: 8594.5,
          throughput_samples_per_sec: 0.4654,
          selected_checkpoint_blocks: ["layer1", "layer2", "layer3", "layer4", "layer5"],
        },
      },
    ],
  },
  "A10G progressive scaling",
);
