# Extension experiments

Follow-up work after the published study. Two new kernels improve on the published `tiled` and
`implicit` variants, and a set of experiments checks them against cuDNN and on real data. The
seven published variants, the tests, and the evidence in `results/` (top level) are unchanged:
the new kernels are built as separate PyTorch extensions (`pynet_ext_tiled_mk`,
`pynet_ext_implicit_reg`), and all new evidence goes to `results/extensions/`.

Findings, tables, and the claims they support: [results/extensions/FINDINGS.md](../results/extensions/FINDINGS.md).

## Kernels

| File | Kernel | Idea |
|---|---|---|
| `tiled_mk.cu` | `conv2d_tiled_mk<T, KB>` | `tiled`, but each thread computes KB output channels from one staged input tile. KB ∈ {1, 4, 8, 16}; KB = 1 has the original structure. |
| `implicit_reg.cu` | `conv2d_implicit_reg<BM, BN, BK, TM, TN>` | `implicit` with a TM×TN register tile per thread, a block tile height chosen from K, and (c, r, s) / output-column decoding done once per block in shared-memory tables. `mode` 0 = auto (falls back to the original implicit kernel when the register tiles would launch fewer blocks than the GPU has SMs), 1 = always register tiles, 2 = always original. |

Both keep the original summation order and add bias after summation, so on every tested shape
their outputs are bitwise equal to the kernel they improve.

## Scripts

Run from the project root in the HOWTO section 1 environment. Each writes JSON to
`results/extensions/` by default (`--output-dir` to change).

| Script | Needs | Output |
|---|---|---|
| `compare_variants.py --reps 200 --warmup 25` | GPU | `single_session.json`: correctness sweep + one-session timing of all variants |
| `audit_cudnn.py` | GPU | `audit.json`: cuDNN settings, kernels per call, full-call vs GPU-only time |
| `repeatability.py --processes 12` | GPU | `repeatability/`: 12 fresh processes, event and CUDA-graph times, paired statistics |
| `real_data.py --count 200` | GPU, weights, ZRR test pairs | `real_data.json`: real PyNET layer inputs and whole-network quality/latency |
| `plot_results.py` | `repeatability/summary.json`, matplotlib | `ext_vs_cudnn.png`, `ext_vs_original.png` |

`repeatability.py` refuses an output directory that already holds run files; delete
`results/extensions/repeatability/` or pass a fresh `--output-dir` to rerun, or use
`--analyze-only` to recompute `summary.json` from existing runs.

`common.py` holds the shared workloads, extension builds, timing flags, and the CUDA-graph timer.

## Two timing metrics

- **event**: CUDA events around one eager call, the report's invocation-path metric. The start
  event is recorded before the host dispatches the call, so CPU-side work (PyTorch dispatch,
  cuDNN setup, extra kernel launches) appears as GPU idle time inside the interval.
- **graph**: 20 calls captured in a CUDA graph and replayed; the GPU runs the recorded kernels
  back to back with no host work, so this is GPU execution time.

Small workloads differ greatly between the two, mostly for cuDNN, which dispatches a convolution
kernel and a separate bias-add kernel through a heavier host path.
