# Reproducing the CUDA Convolution Study

This project implements seven FP32 CUDA convolution variants, evaluates their
latency and scaling, and substitutes two layers of pretrained PyNET for
RAW-to-sRGB inference. The architecture and weights are reused; the CUDA
implementations and experiments are our work. Results and discussion are in
[evaluation_report.md](evaluation_report.md); a Chinese translation is in
[evaluation_report_zh.md](evaluation_report_zh.md).

## 1. Environment

The recorded configuration is Linux, Python 3.11, PyTorch 2.9.1+cu129,
CUDA toolkit 12.9, GCC/G++ 13, and an RTX 3060 (12 GB, compute capability 8.6).
Install a compatible NVIDIA driver, the toolkit including `nvcc`, Python
headers, and `curl`. The PyTorch wheel does not supply the CUDA compiler.
Run the commands below from this project directory.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=gcc-13
export CXX=g++-13
export TORCH_CUDA_ARCH_LIST="8.6"  # Adjust to your GPU.
export MAX_JOBS=2
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

CUDA availability must be `True` for GPU experiments. Compilation occurs on
first use and is excluded from timing. Neither `torchvision` nor `pytest` is
required. Other GPU/toolchain combinations may produce different timings and
cuDNN execution paths.

On the original workspace host, `run.sh` uses the adjacent `pynet_env` and
`cuda_env` environments. For example, `bash run.sh --help` runs `benchmark.py`.
Use `PYNET_SCRIPT=test_convolutions.py bash run.sh ...` for another script, or
`PYNET_SCRIPT=-m bash run.sh unittest discover tests -v` for the unit suite.
On other hosts, use the plain Python commands in this document.

## 2. Weights and Data

```bash
bash setup.sh
python download_zrr_samples.py --count 1204 --full-id 1050 --output data/zrr_test_samples
```

`setup.sh` downloads the official level-0 checkpoint to
`checkpoints/pynet_level_0.pth`. Model loading uses strict parameter matching
and `weights_only=True`; no training is needed. The downloader creates
`data/zrr_test_samples/test/huawei_raw/*.png`, matched `test/canon/*.jpg`, and
`full_resolution/huawei_raw/1050.png`. Downloads and weights are excluded from
the submission ZIP. Existing local downloads can be reused.

Preprocessing packs an RGGB Bayer mosaic into `[B, Gr, R, Gb]` planes, scaled
by 1/1020. `Gr` is green on a red row and `Gb` is green on a blue row.
An input RAW patch of 448×448 becomes a 1×4×224×224 tensor and produces a
1×3×448×448 RGB output after Sigmoid.

## 3. Correctness Checks

```bash
python -m unittest discover tests -v
python test_convolutions.py --output-dir ../pynet_correctness_reproduction
```

The unit suite covers preprocessing, quality metrics, statistical analysis,
model replacement, CLI routing, extension caching, and actual CUDA execution.
The standalone sweep expects **54 passed checks and one rejected
constant-memory capacity case** against float64 PyTorch convolution. These
checks need neither images nor pretrained weights. Skipped CUDA tests do not
validate the GPU build.

Operator checks do not imply universal whole-network agreement. In the
published full-test evaluation, four variants pass all 1,204 image comparisons;
`naive`, `const`, and `gridstride` each exceed the fixed tolerance on image 747.
The 14 channel values occupy seven final RGB pixels after Sigmoid. In the supplied historical JSON, the legacy `violating_pixels` field counts
channel values; the current evaluator writes separate `violating_channel_values`
and spatial `violating_pixels` counts. The report
documents bias-order rounding and its downstream amplification. Aggregate
quality remains unchanged at the displayed precision. Diagnostic interventions
did not alter the kernels used for the published timings and quality scores.

## 4. Performance and Quality Experiments

Always choose a separate output directory. Published evidence is in `results/`;
commands below leave it available for comparison.

**Image size → latency (no weights or data required).** Hold N=1, channels
4→32, filter 3×3, stride 1, and padding 1 fixed. Measure nine feature-map sizes:

```bash
python benchmark.py --operator-size-only \
  --operator-sizes 32 64 96 128 224 320 448 640 896 \
  --reps 120 --warmup 10 --output-dir ../pynet_size_reproduction
```

This writes `image_size_scaling.json` with samples and numerical errors.
All methods share inputs at each size and weights across sizes.

**Original operator, tile, grid, and application experiments.** The full command
runs nine heterogeneous operator workloads, tile sides 8/16/32, grid counts
1/7/14/28/56/112/224/448, six genuine RAW crop sizes, then all 1,204 quality pairs:

```bash
python benchmark.py --data data/zrr_test_samples \
  --raw data/zrr_test_samples/full_resolution/huawei_raw/1050.png \
  --sizes 128 256 384 448 640 896 --reps 60 --warmup 5 --test-count 1204 \
  --output-dir ../pynet_full_reproduction
```

The tile/grid workload is N=1, channels 4→32, input 224×224, filter 3×3,
stride 1, padding 1. Only the named launch parameter changes. Grid counts
are chosen around the original GPU's 28 SMs; each gridstride block has 256
threads. Counts remain fixed on other GPUs. They do not prescribe active SMs
or measure occupancy.

**Quality only (resumable).**

```bash
python benchmark.py --quality-only --test-count 1204 \
  --data data/zrr_test_samples --output-dir ../pynet_quality_reproduction
```

For a short inspection of the known exception, use `--ids 747` instead of
`--test-count 1204`, with a fresh output directory. The quality evaluator
records tolerance failures and continues evaluating; they are not crashes.
Quality is measured against DSLR targets using PSNR, SSIM, MS-SSIM, MAE,
RMSE, and CIEDE2000; agreement separately compares custom outputs with cuDNN.

Progress is saved every 25 images and baseline outputs are cached in
`.quality_cache/`. Resume with the same output directory, data path, checkpoint,
IDs, and GPU/software environment. Use a fresh directory if these change.
A full run writes `evaluation.json`, `full_quality.json`,
`resolution_summary.csv`, and `quality_all_methods.csv`. Quality-only writes
the quality JSON and CSV. CSVs are generated on execution and are not submitted.
Report text and figures are fixed artifacts and are not regenerated automatically.

## 5. Repeated Processes, Statistics, and GPU Traces

**Run the published repeatability protocol.** Requires weights and the genuine
full-resolution RAW image. Run without another GPU workload:

```bash
python benchmark.py --repeatability --processes 12 \
  --raw data/zrr_test_samples/full_resolution/huawei_raw/1050.png \
  --output-dir ../pynet_repeatability_reproduction
```

This starts twelve sequential fresh Python processes. Each measures four
operator shapes (C,H: 4,32; 4,64; 16,32; 4,320; K=32, filter 3×3), six selected
tile/grid settings, and PyNET at RAW 128 and 448. Inputs/weights are identical
across processes; case order and per-round method order are randomized.
Operators use 25 warm-ups and 200 timings, scheduling 10/100, and networks
5/20. Small C=4 operators additionally use 100 ten-call intervals. A second
cuDNN label invokes the exact same callable as a measurement-variation control.

Outputs are `protocol.json`, `run_00.json` through `run_11.json`,
`performance_repeatability.json`, and `performance_repeatability_summary.json`.
Existing repeated evidence is rejected to avoid mixing runs. Choose a fresh
output directory after an interrupted run. `--processes 2` is useful for a
smoke check; use 12 for the reported protocol. Supported counts are 2–16.

**Reanalyze the supplied evidence without running the GPU experiments.**

```bash
mkdir -p ../pynet_statistics_reproduction
cp results/performance_repeatability.json ../pynet_statistics_reproduction/
python benchmark.py --analyze-repeatability --output-dir ../pynet_statistics_reproduction
```

The analysis uses paired **process medians**, not individual calls as independent
samples. It computes the median speedup, a seeded 20,000-resample 95% bootstrap
interval, and exact paired sign-flip tests on mean log speedup. The latter
assume symmetric paired effects under the null. Holm correction uses separate
families of 14 network and four selected operator comparisons. Tile/grid
intervals are descriptive. The ±1% band is a practical reference, not proof of
equivalence. The provided evidence reproduces the supplied summary exactly.

**Inspect the execution paths.** No weights or images required:

```bash
python benchmark.py --profile-paths --output-dir ../pynet_trace_reproduction
```

This warms each path 25 times, then records one invocation using
`torch.profiler`. It writes four Chrome traces under `traces/` and a
`performance_mechanisms.json` summary of kernel names/counts and CPU operators.
The cases are cuDNN/implicit at 32×32 and implicit/im2col_reg at 320×320,
all with channels 4→32 and filter 3×3. On the recorded system, kernel counts
are 2/1 and 1/2 respectively. Counts can change with software or device.
Profiler durations are diagnostic and are not substituted for benchmark timings.

## 6. Measurement Scope and Troubleshooting

All performance experiments use FP32 inference, disabled TF32, and cuDNN
autotuning. CUDA events surround full invocations; allocations, explicit
lowering, and per-call constant-weight uploads are included. Inputs and weights
are already on the GPU. Compilation, loading, decoding, transfers, and metric
computation are excluded. Event intervals can contain GPU idle gaps between
host dispatches; wall time is recorded separately. Clocks are unlocked.
Full-test quality instead uses deterministic cuDNN with autotuning disabled.

The baseline is the tested PyTorch/cuDNN convolution-with-bias path, not every
cuDNN algorithm or fused API. PyNET substitutes only
`conv_l1_d1.conv_3a.conv2d` (4→32) and `conv_l0_d1.conv2d` (16→3), both 3×3.
Their 0.042% share of convolution MACs is not a measured runtime fraction.
The study concerns one GPU and specified workloads, not scaling across GPUs.

`const` supports only the default stream and at most 16,384 FP32 weights
(64 KiB). Other methods also receive non-default-stream checks. The binding
supports dense convolution with bias, batch, stride, and zero padding. It does
not implement a custom backward pass.

For build failures, inspect the PyTorch extension build log and verify `nvcc`,
`CUDA_HOME`, the host compiler, and supported GPU architecture. For memory
limits, reduce the RAW sizes or test only operators. If profiling captures no
kernel events, check the profiler/CUPTI installation. Use a fresh output
directory when a result's protocol or environment differs from the current run.

## 7. Extension Experiments

`extensions/` holds two follow-up kernels (`tiled_mk`, `implicit_reg`) and the experiments that
check them. They build as separate extensions, so the seven published variants and the evidence
above are unchanged; new evidence is written to `results/extensions/`. Details are in
[extensions/README.md](extensions/README.md) and results in
[results/extensions/FINDINGS.md](results/extensions/FINDINGS.md).

```bash
python extensions/compare_variants.py --reps 200 --warmup 25   # correctness + one session
python extensions/audit_cudnn.py                               # cuDNN settings and host overhead
python extensions/repeatability.py --processes 12              # fresh-process statistics
python extensions/real_data.py --count 200                     # needs Section 2 weights and data
python extensions/plot_results.py                              # needs matplotlib
```

These were recorded on an RTX 5070 Ti (sm_120) with PyTorch 2.11.0+cu130, CUDA 13.0, and
cuDNN 9.19, not the RTX 3060 of the published study; set `TORCH_CUDA_ARCH_LIST` for your GPU.
They add a second timing metric, CUDA-graph replay, which excludes host dispatch.

## 8. Upstream attribution

`src/pynet.py` is the unmodified content of upstream `model.py`, copyright 2020
Andrey Ignatov, from [PyNET-PyTorch](https://github.com/aiff22/PyNET-PyTorch).
`src/msssim.py` is copied unchanged from that distribution and retains its
references to [pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim)
and the [SSIM research page](https://ece.uwaterloo.ca/~z70wang/research/ssim/).
The upstream distribution is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Upstream code is provided as-is, without warranties; the linked license contains
the full terms and disclaimer. References for algorithmic techniques and metrics
appear next to method names and in the report bibliography.
