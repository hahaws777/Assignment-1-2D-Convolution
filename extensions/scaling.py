"""Strong and weak SM scaling of the published methods against cuDNN.

SM count is the resource axis, enforced with CUDA green contexts so cuDNN and the custom
kernels are confined to the same SMs. Each SM count runs in a fresh process, so cuDNN
autotunes under that SM count. A probe kernel records which SMs actually execute work.

  strong: fixed problems (1x4x224x224 and 1x4x896x896, 4->32, 3x3), SMs varied.
  weak:   fixed work per SM, input 1x4x(32*SMs)x224, 4->32, 3x3.

`const` is excluded: it requires the default stream, and green-context work runs on the
green context's own stream. Timing is CUDA events around 10 back-to-back calls (1 call when
a call exceeds 1 ms), so host dispatch is largely hidden behind GPU work.

    python extensions/scaling.py --output-dir ../conv2d_scaling
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from runtime import METHODS, extension  # noqa: E402

# Green contexts on the RTX 5070 Ti allocate at least 8 SMs; requests of 1, 2 and 4 measured 8
SM_COUNTS = [8, 16, 24, 32, 48, 64, 70]
STRONG_SIZES = [224, 896]
WEAK_ROWS_PER_SM = 32
WEAK_WIDTH = 224
SCALED = [m for m in METHODS if m != "const"]
REPS = 15

PROBE = r"""
#include <ATen/cuda/CUDAContext.h>
__global__ void probe(int* flags) {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    if (threadIdx.x == 0 && s < 1024) flags[s] = 1;
    long long t0 = clock64();
    while (clock64() - t0 < 200000) {}
}
torch::Tensor run(int blocks) {
    auto flags = torch::zeros({1024}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    probe<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(flags.data_ptr<int>());
    return flags;
}
"""


def time_ms(fn) -> float:
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    inner = 10 if start.elapsed_time(end) < 1.0 else 1
    samples = []
    for _ in range(REPS):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return statistics.median(samples)


def run_shape(ext, h: int, w_dim: int) -> dict:
    x = torch.randn(1, 4, h, w_dim, device="cuda")
    w = torch.randn(32, 4, 3, 3, device="cuda") * 0.03
    b = torch.randn(32, device="cuda")
    functions = {"cudnn": lambda: F.conv2d(x, w, b, padding=1)}
    functions.update({m: (lambda m=m: ext.convolution(x, w, b, m, 1, 1, 16, 0)) for m in SCALED})
    ref = functions["cudnn"]()
    row = {"shape": [1, 4, h, w_dim], "ms": {}, "passed": {}}
    for name, fn in functions.items():
        try:
            # Same tolerance as evaluation.agreement, whose quantile() rejects >16M-element outputs
            torch.testing.assert_close(fn(), ref, rtol=1e-4, atol=2e-5)
            row["passed"][name] = True
        except AssertionError:
            row["passed"][name] = False
        row["ms"][name] = time_ms(fn)
    return row


def worker(sms: int) -> dict:
    from torch.cuda.green_contexts import GreenContext
    from torch.utils.cpp_extension import load_inline

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = extension()
    probe = load_inline("sm_probe", cpp_sources="torch::Tensor run(int blocks);", cuda_sources=PROBE,
                        functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)

    ctx = GreenContext.create(sms, 0)
    ctx.set_context()
    with torch.cuda.stream(ctx.Stream()):
        actual = int(probe.run(4096).sum())
        result = {"requested_sms": sms, "actual_sms": actual,
                  "strong": [run_shape(ext, s, s) for s in STRONG_SIZES],
                  "weak": run_shape(ext, WEAK_ROWS_PER_SM * actual, WEAK_WIDTH)}
        torch.cuda.synchronize()
    ctx.pop_context()
    return result


def plot(rows: list, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    names = ["cudnn"] + SCALED
    sms = [r["actual_sms"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [(f"Strong scaling, 1x4x{s}x{s}", [r["strong"][i]["ms"] for r in rows])
              for i, s in enumerate(STRONG_SIZES)]
    panels.append((f"Weak scaling, 1x4x({WEAK_ROWS_PER_SM}*SMs)x{WEAK_WIDTH}", [r["weak"]["ms"] for r in rows]))
    for ax, (title, series) in zip(axes, panels):
        for name, color, marker in zip(names, colors, markers):
            ax.plot(sms, [s[name] for s in series], color=color, marker=marker, markersize=6,
                    linewidth=2 if name == "cudnn" else 1.5, label=name)
        if title.startswith("Strong"):
            base = series[0]["cudnn"]
            ax.plot(sms, [base * sms[0] / n for n in sms], color="#8a8a85", linestyle="--",
                    linewidth=1, label="ideal (cudnn)")
            ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sms, ["" if n == 64 else str(n) for n in sms])
        ax.set_xlabel("SMs available (measured)")
        ax.set_ylabel("median ms per call")
        ax.set_title(title, fontsize=11)
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling.png", dpi=150)


def summarize(rows: list) -> str:
    names = ["cudnn"] + SCALED
    first, last = rows[0], rows[-1]
    n0, n1 = first["actual_sms"], last["actual_sms"]
    lines = [f"| method | " + " | ".join(f"strong {s} speedup {n0}->{n1} SMs (eff.)" for s in STRONG_SIZES)
             + f" | weak eff. {n1} SMs | {n1}-SM ms {STRONG_SIZES[-1]} / cudnn |",
             "|---" * (3 + len(STRONG_SIZES)) + "|"]
    for name in names:
        cells = []
        for i in range(len(STRONG_SIZES)):
            sp = first["strong"][i]["ms"][name] / last["strong"][i]["ms"][name]
            cells.append(f"{sp:.1f}x ({sp / (n1 / n0):.0%})")
        weak = first["weak"]["ms"][name] / last["weak"]["ms"][name]
        ratio = last["strong"][-1]["ms"][name] / last["strong"][-1]["ms"]["cudnn"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {weak:.0%} | {ratio:.2f}x |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sms in SM_COUNTS:
        proc = subprocess.run([sys.executable, __file__, "--output-dir", str(args.output_dir), "--worker", str(sms)],
                              capture_output=True, text=True)
        if proc.returncode:
            print(f"SMs={sms}: failed\n{proc.stderr[-1500:]}", flush=True)
            continue
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"SMs={sms} (measured {row['actual_sms']}): strong {STRONG_SIZES[-1]} "
              + str({k: round(v, 3) for k, v in row["strong"][-1]["ms"].items()}), flush=True)
    prop = torch.cuda.get_device_properties(0)
    (args.output_dir / "scaling.json").write_text(json.dumps(
        {"gpu": prop.name, "sm_count": prop.multi_processor_count, "torch": torch.__version__,
         "reps": REPS, "rows": rows}, indent=2))
    plot(rows, args.output_dir)
    table = summarize(rows)
    (args.output_dir / "scaling_summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
