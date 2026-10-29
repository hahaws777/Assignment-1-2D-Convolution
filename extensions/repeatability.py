"""Repeated fresh-process comparison of the extension kernels against cuDNN and their originals.

Follows the report's repeatability protocol (Section 3.2) on the nine Table 2 workloads:
fresh Python processes, identical seeded inputs, randomized workload and method order,
25 warm-ups and 200 timed calls per method, and a second cuDNN label invoking the exact
same callable as a measurement-variation control. Every process records two metrics:

  event - CUDA events around one eager call (the report's invocation-path metric; includes
          host dispatch time while the GPU waits for the first kernel);
  graph - per-call time of 20 calls captured in a CUDA graph and replayed (GPU execution only).

Inference uses paired process medians (`evaluation.compare_processes`) with Holm correction
within two families per metric: every method vs cuDNN, and each extension kernel vs its original.

    python extensions/repeatability.py --processes 12

Writes run_XX.json and summary.json to --output-dir (default results/extensions/repeatability/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_OUTPUT, WORKLOADS, environment, graph_time,  # noqa: E402
                    load_extensions, set_timing_flags, workload_label)
from benchmark import repeated_timing  # noqa: E402
from evaluation import compare_processes, holm  # noqa: E402

METRICS = ("event", "graph")
VS_CUDNN = ["cudnn_control", "tiled", "implicit", "implicit_auto", "tiled_mk4", "tiled_mk8", "tiled_mk16"]
VS_ORIGINAL = [("implicit_auto", "implicit"), ("tiled_mk4", "tiled"), ("tiled_mk8", "tiled"),
               ("tiled_mk16", "tiled")]


@torch.inference_mode()
def worker(index: int, output_dir: Path) -> None:
    """Time every workload in this fresh process and write process medians for both metrics."""
    set_timing_flags()
    ext, mk, ir = load_extensions()
    rng = np.random.default_rng(1000 + index)

    medians = {}
    for wid in rng.permutation(len(WORKLOADS)).tolist():
        n, c, h, w_dim, k, r = WORKLOADS[wid]
        gen = torch.Generator().manual_seed(4200 + wid)
        x = torch.randn(n, c, h, w_dim, generator=gen).cuda()
        w = (torch.randn(k, c, r, r, generator=gen) * 0.03).cuda()
        b = torch.randn(k, generator=gen).cuda()
        pad = r // 2

        def conv():
            return F.conv2d(x, w, b, padding=pad)

        functions = {
            "cudnn": conv,
            "cudnn_control": conv,
            "tiled": lambda: ext.convolution(x, w, b, "tiled", 1, pad, 16, 0),
            "implicit": lambda: ir.implicit_reg(x, w, b, 1, pad, 2),
            "implicit_auto": lambda: ir.implicit_reg(x, w, b, 1, pad, 0),
        }
        for kb in (4, 8, 16):
            functions[f"tiled_mk{kb}"] = lambda kb=kb: mk.tiled_mk(x, w, b, 1, pad, 16, kb)

        timing = repeated_timing(functions, 200, 25, seed=10000 + 100 * index + wid)
        graphs = {name: graph_time(functions[name]) for name in rng.permutation(list(functions)).tolist()}
        medians[workload_label(WORKLOADS[wid])] = {
            "event": {m: t["cuda_event"]["median_ms"] for m, t in timing.items()},
            "graph": graphs,
        }

    (output_dir / f"run_{index:02d}.json").write_text(json.dumps({
        "environment": environment(), "medians": medians}, indent=2))


def analyze(output_dir: Path, processes: int) -> dict:
    """Paired process-median comparisons, Holm-corrected within each declared family."""
    runs = [json.loads((output_dir / f"run_{i:02d}.json").read_text()) for i in range(processes)]
    summary = {"processes": processes, "environment": runs[0]["environment"]}
    for metric in METRICS:
        vs_cudnn, vs_original = [], []
        for workload in WORKLOADS:
            name = workload_label(workload)

            def series(method):
                return [run["medians"][name][metric][method] for run in runs]

            for method in VS_CUDNN:
                row = compare_processes(series("cudnn"), series(method))
                row.update({"workload": name, "method": method, "reference": "cudnn"})
                vs_cudnn.append(row)
            for method, original in VS_ORIGINAL:
                row = compare_processes(series(original), series(method))
                row.update({"workload": name, "method": method, "reference": original})
                vs_original.append(row)
        holm([row for row in vs_cudnn if row["method"] != "cudnn_control"])
        holm([row for row in vs_cudnn if row["method"] == "cudnn_control"])
        holm(vs_original)
        summary[metric] = {"vs_cudnn": vs_cudnn, "vs_original": vs_original}
    return summary


def print_tables(summary: dict) -> None:
    for metric in METRICS:
        rows = {(r["workload"], r["method"]): r for r in summary[metric]["vs_cudnn"]}
        print(f"\n[{metric}] speedup vs cuDNN")
        print(f"{'workload':<24}{'cudnn ms':>10}" + "".join(f"{m:>16}" for m in VS_CUDNN))
        for workload in WORKLOADS:
            name = workload_label(workload)
            cells = []
            for method in VS_CUDNN:
                row = rows[(name, method)]
                mark = "+" if row["supported_faster"] else "-" if row["supported_slower"] else " "
                cells.append(f"{row['speedup']:>9.2f}x {row['runs_faster']:>2}/{row['runs']}{mark}")
            print(f"{name:<24}{rows[(name, VS_CUDNN[0])]['reference_median_ms']:>10.4f}"
                  + "".join(f"{c:>16}" for c in cells))
    print("\nSpeedup = reference process median / method process median (median over processes); "
          "n/N = processes where the method was faster.\n"
          "+ / - = supported faster / slower (Holm p < 0.05 and 95% interval excludes 1).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "repeatability")
    parser.add_argument("--processes", type=int, default=12, help="Fresh processes (2..16; report: 12)")
    parser.add_argument("--analyze-only", action="store_true", help="Recompute summary.json from run files")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        worker(args.worker, args.output_dir)
        return
    if not 2 <= args.processes <= 16:
        parser.error("--processes must be in [2, 16]")
    if not args.analyze_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if any(args.output_dir.glob("run_*.json")):
            parser.error("Output directory already holds run files; choose a fresh one or use --analyze-only")
        load_extensions()  # compile once before timing processes start
        for index in range(args.processes):
            print(f"process {index + 1}/{args.processes}", flush=True)
            subprocess.run([sys.executable, __file__, "--worker", str(index), "--output-dir", str(args.output_dir)],
                           check=True)

    summary = analyze(args.output_dir, args.processes)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_tables(summary)


if __name__ == "__main__":
    main()
