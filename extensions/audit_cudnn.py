"""Audit the cuDNN baseline settings and separate host overhead from GPU execution time.

Prints the precision and autotuning settings in force, then for each Table 2 workload and method:
  event   - CUDA events around one eager call (includes host dispatch while the GPU waits);
  graph   - per-call time of 20 calls captured in a CUDA graph and replayed (GPU execution only);
  kernels - kernels launched per call, their summed duration, and their names (torch.profiler).
The kernel names confirm the cuDNN path runs FP32 (float, float) kernels plus a separate bias add.
This is one process; repeatability.py gives process-level statistics for event and graph times.

    python extensions/audit_cudnn.py

Writes audit.json to --output-dir (default results/extensions/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_OUTPUT, WORKLOADS, environment, graph_time,  # noqa: E402
                    load_extensions, set_timing_flags, workload_label)
from evaluation import measure  # noqa: E402


def kernel_profile(fn, calls: int = 10):
    """Kernels per call, summed kernel time per call (ms), and distinct kernel names."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    total_us = sum(e.time_range.elapsed_us() for e in kernels)
    return len(kernels) / calls, total_us / calls / 1000.0, sorted({e.name for e in kernels})


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    set_timing_flags()
    settings = {
        **environment(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "inference_mode": torch.is_inference_mode_enabled(),
    }
    print("Settings")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    ext, mk, ir = load_extensions()
    torch.manual_seed(42)
    rows = []
    for workload in WORKLOADS:
        n, c, h, w_dim, k, r = workload
        x = torch.randn(n, c, h, w_dim, device="cuda")
        w = torch.randn(k, c, r, r, device="cuda") * 0.03
        b = torch.randn(k, device="cuda")
        pad = r // 2
        functions = {
            "cudnn": lambda: F.conv2d(x, w, b, padding=pad),
            "cudnn_nobias": lambda: F.conv2d(x, w, None, padding=pad),
            "implicit": lambda: ir.implicit_reg(x, w, b, 1, pad, 2),
            "implicit_auto": lambda: ir.implicit_reg(x, w, b, 1, pad, 0),
            "tiled_mk4": lambda: mk.tiled_mk(x, w, b, 1, pad, 16, 4),
            "tiled_mk16": lambda: mk.tiled_mk(x, w, b, 1, pad, 16, 16),
        }
        assert functions["cudnn"]().dtype == torch.float32
        event = measure(functions, 200, 25)

        print(f"\n{workload_label(workload)}")
        print(f"  {'method':<14}{'event ms':>10}{'graph ms':>10}{'kernel ms':>11}{'kernels':>9}{'event/graph':>13}")
        methods = {}
        for name, fn in functions.items():
            g = graph_time(fn)
            count, kms, names = kernel_profile(fn)
            e = event[name]["cuda_event"]["median_ms"]
            methods[name] = {"event_ms": e, "graph_ms": g, "kernel_ms": kms, "kernels_per_call": count,
                             "kernel_names": names}
            print(f"  {name:<14}{e:>10.4f}{g:>10.4f}{kms:>11.4f}{count:>9.1f}{e / g:>12.2f}x")
        rows.append({"workload": workload_label(workload), "methods": methods})

    cudnn_names = sorted({nm for row in rows for m in ("cudnn", "cudnn_nobias")
                          for nm in row["methods"][m]["kernel_names"]})
    print("\nKernels launched by the cuDNN path:")
    for name in cudnn_names:
        print("  ", name[:150])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "audit.json"
    path.write_text(json.dumps({"settings": settings, "cudnn_kernels": cudnn_names, "workloads": rows}, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
