"""Shared definitions for the extension experiments.

The two extension kernels (`tiled_mk.cu`, `implicit_reg.cu`) are built as separate PyTorch
extensions, so the published seven-variant build in src/convolution.cu is never replaced.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from runtime import extension  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results/extensions"

# Table 2 / benchmark.py operator workloads: (N, C, H, W, K, filter side)
WORKLOADS = [
    (1, 4, 64, 64, 32, 3),
    (1, 4, 224, 224, 32, 3),
    (1, 16, 448, 448, 3, 3),
    (1, 16, 32, 32, 32, 3),
    (1, 16, 32, 32, 32, 5),
    (1, 16, 32, 32, 32, 7),
    (1, 16, 32, 32, 32, 9),
    (2, 4, 64, 64, 32, 3),
    (4, 4, 64, 64, 32, 3),
]

GRAPH_CALLS = 20
GRAPH_REPLAYS = 100


def workload_label(workload) -> str:
    n, c, h, w_dim, k, r = workload
    return f"{n}x{c}x{h}x{w_dim} K{k} {r}x{r}"


def set_timing_flags() -> None:
    """Performance settings used by benchmark.py: FP32, TF32 off, cuDNN autotuning on."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_extensions() -> Tuple[Any, Any, Any]:
    """Return (published seven-variant extension, tiled_mk extension, implicit_reg extension)."""
    def build(name: str, source: str) -> Any:
        return load(name=name, sources=[str(ROOT / "extensions" / source)],
                    extra_cuda_cflags=["-O3"], verbose=False)

    return (extension(),
            build("pynet_ext_tiled_mk", "tiled_mk.cu"),
            build("pynet_ext_implicit_reg", "implicit_reg.cu"))


def environment() -> dict:
    prop = torch.cuda.get_device_properties(0)
    return {"gpu": prop.name, "sm_count": prop.multi_processor_count, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "tf32": torch.backends.cudnn.allow_tf32}


def graph_time(fn: Callable[[], Any], calls: int = GRAPH_CALLS, replays: int = GRAPH_REPLAYS) -> float:
    """Median per-call time (ms) of `calls` calls captured in one CUDA graph and replayed.

    Replay launches the recorded kernels back to back with no Python or host dispatch,
    so this measures GPU execution time, unlike CUDA events around a single eager call.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(5):
        graph.replay()
    times = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / calls)
    return float(np.median(times))
