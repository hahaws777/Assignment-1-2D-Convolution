"""Single-session comparison of the extension kernels with the published variants and cuDNN.

Step 1 checks `tiled_mk` and `implicit_reg` against float64 PyTorch on the shapes from
test_convolutions.py (plus K=20 and K=64), for every tile/KB and implicit_reg mode, and counts
outputs bitwise equal to the original kernel each one derives from.
Step 2 times the nine Table 2 workloads with the benchmark.py protocol (seeded random method
order, CUDA events around each eager call, TF32 off, cuDNN autotune on). This is a single
session, like Table 2, and is not a repeatability claim; see repeatability.py.

    python extensions/compare_variants.py --reps 200 --warmup 25

Writes single_session.json to --output-dir (default results/extensions/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_OUTPUT, WORKLOADS, environment, load_extensions,  # noqa: E402
                    set_timing_flags, workload_label)
from evaluation import agreement, measure  # noqa: E402

KBS: Sequence[int] = (1, 4, 8, 16)
TILES: Sequence[int] = (8, 16, 32)
IMPLICIT_MODES = {"implicit_reg": 1, "implicit_auto": 0}

# test_convolutions.py shapes plus K=20 (partly empty last tile/group) and K=64 (largest
# implicit_reg configuration). Format: (N, C, H, W, K, R, S, stride, padding)
CHECK_SHAPES = [
    (1, 1, 7, 9, 1, 3, 3, 1, 0),
    (2, 3, 17, 19, 5, 3, 5, 2, 2),
    (1, 4, 19, 21, 32, 3, 3, 1, 1),
    (1, 16, 9, 11, 3, 3, 3, 1, 1),
    (1, 32, 11, 13, 32, 5, 5, 1, 2),
    (1, 4, 20, 23, 20, 3, 3, 1, 1),
    (2, 8, 15, 17, 64, 3, 3, 1, 1),
]


def check_correctness(ext: Any, mk: Any, ir: Any) -> Dict[str, int]:
    """Compare both extension kernels with float64 PyTorch for every shape and setting.

    Raises AssertionError (from `agreement`) on the first value outside tolerance.
    """
    torch.manual_seed(123)
    counts = {"tiled_mk_checks": 0, "tiled_mk_bitwise_equal_to_tiled": 0,
              "implicit_reg_checks": 0, "implicit_reg_bitwise_equal_to_implicit": 0}
    for n, c, h, w_dim, k, r, s, stride, pad in CHECK_SHAPES:
        x = torch.randn(n, c, h, w_dim, device="cuda")
        w = torch.randn(k, c, r, s, device="cuda") * 0.03
        b = torch.randn(k, device="cuda")
        ref = F.conv2d(x.double(), w.double(), b.double(), stride=stride, padding=pad).float()
        for tile in TILES:
            original = ext.convolution(x, w, b, "tiled", stride, pad, tile, 0)
            for kb in KBS:
                out = mk.tiled_mk(x, w, b, stride, pad, tile, kb)
                agreement(ref, out)
                counts["tiled_mk_checks"] += 1
                counts["tiled_mk_bitwise_equal_to_tiled"] += int(torch.equal(out, original))

        original = ir.implicit_reg(x, w, b, stride, pad, 2)
        for mode in (0, 1, 2):
            out = ir.implicit_reg(x, w, b, stride, pad, mode)
            agreement(ref, out)
            counts["implicit_reg_checks"] += 1
            counts["implicit_reg_bitwise_equal_to_implicit"] += int(torch.equal(out, original))

    # Non-default stream, as in the project's stream check
    x = torch.randn(1, 4, 19, 21, device="cuda")
    w = torch.randn(32, 4, 3, 3, device="cuda") * 0.03
    b = torch.randn(32, device="cuda")
    ref = F.conv2d(x.double(), w.double(), b.double(), padding=1).float()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        out_mk = mk.tiled_mk(x, w, b, 1, 1, 16, 8)
        out_ir = ir.implicit_reg(x, w, b, 1, 1, 1)
    side.synchronize()
    agreement(ref, out_mk)
    agreement(ref, out_ir)
    counts["tiled_mk_checks"] += 1
    counts["implicit_reg_checks"] += 1
    return counts


def compare_speed(ext: Any, mk: Any, ir: Any, reps: int, warmup: int, tile: int,
                  kbs: Sequence[int]) -> List[Dict[str, Any]]:
    """Time cuDNN, three published variants, and both extension kernels on the Table 2 workloads."""
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
            "tiled": lambda: ext.convolution(x, w, b, "tiled", 1, pad, tile, 0),
            "implicit": lambda: ir.implicit_reg(x, w, b, 1, pad, 2),
            "im2col_reg": lambda: ext.convolution(x, w, b, "im2col_reg", 1, pad, tile, 0),
        }
        for kb in kbs:
            functions[f"tiled_mk{kb}"] = lambda kb=kb: mk.tiled_mk(x, w, b, 1, pad, tile, kb)
        for name, mode in IMPLICIT_MODES.items():
            functions[name] = lambda mode=mode: ir.implicit_reg(x, w, b, 1, pad, mode)

        ref = functions["cudnn"]()
        for fn in functions.values():
            agreement(ref, fn())
        timing = measure(functions, reps, warmup)
        rows.append({
            "workload": workload_label(workload),
            "median_ms": {name: row["cuda_event"]["median_ms"] for name, row in timing.items()},
            "std_ms": {name: row["cuda_event"]["std_ms"] for name, row in timing.items()},
        })
    return rows


def print_table(rows: List[Dict[str, Any]], kbs: Sequence[int]) -> None:
    """Print median latency per method and each extension kernel's speedup over its original."""
    names = list(rows[0]["median_ms"])
    print(f"\n{'workload':<24}" + "".join(f"{name:>14}" for name in names)
          + f"{'mk/tiled':>10}{'reg/impl':>10}{'auto/impl':>11}")
    for row in rows:
        med = row["median_ms"]
        best_mk = min(med[f"tiled_mk{kb}"] for kb in kbs)
        print(
            f"{row['workload']:<24}"
            + "".join(f"{med[name]:>14.4f}" for name in names)
            + f"{med['tiled'] / best_mk:>9.2f}x"
            + f"{med['implicit'] / med['implicit_reg']:>9.2f}x"
            + f"{med['implicit'] / med['implicit_auto']:>10.2f}x"
        )
    print("\nMedian CUDA-event latency in ms (lower is better). Ratio columns: original time / extension "
          "time (above 1 = extension kernel faster); mk/tiled uses the fastest KB.")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=60, help="Timing repetitions (Table 2: 60)")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up iterations (Table 2: 5)")
    parser.add_argument("--tile", type=int, default=16, choices=TILES, help="Tile side for all tiled runs")
    parser.add_argument("--kbs", type=int, nargs="+", default=[4, 8, 16], choices=KBS,
                        help="Output channels per thread to time")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.reps < 5 or args.warmup < 1:
        parser.error("Use reps >= 5 and warmup >= 1")
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required")

    set_timing_flags()
    print("Building extensions (first run compiles; this can take a few minutes)...")
    ext, mk, ir = load_extensions()

    correctness = check_correctness(ext, mk, ir)
    print(f"Correctness vs float64 PyTorch: tiled_mk {correctness['tiled_mk_checks']} passed "
          f"({correctness['tiled_mk_bitwise_equal_to_tiled']} bitwise equal to tiled); "
          f"implicit_reg {correctness['implicit_reg_checks']} passed "
          f"({correctness['implicit_reg_bitwise_equal_to_implicit']} bitwise equal to implicit).")

    rows = compare_speed(ext, mk, ir, args.reps, args.warmup, args.tile, args.kbs)
    print_table(rows, args.kbs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "single_session.json"
    path.write_text(json.dumps({
        "environment": environment(),
        "protocol": {"reps": args.reps, "warmup": args.warmup, "tile": args.tile, "kbs": args.kbs,
                     "timing": "CUDA events around one eager call, seeded random method order"},
        "correctness": correctness,
        "operators": rows,
    }, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
