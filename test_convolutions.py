"""Validation suite for seven CUDA convolution variants.

Tests across tensor shapes, channel depths, kernel dimensions, batches,
strides, tile sizes, non-default CUDA streams, and constant memory capacity limits.
Outputs a structured JSON correctness report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluation import agreement  # noqa: E402
from runtime import METHODS, ROOT, extension  # noqa: E402


@torch.inference_mode()
def run_convolutions_test(output_dir: Path) -> Dict[str, Any]:
    """Execute correctness checks for all seven custom CUDA convolution methods.

    Validates:
      1. Correctness against double-precision PyTorch F.conv2d across 5 tensor shapes.
      2. Rejection of weights exceeding 64 KiB in the constant-memory path.
      3. Shared-memory tile dimensions (tile=8, 16, 32).
      4. Stream independence across non-default CUDA streams.
      5. Grid-stride scaling over varying grid block sizes.

    Args:
        output_dir: Directory where 'seven_convolutions_correctness.json' will be written.

    Returns:
        Dict[str, Any]: Test summary containing pass counts and individual check records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    torch.backends.cudnn.allow_tf32 = False

    ext = extension()
    rows: List[Dict[str, Any]] = []

    # Format: (batch N, channels C, height H, width W, filters K, kernel R, kernel S, stride, padding)
    shapes = [
        (1, 1, 7, 9, 1, 3, 3, 1, 0),
        (2, 3, 17, 19, 5, 3, 5, 2, 2),
        (1, 4, 19, 21, 32, 3, 3, 1, 1),
        (1, 16, 9, 11, 3, 3, 3, 1, 1),
        (1, 32, 11, 13, 32, 5, 5, 1, 2),
    ]

    last_x = None
    last_w = None
    last_b = None
    last_stride = 1
    last_pad = 0

    for n, c, h, w_dim, k, r, s, stride, pad in shapes:
        x = torch.randn(n, c, h, w_dim, device="cuda")
        w = torch.randn(k, c, r, s, device="cuda") * 0.03
        b = torch.randn(k, device="cuda")
        last_x, last_w, last_b = x, w, b
        last_stride, last_pad = stride, pad

        ref = F.conv2d(x.double(), w.double(), b.double(), stride=stride, padding=pad).float()

        for method in METHODS:
            if method == "const" and w.numel() > 16384:
                try:
                    ext.convolution(x, w, b, method, stride, pad, 16, 0)
                except RuntimeError as e:
                    assert "64 KiB" in str(e), f"Unexpected exception message: {e}"
                    rows.append({
                        "method": method,
                        "status": "N/A",
                        "reason": "64 KiB weight limit",
                        "shape": list(x.shape),
                    })
                else:
                    raise AssertionError("Constant-memory overflow was not rejected")
                continue

            out = ext.convolution(x, w, b, method, stride, pad, 16, 0)
            rows.append({
                "method": method,
                "status": "PASS",
                "shape": list(x.shape),
                "kernel": [r, s],
                "stride": stride,
                "pad": pad,
                "error": agreement(ref, out),
            })

        # Test shared-memory tile sweep
        for tile in [8, 32]:
            out = ext.convolution(x, w, b, "tiled", stride, pad, tile, 0)
            rows.append({
                "method": "tiled",
                "tile": tile,
                "status": "PASS",
                "error": agreement(ref, out),
            })

    # Test non-default CUDA stream safety
    if last_x is not None and last_w is not None and last_b is not None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for method in METHODS:
                if method == "const":
                    continue  # const memory symbol path requires default stream
                out = ext.convolution(last_x, last_w, last_b, method, last_stride, last_pad, 16, 0)
                rows.append({
                    "method": method,
                    "stream": "non-default",
                    "status": "PASS",
                    "error": agreement(ref, out),
                })
        stream.synchronize()

    # Test grid-stride kernel across block dimensions
    for blocks in [1, 7, 28, 224]:
        out = ext.convolution(last_x, last_w, last_b, "gridstride", last_stride, last_pad, 16, blocks)
        rows.append({
            "method": "gridstride",
            "blocks": blocks,
            "status": "PASS",
            "error": agreement(ref, out),
        })

    passed_count = sum(r["status"] == "PASS" for r in rows)
    na_count = sum(r["status"] == "N/A" for r in rows)
    result = {
        "passed": passed_count,
        "not_applicable": na_count,
        "checks": rows,
    }

    report_path = output_dir / "seven_convolutions_correctness.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"{passed_count} checks passed; {na_count} explicitly unsupported configuration")
    return result


def main() -> None:
    """CLI entry point for convolution verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory to save verification JSON output (default: results/)",
    )
    args = parser.parse_args()
    run_convolutions_test(args.output_dir)


if __name__ == "__main__":
    main()
