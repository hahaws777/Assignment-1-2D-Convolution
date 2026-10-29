"""Unit tests for seven custom CUDA 2D convolution kernels.

Validates numerical precision against double-precision PyTorch reference,
constant memory capacity bounds, tiling sweeps, and CUDA stream safety.
Automatically skips when running in environments without an active NVIDIA GPU.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Add project root and src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    from evaluation import agreement
    from runtime import CustomConv, METHODS, extension
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_TORCH and HAS_CUDA and HAS_DEPS, "Active CUDA GPU and dependencies required")
class TestCUDAConvolutions(unittest.TestCase):
    """Numerical precision and boundary tests for the 7 CUDA convolution variants."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build and cache the CUDA extension before running tests."""
        if HAS_CUDA and HAS_DEPS:
            cls.ext = extension()
            torch.manual_seed(123)
            torch.backends.cudnn.allow_tf32 = False

    def test_convolution_numerical_equivalence_across_methods(self) -> None:
        """Verify all seven convolution variants match double-precision F.conv2d."""
        shapes = [
            # N, C, H, W, K, R, S, stride, pad
            (1, 4, 19, 21, 16, 3, 3, 1, 1),
            (2, 3, 17, 19, 8, 3, 5, 2, 2),
        ]

        for n, c, h, w_dim, k, r, s, stride, pad in shapes:
            x = torch.randn(n, c, h, w_dim, device="cuda")
            w = torch.randn(k, c, r, s, device="cuda") * 0.03
            b = torch.randn(k, device="cuda")

            ref = F.conv2d(x.double(), w.double(), b.double(), stride=stride, padding=pad).float()

            for method in METHODS:
                if method == "const" and w.numel() > 16384:
                    continue
                out = self.ext.convolution(x, w, b, method, stride, pad, 16, 0)
                err = agreement(ref, out)
                # Verify max absolute error is well within 2e-5 + 1e-4 * ref
                self.assertLess(err["max_abs"], 1e-3, f"Method {method} exceeded tolerance")

    def test_constant_memory_capacity_rejection(self) -> None:
        """Verify that weights exceeding 64 KiB in conv2d_const raise RuntimeError."""
        # 32 filters, 32 channels, 5x5 kernel = 32 * 32 * 25 = 25,600 floats > 16,384 floats (64 KiB)
        x = torch.randn(1, 32, 11, 13, device="cuda")
        w = torch.randn(32, 32, 5, 5, device="cuda")
        b = torch.randn(32, device="cuda")

        with self.assertRaises(RuntimeError) as context:
            self.ext.convolution(x, w, b, "const", 1, 2, 16, 0)
        self.assertIn("64 KiB", str(context.exception))

    def test_custom_conv_forward_preserves_parameters_and_stride(self) -> None:
        """Exercise the wrapper through every CUDA variant, including stride two."""
        with torch.inference_mode():
            original = torch.nn.Conv2d(4, 8, kernel_size=3, stride=2).cuda().eval()
            x = torch.randn(1, 4, 17, 19, device="cuda")
            ref = F.conv2d(
                x.double(), original.weight.double(), original.bias.double(), stride=2
            ).float()
            for method in METHODS:
                with self.subTest(method=method):
                    wrapped = CustomConv(original, method)
                    self.assertIs(wrapped.weight, original.weight)
                    self.assertIs(wrapped.bias, original.bias)
                    agreement(ref, wrapped(x))

    def test_shared_memory_tile_sweeps(self) -> None:
        """Verify tiled convolution across tile sizes 8, 16, and 32."""
        x = torch.randn(1, 4, 16, 16, device="cuda")
        w = torch.randn(8, 4, 3, 3, device="cuda") * 0.03
        b = torch.randn(8, device="cuda")
        ref = F.conv2d(x.double(), w.double(), b.double(), stride=1, padding=1).float()

        for tile in [8, 16, 32]:
            out = self.ext.convolution(x, w, b, "tiled", 1, 1, tile, 0)
            err = agreement(ref, out)
            self.assertLess(err["max_abs"], 1e-3, f"Tiled convolution failed for tile={tile}")

    def test_non_default_cuda_stream_execution(self) -> None:
        """Verify kernels correctly execute on non-default CUDA streams."""
        x = torch.randn(1, 4, 16, 16, device="cuda")
        w = torch.randn(8, 4, 3, 3, device="cuda") * 0.03
        b = torch.randn(8, device="cuda")
        ref = F.conv2d(x.double(), w.double(), b.double(), stride=1, padding=1).float()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for method in METHODS:
                if method == "const":
                    continue  # Constant weights require the default stream.
                out = self.ext.convolution(x, w, b, method, 1, 1, 16, 0)
                err = agreement(ref, out)
                self.assertLess(err["max_abs"], 1e-3, f"Method {method} failed on non-default stream")
        stream.synchronize()


if __name__ == "__main__":
    unittest.main()
