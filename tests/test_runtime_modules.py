"""Unit tests for PyTorch runtime integration, module substitution, and MAC calculations.

Tests PyTorch-level graph grafting and module contracts.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root and src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import runtime
    from runtime import CustomConv, TARGETS, replace_convolutions
    from benchmark import macs
    import benchmark
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False


@unittest.skipUnless(HAS_TORCH and HAS_RUNTIME, "PyTorch required for runtime module tests")
class TestRuntimeModules(unittest.TestCase):
    """Test suite for CustomConv wrapper and PyNET layer substitution logic."""

    def test_custom_conv_parameter_validation(self) -> None:
        """Verify that CustomConv rejects non-dense or grouped convolutions."""
        # Invalid groups != 1
        grouped_conv = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, groups=2)
        with self.assertRaises(ValueError):
            CustomConv(grouped_conv, method="naive")

        # Invalid dilation != 1
        dilated_conv = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, dilation=2)
        with self.assertRaises(ValueError):
            CustomConv(dilated_conv, method="naive")

        # Invalid padding != 0
        padded_conv = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        with self.assertRaises(ValueError):
            CustomConv(padded_conv, method="naive")

    def test_pynet_replacement_preserves_parameters_and_other_modules(self) -> None:
        """Replace the real PyNET target paths without allocating model weights."""
        with torch.device("meta"):
            model = runtime.PyNET(0, instance_norm=True, instance_norm_level_1=True)
        originals = dict(model.named_modules())
        extension_stub = object()

        # CUDA execution is covered by test_cuda_convolutions; this checks the graph.
        with patch.object(runtime, "extension", return_value=extension_stub):
            result = replace_convolutions(model, "tiled")

        self.assertIs(result, model)
        replaced = {name for name, layer in model.named_modules() if isinstance(layer, CustomConv)}
        self.assertEqual(replaced, set(TARGETS))
        for name, original in originals.items():
            current = model.get_submodule(name)
            if name in TARGETS:
                self.assertIs(current.weight, original.weight)
                self.assertIs(current.bias, original.bias)
                self.assertEqual(current.stride, original.stride[0])
                self.assertEqual(current.method, "tiled")
                self.assertIs(current.ext, extension_stub)
            else:
                self.assertIs(current, original)

    def test_extension_is_loaded_once_per_process(self) -> None:
        """Repeated extension access must reuse the loaded module."""
        extension_stub = object()
        with patch.object(runtime, "_EXTENSION_CACHE", None), patch.object(
            runtime, "load", return_value=extension_stub
        ) as loader:
            self.assertIs(runtime.extension(), extension_stub)
            self.assertIs(runtime.extension(), extension_stub)
            loader.assert_called_once()

    def test_experiment_modes_are_exclusive(self) -> None:
        """Conflicting modes must fail before any CUDA or dataset work."""
        with patch.object(sys, "argv", ["benchmark.py", "--quality-only", "--profile-paths"]), \
                patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            benchmark.main()
        self.assertEqual(error.exception.code, 2)

    def test_repeatability_cli_routes_arguments(self) -> None:
        """The parent entry point must receive the requested input, output, and count."""
        with patch.object(sys, "argv", ["benchmark.py", "--repeatability", "--processes", "2",
                                      "--raw", "sensor.png", "--output-dir", "fresh"]), \
                patch.object(benchmark, "repeatability") as run:
            benchmark.main()
        run.assert_called_once_with(Path("sensor.png"), Path("fresh"), 2)

    def test_macs_calculation_hook(self) -> None:
        """Verify that macs() forward hook calculates exact MAC operations."""
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Conv 1: 4 in, 8 out, 3x3 kernel
                self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=0)
                # Conv 2: 8 in, 16 out, 1x1 kernel
                self.conv2 = nn.Conv2d(8, 16, kernel_size=1, padding=0)

            def forward(self, x):
                return self.conv2(self.conv1(x))

        model = ToyModel()
        # Input tensor: batch 1, 4 channels, 10x10 spatial
        x = torch.randn(1, 4, 10, 10)
        
        # After conv1 (k=3, pad=0, stride=1): output spatial is 8x8.
        # MACs conv1 = out_numel * in_c * k_h * k_w = (1 * 8 * 8 * 8) * 4 * 3 * 3 = 512 * 36 = 18432
        # After conv2 (k=1, pad=0, stride=1): output spatial is 8x8.
        # MACs conv2 = (1 * 16 * 8 * 8) * 8 * 1 * 1 = 1024 * 8 = 8192
        # Total MACs = 18432 + 8192 = 26624
        
        stats = macs(model, x)
        self.assertEqual(stats["all_conv_macs"], 26624)
        self.assertEqual(stats["selected_conv_macs"], 0)


if __name__ == "__main__":
    unittest.main()
