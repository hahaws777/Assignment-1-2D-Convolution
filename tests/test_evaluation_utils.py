"""Unit tests for evaluation metrics, Bayer sensor unpacking, and statistical utilities.

These tests run purely on CPU without requiring an NVIDIA GPU or CUDA runtime.
Gracefully skips if numpy or skimage are not yet installed in the current environment.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from evaluation import ci_ratio, pack, pack_bayer_mosaic, quality, summary, compare_processes, holm
    HAS_EVALUATION = True
except ImportError:
    HAS_EVALUATION = False


@unittest.skipUnless(HAS_NUMPY and HAS_EVALUATION, "numpy and skimage required for evaluation utils")
class TestEvaluationUtils(unittest.TestCase):
    """Test suite for numerical evaluation, statistical, and image processing helpers."""

    def test_bayer_packing_geometry_and_normalization(self) -> None:
        """Verify that pack() converts a 2D Bayer array to 4 channels with /1020 normalization."""
        raw = np.array([
            [1020,  510, 1020,  510],
            [ 255,    0,  255,    0],
            [1020,  510, 1020,  510],
            [ 255,    0,  255,    0],
        ], dtype=np.uint16)

        packed = pack(raw)
        self.assertEqual(packed.shape, (4, 2, 2))
        self.assertEqual(packed.dtype, np.float32)

        # RGGB spatial layout; packed order is [B, Gr, R, Gb]:
        # channel 0: raw[1::2, 1::2] -> Blue = 0
        # channel 1: raw[0::2, 1::2] -> Gr   = 510 / 1020 = 0.5
        # channel 2: raw[0::2, 0::2] -> Red  = 1020 / 1020 = 1.0
        # channel 3: raw[1::2, 0::2] -> Gb   = 255 / 1020 = 0.25
        np.testing.assert_allclose(packed[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(packed[1], 0.5, atol=1e-6)
        np.testing.assert_allclose(packed[2], 1.0, atol=1e-6)
        np.testing.assert_allclose(packed[3], 0.25, atol=1e-6)

        # Verify backward-compatible alias
        packed_alias = pack_bayer_mosaic(raw)
        np.testing.assert_array_equal(packed, packed_alias)

    def test_bayer_packing_rejects_invalid_dimensions(self) -> None:
        """Reject non-2D, empty, and incomplete Bayer mosaics."""
        for shape in [(16,), (4, 4, 3), (0, 4), (4, 0), (3, 4), (4, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                pack(np.zeros(shape, dtype=np.float32))

    def test_summary_statistics_calculation(self) -> None:
        """Verify summary() produces accurate mean, median, percentiles, and std."""
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = summary(samples)

        self.assertAlmostEqual(stats["median_ms"], 30.0)
        self.assertAlmostEqual(stats["mean_ms"], 30.0)
        self.assertAlmostEqual(stats["std_ms"], float(np.std(samples, ddof=1)))
        self.assertAlmostEqual(stats["p10_ms"], float(np.percentile(samples, 10)))
        self.assertAlmostEqual(stats["p90_ms"], float(np.percentile(samples, 90)))
        self.assertEqual(stats["samples_ms"], samples)

    def test_summary_single_sample_has_zero_standard_deviation(self) -> None:
        """A single timing sample must not produce a NaN standard deviation."""
        stats = summary([2.5])
        self.assertEqual(stats["median_ms"], 2.5)
        self.assertEqual(stats["std_ms"], 0.0)

    def test_ci_ratio_bootstrap_interval(self) -> None:
        """Verify bootstrap confidence interval returns valid [lower, upper] bounds."""
        base = [10.0, 10.2, 9.8, 10.1, 10.0] * 10
        custom = [5.0, 5.1, 4.9, 5.0, 5.2] * 10

        ci = ci_ratio(base, custom)
        self.assertEqual(len(ci), 2)
        lower, upper = ci[0], ci[1]
        self.assertLess(lower, upper)
        self.assertGreater(lower, 1.8)
        self.assertLess(upper, 2.2)

    def test_process_comparison_null_and_known_speedup(self) -> None:
        """Independent process effects, rather than nested calls, define inference."""
        base = [10.0 + i for i in range(12)]
        null = compare_processes(base, base)
        self.assertEqual(null["speedup"], 1.0)
        self.assertEqual(null["ci95"], [1.0, 1.0])
        self.assertEqual(null["raw_p"], 1.0)
        self.assertEqual(null["runs_equal"], 12)
        faster = compare_processes(base, [v / 2 for v in base])
        self.assertEqual(faster["speedup"], 2.0)
        self.assertEqual(faster["ci95"], [2.0, 2.0])
        self.assertEqual(faster["runs_faster"], 12)
        self.assertAlmostEqual(faster["raw_p"], 2 / 4096)

    def test_process_comparison_rejects_invalid_samples(self) -> None:
        """Reject unpaired, nonfinite, nonpositive, or infeasible exact-test data."""
        for base, custom in [([1], [1]), ([1, 2], [1]), ([1, 0], [1, 1]),
                             ([1, np.nan], [1, 1]), ([1] * 17, [1] * 17)]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                compare_processes(base, custom)

    def test_holm_correction_and_interval_direction(self) -> None:
        """A small uncorrected p-value alone must not support an acceleration."""
        rows = [{"raw_p": .03, "ci95": [1.1, 1.2]},
                {"raw_p": .001, "ci95": [.8, .9]},
                {"raw_p": .02, "ci95": [.9, 1.2]},
                {"raw_p": .9, "ci95": [1.1, 1.2]}]
        holm(rows)
        np.testing.assert_allclose([r["holm_p"] for r in rows], [.06, .004, .06, .9])
        self.assertTrue(rows[1]["supported_slower"])
        self.assertFalse(any(r["supported_faster"] for r in rows))

    def test_quality_metrics_invariance_on_identical_images(self) -> None:
        """Verify that identical target and prediction give ideal metric values."""
        target = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        pred = target.copy()

        # Zero reconstruction error gives infinite PSNR by definition.
        with np.errstate(divide="ignore"):
            metrics = quality(target, pred)
        self.assertTrue(np.isposinf(metrics["psnr_db"]))
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=4)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=6)
        self.assertAlmostEqual(metrics["delta_e_2000"], 0.0, places=5)

    def test_quality_metrics_on_distinct_images(self) -> None:
        """Verify quality metrics on opposite black and white images."""
        target = np.zeros((32, 32, 3), dtype=np.float32)
        pred = np.ones((32, 32, 3), dtype=np.float32)

        metrics = quality(target, pred)
        self.assertAlmostEqual(metrics["psnr_db"], 0.0, places=3)
        self.assertAlmostEqual(metrics["mae"], 1.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 1.0, places=6)
        self.assertGreater(metrics["delta_e_2000"], 50.0)


if __name__ == "__main__":
    unittest.main()
