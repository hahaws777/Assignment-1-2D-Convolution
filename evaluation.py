"""Evaluation utilities for CUDA convolution benchmarks.

This module provides high-precision timing, Bayer sensor preprocessing,
correctness assertions against PyTorch references, and full-reference
image quality assessment metrics (PSNR, SSIM, MS-SSIM, CIEDE2000).
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch


def summary(v: List[float]) -> Dict[str, Any]:
    """Compute summary statistics for a sequence of latency measurements.

    Args:
        v: List of latency measurements in milliseconds.

    Returns:
        Dict[str, Any]: Dictionary containing median, mean, standard deviation,
            10th percentile, 90th percentile, 95th percentile, and the raw samples.
    """
    arr = np.asarray(v)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "samples_ms": v,
    }


def measure(
    functions: Dict[str, Callable[[], Any]],
    reps: int,
    warmup: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Benchmark multiple CUDA functions with warm-up and randomized order.

    Measures both GPU execution time via CUDA events and CPU wall-clock time
    via high-resolution perf_counter. Shuffles the execution order in each
    repetition to mitigate thermal throttling and persistent cache bias.

    Args:
        functions: Mapping from method name to a zero-argument callable.
        reps: Number of timed evaluation repetitions.
        warmup: Number of unmeasured warm-up iterations.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Nested mapping containing
            'cuda_event' and 'wall' timing summaries for each method.
    """
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()

    samples: Dict[str, List[float]] = {name: [] for name in functions}
    walls: Dict[str, List[float]] = {name: [] for name in functions}
    rng = np.random.default_rng(42)

    for _ in range(reps):
        order = rng.permutation(list(functions)).tolist()
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            tick = time.perf_counter()
            start.record()

            functions[name]()

            end.record()
            end.synchronize()

            walls[name].append((time.perf_counter() - tick) * 1000.0)
            samples[name].append(start.elapsed_time(end))

    return {
        name: {
            "cuda_event": summary(samples[name]),
            "wall": summary(walls[name]),
        }
        for name in functions
    }


def agreement(
    ref: torch.Tensor,
    out: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 2e-5,
) -> Dict[str, float]:
    """Validate numerical closeness against reference output within tolerances.

    Args:
        ref: Expected reference tensor (typically FP32 or converted double).
        out: Custom CUDA output tensor to validate.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.

    Returns:
        Dict[str, float]: Numerical error metrics including max absolute error,
            mean absolute error, RMSE, and 99th percentile absolute error.

    Raises:
        AssertionError: If any value exceeds the specified tolerances.
    """
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    diff = (ref - out).abs()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float((ref - out).square().mean().sqrt()),
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99)),
    }


def ci_ratio(base: List[float], custom: List[float]) -> List[float]:
    """Compute the 95% bootstrap confidence interval of median speedup ratio.

    Resamples paired repetition indices 2,000 times to calculate robust
    confidence bounds for the ratio (base_median / custom_median).

    Args:
        base: Latency samples from the baseline implementation.
        custom: Latency samples from the custom implementation.

    Returns:
        List[float]: [2.5th percentile, 97.5th percentile] speedup ratio bounds.
    """
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(base), (2000, len(base)))
    base_arr = np.asarray(base)
    custom_arr = np.asarray(custom)
    ratios = np.median(base_arr[indices], axis=1) / np.median(custom_arr[indices], axis=1)
    return np.percentile(ratios, [2.5, 97.5]).tolist()


def pack(raw: np.ndarray) -> np.ndarray:
    """Pack a single-channel Bayer sensor mosaic into 4 sub-sampled color planes.

    Converts an HxW RGGB mosaic to a 4x(H/2)x(W/2) tensor and scales
    sensor values by 1/1020, following the upstream PyNET preprocessing.
    Gr denotes green on a red row; Gb denotes green on a blue row.

    Args:
        raw: 2D numpy array of raw sensor values.

    Returns:
        np.ndarray: 4-channel float32 array in [B, Gr, R, Gb] channel order.

    Raises:
        ValueError: If raw is not 2D or either dimension is empty or odd.
    """
    if raw.ndim != 2 or any(size == 0 or size % 2 for size in raw.shape):
        raise ValueError("Expected a non-empty 2D Bayer mosaic with even height and width")
    return np.stack([
        raw[1::2, 1::2],  # Blue
        raw[0::2, 1::2],  # Gr: green on the red row
        raw[0::2, 0::2],  # Red
        raw[1::2, 0::2],  # Gb: green on the blue row
    ]).astype(np.float32) / 1020.0


def native_input(path: str | Path, size: int) -> Tuple[torch.Tensor, List[int]]:
    """Load and center-crop a full-resolution RAW sensor image.

    Args:
        path: Path to the RAW Bayer PNG file.
        size: Desired spatial dimension (must be multiple of 32 for Bayer alignment).

    Returns:
        Tuple[torch.Tensor, List[int]]:
            - GPU Tensor of shape [1, 4, size // 2, size // 2].
            - Crop coordinates [top, left].

    Raises:
        ValueError: If image is not 2D or size is unaligned.
    """
    sensor = np.asarray(Image.open(path))
    if sensor.ndim != 2 or size > min(sensor.shape) or size % 32:
        raise ValueError("Expected a large 2D Bayer sensor and aligned crop size (divisible by 32)")

    top = ((sensor.shape[0] - size) // 2) // 2 * 2
    left = ((sensor.shape[1] - size) // 2) // 2 * 2
    crop = sensor[top : top + size, left : left + size]
    tensor = torch.from_numpy(pack(crop)[None]).cuda()
    return tensor, [top, left]


def pair(
    root: Path,
    index: int,
    size: int,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, str]]:
    """Load a matched pair of Huawei RAW patch and Canon DSLR reference target.

    Args:
        root: Root directory of Zurich RAW-to-DSLR dataset.
        index: Image identifier index.
        size: Crop dimension (<= 448).

    Returns:
        Tuple[torch.Tensor, np.ndarray, Dict[str, str]]:
            - Preprocessed RAW input tensor on GPU [1, 4, size // 2, size // 2].
            - Canon DSLR reference sRGB array of shape [size, size, 3] in [0, 1].
            - SHA-256 hash dictionary of source image files.

    Raises:
        ValueError: If files do not match expected 448x448 patch format.
    """
    raw_path = root / "test/huawei_raw" / f"{index}.png"
    target_path = root / "test/canon" / f"{index}.jpg"

    raw = np.asarray(Image.open(raw_path))
    target = np.asarray(Image.open(target_path).convert("RGB")).astype(np.float32) / 255.0

    if raw.shape != (448, 448) or target.shape != (448, 448, 3) or size > 448:
        raise ValueError(f"Expected matched 448x448 test pair for index {index}")

    offset = ((448 - size) // 2) // 2 * 2
    raw_crop = raw[offset : offset + size, offset : offset + size]
    target_crop = target[offset : offset + size, offset : offset + size]

    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [raw_path, target_path]
    }
    raw_tensor = torch.from_numpy(pack(raw_crop)[None]).cuda()
    return raw_tensor, target_crop, hashes


def quality(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute standard perceptual and objective image quality metrics.

    Evaluates Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM),
    Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and CIEDE2000
    perceptual color difference.

    Args:
        target: Reference ground truth sRGB array [H, W, 3] in [0, 1].
        pred: Predicted reconstructed sRGB array [H, W, 3] in [0, 1].

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'psnr_db': Peak Signal-to-Noise Ratio (dB).
            - 'ssim': Structural Similarity index.
            - 'mae': Mean absolute error.
            - 'rmse': Root mean squared error.
            - 'delta_e_2000': CIEDE2000 perceptual color difference in Lab space.
    """
    return {
        "psnr_db": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=1.0)),
        "mae": float(np.abs(target - pred).mean()),
        "rmse": float(np.sqrt(((target - pred) ** 2).mean())),
        "delta_e_2000": float(deltaE_ciede2000(rgb2lab(target), rgb2lab(pred)).mean()),
    }


# Backwards-compatible descriptive aliases
pack_bayer_mosaic = pack
check_numerical_agreement = agreement
compute_image_quality = quality
load_test_pair = pair


def compare_processes(base, custom):
    """Compare paired process medians with bootstrap CI and an exact sign-flip test.

    Calls within a process are nested observations. The sign-flip test assumes
    symmetric paired log effects under the null; it is not an equivalence test.
    """
    base, custom = np.asarray(base, dtype=float), np.asarray(custom, dtype=float)
    if (base.ndim != 1 or base.shape != custom.shape or not 2 <= len(base) <= 16
            or not np.isfinite(base).all() or not np.isfinite(custom).all()
            or (base <= 0).any() or (custom <= 0).any()):
        raise ValueError("Expected 2 to 16 paired, finite, positive process medians")
    ratios = base / custom
    count = len(ratios)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, count, (20000, count))
    boot = np.median(ratios[indices], axis=1)
    log_ratios = np.log(ratios)
    # Exact paired sign-flip test on the mean log ratio, assuming symmetric
    # paired run effects under the null. Raw p-values remain descriptive until
    # Holm correction within the explicitly declared comparison family.
    signs = np.array(list(itertools.product([-1, 1], repeat=count)))
    permutation = np.abs(signs @ log_ratios / count)
    observed = abs(log_ratios.mean())
    p = float(np.mean(permutation >= observed - 1e-15))
    return {
        'speedup': float(np.median(ratios)),
        'ci95': np.percentile(boot, [2.5,97.5]).tolist(),
        'run_speedups': ratios.tolist(), 'run_range': [float(ratios.min()),float(ratios.max())],
        'runs_faster': int((ratios>1).sum()), 'runs_slower': int((ratios<1).sum()),
        'runs_equal': int((ratios==1).sum()),
        'reference_median_ms': float(np.median(base)),
        'custom_median_ms': float(np.median(custom)),
        'raw_p': p, 'runs': count,
    }


def holm(rows):
    """Apply Holm correction in place within one declared comparison family."""
    ordered = sorted(rows, key=lambda x:x['raw_p'])
    previous = 0.0
    for index,row in enumerate(ordered):
        adjusted = min(1.0,max(previous, (len(rows)-index)*row['raw_p']))
        row['holm_p'] = adjusted
        row['supported_faster'] = adjusted < .05 and row['ci95'][0] > 1
        row['supported_slower'] = adjusted < .05 and row['ci95'][1] < 1
        previous = adjusted


def _process_median(row,method,field='timings'):
    """Read one process-level CUDA-event median."""
    return row[field][method]['cuda_event']['median_ms']


def analyze_repeatability(evidence):
    """Recompute published process-level statistics from raw repeated evidence.

    Network (14) and selected operator (4) comparisons form separate Holm
    families. Scheduling and other intervals remain descriptive.
    """
    runs = evidence["runs"]
    if not evidence.get("complete") or not 2 <= len(runs) <= 16:
        raise ValueError("Expected complete evidence from 2 to 16 fresh processes")
    result = {'protocol': evidence['protocol'], 'complete': evidence['complete'],
              'analysis': {'point_estimate': 'median of paired per-process median-time ratios',
                           'ci': '20,000 bootstrap resamples of paired process-level speedups; 95% percentile interval',
                           'p_test': 'exact paired sign-flip test of mean log speedup; Holm correction within family',
                           'families': '14 network custom-versus-cuDNN comparisons; 4 selected operator comparisons separately',
                           'limits': 'Same GPU and consecutive fresh-process runs; intervals do not cover other devices or arbitrary operating conditions'},
              'operators': [], 'network': [], 'scheduling': []}
    methods = ['naive','const','tiled','im2col','im2col_reg','implicit','gridstride','cudnn_repeat']
    for c,size in [(4,32),(4,64),(16,32),(4,320)]:
        rows = [next(v for v in r['operators'] if (v['channels'],v['size'])==(c,size)) for r in runs]
        entry = {'channels':c,'size':size,'comparisons':{},'ten_call_comparisons':{}}
        for method in methods:
            entry['comparisons'][method] = compare_processes([_process_median(v,'cudnn') for v in rows],[_process_median(v,method) for v in rows])
            if 'ten_call_timings' in rows[0]:
                entry['ten_call_comparisons'][method] = compare_processes(
                    [_process_median(v,'cudnn','ten_call_timings') for v in rows],
                    [_process_median(v,method,'ten_call_timings') for v in rows])
        if size==320:
            entry['reg_vs_implicit'] = compare_processes([_process_median(v,'implicit') for v in rows],[_process_median(v,'im2col_reg') for v in rows])
        result['operators'].append(entry)

    selected = [entry['comparisons']['implicit'] for entry in result['operators'][:3]]
    selected.append(result['operators'][3]['reg_vs_implicit'])
    holm(selected)
    family=[]
    for size in [128,448]:
        rows = [next(v for v in r['network'] if v['raw_size']==size) for r in runs]
        entry={'raw_size':size,'comparisons':{}}
        for method in methods:
            comparison=compare_processes([_process_median(v,'cudnn') for v in rows],[_process_median(v,method) for v in rows])
            entry['comparisons'][method] = comparison
            if method!='cudnn_repeat':family.append(comparison)
        result['network'].append(entry)
    holm(family)
    for baseline,custom in [('tile_8','tile_16'),('tile_16','tile_32'),('grid_112','grid_224'),('grid_112','grid_448'),('grid_224','grid_448')]:
        result['scheduling'].append({'reference':baseline,'custom':custom,
            **compare_processes([r['scheduling'][baseline]['cuda_event']['median_ms'] for r in runs],
                      [r['scheduling'][custom]['cuda_event']['median_ms'] for r in runs])})
    return result
