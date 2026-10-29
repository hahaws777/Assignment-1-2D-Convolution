"""Benchmarking suite for seven CUDA convolutions and PyNET RAW-to-sRGB pipeline.

Conducts rigorous evaluations across:
  1. Synthetic operator sweeps (varying channels, spatial sizes, kernels, batches).
  2. Tiling sweeps (tile = 8, 16, 32) and grid-stride block scaling.
  3. Real RAW resolution scaling (128x128 to 896x896) on genuine Huawei sensor crops.
  4. Resumable quality evaluation across all 1,204 native 448x448 RAW/DSLR test pairs.
  5. Fresh-process repeatability, paired statistical analysis, and GPU path traces.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluation import (  # noqa: E402
    agreement,
    ci_ratio,
    measure,
    native_input,
    pair,
    quality,
    summary,
    analyze_repeatability,
)
from msssim import msssim  # noqa: E402
from runtime import (  # noqa: E402
    METHODS,
    ROOT,
    TARGETS,
    extension,
    load_model,
    replace_convolutions,
)


def macs(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, int]:
    """Calculate total multiply-accumulate operations (MACs) for convolution layers.

    Registers forward hooks on all `torch.nn.Conv2d` modules to calculate exact
    MAC operations during a forward pass with input tensor `x`.

    Args:
        model: PyTorch model instance (e.g., PyNET).
        x: Sample input tensor with target batch and spatial shape.

    Returns:
        Dict[str, int]: Dictionary containing:
            - 'all_conv_macs': Total MACs across all Conv2d modules in the network.
            - 'selected_conv_macs': Total MACs within the replaced TARGETS layers.
    """
    stats = {"all_conv_macs": 0, "selected_conv_macs": 0}
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):

            def hook(layer, args, out, layer_name=name):
                value = (
                    out.numel()
                    * layer.in_channels
                    * layer.kernel_size[0]
                    * layer.kernel_size[1]
                )
                stats["all_conv_macs"] += value
                if layer_name in TARGETS:
                    stats["selected_conv_macs"] += value

            handles.append(module.register_forward_hook(hook))

    model(x)
    for h in handles:
        h.remove()
    return stats


def write_resolution_csv(result: Dict[str, Any], output_dir: Path) -> Path:
    """Export recorded network timings as one row per RAW size and method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolution_summary.csv"
    temporary = path.with_suffix(".csv.part")
    fields = ["raw_size", "method", "median_ms", "std_ms", "p95_ms", "speedup", "ci95_low", "ci95_high"]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for resolution in result["resolution"]:
            for method, timing in resolution["timings"].items():
                writer.writerow({
                    "raw_size": resolution["raw_size"],
                    "method": method,
                    "median_ms": timing["cuda_event"]["median_ms"],
                    "std_ms": timing["cuda_event"]["std_ms"],
                    "p95_ms": timing["cuda_event"]["p95_ms"],
                    "speedup": timing["speedup"],
                    "ci95_low": timing["speedup_ci95"][0],
                    "ci95_high": timing["speedup_ci95"][1],
                })
    temporary.replace(path)
    return path


def write_quality_csv(result: Dict[str, Any], output_dir: Path) -> Path:
    """Export completed reconstruction metrics as one row per image and method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "quality_all_methods.csv"
    temporary = path.with_suffix(".csv.part")
    metric_keys = list(result["methods"]["cudnn"]["mean"])
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["raw_size", "image_id", "method"] + metric_keys)
        writer.writeheader()
        for method, entry in result["methods"].items():
            for index in result["protocol"]["ids"]:
                writer.writerow({
                    "raw_size": result["protocol"]["raw_size"],
                    "image_id": index,
                    "method": method,
                    **entry["images"][str(index)]["metrics"],
                })
    temporary.replace(path)
    return path


@torch.inference_mode()
def full_quality(data: Path, ids: Sequence[int], output_dir: Path) -> None:
    """Evaluate end-to-end reconstruction quality on native 448-pixel test images.

    Computes PSNR, SSIM, MS-SSIM, MAE, RMSE, and CIEDE2000 for each test patch
    against Canon DSLR ground truth. Supports interruption and resumption by caching
    baseline outputs in `.quality_cache/` and saving progress every 25 images.

    Args:
        data: Directory path containing the extracted Zurich RAW-to-DSLR test split.
        ids: Sequence of image indices to evaluate.
        output_dir: Directory where 'full_quality.json' and 'quality_all_methods.csv' are written.

    Raises:
        ValueError: If an existing output file was generated with a different protocol or GPU.
    """
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    output_path = output_dir / "full_quality.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".quality_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_bytes = (ROOT / "checkpoints/pynet_level_0.pth").read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()

    protocol = {
        "raw_size": 448,
        "ids": list(ids),
        "count": len(ids),
        "tf32": False,
        "deterministic": True,
        "checkpoint_sha256": checkpoint_hash,
        "data": str(data.resolve()),
        "tolerance": {"rtol": 1e-4, "atol": 2e-5},
        "batch_size": 1,
    }

    device_prop = torch.cuda.get_device_properties(0)
    environment = {
        "gpu": device_prop.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }

    if output_path.exists():
        result = json.loads(output_path.read_text())
        if result["protocol"] != protocol:
            raise ValueError(
                "Existing full_quality.json uses a different protocol; use a new --output-dir"
            )
        if result["environment"] != environment:
            raise ValueError(
                "Existing full_quality.json uses a different GPU/environment; use a new --output-dir"
            )
    else:
        result = {
            "protocol": protocol,
            "environment": environment,
            "methods": {},
            "complete": False,
        }

    def save_state() -> None:
        temp_file = output_path.with_suffix(".json.part")
        temp_file.write_text(json.dumps(result, indent=2, allow_nan=False))
        temp_file.replace(output_path)

    base_model = load_model()
    started = time.perf_counter()
    metric_pool = ThreadPoolExecutor(max_workers=1)

    for method in ("cudnn",) + tuple(METHODS):
        entry = result["methods"].setdefault(method, {"images": {}})
        model = base_model if method == "cudnn" else replace_convolutions(
            copy.deepcopy(base_model), method
        )
        pending: Optional[tuple] = None

        def finish(item: tuple) -> None:
            key, row_dict, future = item
            metrics = future.result()
            metrics["ms_ssim"] = row_dict.pop("ms_ssim")
            row_dict["metrics"] = metrics
            entry["images"][key] = row_dict
            done = len(entry["images"])
            if done % 25 == 0 or done == len(ids):
                save_state()
                elapsed = time.perf_counter() - started
                print(
                    f"Full quality {method}: {done}/{len(ids)} pairs; elapsed {elapsed:.0f}s",
                    flush=True,
                )

        for index in ids:
            key = str(index)
            cached_pred = cache_dir / f"{index}.npy"
            if key in entry["images"] and (method != "cudnn" or cached_pred.is_file()):
                continue

            x, target, _ = pair(data, index, 448)
            out = model(x)

            if pending is not None:
                finish(pending)
                pending = None

            prediction = out[0].permute(1, 2, 0).cpu().numpy()
            future = metric_pool.submit(quality, target, prediction)
            target_tensor = torch.from_numpy(target.copy()).permute(2, 0, 1)[None].cuda()
            row = {"ms_ssim": float(msssim(target_tensor, out, val_range=1))}

            if method == "cudnn":
                temp_cache = cached_pred.with_suffix(".npy.part")
                with temp_cache.open("wb") as handle:
                    np.save(handle, out[0].cpu().numpy())
                temp_cache.replace(cached_pred)
            else:
                ref = torch.from_numpy(np.load(cached_pred)[None]).cuda()
                diff = (out - ref).abs()
                tol = 2e-5 + 1e-4 * ref.abs()
                row["agreement"] = {
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                    "rmse": float(diff.square().mean().sqrt()),
                    "p99_abs": float(torch.quantile(diff.flatten(), 0.99)),
                    "passed": bool((diff <= tol).all()),
                    "violating_channel_values": int((diff > tol).sum()),
                    "violating_pixels": int((diff > tol).any(dim=1).sum()),
                    "max_tolerance_ratio": float((diff / tol).max()),
                }

            pending = (key, row, future)

        if pending is not None:
            finish(pending)

        first_metrics = next(iter(entry["images"].values()))["metrics"]
        entry["mean"] = {
            k: float(np.mean([v["metrics"][k] for v in entry["images"].values()]))
            for k in first_metrics
        }

        if method != "cudnn":
            entry["agreement_summary"] = {
                "passed": sum(v["agreement"]["passed"] for v in entry["images"].values()),
                "failed": sum(not v["agreement"]["passed"] for v in entry["images"].values()),
                "worst_abs": max(v["agreement"]["max_abs"] for v in entry["images"].values()),
            }

        entry["complete"] = True
        save_state()
        print(f"Completed {method}: {entry['mean']}", flush=True)

        if method != "cudnn":
            del model

    metric_pool.shutdown(wait=True)

    write_quality_csv(result, output_dir)
    result["complete"] = True
    save_state()
    print(f"Complete native-patch quality evaluation saved to {output_path}", flush=True)


@torch.inference_mode()
def image_size_scaling(sizes: Sequence[int], reps: int, warmup: int, output_dir: Path) -> None:
    """Measure spatial scaling with fixed N=1, C=4, K=32 and a 3x3 filter.

    Uses identical weights across sizes and identical inputs across methods.
    Saves timings, raw samples, and numerical errors without requiring PyNET
    weights or image data. Sizes are visited sequentially; method order is
    randomized within each repetition by ``measure``.
    """
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = extension()
    prop = torch.cuda.get_device_properties(0)
    result = {
        "environment": {
            "gpu": prop.name, "sm_count": prop.multi_processor_count,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "protocol": {
            "reps": reps, "warmup": warmup, "seed": 42, "tf32": False,
            "cudnn_benchmark": True, "deterministic": False,
            "sizes": list(sizes), "size_order": "as listed, sequential",
            "timing_order": "seeded random permutation per repetition",
            "fixed": {"batch": 1, "in_channels": 4, "out_channels": 32,
                      "kernel": 3, "stride": 1, "padding": 1, "tile": 16,
                      "grid_blocks": 0},
            "weights": "same weights and bias at every size",
            "tolerance": {"rtol": 1e-4, "atol": 2e-5},
        },
        "operators": [], "complete": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "image_size_scaling.json"
    weights = torch.randn(32, 4, 3, 3, device="cuda") * 0.03
    bias = torch.randn(32, device="cuda")
    for size in sizes:
        x = torch.randn(1, 4, size, size, device="cuda")
        functions = {"cudnn": lambda: F.conv2d(x, weights, bias, padding=1)}
        functions.update({
            method: lambda method=method: ext.convolution(x, weights, bias, method, 1, 1, 16, 0)
            for method in METHODS
        })
        ref = functions["cudnn"]()
        errors = {}
        for method, fn in functions.items():
            out = fn()
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=2e-5)
            diff = out - ref
            errors[method] = {"passed": True, "max_abs": float(diff.abs().max()),
                              "rmse": float(diff.square().mean().sqrt())}
        del ref, out, diff
        timings = measure(functions, reps, warmup)
        flops = 2 * 32 * size * size * 4 * 3 * 3
        for row in timings.values():
            ms = row["cuda_event"]["median_ms"]
            row["gflops"] = flops / (ms * 1e6)
            row["speedup"] = timings["cudnn"]["cuda_event"]["median_ms"] / ms
        result["operators"].append({
            "shape": [1, 4, size, size], "out_channels": 32, "kernel": 3,
            "flops": flops, "errors": errors, "timings": timings,
        })
        result["complete"] = len(result["operators"]) == len(sizes)
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(path)
        print(f"Image size {size}: " + str({
            m: round(t["cuda_event"]["median_ms"], 6) for m, t in timings.items()
        }), flush=True)
    print(f"Spatial scaling measurements saved to {path}", flush=True)


def save_json(path, result):
    """Write evidence atomically without leaving a partial result file."""
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(path)


@torch.inference_mode()
def repeated_timing(functions, repetitions, warmup, seed, calls=1):
    """Randomize labels per round; retain paired event/wall samples.

    calls=1 matches the published call interval. calls=10 is a separate
    sensitivity test, reporting the interval divided by ten (not pure kernel
    time). All event objects are allocated before each timed interval.
    """
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    samples = {m: [] for m in functions}
    walls = {m: [] for m in functions}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        for method in rng.permutation(list(functions)).tolist():
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            begin.record()
            for _ in range(calls):
                functions[method]()
            end.record()
            end.synchronize()
            samples[method].append(begin.elapsed_time(end) / calls)
            walls[method].append((time.perf_counter() - tick) * 1000 / calls)
    return {m: {'cuda_event': summary(samples[m]), 'wall': summary(walls[m])}
            for m in functions}


def telemetry():
    """Record GPU telemetry when nvidia-smi is available."""
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Unavailable: {type(exc).__name__}"


@torch.inference_mode()
def repeatability_worker(index, raw, output_dir):
    """Run the fixed protocol once in a fresh Python process."""
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = extension()
    prop = torch.cuda.get_device_properties(0)
    result = {'run': index, 'pid': os.getpid(), 'started': time.time(),
              'environment': {'gpu': prop.name, 'torch': torch.__version__,
                              'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
              'telemetry_before': telemetry(), 'operators': [], 'scheduling': [], 'network': []}
    # A fixed CPU generator for each case makes tensors independent of case order.
    cases = [(4, 32), (4, 64), (16, 32), (4, 320)]
    case_order = np.random.default_rng(1000 + index).permutation(len(cases))
    for case_id in case_order:
        channels, size = cases[case_id]
        generator = torch.Generator().manual_seed(4200 + int(case_id))
        x = torch.randn(1, channels, size, size, generator=generator).cuda()
        w = (torch.randn(32, channels, 3, 3, generator=generator) * .03).cuda()
        b = torch.randn(32, generator=generator).cuda()
        functions = {'cudnn': lambda: F.conv2d(x, w, b, padding=1)}
        functions['cudnn_repeat'] = functions['cudnn']
        functions.update({m: lambda m=m: ext.convolution(x, w, b, m, 1, 1, 16, 0)
                          for m in METHODS})
        ref = functions['cudnn']()
        for fn in functions.values():
            torch.testing.assert_close(fn(), ref, rtol=1e-4, atol=2e-5)
        row = {'channels': channels, 'size': size,
               'timings': repeated_timing(functions, 200, 25, 2000 + index * 10 + int(case_id))}
        if channels == 4 and size in (32, 64):
            row['ten_call_timings'] = repeated_timing(functions, 100, 25, 3000 + index * 10 + int(case_id), calls=10)
        result['operators'].append(row)
        print(f'run {index}: operator C{channels}, {size} done', flush=True)

    generator = torch.Generator().manual_seed(7300)
    x = torch.randn(1, 4, 224, 224, generator=generator).cuda()
    w = (torch.randn(32, 4, 3, 3, generator=generator) * .03).cuda()
    b = torch.randn(32, generator=generator).cuda()
    functions = {f'tile_{t}': lambda t=t: ext.convolution(x, w, b, 'tiled', 1, 1, t, 0)
                 for t in (8, 16, 32)}
    functions.update({f'grid_{g}': lambda g=g: ext.convolution(x, w, b, 'gridstride', 1, 1, 16, g)
                      for g in (112, 224, 448)})
    result['scheduling'] = repeated_timing(functions, 100, 10, 4000 + index)
    del functions, x, w, b, ref
    gc.collect()
    torch.cuda.empty_cache()

    base = load_model()
    models = {'cudnn': base}
    models.update({m: replace_convolutions(copy.deepcopy(base), m) for m in METHODS})
    for size in np.random.default_rng(5000 + index).permutation([128, 448]).tolist():
        x, _ = native_input(raw, size)
        functions = {m: lambda model=model: model(x) for m, model in models.items()}
        functions['cudnn_repeat'] = functions['cudnn']
        ref = functions['cudnn']()
        correctness = {}
        for name, fn in functions.items():
            output = fn()
            torch.testing.assert_close(output, ref, rtol=1e-4, atol=2e-5)
            correctness[name] = float((output-ref).abs().max())
        del output, ref
        measured = repeated_timing(functions, 20, 5, 6000 + index * 10 + size)
        result['network'].append({'raw_size': size, 'timings': measured,
                                  'max_abs': correctness})
        print(f'run {index}: PyNET RAW {size} done', flush=True)
    result['telemetry_after'] = telemetry()
    result['elapsed_seconds'] = time.time() - result['started']
    path = output_dir / f'run_{index:02}.json'
    save_json(path, result)
    print(f'run {index}: complete in {result["elapsed_seconds"]:.1f}s', flush=True)



def repeatability(raw, output_dir, processes):
    """Run sequential fresh processes and analyze their paired timing medians."""
    if not raw.is_file() or not (ROOT / "checkpoints/pynet_level_0.pth").is_file():
        raise ValueError("RAW source or checkpoint missing; follow HOWTO.md setup")
    if not 2 <= processes <= 16:
        raise ValueError("Use 2 to 16 processes (12 for the published protocol)")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (any(output_dir.glob("run_*.json"))
            or any((output_dir / name).exists() for name in ("protocol.json", "performance_repeatability.json", "performance_repeatability_summary.json"))):
        raise ValueError("Repeatability evidence already exists; use a fresh --output-dir")
    protocol = {
        'independent_processes': processes,
        'inputs': 'Identical seeded tensors, weights, and bias in every process; same RAW source and model weights',
        'precision': 'FP32; TF32 disabled; cuDNN autotuning on; deterministic=False',
        'operator_cases': [{'channels': c, 'size': h, 'out_channels': 32, 'filter': [3,3]}
                           for c,h in [(4,32),(4,64),(16,32),(4,320)]],
        'operator_repetitions': 200, 'operator_warmup': 25,
        'sensitivity': {'calls_per_interval': 10, 'repetitions': 100,
                        'cases': 'C4, sizes 32 and 64',
                        'scope': 'Ten sequential full invocations divided by ten; separate from single-call latency'},
        'network_sizes': [128,448], 'network_repetitions': 20, 'network_warmup': 5,
        'scheduling_repetitions': 100, 'scheduling_warmup': 10,
        'order': 'Case order and per-repetition method order independently randomized by process seed',
        'null_control': 'cudnn_repeat is the same callable as cudnn under a second randomized label',
        'timing_scope': 'CUDA event interval around full invocation, allocations included; synchronization before each interval',
        'primary_unit': 'fresh-process median; individual calls are nested observations, not independent runs',
        'inference': '95% bootstrap intervals across paired process medians; same GPU, no cross-device claim',
        'practical_margin': 'Network speedup between 0.99 and 1.01 is a predeclared ±1% practical band, separate from significance',
    }
    save_json(output_dir / 'protocol.json', protocol)
    for index in range(processes):
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--repeatability-worker', str(index),
                        '--raw', str(raw.resolve()), '--output-dir', str(output_dir.resolve())], check=True)
    destination = output_dir / 'performance_repeatability.json'
    evidence = {
        'protocol': protocol, 'runs': [json.loads((output_dir/f'run_{i:02}.json').read_text()) for i in range(processes)],
        'complete': True,
    }
    save_json(destination, evidence)
    save_json(output_dir / 'performance_repeatability_summary.json', analyze_repeatability(evidence))
    print('Saved complete repeated-process evidence to', destination, flush=True)


@torch.inference_mode()
def profile_paths(output_dir):
    """Trace warmed execution structure; profiler durations are not benchmarks."""
    from torch.profiler import profile, ProfilerActivity
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = output_dir / "traces"
    traces.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = extension()
    rows = []
    for case_id, size, methods in [(0, 32, ['cudnn', 'implicit']),
                                   (3, 320, ['implicit', 'im2col_reg'])]:
        gen = torch.Generator().manual_seed(4200 + case_id)
        x = torch.randn(1, 4, size, size, generator=gen).cuda()
        w = (torch.randn(32, 4, 3, 3, generator=gen) * .03).cuda()
        b = torch.randn(32, generator=gen).cuda()
        for method in methods:
            def invoke():
                if method == 'cudnn':
                    return F.conv2d(x, w, b, padding=1)
                return ext.convolution(x, w, b, method, 1, 1, 16, 0)
            for _ in range(25):
                invoke()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                invoke()
                torch.cuda.synchronize()
            trace = traces / f'{method}_{size}.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel']
            if not kernels:
                raise RuntimeError('No CUDA kernel events captured; check profiler/CUPTI installation')
            rows.append({'method': method, 'input_size': size,
                         'kernel_count': len(kernels),
                         'kernel_names': [e['name'] for e in kernels],
                         'cpu_operators': [e.key for e in prof.key_averages() if e.key.startswith('aten::')]})
            print(method, size, len(kernels), [e['name'][:180] for e in kernels], flush=True)
    result = {'scope': 'One warmed invocation per path; diagnostic launch structure only, not a latency benchmark.',
              'precision': 'FP32; TF32 disabled; cuDNN autotuning enabled',
              'cases': rows}
    save_json(output_dir / 'performance_mechanisms.json', result)


@torch.inference_mode()
def main() -> None:
    """CLI orchestrator for operator, resolution, and end-to-end benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Result directory (default: results/)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/zrr_test_samples",
        help="Path to downloaded test samples directory",
    )

    default_raw = ROOT / "data/zrr_test_samples/full_resolution/huawei_raw/1050.png"
    legacy_raw = ROOT / "data/full_resolution/huawei_raw/1050.png"
    if not default_raw.is_file() and legacy_raw.is_file():
        default_raw = legacy_raw

    parser.add_argument(
        "--raw",
        type=Path,
        default=default_raw,
        help="Path to full-resolution RAW image (e.g. 1050.png)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256, 384, 448, 640, 896],
        help="Bayer crop dimensions for resolution scaling tests",
    )
    parser.add_argument("--ids", type=int, nargs="+", help="Explicit subset of test IDs")
    parser.add_argument(
        "--test-count",
        type=int,
        default=1204,
        help="Evaluate IDs 0 through count-1 (default: 1204)",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="Resume native 448-pixel quality evaluation without rerunning timing",
    )
    parser.add_argument("--reps", type=int, default=60, help="Timing repetitions (default: 60)")
    parser.add_argument("--warmup", type=int, default=5, help="Timing warm-up iterations (default: 5)")
    parser.add_argument("--operator-size-only", action="store_true",
                        help="Run only fixed-channel convolution image-size scaling; no model or dataset needed")
    parser.add_argument("--operator-sizes", type=int, nargs="+",
                        default=[32, 64, 96, 128, 224, 320, 448, 640, 896],
                        help="Feature-map side lengths for --operator-size-only")

    parser.add_argument("--repeatability", action="store_true",
                        help="Run the fixed fresh-process protocol and statistical analysis")
    parser.add_argument("--processes", type=int, default=12,
                        help="Fresh processes for --repeatability (2..16; published: 12)")
    parser.add_argument("--repeatability-worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--analyze-repeatability", action="store_true",
                        help="Reanalyze performance_repeatability.json in --output-dir; no GPU needed")
    parser.add_argument("--profile-paths", action="store_true",
                        help="Export warmed GPU traces and execution-structure summary")
    args = parser.parse_args()
    modes = [args.operator_size_only, args.quality_only, args.repeatability,
             args.analyze_repeatability, args.profile_paths, args.repeatability_worker is not None]
    if sum(modes) > 1:
        parser.error("Choose only one experiment mode")
    if args.repeatability_worker is not None:
        if not 0 <= args.repeatability_worker < 16 or not args.output_dir.is_dir():
            parser.error("Invalid internal worker index or output directory")
        repeatability_worker(args.repeatability_worker, args.raw, args.output_dir)
        return
    if args.repeatability:
        try:
            repeatability(args.raw, args.output_dir, args.processes)
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.analyze_repeatability:
        evidence = json.loads((args.output_dir / "performance_repeatability.json").read_text())
        save_json(args.output_dir / "performance_repeatability_summary.json",
                  analyze_repeatability(evidence))
        print("Recomputed process-level statistics in", args.output_dir)
        return
    if args.profile_paths:
        profile_paths(args.output_dir)
        return

    if args.operator_size_only:
        if (args.reps < 5 or args.warmup < 1 or any(s < 1 for s in args.operator_sizes)
                or len(set(args.operator_sizes)) != len(args.operator_sizes)):
            parser.error("Use positive, distinct operator sizes, reps >= 5 and warmup >= 1")
        image_size_scaling(args.operator_sizes, args.reps, args.warmup, args.output_dir)
        return

    if not (1 <= args.test_count <= 1204):
        parser.error("--test-count must be in [1, 1204]")
    if args.ids is None:
        args.ids = list(range(args.test_count))
    if not args.ids or len(set(args.ids)) != len(args.ids) or any(i < 0 or i >= 1204 for i in args.ids):
        parser.error("Invalid or duplicated test IDs")

    if args.quality_only:
        full_quality(args.data, args.ids, args.output_dir)
        return

    if args.reps < 5 or args.warmup < 1 or any(s < 128 or s % 32 for s in args.sizes) or not args.ids:
        parser.error("Invalid repetitions, sizes or image IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    ext = extension()
    base = load_model()
    models = {"cudnn": base}
    for method in METHODS:
        models[method] = replace_convolutions(copy.deepcopy(base), method)

    prop = torch.cuda.get_device_properties(0)
    result = {
        "environment": {
            "gpu": prop.name,
            "sm_count": prop.multi_processor_count,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "protocol": {
            "reps": args.reps,
            "warmup": args.warmup,
            "seed": 42,
            "tf32": False,
            "targets": list(TARGETS),
            "timing_order": "seeded random permutation per repetition",
            "constant_upload": "D2D upload included every invocation; default stream only",
            "model_scope": "Only two Conv2d modules replaced; every non-convolution operator unchanged",
        },
        "source": {
            "path": str(args.raw),
            "sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        },
        "checkpoint_sha256": hashlib.sha256(
            (ROOT / "checkpoints/pynet_level_0.pth").read_bytes()
        ).hexdigest(),
        "operators": [],
        "tiles": [],
        "grid_scaling": [],
        "resolution": [],
        "quality_results": "full_quality.json",
    }

    def save_benchmark() -> None:
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        write_resolution_csv(result, args.output_dir)

    # 1. Synthetic convolution inputs
    shapes = [
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

    for n, c, h, q, k, r in shapes:
        x = torch.randn(n, c, h, q, device="cuda")
        w = torch.randn(k, c, r, r, device="cuda") * 0.03
        b = torch.randn(k, device="cuda")

        functions = {"cudnn": lambda: F.conv2d(x, w, b, padding=r // 2)}
        unsupported = {}

        for method in METHODS:
            if method == "const" and w.numel() > 16384:
                unsupported[method] = "Weight tensor exceeds 64 KiB constant-memory capacity"
                continue
            functions[method] = (
                lambda method=method: ext.convolution(x, w, b, method, 1, r // 2, 16, 0)
            )

        ref = functions["cudnn"]()
        errors = {m_name: agreement(ref, fn()) for m_name, fn in functions.items()}
        timing = measure(functions, args.reps, args.warmup)
        flops = 2 * n * k * h * q * c * r * r

        for method, row in timing.items():
            ms = row["cuda_event"]["median_ms"]
            row.update({
                "gflops": flops / (ms * 1e6),
                "speedup": timing["cudnn"]["cuda_event"]["median_ms"] / ms,
                "workspace_bytes": 0 if method not in ("im2col", "im2col_reg") else 4 * c * r * r * h * q,
            })

        result["operators"].append({
            "shape": [n, c, h, q],
            "out_channels": k,
            "kernel": r,
            "flops": flops,
            "errors": errors,
            "not_applicable": unsupported,
            "timings": timing,
        })
        formatted_timing = {k_name: round(t["cuda_event"]["median_ms"], 4) for k_name, t in timing.items()}
        print(f"Operator {n}x{c}x{h}x{q}, K={k}, R={r}: {formatted_timing}", flush=True)

    # 2. Tile size sweep
    x = torch.randn(1, 4, 224, 224, device="cuda")
    w = torch.randn(32, 4, 3, 3, device="cuda") * 0.03
    b = torch.randn(32, device="cuda")
    ref = F.conv2d(x, w, b, padding=1)

    for tile in [8, 16, 32]:
        fn = lambda tile=tile: ext.convolution(x, w, b, "tiled", 1, 1, tile, 0)
        result["tiles"].append({
            "tile": tile,
            "threads": tile * tile,
            "error": agreement(ref, fn()),
            "timing": measure({"tiled": fn}, args.reps, args.warmup)["tiled"],
        })

    # 3. Grid-stride block scaling sweep
    for blocks in [1, 7, 14, 28, 56, 112, 224, 448]:
        fn = lambda blocks=blocks: ext.convolution(x, w, b, "gridstride", 1, 1, 16, blocks)
        result["grid_scaling"].append({
            "blocks": blocks,
            "threads_per_block": 256,
            "error": agreement(ref, fn()),
            "timing": measure({"gridstride": fn}, args.reps, args.warmup)["gridstride"],
        })

    # 4. RAW resolution scaling sweep
    for size in args.sizes:
        x, origin = native_input(args.raw, size)
        ref = base(x)
        errors = {m_name: agreement(ref, m_mod(x)) for m_name, m_mod in models.items()}
        work = macs(base, x)
        timing = measure({m_name: (lambda m_mod=m_mod: m_mod(x)) for m_name, m_mod in models.items()}, args.reps, args.warmup)

        for m_name, m_mod in models.items():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            m_mod(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            ms = timing[m_name]["cuda_event"]["median_ms"]

            timing[m_name].update({
                "speedup": timing["cudnn"]["cuda_event"]["median_ms"] / ms,
                "speedup_ci95": ci_ratio(
                    timing["cudnn"]["cuda_event"]["samples_ms"],
                    timing[m_name]["cuda_event"]["samples_ms"],
                ),
                "fps": 1000.0 / ms,
                "megapixels_per_s": (size * size) / (ms * 1000.0),
                "conv_equivalent_gflops": (2 * work["all_conv_macs"]) / (ms * 1e6),
                "peak_allocated_mib": peak / (1024**2),
                "incremental_peak_mib": (peak - before) / (1024**2),
            })

        result["resolution"].append({
            "raw_size": size,
            "packed_shape": list(x.shape),
            "source": "native_Bayer_center_crop",
            "origin": origin,
            "work": work,
            "errors": errors,
            "timings": timing,
        })
        save_benchmark()
        formatted_res = {k_name: round(t["cuda_event"]["median_ms"], 3) for k_name, t in timing.items()}
        print(f"RAW {size}: {formatted_res}", flush=True)

    # 5. Full native-patch quality phase
    save_benchmark()
    models.clear()
    del base, m_mod, ref, x
    torch.cuda.empty_cache()
    full_quality(args.data, args.ids, args.output_dir)


if __name__ == "__main__":
    main()
