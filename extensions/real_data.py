"""Check the extension kernels on the public Zurich RAW-to-RGB test set with pretrained PyNET.

Part 1 (real activations): runs pretrained PyNET on test images, captures the tensors that reach
the two replaced layers, then on those real tensors compares each method with float64 PyTorch
and times it two ways: event (one eager call, including host dispatch) and graph (CUDA-graph
replay of 20 calls on 20 different images, i.e. GPU execution time only).

Part 2 (whole network): replaces both layers and reconstructs every test image with deterministic
cuDNN, as in the report's quality protocol; reports quality against the Canon DSLR photo,
agreement with the cuDNN network output, and network latency (event and graph, one process).

Requires `bash setup.sh` and `python download_zrr_samples.py --count N ...` (HOWTO section 2).

    python extensions/real_data.py --count 200

Writes real_data.json to --output-dir (default results/extensions/).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_OUTPUT, ROOT, environment, graph_time, load_extensions,  # noqa: E402
                    set_timing_flags)
from evaluation import agreement, measure, pair, quality  # noqa: E402
from runtime import TARGETS, load_model  # noqa: E402


class ExtensionConv(nn.Module):
    """nn.Conv2d replacement that calls an extension kernel; PyNET pads outside the convolution."""

    def __init__(self, original: nn.Conv2d, call: Callable) -> None:
        super().__init__()
        if (original.groups != 1 or original.dilation != (1, 1) or original.padding != (0, 0)
                or original.stride[0] != original.stride[1] or original.bias is None):
            raise ValueError("Expected dense square-stride PyNET convolution with bias and padding=0")
        self.weight = original.weight
        self.bias = original.bias
        self.stride = original.stride[0]
        self.call = call

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.call(x, self.weight, self.bias, self.stride)


def replace_extension(model: nn.Module, make_call: Callable[[nn.Conv2d], Callable]) -> nn.Module:
    """Swap the two report target layers for ExtensionConv wrappers."""
    for path in TARGETS:
        parent_path, attribute = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        original = getattr(parent, attribute)
        setattr(parent, attribute, ExtensionConv(original, make_call(original)))
    return model


def mk_kb(original: nn.Conv2d) -> int:
    """KB fixed before measuring: 4 for the 3-output layer, 16 for the 32-output layer."""
    return 4 if original.out_channels <= 4 else 16


def cycling(fn: Callable, inputs: List[torch.Tensor]) -> Callable[[], Any]:
    """Zero-argument callable feeding a different real input on every call."""
    state = {"i": 0}

    def call():
        x = inputs[state["i"] % len(inputs)]
        state["i"] += 1
        return fn(x)

    return call


def capture_activations(model: nn.Module, root: Path, ids: List[int]) -> Dict[str, List[torch.Tensor]]:
    """Record (on the CPU) the input reaching each target layer for every test image."""
    acts: Dict[str, List[torch.Tensor]] = {path: [] for path in TARGETS}
    hooks = [
        model.get_submodule(path).register_forward_hook(
            lambda module, inputs, output, path=path: acts[path].append(inputs[0].cpu()))
        for path in TARGETS
    ]
    for index in ids:
        x, _, _ = pair(root, index, 448)
        model(x)
    for hook in hooks:
        hook.remove()
    return acts


def layer_check(model: nn.Module, acts: Dict[str, List[torch.Tensor]], ext: Any, mk: Any, ir: Any) -> List[dict]:
    """Part 1: correctness and timing of each method on real inputs of the two layers."""
    results = []
    for path in TARGETS:
        layer = model.get_submodule(path)
        w, b = layer.weight.detach(), layer.bias.detach()
        kb = mk_kb(layer)
        methods = {
            "cudnn": lambda x: F.conv2d(x, w, b),
            "tiled": lambda x: ext.convolution(x, w, b, "tiled", 1, 0, 16, 0),
            "implicit": lambda x: ir.implicit_reg(x, w, b, 1, 0, 2),
            "implicit_auto": lambda x: ir.implicit_reg(x, w, b, 1, 0, 0),
            f"tiled_mk{kb}": lambda x: mk.tiled_mk(x, w, b, 1, 0, 16, kb),
        }
        shape = list(acts[path][0].shape)
        print(f"\nLayer {path}: input {tuple(shape)} -> {tuple(w.shape[:2])} outputs, {len(acts[path])} real inputs")

        worst = {name: 0.0 for name in methods}
        passed = {name: 0 for name in methods}
        bitwise_cudnn = {name: 0 for name in methods}
        for cpu_x in acts[path]:
            x = cpu_x.cuda()
            ref = F.conv2d(x.double(), w.double(), b.double()).float()
            base = methods["cudnn"](x)
            for name, fn in methods.items():
                out = fn(x)
                worst[name] = max(worst[name], float((out - ref).abs().max()))
                bitwise_cudnn[name] += int(torch.equal(out, base))
                try:
                    agreement(ref, out)
                    passed[name] += 1
                except AssertionError:
                    pass

        inputs = [t.cuda() for t in acts[path][:20]]
        functions = {name: cycling(fn, inputs) for name, fn in methods.items()}
        event = measure(functions, 200, 25)
        graphs = {name: graph_time(cycling(fn, inputs)) for name, fn in methods.items()}
        base_event = event["cudnn"]["cuda_event"]["median_ms"]
        print(f"  {'method':<14}{'max err vs fp64':>16}{'within tol':>12}{'event ms':>10}{'graph ms':>10}"
              f"{'event speedup':>15}{'GPU speedup':>13}")
        rows = {}
        for name in methods:
            e = event[name]["cuda_event"]["median_ms"]
            rows[name] = {"max_abs_err_vs_fp64": worst[name], "within_tolerance": passed[name],
                          "bitwise_equal_to_cudnn": bitwise_cudnn[name], "event_ms": e,
                          "graph_ms": graphs[name], "event_speedup_vs_cudnn": base_event / e,
                          "graph_speedup_vs_cudnn": graphs["cudnn"] / graphs[name]}
            print(f"  {name:<14}{worst[name]:>16.2e}{passed[name]:>8}/{len(acts[path]):<3}{e:>10.4f}"
                  f"{graphs[name]:>10.4f}{base_event / e:>14.2f}x{graphs['cudnn'] / graphs[name]:>12.2f}x")
        results.append({"layer": path, "input_shape": shape, "out_channels": int(w.shape[0]),
                        "images": len(acts[path]), "methods": rows})
    return results


def network_check(base: nn.Module, root: Path, ids: List[int], ext: Any, mk: Any, ir: Any) -> dict:
    """Part 2: whole-network quality, agreement with cuDNN, and latency."""
    models = {
        "cudnn": base,
        "implicit": replace_extension(copy.deepcopy(base),
                                      lambda o: lambda x, w, b, s: ir.implicit_reg(x, w, b, s, 0, 2)),
        "implicit_auto": replace_extension(copy.deepcopy(base),
                                           lambda o: lambda x, w, b, s: ir.implicit_reg(x, w, b, s, 0, 0)),
        "tiled_mk": replace_extension(copy.deepcopy(base),
                                      lambda o: lambda x, w, b, s, kb=mk_kb(o): mk.tiled_mk(x, w, b, s, 0, 16, kb)),
    }

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    scores = {name: [] for name in models}
    within = {name: 0 for name in models}
    worst = {name: 0.0 for name in models}
    for index in ids:
        x, target, _ = pair(root, index, 448)
        ref = models["cudnn"](x)
        for name, model in models.items():
            out = ref if name == "cudnn" else model(x)
            scores[name].append(quality(target, out[0].permute(1, 2, 0).cpu().numpy()))
            worst[name] = max(worst[name], float((out - ref).abs().max()))
            try:
                agreement(ref, out)
                within[name] += 1
            except AssertionError:
                pass

    set_timing_flags()
    x, _, _ = pair(root, ids[0], 448)
    event = measure({name: (lambda m=model: m(x)) for name, model in models.items()}, 50, 10)
    graphs = {name: graph_time(lambda m=model: m(x), calls=1, replays=50) for name, model in models.items()}

    print(f"\nWhole PyNET on {len(ids)} public test images (448x448 RAW -> sRGB), both layers replaced")
    print(f"  {'method':<14}{'PSNR dB':>9}{'SSIM':>8}{'dE2000':>8}{'max diff vs cudnn':>19}{'within tol':>12}"
          f"{'event ms':>10}{'graph ms':>10}")
    rows = {}
    for name in models:
        mean = {key: float(np.mean([s[key] for s in scores[name]])) for key in scores[name][0]}
        e = event[name]["cuda_event"]["median_ms"]
        rows[name] = {"mean_quality": mean, "max_abs_diff_vs_cudnn": worst[name], "within_tolerance": within[name],
                      "event_ms": e, "graph_ms": graphs[name]}
        print(f"  {name:<14}{mean['psnr_db']:>9.4f}{mean['ssim']:>8.4f}{mean['delta_e_2000']:>8.4f}"
              f"{worst[name]:>19.2e}{within[name]:>8}/{len(ids):<3}{e:>10.3f}{graphs[name]:>10.3f}")
    return {"images": len(ids), "methods": rows}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/zrr_test_samples")
    parser.add_argument("--count", type=int, default=200, help="Test images 0..count-1")
    parser.add_argument("--act-images", type=int, default=50, help="Images whose layer inputs are captured")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    set_timing_flags()
    ext, mk, ir = load_extensions()
    base = load_model()
    ids = list(range(args.count))

    print(f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, "
          f"TF32 off, inference_mode on")
    acts = capture_activations(base, args.data, ids[:args.act_images])
    layers = layer_check(base, acts, ext, mk, ir)
    del acts
    network = network_check(base, args.data, ids, ext, mk, ir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "real_data.json"
    path.write_text(json.dumps({"environment": environment(), "dataset": "Zurich RAW-to-RGB test split",
                                "test_ids": [ids[0], ids[-1]], "layers": layers, "network": network}, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
