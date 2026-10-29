"""Draw the extension-experiment figures from results/extensions/repeatability/summary.json.

  ext_vs_cudnn.png        speedup over cuDNN per workload, full call (event) vs GPU only (graph)
  ext_vs_original.png     GPU-only speedup of each extension kernel over the kernel it improves

Points are medians of twelve paired process ratios; whiskers are pointwise 95% process-bootstrap
intervals. Logarithmic x axes; the vertical line marks no change.

    python extensions/plot_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUTPUT, WORKLOADS, workload_label  # noqa: E402

# Validated categorical slots 1-3 (light mode, all-pairs); identity also carried by marker shape.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
TICKS = [0.25, 0.5, 1, 2, 4]


def pretty(workload) -> str:
    n, c, h, _, k, r = workload
    batch = f", N={n}" if n > 1 else ""
    return f"{c}→{k}, {h}×{h}, {r}×{r}{batch}"


def style(ax, xlabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.axvline(1, color=INK_2, linewidth=1, zorder=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def dots(ax, rows, label, slot, offset) -> None:
    color, marker = SERIES[slot]
    ys = [i + offset for i in range(len(rows))]
    xs = [r["speedup"] for r in rows]
    lo = [r["speedup"] - r["ci95"][0] for r in rows]
    hi = [r["ci95"][1] - r["speedup"] for r in rows]
    ax.errorbar(xs, ys, xerr=[lo, hi], fmt=marker, color=color, markersize=6.5, elinewidth=1.4,
                capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)


def lookup(summary, metric, family, method):
    rows = {(r["workload"], r["method"]): r for r in summary[metric][family]}
    return [rows[(workload_label(w), method)] for w in WORKLOADS]


def finish(fig, axes, title: str, path: Path, legend_ax=None) -> None:
    """Label rows, add one legend (figure-wide, or above `legend_ax` only), and save."""
    labels = [pretty(w) for w in WORKLOADS]
    axes[0].set_yticks(range(len(labels)), labels, color=INK, fontsize=8.5)
    axes[0].invert_yaxis()
    legend = dict(frameon=False, fontsize=8.5, labelcolor=INK)
    if legend_ax is not None:
        handles, names = legend_ax.get_legend_handles_labels()
        legend_ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=len(names),
                         borderaxespad=0, handletextpad=0.3, columnspacing=1.2, **legend)
    else:
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(names), **legend)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Wrote", path)


def plot_vs_cudnn(summary, path: Path) -> None:
    panels = [("implicit", "Original implicit"), ("implicit_auto", "implicit_auto (new)"),
              ("tiled_mk4", "tiled_mk, KB=4 (new)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (method, title) in zip(axes, panels):
        dots(ax, lookup(summary, "event", "vs_cudnn", method), "Full call (CUDA events, includes host dispatch)", 0, -0.14)
        dots(ax, lookup(summary, "graph", "vs_cudnn", method), "GPU only (CUDA-graph replay)", 1, 0.14)
        style(ax, "Speedup over cuDNN (right of 1 = faster)")
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    fig.subplots_adjust(top=0.78, wspace=0.08)
    finish(fig, axes, "Speedup over PyTorch/cuDNN: most of the full-call gain is host overhead", path)


def plot_vs_original(summary, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, facecolor=SURFACE)
    dots(axes[0], lookup(summary, "graph", "vs_original", "implicit_auto"), "implicit_auto", 0, 0)
    axes[0].set_title("implicit_auto vs original implicit", color=INK, fontsize=9.5, loc="left")
    for slot, (kb, offset) in enumerate([(4, -0.2), (8, 0), (16, 0.2)]):
        dots(axes[1], lookup(summary, "graph", "vs_original", f"tiled_mk{kb}"), f"tiled_mk KB={kb}", slot, offset)
    axes[1].set_title("tiled_mk vs original tiled", color=INK, fontsize=9.5, loc="left")
    for ax in axes:
        style(ax, "GPU-only speedup over the original kernel")
    axes[0].set_title("implicit_auto vs original implicit", color=INK, fontsize=9.5, loc="left", pad=26)
    axes[1].set_title("tiled_mk vs original tiled", color=INK, fontsize=9.5, loc="left", pad=26)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    finish(fig, axes, "GPU-only speedup of the extension kernels over the kernels they improve", path,
           legend_ax=axes[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads((args.output_dir / "repeatability/summary.json").read_text())
    plot_vs_cudnn(summary, args.output_dir / "ext_vs_cudnn.png")
    plot_vs_original(summary, args.output_dir / "ext_vs_original.png")


if __name__ == "__main__":
    main()
