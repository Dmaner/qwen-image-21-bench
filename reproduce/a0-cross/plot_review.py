#!/usr/bin/env python3
"""Render A0 review figures from the published 2026-10-10 trace summaries.

Requires matplotlib 3.9.4. Run from any directory; no GPU or model required.
SVGs go to figures/a0-review. --preview-dir also writes local PNG previews.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter


REPO = Path(__file__).resolve().parents[2]
COLORS = ["#2877d4", "#ef6335", "#009e73", "#e6a000", "#cc79a7"]
TEXT = "#17212b"
MUTED = "#52606d"


def save(fig, name, preview_dir):
    output = REPO / "figures/a0-review"
    output.mkdir(parents=True, exist_ok=True)
    svg = output / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=140)
    plt.close(fig)


def axes_style(ax):
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8, colors=MUTED)


def step_plot(data, preview_dir):
    fig = plt.figure(figsize=(12, 6.2), facecolor="white")
    fig.text(0.075, 0.932, "GPU-event duration per denoising step",
             fontsize=17, weight="bold", color=TEXT)
    fig.text(0.075, 0.885,
             "H200 BF16 · 1024×1024 · 50 steps · seed 42 · one profiled request per workload",
             fontsize=10.5, color=MUTED)
    ax = fig.add_axes([0.075, 0.185, 0.69, 0.645])
    for mode, color in zip(("t2i", "edit"), COLORS):
        steps = data[mode]["steps"]
        values = [row["gpu_busy_ms"] for row in steps["per_step"]]
        assert len(values) == steps["n"] == 50
        assert math.isclose(sum(values[1:]) / 49,
                            steps["rest_mean_gpu_busy_ms"], rel_tol=1e-10)
        ax.plot(range(50), values, color=color, linewidth=1.8)
        ax.scatter([0], values[:1], color=color, s=27,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set(xlim=(-0.6, 49.6), ylim=(110, 305),
           xticks=[0, 10, 20, 30, 40, 49], yticks=[125, 150, 200, 250, 300],
           xlabel="Denoising step", ylabel="GPU-event duration (ms)")
    ax.grid(axis="y", color="#e7ebef", linewidth=0.8)
    axes_style(ax)
    first = data["edit"]["steps"]["first"]["gpu_busy_ms"]
    ax.annotate(f"Edit step 0: {first:.1f} ms\nIncludes reference prefill and other first-step work",
                xy=(0, first), xytext=(3, 280), fontsize=10.5, color=TEXT,
                va="center", arrowprops={"arrowstyle": "-", "color": COLORS[1], "lw": 1})
    for mode, label, color, y in [
        ("edit", "Edit (1 reference)", COLORS[1], 0.39),
        ("t2i", "T2I", COLORS[0], 0.285),
    ]:
        avg = data[mode]["steps"]["rest_mean_gpu_busy_ms"]
        fig.text(0.79, y, "●", fontsize=12, color=color)
        fig.text(0.815, y, label, fontsize=11, weight="bold", color=TEXT)
        fig.text(0.815, y - 0.033, f"Steps 1–49: {avg:.1f} ms", fontsize=10.5, color=MUTED)
    fig.text(0.075, 0.052,
             "Duration sums of correlated kernel/memcpy/memset events; not wall time or GPU utilization.",
             fontsize=9.5, color=MUTED)
    save(fig, "denoising_steps", preview_dir)


def kernel_plot(data, preview_dir):
    fig = plt.figure(figsize=(12, 4.8), facecolor="white")
    fig.text(0.105, 0.925, "GPU-event duration by kernel class", fontsize=17,
             weight="bold", color=TEXT)
    fig.text(0.105, 0.865,
             "Whole profiled request · segment labels show each class's share of that request",
             fontsize=10.5, color=MUTED)
    groups = [
        ("GEMM", ("gemm",)),
        ("Elementwise / copy", ("elementwise/copy",)),
        ("Attention (FA3 + SDPA)", ("attention",)),
        ("Reduce + norm", ("reduce", "norm")),
        ("Other", None),
    ]
    fig.legend([Patch(color=c) for c in COLORS], [label for label, _ in groups],
               loc="upper left", bbox_to_anchor=(0.105, 0.825), ncol=5,
               frameon=False, fontsize=9.4, handlelength=0.8,
               handletextpad=0.5, columnspacing=1.5, borderaxespad=0)
    ax = fig.add_axes([0.105, 0.235, 0.84, 0.49])
    known = {key for _, keys in groups if keys for key in keys}
    for mode, y in (("t2i", 1), ("edit", 0)):
        kernels = data[mode]["kernels"]
        cats = kernels["by_category"]
        total = kernels["total_gpu_ms"]
        values = [sum(cats[k]["ms"] for k in keys) if keys else
                  sum(v["ms"] for k, v in cats.items() if k not in known)
                  for _, keys in groups]
        # Other includes the small editing convolution category, keeping totals exact.
        assert math.isclose(sum(values), total, rel_tol=1e-10)
        left = 0
        for value, color in zip(values, COLORS):
            ax.barh(y, value, height=0.46, left=left, color=color,
                    edgecolor="white", linewidth=1)
            fraction = value / total * 100
            if fraction >= 7:
                ax.text(left + value / 2, y, f"{fraction:.1f}%", va="center",
                        ha="center", fontsize=11, weight="bold", color="white")
            left += value
        ax.text(total + 100, y, f"{total:,.0f} ms", va="center", fontsize=10.5, color=TEXT)
    ax.set(xlim=(0, 8650), ylim=(-0.6, 1.6), xticks=[0, 2000, 4000, 6000, 8000],
           yticks=[1, 0], yticklabels=["T2I", "Edit (1 ref)"],
           xlabel="Summed GPU-event duration (ms)")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", color="#e7ebef", linewidth=0.8)
    axes_style(ax)
    fig.text(0.105, 0.053,
             "Source: 2026-10-10 A0 cross-check trace summaries. Attention includes FA3 and PyTorch SDPA.",
             fontsize=9.5, color=MUTED)
    save(fig, "kernel_classes", preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": MUTED, "axes.labelpad": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "a0-review-20261010"})
    data = {mode: json.loads((REPO / f"data/a0-cross/{mode}_trace_analysis.json").read_text())
            for mode in ("t2i", "edit")}
    step_plot(data, args.preview_dir)
    kernel_plot(data, args.preview_dir)


if __name__ == "__main__":
    main()
